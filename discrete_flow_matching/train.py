import os
from contextlib import nullcontext
from time import perf_counter, time

import datasets
import pytorch_lightning as pl
import torch
import torch._dynamo.cache_size
import torch.nn.functional as F
import typer
from lightning.pytorch.loggers import WandbLogger
from pyinstrument import Profiler
from pyinstrument.renderers.speedscope import SpeedscopeRenderer
from structlog import get_logger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = get_logger()

app = typer.Typer(pretty_exceptions_show_locals=False)


class FlopCounterCallback(pl.Callback):
    def __init__(self, train_step_flops: float | None = None):
        self.t_start = None
        self.device_flops_per_second = 142.5e12  # RTX 3090 142.5 TFLOPS bf16
        self.flop_counter = None
        self.trained_flops = 0
        self.trained_optimal_flops = 0
        self.train_batch_t_start = None
        self.trained_duration = 0
        self.train_step_flops = train_step_flops

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.train_step_flops is None:
            self.flop_counter = FlopCounterMode(depth=None, display=False)
        else:
            self.flop_counter = None

        self.t_start = perf_counter()

    def on_train_batch_start(self, *args, **kwargs) -> None:
        self.train_batch_t_start = perf_counter()

        if self.flop_counter is not None:
            self.flop_counter.__enter__()

    def on_train_batch_end(self, *args, **kwargs) -> None:
        if self.flop_counter is not None:
            self.flop_counter.__exit__(None, None, None)
            train_step_flops = self.flop_counter.get_total_flops()
            if self.train_step_flops is None:
                logger.info(
                    "Estimated train step flops", train_step_flops=train_step_flops
                )
            self.train_step_flops = train_step_flops
        t = perf_counter()

        # Train step flops
        self.trained_flops += self.train_step_flops

        # Optimal train step flops
        train_step_duration = t - self.train_batch_t_start
        train_step_optimal_flops = train_step_duration * self.device_flops_per_second
        self.trained_optimal_flops += train_step_optimal_flops

        # Optimal total flops
        duration = t - self.t_start
        optimal_flops = duration * self.device_flops_per_second

        # MFU
        trained_mfu = self.trained_flops / self.trained_optimal_flops
        mfu = self.trained_flops / optimal_flops

        # Trained duration fraction
        self.trained_duration += train_step_duration
        trained_duration_fraction = self.trained_duration / duration

        self.log_dict(
            {
                "train/step_pflops": self.train_step_flops / 1e15,
                "train/step_optimal_pflops": train_step_optimal_flops / 1e15,
                "train/trained_pflops": self.trained_flops / 1e15,
                "train/trained_optimal_pflops": self.trained_optimal_flops / 1e15,
                "train/trained_mfu": trained_mfu,
                "mfu": mfu,
                "train/trained_duration_fraction": trained_duration_fraction,
            }
        )


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Reshape(nn.Module):
    def __init__(self, shape: list[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)


class DiscreteFlowMatchingNet(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_timesteps: int,
        num_layers: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        self.scheduler = torch.linspace(
            1 / num_timesteps, 1, steps=num_timesteps, dtype=torch.float32
        )  # Probability path scheduler

        # x: B, L

        # Input embedding, B L -> B L C
        self.input_projection = nn.Embedding(vocab_size, hidden_dim)

        # Embed timestep to B, 1, C
        self.embed_timestep = nn.Sequential(
            nn.Embedding(num_timesteps, hidden_dim),
            Unsqueeze(1),
        )

        self.blocks = nn.ModuleList()
        self.timestep_embedding_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.blocks.append(
                nn.Sequential(
                    Transpose(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    Transpose(),
                    nn.LayerNorm([hidden_dim]),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm([hidden_dim]),
                    nn.GELU(),
                ),
            )
            self.timestep_embedding_norms.append(nn.LayerNorm([hidden_dim]))

        # Output projection, B L C -> B L V
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        self.scheduler = self.scheduler.to(self.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: B, L, V, t: B
        x = self.input_projection(x)  # BLC

        for block, timestep_embedding_norm in zip(
            self.blocks, self.timestep_embedding_norms
        ):
            x = x + block(x + timestep_embedding_norm(self.embed_timestep(t)))  # BLC

        x = self.output_projection(x)  # BLV

        return x

    def forward_noising(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mask x (BL) depending on time step t (BL)."""

        # t is the masking probability. t=0%: dont mask anything, t=100%: mask everything
        mask_prob = self.scheduler[t].expand(-1, x.shape[1])
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool)
        is_not_pad = x != self.pad_token_id

        noised_x = x.clone()
        noised_x[will_mask & is_not_pad] = self.mask_token_id

        return noised_x

    @torch._dynamo.disable
    def log_training_step(self, log_dict):
        self.log_dict(
            {
                **log_dict,
                "train/learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
        )

    def training_step(self, batch, batch_idx: int):
        # x: B L
        x = batch["input_ids"]

        # t: B
        t = torch.randint(0, len(self.scheduler), [x.size(0)], device=x.device)

        # noised_x: B L
        noised_x = self.forward_noising(x, t.unsqueeze(1))

        # Unmasking logits: B L V
        logits = self(noised_x, t)  # .to(torch.float32)

        # Only calculate loss on tokens that were masked
        target = x.clone()
        target[noised_x != self.mask_token_id] = -100

        loss = F.cross_entropy(
            # CE expects input BVL, target BL
            input=logits.transpose(-1, -2),
            target=target,
            reduction="mean",
        )
        self.log_training_step({"train/loss": loss})

        return loss

    @torch._dynamo.disable
    def log_validation_step(
        self,
        num_samples,
        input_text_tokenized,
        generated_texts_tokenized,
        noised_texts_tokenized,
        sampling_timesteps,
        losses,
    ):
        # input_text: B
        # generated_texts: T B
        # noised_texts: T B
        # sampling_timesteps: T

        input_text = self.tokenizer.batch_decode(input_text_tokenized)
        generated_texts = [
            self.tokenizer.batch_decode(t) for t in generated_texts_tokenized
        ]
        noised_texts = [self.tokenizer.batch_decode(t) for t in noised_texts_tokenized]

        self.log_dict(losses)

        num_samples = min(num_samples, len(input_text))

        if isinstance(self.logger, WandbLogger):
            for i_t, t in enumerate(sampling_timesteps):
                self.logger.log_table(
                    f"validation-texts/{t}",
                    columns=["input_text", "generated_text", "generated_text_inputs"],
                    data=[
                        [
                            input_text[i],
                            generated_texts[i_t][i],
                            noised_texts[i_t][i],
                        ]
                        for i in range(num_samples)
                    ],
                )

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.validation_step_without_compile(batch, batch_idx)

    @torch._dynamo.disable
    def validation_step_without_compile(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ):
        # x: B L
        x = batch["input_ids"]

        num_samples = 5  # Number of samples to visualize

        num_sampling_steps = 8
        sampling_timesteps = torch.flip(
            torch.linspace(
                len(self.scheduler) // num_sampling_steps,
                len(self.scheduler) - 1,
                num_sampling_steps,
                device=x.device,
                dtype=torch.int,
            ),
            [0],
        )
        # step_sizes = -torch.diff(
        #     sampling_timesteps + 1,
        #     append=torch.zeros([1], device=sampling_timesteps.device, dtype=torch.int),
        # ).to(torch.float32) / len(self.scheduler)
        # assert torch.allclose(
        #     torch.sum(step_sizes),
        #     torch.ones([1], dtype=torch.float32, device=x.device),
        # )

        losses = {}

        # Apply forward noising and reverse process
        noised_texts_tokenized = []
        generated_texts_tokenized = []

        # for t, step_size in zip(sampling_timesteps, step_sizes, strict=True):
        for t in sampling_timesteps:
            t = t.repeat(x.shape[0])
            assert t.shape == x.shape[:1], t.shape

            # B L
            noised_x = self.forward_noising(x, t.unsqueeze(1))

            # Only calculate loss on tokens that were masked
            # B L
            target = x.clone()
            target[noised_x != self.mask_token_id] = -100

            # B L V
            logits = self(noised_x, t)  # .to(torch.float32)

            # Get samples for each token
            # B L
            # samples = torch.distributions.Categorical(logits=logits).sample()
            samples = torch.argmax(logits, dim=-1)

            # Unmask the masked tokens
            # B L
            generated_tokens = noised_x.clone()
            generated_tokens[noised_x == self.mask_token_id] = samples[
                noised_x == self.mask_token_id
            ]

            generated_texts_tokenized.append(generated_tokens)
            noised_texts_tokenized.append(noised_x)

            losses[f"validation-losses/loss_{t[0]}"] = F.cross_entropy(
                input=logits.transpose(-1, -2), target=target, reduction="mean"
            )
        losses["validation/loss_mean"] = torch.mean(torch.tensor(list(losses.values())))

        self.log_validation_step(
            num_samples=num_samples,
            input_text_tokenized=x,
            generated_texts_tokenized=generated_texts_tokenized,
            noised_texts_tokenized=noised_texts_tokenized,
            sampling_timesteps=sampling_timesteps,
            losses=losses,
        )

        return losses["validation/loss_mean"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=5),
                "monitor": "train/loss",
                "interval": "step",
                "frequency": 50,
            },
        }


@app.command()
def main(
    no_compile: bool = False,
    max_steps: int = -1,
    profile: bool = False,
    train_step_flops: str = "",
):
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    # Load dataset
    def load_split(split):
        dataset = datasets.load_dataset("roneneldan/TinyStories", split=split)
        dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count())
        dataset.set_format(type="torch", columns=["input_ids"])
        return dataset

    logger.info("Loading datasets")
    train_data = load_split("train")
    val_data = load_split("validation")

    # Dataloader
    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_data, batch_size=256, shuffle=True, num_workers=2, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_data, batch_size=64, shuffle=False, num_workers=2, prefetch_factor=2
    )

    # Create model
    logger.info("Creating model")
    model = DiscreteFlowMatchingNet(
        vocab_size=len(tokenizer),  # .vocab_size excludes the new tokens
        hidden_dim=1024,
        num_timesteps=1024,
        num_layers=6,
        tokenizer=tokenizer,
    ).to(dtype=torch.bfloat16)

    if not no_compile:
        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.fx_graph_cache = True
        model = torch.compile(model)

    # Train model
    train_step_flops = float(train_step_flops) if train_step_flops else None
    if train_step_flops is not None:
        logger.info("Using train step flops", train_step_flops=train_step_flops)
    else:
        logger.info("Using dynamic train step flops")

    trainer = pl.Trainer(
        max_epochs=-1,
        max_steps=max_steps,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else 0,
        limit_val_batches=1,
        callbacks=[
            FlopCounterCallback(train_step_flops=train_step_flops),
        ],
        logger=WandbLogger(project="flow-matching-tiny-stories"),
        precision="bf16",
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.1,
        check_val_every_n_epoch=None,
        val_check_interval=200,
    )

    fit_context = nullcontext() if not profile else Profiler(async_mode="disabled")
    with fit_context:
        trainer.fit(model, train_loader, val_loader)

    if profile:
        with open(f"speedscope-{time()}.json", "w") as f:
            f.write(fit_context.output(SpeedscopeRenderer()))


if __name__ == "__main__":
    app()
