from typing import Literal

import lightning.pytorch as pl
import torch
import torch._dynamo.cache_size
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from transformers import PreTrainedTokenizerBase

from discrete_flow_matching_pytorch.model import ConvNet


def get_timestep_step_sizes(timesteps: torch.Tensor) -> torch.Tensor:
    return -torch.diff(
        timesteps,
        append=torch.zeros([1], device=timesteps.device, dtype=timesteps.dtype),
    )


class DiscreteFlowMatchingNet(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_timesteps: int,
        num_layers: int,
        tokenizer: PreTrainedTokenizerBase,
        val_num_sampling_steps: int = 8,
        scheduler_type: Literal["linear", "square"] = "square",
        learning_rate: float = 1e-2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.val_num_sampling_steps = val_num_sampling_steps
        self.learning_rate = learning_rate

        self.scheduler = torch.linspace(
            1 / num_timesteps, 1, steps=num_timesteps, dtype=torch.float32
        )  # Probability path scheduler

        match scheduler_type:
            case "linear":
                pass
            case "square":
                # Put more weight on higher (=more noisy) timesteps.
                # Examples:
                # 0 -> 0 (no noise)
                # 0.5 -> 0.75 (50% noise moved to 75% noise)
                # 1 -> 1 (all noise)
                self.scheduler = 1 - torch.square(1 - self.scheduler)
            case _:
                raise ValueError(f"Invalid scheduler type: {scheduler_type}")

        self.model = ConvNet(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_timesteps=num_timesteps,
            num_layers=num_layers,
        )

        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        self.scheduler = self.scheduler.to(self.device)

        print("Setting learning rate to", self.learning_rate)
        opt = self.optimizers(False)
        assert isinstance(opt, torch.optim.Optimizer)
        opt.param_groups[0]["lr"] = self.learning_rate

    def on_validation_model_eval(self) -> None:
        self.scheduler = self.scheduler.to(self.device)
        return super().on_validation_model_eval()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: B, L, V
        # t: B
        # model: B, L, V
        return self.model(x, t)

    def forward_noising(
        self, x: torch.Tensor, t: torch.Tensor, should_noise: torch.Tensor | None
    ) -> torch.Tensor:
        """Mask x (BL) depending on time step t (BL)."""

        # t is the masking probability. t=0%: dont mask anything, t=100%: mask everything
        mask_prob = self.scheduler[t].expand(-1, x.shape[1])
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool)

        # Don't mask padding tokens
        will_mask &= x != self.pad_token_id

        # Don't mask tokens that should not be noised
        if should_noise is not None:
            will_mask &= should_noise

        noised_x = x.clone()
        noised_x[will_mask] = self.mask_token_id

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
        # B L
        x: torch.Tensor = batch["input_ids"]
        should_noise: torch.Tensor | None = batch.get("should_noise")

        # t: B
        t = torch.randint(0, len(self.scheduler), [x.size(0)], device=x.device)

        # noised_x: B L
        noised_x = self.forward_noising(
            x=x, t=t.unsqueeze(1), should_noise=should_noise
        )

        # Unmasking logits: B L V
        logits = self(noised_x, t)  # .to(torch.float32)

        target = x.clone()
        # Only calculate loss on tokens that were masked
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

    def _get_sampling_timesteps(self, num_sampling_steps):
        return torch.linspace(
            len(self.scheduler) - 1,
            len(self.scheduler) // num_sampling_steps,
            num_sampling_steps,
            device=self.device,
            dtype=torch.long,
        )

    @torch._dynamo.disable
    def validation_step_without_compile(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ):
        # x: B L
        x = batch["input_ids"]
        should_noise = batch.get("should_noise")

        num_samples = 5  # Number of samples to visualize

        sampling_timesteps = self._get_sampling_timesteps(self.val_num_sampling_steps)

        losses = {}

        # Apply forward noising and reverse process
        noised_texts_tokenized = []
        generated_texts_tokenized = []

        # for t, step_size in zip(sampling_timesteps, step_sizes, strict=True):
        for t in sampling_timesteps:
            t = t.repeat(x.shape[0])
            assert t.shape == x.shape[:1], t.shape

            # B L
            noised_x = self.forward_noising(
                x, t.unsqueeze(1), should_noise=should_noise
            )

            # Only calculate loss on tokens that were masked
            # B L
            target = x.clone()
            # Only calculate loss on tokens that were masked
            target[noised_x != self.mask_token_id] = -100

            # B L V
            logits = self(noised_x, t)

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

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.validation_step_without_compile(batch, batch_idx)

    def sample(
        self,
        num_sampling_steps: int,
        num_samples: int | None = None,
        sequence_length: int | None = None,
        x: torch.Tensor | None = None,
        stochasticity: float = 0.0,
        yield_intermediate: bool = False,
        yield_logits: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
    ):
        assert (
            num_samples is not None and sequence_length is not None
        ) or x is not None, "Must pass either (num_samples and sequence_length) or x"

        assert not (
            yield_intermediate and yield_logits
        ), "Can't yield both logits and intermediate results"

        # B L
        if x is None:
            # Start fully masked
            x = torch.full(
                [num_samples, sequence_length],
                fill_value=self.tokenizer.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            should_noise = None
        else:
            should_noise = x == self.mask_token_id

        # Create the integer timesteps and step sizes for the given num_sampling_steps
        # S
        sampling_timesteps = self._get_sampling_timesteps(num_sampling_steps)
        relative_ts = self.scheduler[sampling_timesteps]
        relative_dts = get_timestep_step_sizes(relative_ts)

        for t, relative_t, relative_dt in zip(
            sampling_timesteps, relative_ts, relative_dts
        ):
            is_last_step = t == sampling_timesteps[-1]
            if yield_intermediate:
                yield t, x

            # B
            t = t.repeat(x.shape[0])
            assert t.shape == x.shape[:1], t.shape

            # B L V
            logits = self(x, t)

            if cfg_scale != 1.0:
                assert should_noise is not None
                x_uncond = x.clone()
                x_uncond[~should_noise] = self.mask_token_id

                # Classifier-free guidance
                # Run model unconditionally (conditioning fully masked)
                logits_uncond = self(x_uncond, t)

                # Mix the logits according to cfg_scale
                logits = logits_uncond + cfg_scale * (logits - logits_uncond)

            if yield_logits:
                yield t, logits

            # B L
            samples = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()

            # B L
            # Chance to unmask proportional to
            # - step size: higher step size means higher chance
            # - timestep: lower timestep means higher chance (so in the end the chance is 100%)
            unmask_threshold = relative_dt / relative_t

            # With remasking, the unmasking probability is changed
            if stochasticity != 0:
                unmask_threshold *= 1 + stochasticity * (1 - relative_t)

            was_masked = x == self.mask_token_id

            # Unmask
            will_unmask = (
                torch.rand(
                    x.shape[:2],
                    device=unmask_threshold.device,
                    dtype=unmask_threshold.dtype,
                )
                < unmask_threshold
            )
            # Only unmask the tokens that were masked
            will_unmask &= was_masked

            # Remask when stochasticity is non-zero
            if stochasticity != 0 and not is_last_step:
                remask_threshold = relative_dt * stochasticity
                will_remask = (
                    torch.rand(
                        x.shape[:2],
                        device=unmask_threshold.device,
                        dtype=unmask_threshold.dtype,
                    )
                    < remask_threshold
                )
                # Only remask the tokens that were unmasked
                will_remask &= ~was_masked

                # Only remask tokens that aren't constant
                if should_noise is not None:
                    will_remask &= should_noise

                x[will_remask] = self.mask_token_id

            # B L
            x[will_unmask] = samples[will_unmask]

        if yield_intermediate:
            yield torch.zeros_like(t), x
        else:
            return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
