import os
from contextlib import nullcontext
from time import time

import datasets
import lightning.pytorch as pl
import torch
import torch._dynamo.cache_size
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pyinstrument import Profiler
from pyinstrument.renderers.speedscope import SpeedscopeRenderer
from structlog import get_logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from discrete_flow_matching_pytorch.flops import FlopCounterCallback
from discrete_flow_matching_pytorch.model import DiscreteFlowMatchingNet

logger = get_logger()

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    no_compile: bool = False,
    max_steps: int = -1,
    profile: bool = False,
    train_step_flops: str = "",
    ckpt_path: str = "",
    val_interval: int = 1_000,
    checkpoint_interval: int = 1_000,
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
        val_data, batch_size=64, shuffle=False, num_workers=1, prefetch_factor=2
    )

    # Create model
    logger.info("Creating model")
    model = DiscreteFlowMatchingNet(
        vocab_size=len(tokenizer),  # .vocab_size excludes the new tokens
        hidden_dim=768,
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
            ModelCheckpoint(every_n_train_steps=checkpoint_interval, save_top_k=-1),
        ],
        logger=WandbLogger(project="flow-matching-tiny-stories"),
        precision="bf16",
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.1,
        check_val_every_n_epoch=None,
        val_check_interval=val_interval,
    )

    fit_context = nullcontext() if not profile else Profiler(async_mode="disabled")
    with fit_context:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path if ckpt_path else None,
        )

    if profile:
        with open(f"speedscope-{time()}.json", "w") as f:
            f.write(fit_context.output(SpeedscopeRenderer()))


if __name__ == "__main__":
    app()
