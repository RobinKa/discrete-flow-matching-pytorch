from contextlib import nullcontext
from time import time

import lightning.pytorch as pl
import torch
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pyinstrument import Profiler
from pyinstrument.renderers.speedscope import SpeedscopeRenderer
from structlog import get_logger
from torch.utils.data import DataLoader

from discrete_flow_matching_pytorch.data import (
    get_default_tokenizer,
    load_dataset_by_name,
)
from discrete_flow_matching_pytorch.flops import FlopCounterCallback
from discrete_flow_matching_pytorch.model import DiscreteFlowMatchingNet

logger = get_logger()

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_run_name(
    dataset: str,
    train_batch_size: int,
    hidden_dim: int,
    num_layers: int,
    scheduler_type: str,
):
    return f"{dataset}-bs={train_batch_size}-h={hidden_dim}-l={num_layers}-s={scheduler_type}"


@app.command()
def main(
    compile: bool = True,
    wandb: bool = True,
    max_steps: int = -1,
    profile: bool = False,
    train_step_flops: str = "",
    ckpt_path: str = "",
    val_interval: int = 500,
    checkpoint_interval: int = 1_000,
    dataset: str = "tiny_stories",
    train_batch_size: int = 256,
    shuffle_train: bool = True,  # Needs to be false for IterableDataset
    train_workers: int = 2,
    val_split_name: str = "validation",  # Some datasets don't have validation, but we can still use train
    hidden_dim: int = 768,
    num_layers: int = 6,
    scheduler_type: str = "square",
):
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = get_default_tokenizer()

    logger.info("Loading dataset", dataset=dataset)
    train_data = load_dataset_by_name(
        dataset=dataset, tokenizer=tokenizer, split="train"
    )
    val_data = load_dataset_by_name(
        dataset=dataset, tokenizer=tokenizer, split=val_split_name
    )

    # Dataloader
    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        num_workers=train_workers,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_data, batch_size=64, shuffle=False, num_workers=1, prefetch_factor=2
    )

    # Create model
    logger.info("Creating model")
    model = DiscreteFlowMatchingNet(
        vocab_size=len(tokenizer),  # .vocab_size excludes the new tokens
        hidden_dim=hidden_dim,
        num_timesteps=1024,
        num_layers=num_layers,
        tokenizer=tokenizer,
        scheduler_type=scheduler_type,
    ).to(dtype=torch.bfloat16)

    if compile:
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
        logger=WandbLogger(
            project="discrete-flow-matching",
            name=get_run_name(
                dataset=dataset,
                train_batch_size=train_batch_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                scheduler_type=scheduler_type,
            ),
        )
        if wandb
        else None,
        precision="bf16",
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.1,
        check_val_every_n_epoch=None,
        val_check_interval=val_interval,
        log_every_n_steps=25,
        num_sanity_val_steps=0,
    )

    fit_context = nullcontext() if not profile else Profiler(async_mode="disabled")
    with fit_context:
        # Run validation before training to get the initial loss
        trainer.validate(model=model, dataloaders=val_loader)

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
