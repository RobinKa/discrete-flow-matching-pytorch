from time import perf_counter

import lightning.pytorch as pl
from structlog import get_logger
from torch.utils.flop_counter import FlopCounterMode

logger = get_logger()


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
