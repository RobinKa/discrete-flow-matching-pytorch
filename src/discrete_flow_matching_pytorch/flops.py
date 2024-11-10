from time import perf_counter

import lightning.pytorch as pl
from structlog import get_logger
from torch.utils.flop_counter import FlopCounterMode

logger = get_logger()


class FlopCounterCallback(pl.Callback):
    def __init__(self, train_step_flops: float | None = None):
        # Constants
        self.device_flops_per_second = 142.5e12  # RTX 3090 142.5 TFLOPS bf16
        self.train_step_flops = train_step_flops

        # Needed for calculating durations
        self.t_start = None
        self.previous_t_train_batch_end = None
        self.train_batch_t_start = None

        # Dynamic flop counter
        self.flop_counter = None

        # State
        self.trained_flops = 0
        self.trained_optimal_flops = 0
        self.trained_duration = 0
        self.duration = 0

    def load_state_dict(self, state_dict):
        self.trained_flops = state_dict.get("trained_flops", 0)
        self.trained_optimal_flops = state_dict.get("trained_optimal_flops", 0)
        self.trained_duration = state_dict.get("trained_duration", 0)
        self.duration = state_dict.get("duration", 0)

    def state_dict(self):
        return dict(
            trained_flops=self.trained_flops,
            trained_optimal_flops=self.trained_optimal_flops,
            trained_duration=self.trained_duration,
            duration=self.duration,
        )

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

        # Accumulate total runtime
        if self.previous_t_train_batch_end is not None:
            end_to_end_time = t - self.previous_t_train_batch_end
            self.duration += end_to_end_time
            end_to_start_time = (
                self.train_batch_t_start - self.previous_t_train_batch_end
            )
        else:
            self.duration += t - self.t_start
            end_to_end_time = None
            end_to_start_time = None

        self.previous_t_train_batch_end = t

        # Optimal total flops
        optimal_flops = self.duration * self.device_flops_per_second

        # MFU
        trained_mfu = self.trained_flops / self.trained_optimal_flops
        mfu = self.trained_flops / optimal_flops

        # Trained duration fraction
        self.trained_duration += train_step_duration
        trained_duration_fraction = self.trained_duration / self.duration

        self.log_dict(
            {
                "flops/train_step_pflops": self.train_step_flops / 1e15,
                "flops/train_step_optimal_pflops": train_step_optimal_flops / 1e15,
                "flops/trained_pflops": self.trained_flops / 1e15,
                "flops/trained_optimal_pflops": self.trained_optimal_flops / 1e15,
                "flops/trained_mfu": trained_mfu,
                "flops/trained_duration_fraction": trained_duration_fraction,
                "flops/train_duration": self.trained_duration,
                "flops/duration": self.duration,
                "flops/mfu": mfu,
                "flops/train_start_to_end_time": train_step_duration,
                **(
                    {
                        "flops/train_end_to_start_time": end_to_start_time,
                        "flops/train_end_to_end_time": end_to_end_time,
                    }
                    if end_to_start_time is not None and end_to_end_time is not None
                    else {}
                ),
            }
        )

    def on_validation_batch_end(self, *args, **kwargs) -> None:
        if self.train_step_flops is not None:
            assert self.trained_flops is not None
            assert self.trained_optimal_flops is not None
            self.log_dict(
                {
                    "flops/train_step_pflops": self.train_step_flops / 1e15,
                    "flops/trained_pflops": self.trained_flops / 1e15,
                    "flops/trained_optimal_pflops": self.trained_optimal_flops / 1e15,
                    "flops/train_duration": self.trained_duration,
                    "flops/duration": self.duration,
                }
            )
