"""Model-agnostic training loop: AdamW, warm-up + cosine LR, AMP, clipping, early stopping."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from stlf.config import TrainConfig
from stlf.data.windowing import WindowBatcher
from stlf.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EpochRecord:
    """Metrics logged at the end of one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    seconds: float


@dataclass
class TrainingHistory:
    """Per-epoch log plus the epoch whose weights were kept."""

    records: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.records])


EpochCallback = Callable[[EpochRecord], None]


class Trainer:
    """Fit any network called as ``model(past, future, holiday_past, holiday_future, lags)``."""

    def __init__(self, cfg: TrainConfig, loss_fn: nn.Module, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.loss_fn = loss_fn.to(device)
        self.use_amp = cfg.amp and device.type == "cuda"

    @staticmethod
    def _forward(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return model(
            batch["past"],
            batch["future"],
            batch["holiday_past"],
            batch["holiday_future"],
            batch["future_lags"],
        )

    def _schedule(self, total_steps: int, steps_per_epoch: int) -> Callable[[int], float]:
        """Linear warm-up then cosine decay to 1 % of the peak rate (or a constant rate)."""
        warmup = max(1, int(self.cfg.warmup_epochs * steps_per_epoch))

        def factor(step: int) -> float:
            if self.cfg.schedule == "constant":
                return 1.0
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return factor

    def fit(
        self,
        model: nn.Module,
        train: WindowBatcher,
        val: WindowBatcher,
        callbacks: Sequence[EpochCallback] = (),
    ) -> TrainingHistory:
        """Train until ``max_epochs`` or ``patience`` epochs without validation improvement.

        The weights of the best validation epoch are restored before returning. Validation
        data is a held-out tail of the training period — never the test fold — so early
        stopping cannot peek at the numbers that are finally reported. With ``ema_decay > 0``
        an exponential moving average of the weights is what gets validated and kept: it
        smooths the epoch-to-epoch noise of a small, quickly over-fitting network.
        """
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            fused=self.device.type == "cuda",
        )
        total_steps = self.cfg.max_epochs * len(train)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._schedule(total_steps, len(train))
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        ema = (
            AveragedModel(
                model, multi_avg_fn=get_ema_multi_avg_fn(self.cfg.ema_decay), use_buffers=True
            )
            if self.cfg.ema_decay > 0
            else None
        )
        evaluated = ema.module if ema is not None else model
        history = TrainingHistory()
        best_state = copy.deepcopy(evaluated.state_dict())
        stale_epochs = 0

        for epoch in range(1, self.cfg.max_epochs + 1):
            started = time.perf_counter()
            model.train()
            running, batches = 0.0, 0
            for batch in train:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                    loss = self.loss_fn(self._forward(model, batch).float(), batch["target"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if ema is not None:
                    ema.update_parameters(model)
                running += float(loss.detach())
                batches += 1

            val_loss = self.evaluate(evaluated, val)
            record = EpochRecord(
                epoch=epoch,
                train_loss=running / max(1, batches),
                val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - started,
            )
            history.records.append(record)
            improved = val_loss < history.best_val_loss - 1e-6
            if improved:
                history.best_val_loss, history.best_epoch = val_loss, epoch
                best_state = copy.deepcopy(evaluated.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            logger.info(
                "epoch %3d | train %.4f | val %.4f%s | lr %.2e | %.1fs",
                epoch,
                record.train_loss,
                val_loss,
                " *" if improved else "  ",
                record.learning_rate,
                record.seconds,
            )
            for callback in callbacks:
                callback(record)
            if stale_epochs >= self.cfg.patience:
                logger.info("early stop: no improvement for %d epochs", self.cfg.patience)
                break

        model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def evaluate(self, model: nn.Module, batcher: WindowBatcher) -> float:
        """Mean loss over ``batcher`` (weighted by batch size)."""
        model.eval()
        total, count = 0.0, 0
        for batch in batcher:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                prediction = self._forward(model, batch).float()
            size = batch["target"].shape[0]
            total += float(self.loss_fn(prediction, batch["target"])) * size
            count += size
        return total / max(1, count)

    @torch.no_grad()
    def predict(self, model: nn.Module, batcher: WindowBatcher) -> np.ndarray:
        """Stack predictions for every origin of a non-shuffled batcher: ``[N, H, Q]``."""
        model.eval()
        outputs = []
        for batch in batcher:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                outputs.append(self._forward(model, batch).float().cpu())
        return torch.cat(outputs).numpy()
