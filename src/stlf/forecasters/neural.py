"""Neural forecaster: scaling, windowing, model construction, training and calibration glue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from stlf.config import ExperimentConfig
from stlf.data.splits import FoldSplit
from stlf.data.windowing import WindowBatcher, WindowTensors, build_tensors, select_origins
from stlf.features.engineering import PreparedData
from stlf.features.scaling import StandardScaler
from stlf.forecasters.base import (
    Forecaster,
    apply_margin,
    conformal_margin,
    target_positions,
)
from stlf.logging_utils import get_logger
from stlf.models import build_network, count_parameters
from stlf.reproducibility import resolve_device, seed_everything
from stlf.training.engine import EpochCallback, Trainer, TrainingHistory
from stlf.training.losses import build_loss

logger = get_logger(__name__)

_WEIGHTS = "model.pt"
_STATE = "state.json"
_PREDICT_BATCH = 1024


class NeuralForecaster(Forecaster):
    """Wraps :class:`~stlf.models.HybridGRUCNN` or the legacy network behind ``Forecaster``."""

    persistable: ClassVar[bool] = True

    def __init__(self, cfg: ExperimentConfig, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        self.device = resolve_device(cfg.train.device)
        self.scaler: StandardScaler | None = None
        self.network: torch.nn.Module | None = None
        self.input_widths: tuple[int, int] = (0, 0)
        self.margin = 0.0
        self.callbacks: list[EpochCallback] = []

    @property
    def probabilistic(self) -> bool:  # type: ignore[override]
        return len(self.quantiles) > 1

    @property
    def required_history(self) -> int:
        return max(self.cfg.window.lookback, *self.cfg.features.target_lags, 1)

    def _tensors(self, data: PreparedData) -> WindowTensors:
        if self.scaler is None:
            raise RuntimeError("forecaster is not fitted")
        return build_tensors(self.scaler.transform(data.frame), data.spec, self.device)

    def _batcher(
        self,
        tensors: WindowTensors,
        origins: np.ndarray,
        batch_size: int = _PREDICT_BATCH,
        shuffle: bool = False,
    ) -> WindowBatcher:
        return WindowBatcher(
            tensors,
            origins,
            lookback=self.cfg.window.lookback,
            horizon=self.cfg.window.horizon,
            batch_size=batch_size,
            target_lags=self.cfg.features.target_lags,
            shuffle=shuffle,
            seed=self.seed,
        )

    def _trainer(self) -> Trainer:
        return Trainer(self.cfg.train, build_loss(self.cfg.train.loss, self.quantiles), self.device)

    def fit(self, data: PreparedData, split: FoldSplit) -> TrainingHistory:
        seed_everything(self.seed, self.cfg.train.deterministic)
        window = self.cfg.window
        self.scaler = StandardScaler.fit(data.frame.loc[: split.train_end], data.spec.scaled)
        tensors = self._tensors(data)

        common = {"lookback": self.required_history, "horizon": window.horizon}
        train_origins = select_origins(
            data.index, **common, last_target=split.train_end, stride=window.train_stride
        )
        val_origins = select_origins(
            data.index, **common, first_target=split.val_start, last_target=split.val_end
        )
        batch_size = self.cfg.train.batch_size
        train = self._batcher(tensors, train_origins, batch_size, shuffle=True)
        val = self._batcher(tensors, val_origins, batch_size)

        self.input_widths = (tensors.n_past, tensors.n_known)
        self.network = build_network(self.cfg, n_past=tensors.n_past, n_known=tensors.n_known)
        logger.info(
            "%s | seed %d | %s params | %d train / %d val origins | train labels <= %s",
            self.cfg.model.name,
            self.seed,
            f"{count_parameters(self.network):,}",
            len(train_origins),
            len(val_origins),
            split.train_end,
        )
        history = self._trainer().fit(self.network, train, val, self.callbacks)

        self.margin = 0.0
        if self.cfg.model.conformal and self.probabilistic:
            forecast = self.predict(data, val_origins)
            actual = data.frame[data.spec.target].to_numpy()[
                target_positions(val_origins, window.horizon)
            ]
            nominal = self.quantiles[-1] - self.quantiles[0]
            self.margin = conformal_margin(actual, forecast[..., 0], forecast[..., -1], nominal)
            logger.info("conformal interval margin: %+.1f MW", self.margin)
        return history

    def predict(self, data: PreparedData, origins: np.ndarray) -> np.ndarray:
        if self.network is None or self.scaler is None:
            raise RuntimeError("forecaster is not fitted")
        scaled = self._trainer().predict(self.network, self._batcher(self._tensors(data), origins))
        megawatts = self.scaler.inverse(scaled, data.spec.target)
        # Monotone rearrangement removes any quantile crossing, then the conformal margin.
        return apply_margin(np.sort(megawatts, axis=-1), self.margin)

    @torch.no_grad()
    def attention_map(self, data: PreparedData, origins: np.ndarray) -> np.ndarray:
        """Head-averaged cross-attention ``[N, H, L]`` of the hybrid network (diagnostics)."""
        if self.network is None or not getattr(self.network, "use_attention", False):
            raise RuntimeError("attention maps need a fitted gru_cnn model with attention")
        self.network.eval()
        maps = []
        for batch in self._batcher(self._tensors(data), origins):
            _, weights = self.network(
                batch["past"],
                batch["future"],
                batch["holiday_past"],
                batch["holiday_future"],
                batch["future_lags"],
                return_attention=True,
            )
            maps.append(weights.float().cpu())
        return torch.cat(maps).numpy()

    def save(self, directory: Path) -> None:
        if self.network is None or self.scaler is None:
            raise RuntimeError("cannot save an unfitted forecaster")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), directory / _WEIGHTS)
        n_past, n_known = self.input_widths
        state = {
            "seed": self.seed,
            "n_past": n_past,
            "n_known": n_known,
            "conformal_margin": self.margin,
            "scaler": self.scaler.to_dict(),
        }
        (directory / _STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, cfg: ExperimentConfig) -> NeuralForecaster:
        state = json.loads((directory / _STATE).read_text(encoding="utf-8"))
        forecaster = cls(cfg, seed=int(state["seed"]))
        forecaster.scaler = StandardScaler.from_dict(state["scaler"])
        forecaster.input_widths = (int(state["n_past"]), int(state["n_known"]))
        forecaster.margin = float(state["conformal_margin"])
        weights = torch.load(
            directory / _WEIGHTS, map_location=forecaster.device, weights_only=True
        )
        n_past, n_known = forecaster.input_widths
        forecaster.network = build_network(cfg, n_past=n_past, n_known=n_known)
        forecaster.network.load_state_dict(weights)
        forecaster.network.to(forecaster.device).eval()
        return forecaster
