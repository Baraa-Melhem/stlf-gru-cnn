"""The forecaster abstraction the backtester and CLIs are written against."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np

from stlf.config import ExperimentConfig
from stlf.data.splits import FoldSplit
from stlf.features.engineering import PreparedData
from stlf.training.engine import TrainingHistory


class Forecaster(ABC):
    """Fit on the label periods of a :class:`FoldSplit`, then forecast arbitrary origins.

    Implementations own their preprocessing (scaling, feature design) and must derive every
    statistic from rows up to ``split.train_end`` only. Predictions are always returned in MW
    with shape ``[n_origins, horizon, n_quantiles]`` and non-decreasing along the last axis.
    """

    probabilistic: ClassVar[bool] = True
    persistable: ClassVar[bool] = False

    def __init__(self, cfg: ExperimentConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed

    @property
    def quantiles(self) -> tuple[float, ...]:
        return self.cfg.model.quantiles

    @property
    def required_history(self) -> int:
        """Rows needed at and before an origin to build its inputs."""
        return self.cfg.window.lookback

    @abstractmethod
    def fit(self, data: PreparedData, split: FoldSplit) -> TrainingHistory | None:
        """Train on labels up to ``split.train_end``; ``split.val_*`` is for early stopping."""

    @abstractmethod
    def predict(self, data: PreparedData, origins: np.ndarray) -> np.ndarray:
        """Forecast ``[len(origins), horizon, n_quantiles]`` in MW."""

    def save(self, directory: Path) -> None:
        """Persist everything :meth:`load` needs (only for ``persistable`` forecasters)."""
        raise TypeError(f"{type(self).__name__} does not produce a deployable artifact")

    @classmethod
    def load(cls, directory: Path, cfg: ExperimentConfig) -> Forecaster:
        """Restore a forecaster written by :meth:`save`."""
        raise TypeError(f"{cls.__name__} does not produce a deployable artifact")


def target_positions(origins: np.ndarray, horizon: int) -> np.ndarray:
    """``[N, H]`` integer positions of the targets of each origin."""
    return np.asarray(origins)[:, None] + np.arange(1, horizon + 1)[None, :]


def conformal_margin(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float
) -> float:
    """Conformalized quantile regression margin (Romano, Patterson & Candès, 2019).

    With conformity scores ``E = max(lower - y, y - upper)`` on held-out data, widening (or,
    when negative, narrowing) both interval ends by the ``nominal·(1 + 1/n)`` empirical quantile
    of ``E`` restores the target coverage that over-confident quantile heads under-shoot.
    """
    y, lo, hi = (np.asarray(a, dtype=np.float64).reshape(-1) for a in (y_true, lower, upper))
    keep = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    scores = np.maximum(lo[keep] - y[keep], y[keep] - hi[keep])
    if scores.size == 0:
        return 0.0
    level = min(1.0, nominal * (1.0 + 1.0 / scores.size))
    return float(np.quantile(scores, level, method="higher"))


def apply_margin(forecast: np.ndarray, margin: float) -> np.ndarray:
    """Shift the outermost quantiles of ``[..., Q]`` forecasts outward by ``margin``."""
    if forecast.shape[-1] < 3 or margin == 0.0:
        return forecast
    adjusted = forecast.copy()
    adjusted[..., 0] -= margin
    adjusted[..., -1] += margin
    return np.sort(adjusted, axis=-1)
