"""Reference forecasts every learned model must beat."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from stlf.config import ExperimentConfig
from stlf.data.loader import load_operator_forecast
from stlf.data.splits import FoldSplit
from stlf.features.engineering import PreparedData
from stlf.forecasters.base import Forecaster, target_positions


class SeasonalNaiveForecaster(Forecaster):
    """``ŷ(t) = y(t - s)``: the same hour one season ago (``s = 168`` → last week).

    Prediction intervals come from empirical quantiles of the seasonal differences observed in
    the training period, so the baseline is probabilistic too and its interval scores are
    directly comparable.
    """

    def __init__(self, cfg: ExperimentConfig, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        self.offsets: np.ndarray | None = None

    @property
    def required_history(self) -> int:
        return self.cfg.model.seasonal_period

    def fit(self, data: PreparedData, split: FoldSplit) -> None:
        season = self.cfg.model.seasonal_period
        load = data.frame[data.spec.target].loc[: split.train_end].to_numpy(dtype=np.float64)
        differences = load[season:] - load[:-season]
        offsets = np.quantile(differences[~np.isnan(differences)], self.quantiles)
        offsets[self.quantiles.index(0.5)] = 0.0  # the point forecast stays the pure naive
        self.offsets = offsets
        return None

    def predict(self, data: PreparedData, origins: np.ndarray) -> np.ndarray:
        if self.offsets is None:
            raise RuntimeError("forecaster is not fitted")
        load = data.frame[data.spec.target].to_numpy(dtype=np.float64)
        base = load[
            target_positions(origins, self.cfg.window.horizon) - self.cfg.model.seasonal_period
        ]
        return np.sort(base[..., None] + self.offsets, axis=-1)


class OperatorForecaster(Forecaster):
    """The grid operator's (CND Panama) official weekly pre-dispatch forecast.

    It is a *week-ahead* forecast made with weather forecasts, published before each week,
    so it is an industry reference point rather than a like-for-like day-ahead competitor.
    """

    probabilistic: ClassVar[bool] = False

    def __init__(self, cfg: ExperimentConfig, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        self.series: pd.Series | None = None

    @property
    def required_history(self) -> int:
        return 1

    def fit(self, data: PreparedData, split: FoldSplit) -> None:
        path = f"{self.cfg.data.raw_dir}/{self.cfg.data.operator_forecast_file}"
        self.series = load_operator_forecast(path)
        return None

    def predict(self, data: PreparedData, origins: np.ndarray) -> np.ndarray:
        if self.series is None:
            raise RuntimeError("forecaster is not fitted")
        timestamps = data.index[target_positions(origins, self.cfg.window.horizon).reshape(-1)]
        values = self.series.reindex(timestamps).to_numpy(dtype=np.float64)
        values = values.reshape(len(origins), self.cfg.window.horizon, 1)
        return np.repeat(values, len(self.quantiles), axis=-1)
