"""Walk-forward backtesting of any :class:`~stlf.forecasters.Forecaster`."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stlf.config import ExperimentConfig
from stlf.data.splits import FoldSplit, make_walk_forward_folds
from stlf.data.windowing import select_origins
from stlf.features.engineering import PreparedData
from stlf.forecasters import Forecaster, build_forecaster
from stlf.forecasters.base import target_positions
from stlf.logging_utils import get_logger
from stlf.training.engine import TrainingHistory

logger = get_logger(__name__)


def quantile_column(q: float) -> str:
    """Column name of a quantile forecast, e.g. ``q0.50``."""
    return f"q{q:.2f}"


@dataclass
class BacktestResult:
    """Long-format predictions of every fold and seed, plus training logs."""

    name: str
    quantiles: tuple[float, ...]
    probabilistic: bool
    predictions: pd.DataFrame
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    fit_seconds: dict[str, float] = field(default_factory=dict)


ForecasterFactory = Callable[[ExperimentConfig, int], Forecaster]


def _frame(
    data: PreparedData,
    origins: np.ndarray,
    forecast: np.ndarray,
    quantiles: Sequence[float],
    fold: str,
    seed: int,
) -> pd.DataFrame:
    n_origins, horizon, _ = forecast.shape
    targets = target_positions(origins, horizon).reshape(-1)
    frame = pd.DataFrame(
        {
            "origin": np.repeat(data.index[origins], horizon),
            "target_time": data.index[targets],
            "horizon": np.tile(np.arange(1, horizon + 1), n_origins),
            "fold": fold,
            "seed": seed,
            "actual": data.actual.to_numpy()[targets],
            "anomaly": data.anomaly.to_numpy()[targets],
        }
    )
    for k, q in enumerate(quantiles):
        frame[quantile_column(q)] = forecast[..., k].reshape(-1)
    return frame


def run_backtest(
    cfg: ExperimentConfig,
    data: PreparedData,
    *,
    folds: Sequence[FoldSplit] | None = None,
    seeds: Sequence[int] | None = None,
    factory: ForecasterFactory = build_forecaster,
) -> BacktestResult:
    """Train one forecaster per (fold, seed) and forecast every day-ahead origin of its fold.

    Test origins are the ``window.origin_hour`` (midnight by default) of each day whose 24
    target hours lie inside the fold, i.e. the operational day-ahead setting.
    """
    folds = list(folds) if folds is not None else make_walk_forward_folds(data.index, cfg.backtest)
    seeds = list(seeds) if seeds is not None else list(cfg.backtest.seeds)
    frames: list[pd.DataFrame] = []
    histories: dict[str, pd.DataFrame] = {}
    fit_seconds: dict[str, float] = {}
    probabilistic = True

    for fold in folds:
        for seed in seeds:
            forecaster = factory(cfg, seed)
            probabilistic = bool(forecaster.probabilistic)
            origins = select_origins(
                data.index,
                lookback=forecaster.required_history,
                horizon=cfg.window.horizon,
                first_target=fold.test_start,
                last_target=fold.test_end,
                hour=cfg.window.origin_hour,
            )
            logger.info(
                "[%s] fold %s seed %d | test %s -> %s | %d origins",
                cfg.name,
                fold.name,
                seed,
                fold.test_start,
                fold.test_end,
                len(origins),
            )
            started = time.perf_counter()
            history = forecaster.fit(data, fold)
            key = f"{fold.name}/seed{seed}"
            fit_seconds[key] = time.perf_counter() - started
            if isinstance(history, TrainingHistory):
                histories[key] = history.to_frame()
            forecast = forecaster.predict(data, origins)
            frames.append(_frame(data, origins, forecast, cfg.model.quantiles, fold.name, seed))

    return BacktestResult(
        name=cfg.name,
        quantiles=cfg.model.quantiles,
        probabilistic=probabilistic,
        predictions=pd.concat(frames, ignore_index=True),
        histories=histories,
        fit_seconds=fit_seconds,
    )
