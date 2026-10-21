"""Forecasters share one interface; :func:`build_forecaster` picks one from the config."""

from __future__ import annotations

from pathlib import Path

from stlf.config import NEURAL_MODELS, ExperimentConfig
from stlf.forecasters.base import Forecaster
from stlf.forecasters.gbm import LightGBMForecaster
from stlf.forecasters.naive import OperatorForecaster, SeasonalNaiveForecaster
from stlf.forecasters.neural import NeuralForecaster

__all__ = [
    "Forecaster",
    "LightGBMForecaster",
    "NeuralForecaster",
    "OperatorForecaster",
    "SeasonalNaiveForecaster",
    "build_forecaster",
    "forecaster_class",
    "load_forecaster",
]


def forecaster_class(cfg: ExperimentConfig) -> type[Forecaster]:
    """Return the :class:`Forecaster` subclass that implements ``cfg.model.name``."""
    if cfg.model.name in NEURAL_MODELS:
        return NeuralForecaster
    classes: dict[str, type[Forecaster]] = {
        "lightgbm": LightGBMForecaster,
        "seasonal_naive": SeasonalNaiveForecaster,
        "operator": OperatorForecaster,
    }
    return classes[cfg.model.name]


def build_forecaster(cfg: ExperimentConfig, seed: int = 0) -> Forecaster:
    """Instantiate an unfitted forecaster for ``cfg``."""
    return forecaster_class(cfg)(cfg, seed)


def load_forecaster(directory: Path, cfg: ExperimentConfig) -> Forecaster:
    """Restore a saved forecaster of the family named in ``cfg``."""
    return forecaster_class(cfg).load(directory, cfg)
