"""Shared fixtures: a small synthetic hourly dataset and a CPU-sized experiment config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stlf.config import WEATHER_COLUMNS, ExperimentConfig, config_from_dict
from stlf.data.loader import TARGET_COLUMN
from stlf.features.engineering import PreparedData, prepare_data


def make_raw(days: int = 90, seed: int = 0) -> pd.DataFrame:
    """Synthetic load with daily + weekly seasonality, a trend, weather and holidays."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-01 01:00", periods=days * 24, freq="h", name="datetime")
    hours = np.arange(len(index))
    temperature = (
        27 + 3 * np.sin(2 * np.pi * (hours % 24 - 9) / 24) + rng.normal(0, 0.3, len(index))
    )
    load = (
        1100
        + 150 * np.sin(2 * np.pi * (hours % 24 - 6) / 24)
        - 80 * (index.dayofweek >= 5)
        + 0.02 * hours
        + 12 * (temperature - 27)
        + rng.normal(0, 8, len(index))
    )
    frame = pd.DataFrame(index=index)
    frame[TARGET_COLUMN] = load
    for k, column in enumerate(WEATHER_COLUMNS):
        frame[column] = temperature + k if column.startswith("T2M") else rng.random(len(index))
    holiday = (index.dayofyear % 30 == 0).astype(float)
    frame["Holiday_ID"] = holiday * 5
    frame["holiday"] = holiday
    frame["school"] = (index.month != 1).astype(float)
    return frame


def tiny_config(**sections: dict) -> ExperimentConfig:
    """A config small enough to train in seconds on a CPU."""
    values: dict = {
        "name": "test",
        "window": {"lookback": 48, "horizon": 24, "train_stride": 6},
        "features": {"target_lags": [24, 48]},
        "model": {
            "name": "gru_cnn",
            "gru_cnn": {"d_model": 16, "gru_hidden": 16, "cnn_blocks": 2, "attention_heads": 2},
            "gbm": {"n_estimators": 30, "lags": [24, 48], "early_stopping_rounds": 10},
        },
        "train": {"max_epochs": 2, "batch_size": 64, "device": "cpu", "amp": False},
        "backtest": {"test_start": "2018-03-01 01:00", "fold_months": 1, "val_days": 7},
    }
    for section, overrides in sections.items():
        if isinstance(values.get(section), dict):
            values[section] = {**values[section], **overrides}
        else:
            values[section] = overrides
    return config_from_dict(values)


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def cfg() -> ExperimentConfig:
    return tiny_config()


@pytest.fixture
def data(raw: pd.DataFrame, cfg: ExperimentConfig) -> PreparedData:
    return prepare_data(raw, cfg)
