from __future__ import annotations

import numpy as np
import pytest

from stlf.evaluation.metrics import (
    interval_metrics,
    mae,
    mape,
    mase,
    pinball,
    point_metrics,
    r2,
    rmse,
    seasonal_naive_scale,
    smape,
)
from stlf.forecasters.base import apply_margin, conformal_margin


def test_point_metrics_on_known_values() -> None:
    y, p = np.array([100.0, 200.0, 400.0]), np.array([110.0, 190.0, 400.0])
    assert mae(y, p) == pytest.approx(20 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(200 / 3))
    assert mape(y, p) == pytest.approx(100 * (0.1 + 0.05) / 3)
    assert smape(y, p) == pytest.approx(100 * (20 / 210 + 20 / 390) / 3)
    assert r2(y, y) == 1.0
    assert mase(y, p, scale=20 / 3) == pytest.approx(1.0)


def test_metrics_ignore_missing_values() -> None:
    y, p = np.array([100.0, np.nan, 300.0]), np.array([90.0, 5.0, np.nan])
    assert point_metrics(y, p)["mae"] == pytest.approx(10.0)


def test_seasonal_naive_scale() -> None:
    history = np.tile(np.arange(168, dtype=float), 3)
    assert seasonal_naive_scale(history) == 0.0
    assert seasonal_naive_scale(history + np.arange(len(history)), 168) == pytest.approx(168.0)


def test_pinball_median_is_half_mae() -> None:
    y = np.array([1.0, 2.0, 5.0])
    preds = np.array([[0.0], [3.0], [5.0]])
    assert pinball(y, preds, [0.5]) == pytest.approx(0.5 * mae(y, preds[:, 0]))


def test_interval_metrics() -> None:
    y = np.array([0.0, 5.0, 20.0])
    result = interval_metrics(y, np.full(3, 0.0), np.full(3, 10.0), nominal=0.9)
    assert result["picp"] == pytest.approx(200 / 3)
    assert result["mpiw"] == 10.0
    assert result["winkler"] == pytest.approx(10.0 + (2 / 0.1) * 10.0 / 3)


def test_conformal_margin_restores_coverage() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=20_000)
    lower, upper = np.full_like(y, -1.0), np.full_like(y, 1.0)  # only ~68 % coverage
    margin = conformal_margin(y, lower, upper, nominal=0.9)
    assert margin == pytest.approx(1.645 - 1.0, abs=0.05)
    fresh = rng.normal(size=20_000)
    coverage = np.mean((fresh >= -1 - margin) & (fresh <= 1 + margin))
    assert coverage == pytest.approx(0.9, abs=0.01)


def test_apply_margin_only_moves_the_outer_quantiles() -> None:
    forecast = np.array([[[9.0, 10.0, 11.0]]])
    assert apply_margin(forecast, 2.0).tolist() == [[[7.0, 10.0, 13.0]]]
    assert apply_margin(forecast[..., 1:2], 2.0).tolist() == [[[10.0]]]
