"""Point, probabilistic and horizon-wise forecast accuracy metrics (all NaN-safe)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _valid(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[keep], y_pred[keep]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (MW)."""
    t, p = _valid(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (MW)."""
    t, p = _valid(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%). Load is strictly positive, so it is well defined."""
    t, p = _valid(y_true, y_pred)
    return float(100.0 * np.mean(np.abs(t - p) / np.abs(t)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (%), bounded in [0, 200]."""
    t, p = _valid(y_true, y_pred)
    denominator = np.abs(t) + np.abs(p)
    ratio = np.divide(2.0 * np.abs(t - p), denominator, out=np.zeros_like(t), where=denominator > 0)
    return float(100.0 * np.mean(ratio))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    t, p = _valid(y_true, y_pred)
    total = np.sum((t - t.mean()) ** 2)
    return float(1.0 - np.sum((t - p) ** 2) / total) if total > 0 else float("nan")


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    """MAE scaled by the in-sample MAE of the weekly seasonal-naive forecast (< 1 beats it)."""
    return mae(y_true, y_pred) / scale if scale > 0 else float("nan")


def seasonal_naive_scale(history: np.ndarray, season: int = 168) -> float:
    """In-sample MAE of ``y(t) = y(t - season)`` on a training history — the MASE denominator."""
    history = np.asarray(history, dtype=np.float64)
    return mae(history[season:], history[:-season])


def pinball(y_true: np.ndarray, quantile_preds: np.ndarray, quantiles: Sequence[float]) -> float:
    """Mean pinball loss over ``[N, Q]`` predictions and all quantile levels (MW)."""
    y = np.asarray(y_true, dtype=np.float64)[:, None]
    q = np.asarray(quantiles, dtype=np.float64)[None, :]
    keep = np.isfinite(y[:, 0]) & np.all(np.isfinite(quantile_preds), axis=1)
    error = y[keep] - quantile_preds[keep]
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))


def interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal: float
) -> dict[str, float]:
    """Coverage (PICP), mean width (MPIW) and Winkler interval score of a central interval.

    The Winkler score adds ``2/alpha`` x the miss distance to the width, so it rewards narrow
    intervals only when they still cover — a single number to compare interval quality.
    """
    y, lo, hi = (np.asarray(a, dtype=np.float64) for a in (y_true, lower, upper))
    keep = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[keep], lo[keep], hi[keep]
    alpha = 1.0 - nominal
    width = hi - lo
    penalty = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return {
        "picp": float(100.0 * np.mean((y >= lo) & (y <= hi))),
        "nominal": float(100.0 * nominal),
        "mpiw": float(np.mean(width)),
        "winkler": float(np.mean(width + penalty)),
    }


def point_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mase_scale: float | None = None
) -> dict[str, float]:
    """All point metrics in one dictionary."""
    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    if mase_scale is not None:
        result["mase"] = mase(y_true, y_pred, mase_scale)
    return result
