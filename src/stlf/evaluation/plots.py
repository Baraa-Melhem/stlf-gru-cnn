"""Publication-quality diagnostic figures (matplotlib, PNG).

Colour follows the job it does: actual load is neutral ink; models take categorical slots in a
fixed order (flagship first) that is colour-blind-validated; magnitudes use one blue ramp.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from stlf.evaluation.backtest import quantile_column
from stlf.evaluation.metrics import mae, mape

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
NEUTRAL_BAR = "#b9b8b2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def apply_style() -> None:
    """Recessive axes and grid, readable type, consistent surface."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def model_colors(names: Sequence[str]) -> dict[str, str]:
    """Fixed categorical slot per model, in the given (stable) order."""
    return {name: SERIES[i % len(SERIES)] for i, name in enumerate(names)}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_forecast_windows(
    predictions: pd.DataFrame,
    windows: Sequence[tuple[str, str, str]],
    path: Path,
    *,
    label: str,
    quantiles: Sequence[float],
    operator: pd.Series | None = None,
) -> Path:
    """Actual vs median forecast with the central prediction interval, one panel per window.

    ``windows`` holds ``(title, first_timestamp, last_timestamp)`` triples.
    """
    apply_style()
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 3.3 * len(windows)), squeeze=False)
    low, mid, high = (quantile_column(q) for q in (quantiles[0], 0.5, quantiles[-1]))
    for ax, (title, start, end) in zip(axes[:, 0], windows, strict=True):
        part = predictions[
            (predictions["target_time"] >= start) & (predictions["target_time"] <= end)
        ].sort_values("target_time")
        x = part["target_time"]
        nominal = round(100 * (quantiles[-1] - quantiles[0]))
        ax.fill_between(x, part[low], part[high], color=SERIES[0], alpha=0.16, linewidth=0,
                        label=f"{nominal}% prediction interval")  # fmt: skip
        ax.plot(x, part["actual"], color=INK, linewidth=1.6, label="Actual load")
        ax.plot(x, part[mid], color=SERIES[0], label=f"{label} (median)")
        if operator is not None:
            op = operator.reindex(pd.DatetimeIndex(x))
            ax.plot(x, op.to_numpy(), color=SERIES[1], linewidth=1.4, linestyle=(0, (4, 2)),
                    label="Operator pre-dispatch")  # fmt: skip
        week_mape = mape(part["actual"].to_numpy(), part[mid].to_numpy())
        ax.set_title(f"{title}  ·  MAPE {week_mape:.2f}%")
        ax.set_ylabel("Load (MW)")
        ax.margins(x=0)
    axes[0, 0].legend(loc="upper left", ncols=4, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    return _save(fig, path)


def plot_benchmark_bars(table: pd.DataFrame, path: Path, *, highlight: str, metric: str) -> Path:
    """Horizontal bars of one metric per model; the flagship highlighted, others neutral."""
    apply_style()
    ordered = table.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 0.48 * len(ordered) + 1.2))
    colors = [SERIES[0] if name == highlight else NEUTRAL_BAR for name in ordered["model"]]
    bars = ax.barh(ordered["model"], ordered[metric], color=colors, height=0.62)
    for bar, value in zip(bars, ordered[metric], strict=True):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {value:.2f}",
                va="center", ha="left", color=INK_SECONDARY, fontsize=9)  # fmt: skip
    ax.set_xlabel(metric)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ordered[metric].max() * 1.15)
    ax.set_title(f"Walk-forward day-ahead accuracy ({metric}, lower is better)")
    return _save(fig, path)


def plot_metric_by_group(
    series: dict[str, pd.Series],
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: dict[str, str] | None = None,
) -> Path:
    """One line per model (e.g. MAPE per fold or MAE per horizon).

    Pass ``colors`` so a model keeps the same colour in every figure even when some models are
    left out of this one.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = colors or model_colors(list(series))
    for name, values in series.items():
        ax.plot(values.index, values.to_numpy(), color=colors[name], marker="o", markersize=4,
                label=name)  # fmt: skip
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", ncols=min(2, len(series)))
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def plot_hour_dow_heatmap(predictions: pd.DataFrame, path: Path, *, label: str) -> Path:
    """MAPE by hour of day x day of week for one model (sequential blue ramp)."""
    apply_style()
    frame = predictions[~predictions["anomaly"].astype(bool)].copy()
    frame["ape"] = (frame["actual"] - frame[quantile_column(0.5)]).abs() / frame["actual"] * 100
    frame["hour"] = frame["target_time"].dt.hour
    frame["dow"] = frame["target_time"].dt.dayofweek
    grid = frame.pivot_table(index="dow", columns="hour", values="ape", aggfunc="mean")
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    values = grid.to_numpy()
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=low, vmax=high)
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day (interval end)")
    ax.grid(False)
    for (row, col), value in np.ndenumerate(values):
        shade = "white" if (value - low) > 0.5 * (high - low) else INK
        ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=6.5, color=shade)
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("MAPE %", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    ax.set_title(f"{label}: where the error lives (MAPE % by hour and weekday)")
    return _save(fig, path)


def plot_residuals(predictions: pd.DataFrame, path: Path, *, label: str) -> Path:
    """Residual distribution and predicted-vs-actual scatter."""
    apply_style()
    frame = predictions[~predictions["anomaly"].astype(bool)]
    y, p = frame["actual"].to_numpy(), frame[quantile_column(0.5)].to_numpy()
    residual = y - p
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.hist(residual, bins=80, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.4)
    left.axvline(0, color=INK, linewidth=1)
    left.set_title(f"Residuals (actual − forecast)  ·  bias {residual.mean():+.1f} MW")
    left.set_xlabel("MW")
    left.set_ylabel("Hours")
    lo, hi = np.nanpercentile(np.concatenate([y, p]), [0.2, 99.8])
    right.scatter(y, p, s=3, alpha=0.18, color=SERIES[0], linewidths=0)
    right.plot([lo, hi], [lo, hi], color=INK, linewidth=1)
    right.set_xlim(lo, hi)
    right.set_ylim(lo, hi)
    right.set_title(f"{label}: forecast vs actual  ·  MAE {mae(y, p):.1f} MW")
    right.set_xlabel("Actual (MW)")
    right.set_ylabel("Forecast (MW)")
    fig.tight_layout()
    return _save(fig, path)


def plot_attention(weights: np.ndarray, path: Path) -> Path:
    """Mean cross-attention over the look-back window, by lag (hours before the target)."""
    apply_style()
    _, horizon, lookback = weights.shape
    # weights[:, h, j]: target hour h+1 attending to history position j (j = L-1 is the origin).
    lag_profile = np.zeros(lookback + horizon)
    counts = np.zeros(lookback + horizon)
    for h in range(horizon):
        lags = (h + 1) + (lookback - 1 - np.arange(lookback))
        lag_profile[lags] += weights[:, h, :].mean(axis=0)
        counts[lags] += 1
    profile = np.divide(lag_profile, counts, out=np.zeros_like(lag_profile), where=counts > 0)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    lags = np.arange(len(profile))
    ax.bar(lags, profile, width=1.0, color=SERIES[0])
    for marker in (24, 48, 168, 336):
        if marker < len(profile):
            ax.axvline(marker, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.text(marker, profile.max() * 1.02, f"{marker} h", ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY)  # fmt: skip
    ax.set_xlim(0, len(profile))
    ax.set_ylim(0, profile.max() * 1.12)
    ax.set_xlabel("Lag between target hour and attended history hour (hours)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("What the decoder looks at: cross-attention by lag")
    return _save(fig, path)


def plot_training_curves(history: pd.DataFrame, path: Path, *, label: str) -> Path:
    """Train/validation loss per epoch for every fold of a run (validation solid)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    runs = list(dict.fromkeys(history["run"]))
    shades = np.linspace(0.35, 1.0, len(runs))
    for run, shade in zip(runs, shades, strict=True):
        part = history[history["run"] == run]
        ax.plot(part["epoch"], part["val_loss"], color=SERIES[0], alpha=shade, linewidth=1.6)
        ax.plot(part["epoch"], part["train_loss"], color=SERIES[1], alpha=shade, linewidth=1.0,
                linestyle=(0, (3, 2)))  # fmt: skip
    ax.plot([], [], color=SERIES[0], label="validation")
    ax.plot([], [], color=SERIES[1], linestyle=(0, (3, 2)), label="train")
    ax.legend(loc="upper right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pinball loss (scaled)")
    ax.set_title(f"{label}: training curves, all folds and seeds (darker = later fold)")
    return _save(fig, path)


def plot_grouped_bars(
    table: pd.DataFrame, path: Path, *, title: str, ylabel: str, colors: dict[str, str]
) -> Path:
    """Grouped vertical bars: rows are groups (e.g. weeks), columns are models."""
    apply_style()
    groups, models = table.index.tolist(), table.columns.tolist()
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(groups))
    for k, model in enumerate(models):
        ax.bar(x + (k - (len(models) - 1) / 2) * width, table[model].to_numpy(),
               width=width * 0.92, color=colors[model], label=model)  # fmt: skip
    ax.set_xticks(x, groups, rotation=35, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=len(models))
    return _save(fig, path)
