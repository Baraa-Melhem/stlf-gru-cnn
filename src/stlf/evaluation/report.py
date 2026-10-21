"""Turn backtest predictions into metric tables and persist/reload runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stlf.config import ExperimentConfig, load_config, save_config
from stlf.data.loader import official_week_label, official_week_mask
from stlf.evaluation.backtest import BacktestResult, quantile_column
from stlf.evaluation.metrics import interval_metrics, pinball, point_metrics, seasonal_naive_scale
from stlf.features.engineering import PreparedData

ENSEMBLE_SEED = -1
_PREDICTIONS = "predictions.csv.gz"
_METRICS = "metrics.json"
_CONFIG = "config.yaml"
_HISTORY = "training_history.csv"


def ensemble_seeds(predictions: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    """Average the quantile forecasts of all seeds (a free, variance-reducing ensemble)."""
    columns = [quantile_column(q) for q in quantiles]
    keys = ["origin", "target_time", "horizon", "fold"]
    grouped = predictions.groupby(keys, sort=True)
    merged = grouped[columns].mean()
    merged[["actual", "anomaly"]] = grouped[["actual", "anomaly"]].first()
    merged = merged.reset_index()
    merged["seed"] = ENSEMBLE_SEED
    merged[columns] = np.sort(merged[columns].to_numpy(), axis=1)
    return merged


def score(
    frame: pd.DataFrame,
    quantiles: Sequence[float],
    probabilistic: bool,
    mase_scale: float | None = None,
) -> dict[str, float]:
    """Score one prediction frame, excluding flagged anomalies (blackouts) and missing values."""
    median = quantile_column(0.5)
    keep = ~frame["anomaly"].astype(bool) & frame["actual"].notna() & frame[median].notna()
    scored = frame[keep]
    y = scored["actual"].to_numpy()
    result = point_metrics(y, scored[median].to_numpy(), mase_scale)
    result["hours"] = len(scored)
    if probabilistic and len(quantiles) >= 3:
        columns = [quantile_column(q) for q in quantiles]
        result["pinball"] = pinball(y, scored[columns].to_numpy(), quantiles)
        result.update(
            interval_metrics(
                y,
                scored[columns[0]].to_numpy(),
                scored[columns[-1]].to_numpy(),
                nominal=quantiles[-1] - quantiles[0],
            )
        )
    return result


def summarize(result: BacktestResult, data: PreparedData, cfg: ExperimentConfig) -> dict[str, Any]:
    """Overall, per-seed, per-fold, per-horizon and official-week metrics of one backtest."""
    predictions = result.predictions
    seeds = sorted(predictions["seed"].unique())
    best = (
        ensemble_seeds(predictions, result.quantiles)
        if len(seeds) > 1
        else predictions.assign(seed=seeds[0])
    )
    history = data.frame[data.spec.target].loc[: pd.Timestamp(cfg.backtest.test_start)].iloc[:-1]
    scale = seasonal_naive_scale(history.to_numpy())

    def run(frame: pd.DataFrame) -> dict[str, float]:
        return score(frame, result.quantiles, result.probabilistic, scale)

    per_seed = {int(seed): run(predictions[predictions["seed"] == seed]) for seed in seeds}
    seed_table = pd.DataFrame(per_seed).T
    official = best[official_week_mask(best["target_time"])]
    week_labels = official_week_label(official["target_time"])

    horizon_rows = []
    for horizon, group in best.groupby("horizon"):
        metrics = run(group)
        horizon_rows.append(
            {"horizon": int(horizon), **{k: metrics[k] for k in ("mae", "rmse", "mape")}}
        )

    return {
        "name": result.name,
        "model": cfg.model.name,
        "seeds": [int(seed) for seed in seeds],
        "probabilistic": result.probabilistic,
        "mase_scale": scale,
        "overall": run(best),
        "seed_mean": seed_table.mean().to_dict(),
        "seed_std": seed_table.std(ddof=1).fillna(0.0).to_dict() if len(seeds) > 1 else {},
        "per_seed": {str(seed): metrics for seed, metrics in per_seed.items()},
        "folds": {fold: run(group) for fold, group in best.groupby("fold", sort=True)},
        "horizon": horizon_rows,
        "official_weeks": {
            "overall": run(official) if len(official) else {},
            "weeks": {
                label: run(group)["mape"]
                for label, group in official.groupby(week_labels, sort=True)
            },
        },
        "anomalies_excluded": int(best["anomaly"].astype(bool).sum()),
        "fit_seconds_total": float(sum(result.fit_seconds.values())),
    }


def save_run(
    result: BacktestResult, summary: dict[str, Any], cfg: ExperimentConfig, directory: Path
) -> None:
    """Persist predictions, metrics, training curves and the resolved config of a run."""
    directory.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(directory / _PREDICTIONS, index=False, compression="gzip")
    (directory / _METRICS).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_config(cfg, directory / _CONFIG)
    if result.histories:
        curves = pd.concat(
            [frame.assign(run=key) for key, frame in result.histories.items()], ignore_index=True
        )
        curves.to_csv(directory / _HISTORY, index=False)


@dataclass(frozen=True)
class SavedRun:
    """A run reloaded from disk for cross-model comparison."""

    directory: Path
    cfg: ExperimentConfig
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    history: pd.DataFrame | None

    @property
    def name(self) -> str:
        return self.cfg.name

    def best_predictions(self) -> pd.DataFrame:
        """Seed-ensembled predictions when several seeds were run, else the single seed."""
        if self.predictions["seed"].nunique() > 1:
            return ensemble_seeds(self.predictions, self.cfg.model.quantiles)
        return self.predictions


def load_run(directory: Path) -> SavedRun:
    """Reload a run written by :func:`save_run`."""
    predictions = pd.read_csv(
        directory / _PREDICTIONS, parse_dates=["origin", "target_time"], compression="gzip"
    )
    history_path = directory / _HISTORY
    return SavedRun(
        directory=directory,
        cfg=load_config(directory / _CONFIG),
        metrics=json.loads((directory / _METRICS).read_text(encoding="utf-8")),
        predictions=predictions,
        history=pd.read_csv(history_path) if history_path.exists() else None,
    )


def benchmark_table(runs: Sequence[SavedRun]) -> pd.DataFrame:
    """One row per run: walk-forward and official-week accuracy and interval quality."""
    rows = []
    for saved in runs:
        overall = saved.metrics["overall"]
        official = saved.metrics["official_weeks"]["overall"]
        std = saved.metrics.get("seed_std", {})
        rows.append(
            {
                "model": saved.name,
                "MAPE %": overall["mape"],
                "MAPE σ (seeds)": std.get("mape", np.nan),
                "MAE": overall["mae"],
                "RMSE": overall["rmse"],
                "sMAPE %": overall["smape"],
                "R²": overall["r2"],
                "MASE": overall.get("mase", np.nan),
                "PICP %": overall.get("picp", np.nan),
                "MPIW": overall.get("mpiw", np.nan),
                "Official-weeks MAPE %": official.get("mape", np.nan),
                "Seeds": "+".join(map(str, saved.metrics.get("member_seeds", [])))
                or str(len(saved.metrics["seeds"])),
            }
        )
    return pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)


def to_markdown(table: pd.DataFrame, decimals: int = 2) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""

    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "–" if np.isnan(value) else f"{value:,.{decimals}f}"
        return str(value)

    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(table.shape[1])) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, divider, *body])
