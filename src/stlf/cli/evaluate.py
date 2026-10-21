"""``evaluate.py`` — walk-forward backtests and the cross-model benchmark report.

Run one or more experiments (each is retrained from scratch on every fold)::

    python evaluate.py --config configs/gru_cnn.yaml configs/lightgbm.yaml

Optionally average saved runs into an equal-weight ensemble (no weights are fitted, so
nothing is learned from the test folds)::

    python evaluate.py --combine gru_cnn lightgbm --name ensemble_gru_cnn_lightgbm

Then aggregate every saved run into ``reports/benchmark.md`` plus comparison figures::

    python evaluate.py --benchmark
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from stlf.cli.common import add_override_argument, load_experiment, load_prepared, quiet_warnings
from stlf.config import ExperimentConfig
from stlf.data.loader import load_operator_forecast, official_week_label, official_week_mask
from stlf.data.splits import make_walk_forward_folds
from stlf.evaluation import plots
from stlf.evaluation.backtest import BacktestResult, quantile_column, run_backtest
from stlf.evaluation.metrics import mape
from stlf.evaluation.report import (
    SavedRun,
    benchmark_table,
    load_run,
    save_run,
    summarize,
    to_markdown,
)
from stlf.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

FLAGSHIP = "gru_cnn"
# Display order = fixed colour slot order in every comparison figure.
HEADLINE_MODELS = (
    "gru_cnn", "lightgbm", "operator_predispatch", "seasonal_naive_weekly", "legacy_gru_cnn",
)  # fmt: skip
ENSEMBLE = "ensemble_gru_cnn_lightgbm"
EXAMPLE_WEEKS = (
    ("Typical week — 12-18 Oct 2019", "2019-10-12 01:00", "2019-10-19 00:00"),
    ("COVID-19 lockdown — 16-22 May 2020", "2020-05-16 01:00", "2020-05-23 00:00"),
)


def run_experiments(args: argparse.Namespace) -> None:
    for config_path in args.config:
        cfg, data = load_experiment(config_path, args.overrides)
        folds = make_walk_forward_folds(data.index, cfg.backtest)
        if args.max_folds:
            folds = folds[: args.max_folds]
        result = run_backtest(cfg, data, folds=folds, seeds=args.seeds or None)
        summary = summarize(result, data, cfg)
        directory = Path(args.runs_dir or cfg.output_dir) / cfg.name
        save_run(result, summary, cfg, directory)
        overall = summary["overall"]
        logger.info(
            "[%s] MAPE %.3f%% | MAE %.1f MW | RMSE %.1f MW | saved to %s",
            cfg.name,
            overall["mape"],
            overall["mae"],
            overall["rmse"],
            directory,
        )
        if not args.no_plots:
            run_figures(load_run(directory), cfg)


def combine_runs(args: argparse.Namespace) -> None:
    """Equal-weight average of the (seed-ensembled) quantile forecasts of saved runs."""
    runs_dir = Path(args.runs_dir or "reports/runs")
    members = [load_run(runs_dir / name) for name in args.combine]
    quantiles = members[0].cfg.model.quantiles
    if any(member.cfg.model.quantiles != quantiles for member in members):
        raise SystemExit("all combined runs must forecast the same quantiles")
    keys = ["origin", "target_time", "horizon", "fold"]
    columns = [quantile_column(q) for q in quantiles]
    frames = [m.best_predictions().set_index(keys).sort_index() for m in members]
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    stacked = sum(frame.loc[common, columns] for frame in frames) / len(frames)
    combined = frames[0].loc[common, ["actual", "anomaly"]].join(stacked).reset_index()
    combined["seed"] = 0
    name = args.name or "ensemble_" + "_".join(args.combine)
    cfg = dataclasses.replace(members[0].cfg, name=name)
    result = BacktestResult(
        name=name,
        quantiles=quantiles,
        probabilistic=all(m.metrics["probabilistic"] for m in members),
        predictions=combined[[*keys, "seed", "actual", "anomaly", *columns]],
        fit_seconds={m.name: m.metrics["fit_seconds_total"] for m in members},
    )
    summary = summarize(result, load_prepared(cfg), cfg)
    summary["members"] = list(args.combine)
    summary["member_seeds"] = [len(m.metrics["seeds"]) for m in members]
    directory = runs_dir / name
    save_run(result, summary, cfg, directory)
    logger.info(
        "[%s] MAPE %.3f%% from %s", name, summary["overall"]["mape"], ", ".join(args.combine)
    )


def run_figures(saved: SavedRun, cfg: ExperimentConfig) -> None:
    """Per-run diagnostics written next to the run's metrics."""
    figures = saved.directory / "figures"
    predictions = saved.best_predictions()
    if quantile_column(0.5) not in predictions or not saved.metrics["overall"]:
        return
    plots.plot_residuals(predictions, figures / "residuals.png", label=saved.name)
    plots.plot_hour_dow_heatmap(predictions, figures / "error_heatmap.png", label=saved.name)
    if saved.history is not None:
        plots.plot_training_curves(saved.history, figures / "training_curves.png", label=saved.name)
    if saved.metrics["probabilistic"]:
        windows = [w for w in EXAMPLE_WEEKS if (predictions["target_time"] == w[1]).any()]
        if windows:
            plots.plot_forecast_windows(
                predictions,
                windows,
                figures / "forecast_weeks.png",
                label=saved.name,
                quantiles=cfg.model.quantiles,
            )


def _metric_by(saved: SavedRun, key: str, metric: str) -> pd.Series:
    if key == "fold":
        return pd.Series({fold: m[metric] for fold, m in saved.metrics["folds"].items()})
    rows = saved.metrics["horizon"]
    return pd.Series({row["horizon"]: row[metric] for row in rows})


def build_benchmark(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir or "reports/runs")
    report_dir = Path(args.report_dir)
    runs = {
        path.name: load_run(path)
        for path in sorted(runs_dir.iterdir())
        if (path / "metrics.json").exists()
    }
    if not runs:
        raise SystemExit(f"no saved runs under {runs_dir}")
    figures = report_dir / "figures"
    models = [saved for name, saved in runs.items() if not name.startswith("ablation_")]
    table = benchmark_table(models)
    sections = [
        "# Benchmark — day-ahead walk-forward backtest (Jan 2019 – Jun 2020)\n",
        "All models forecast the next 24 hours at every midnight. Each of the 6 quarterly folds "
        "is retrained from scratch on all earlier data. Anomalous hours (blackouts) are "
        "excluded from scoring for every model. MW units; PICP/MPIW refer to the 90% interval.\n",
        to_markdown(table),
    ]
    plots.plot_benchmark_bars(
        table, figures / "benchmark_mape.png", highlight=FLAGSHIP, metric="MAPE %"
    )

    if FLAGSHIP in runs:
        flagship = runs[FLAGSHIP]
        ablations = [saved for name, saved in runs.items() if name.startswith("ablation_")]
        if ablations:
            seed0 = flagship.metrics["per_seed"][str(flagship.metrics["seeds"][0])]
            rows = [{"variant": "gru_cnn (full model, seed 0)", **_ablation_row(seed0)}]
            rows += [
                {"variant": saved.name, **_ablation_row(saved.metrics["overall"])}
                for saved in ablations
            ]
            sections += [
                "\n## Ablations (single seed, same folds)\n",
                to_markdown(pd.DataFrame(rows)),
            ]
        sections += _official_week_section(runs, figures)
        flagship_predictions = flagship.best_predictions()
        operator = None
        if "operator_predispatch" in runs:
            cfg = runs["operator_predispatch"].cfg
            operator = load_operator_forecast(
                Path(cfg.data.raw_dir) / cfg.data.operator_forecast_file
            )
        plots.plot_forecast_windows(
            flagship_predictions,
            EXAMPLE_WEEKS,
            figures / "forecast_weeks.png",
            label="GRU-CNN",
            quantiles=flagship.cfg.model.quantiles,
            operator=operator,
        )
        plots.plot_hour_dow_heatmap(
            flagship_predictions, figures / "error_heatmap.png", label="GRU-CNN"
        )
        plots.plot_residuals(flagship_predictions, figures / "residuals.png", label="GRU-CNN")
        if flagship.history is not None:
            plots.plot_training_curves(
                flagship.history, figures / "training_curves.png", label="GRU-CNN"
            )

    # The legacy model (10-20 % MAPE) would flatten the y-axis; it is in the bar chart instead.
    headline = {n: runs[n] for n in HEADLINE_MODELS if n in runs and n != "legacy_gru_cnn"}
    colors = plots.model_colors(list(HEADLINE_MODELS))
    plots.plot_metric_by_group(
        {name: _metric_by(saved, "fold", "mape") for name, saved in headline.items()},
        figures / "mape_by_fold.png",
        title="MAPE by walk-forward fold (test quarter)",
        xlabel="Test quarter (start month)",
        ylabel="MAPE %",
        colors=colors,
    )
    by_horizon = {n: s for n, s in headline.items() if n != "operator_predispatch"}
    plots.plot_metric_by_group(
        {name: _metric_by(saved, "horizon", "mae") for name, saved in by_horizon.items()},
        figures / "mae_by_horizon.png",
        title="MAE by forecast horizon (hours after the midnight origin)",
        xlabel="Horizon (h)",
        ylabel="MAE (MW)",
        colors=colors,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "benchmark.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    table.to_csv(report_dir / "benchmark.csv", index=False)
    logger.info("Benchmark of %d runs written to %s", len(runs), report_dir / "benchmark.md")


def _ablation_row(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "MAPE %": metrics["mape"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "PICP %": metrics.get("picp", float("nan")),
        "Winkler": metrics.get("winkler", float("nan")),
    }


def _official_week_section(runs: dict[str, SavedRun], figures: Path) -> list[str]:
    """MAPE on the 14 test weeks proposed by the dataset authors, per week and model."""
    candidates = ("gru_cnn", "lightgbm", ENSEMBLE, "operator_predispatch")
    names = [n for n in candidates if n in runs]
    columns: dict[str, pd.Series] = {}
    for name in names:
        frame = runs[name].best_predictions()
        frame = frame[official_week_mask(frame["target_time"]) & ~frame["anomaly"].astype(bool)]
        labels = official_week_label(frame["target_time"])
        columns[name] = frame.groupby(labels).apply(
            lambda g: mape(g["actual"].to_numpy(), g[quantile_column(0.5)].to_numpy())
        )
    table = pd.DataFrame(columns)
    if table.empty:
        return []
    plots.plot_grouped_bars(
        table,
        figures / "official_weeks.png",
        title="MAPE on the dataset's 14 official test weeks",
        ylabel="MAPE %",
        colors={**plots.model_colors(list(HEADLINE_MODELS)), ENSEMBLE: plots.SERIES[6]},
    )
    shown = table.reset_index(names="week starting")
    shown.loc[len(shown)] = ["**mean**", *table.mean().to_list()]
    note = (
        "\n## The 14 official test weeks\n\n"
        "Weeks proposed in the dataset's metadata (Aguilar Madrid, 2021). The operator column is "
        "the official *week-ahead* pre-dispatch forecast, made with weather forecasts; the models "
        "here forecast day-ahead with observed weather, so it is a reference, not a like-for-like "
        "race.\n"
    )
    return [note, to_markdown(shown)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", nargs="+", help="experiment YAML file(s) to backtest")
    mode.add_argument("--benchmark", action="store_true", help="aggregate saved runs")
    mode.add_argument("--combine", nargs="+", metavar="RUN", help="average saved runs")
    parser.add_argument("--name", help="name of the combined run (with --combine)")
    add_override_argument(parser)
    parser.add_argument("--seeds", type=int, nargs="+", help="override backtest.seeds")
    parser.add_argument("--max-folds", type=int, default=0, help="only run the first N folds")
    parser.add_argument("--runs-dir", help="where runs are saved (default: config output_dir)")
    parser.add_argument("--report-dir", default="reports", help="benchmark output directory")
    parser.add_argument("--no-plots", action="store_true", help="skip per-run figures")
    parser.add_argument("--log-file", help="also write the log to this file")
    args = parser.parse_args(argv)
    setup_logging(log_file=args.log_file)
    quiet_warnings()
    if args.benchmark:
        build_benchmark(args)
    elif args.combine:
        combine_runs(args)
    else:
        run_experiments(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
