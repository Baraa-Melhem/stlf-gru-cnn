"""``tune.py`` — Optuna search over the hybrid network's hyper-parameters.

Each trial is scored by the day-ahead MAPE of an *inner* walk-forward backtest that ends before
the reported test period (by default two quarterly folds in the second half of 2018, each
trained only on earlier data). The 2019-2020 test folds are never touched, so tuned settings
cannot leak test information into the benchmark::

    python tune.py --config configs/gru_cnn.yaml --trials 30
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from stlf.cli.common import add_override_argument, load_experiment, quiet_warnings
from stlf.config import ExperimentConfig, apply_overrides, config_from_dict, to_dict
from stlf.data.splits import make_walk_forward_folds
from stlf.evaluation.backtest import run_backtest
from stlf.evaluation.report import score
from stlf.features.engineering import PreparedData
from stlf.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def suggest(trial: Any) -> list[str]:
    """Sample one configuration as ``section.key=value`` overrides."""
    flag = trial.suggest_categorical
    return [
        f"model.gru_cnn.d_model={trial.suggest_categorical('d_model', [32, 64, 96])}",
        f"model.gru_cnn.attention_heads={trial.suggest_categorical('attention_heads', [2, 4])}",
        f"model.gru_cnn.cnn_blocks={trial.suggest_int('cnn_blocks', 1, 4)}",
        f"model.gru_cnn.kernel_size={trial.suggest_categorical('kernel_size', [3, 5])}",
        f"model.gru_cnn.gru_hidden={trial.suggest_categorical('gru_hidden', [64, 128, 192])}",
        f"model.gru_cnn.gru_layers={trial.suggest_int('gru_layers', 1, 2)}",
        f"model.gru_cnn.dropout={trial.suggest_float('dropout', 0.0, 0.4)}",
        f"model.gru_cnn.origin_anchors={flag('origin_anchors', [True, False])}",
        f"model.gru_cnn.revin_exclude_holidays={flag('revin_exclude_holidays', [True, False])}",
        f"train.learning_rate={trial.suggest_float('learning_rate', 2e-4, 3e-3, log=True)}",
        f"train.weight_decay={trial.suggest_float('weight_decay', 1e-6, 5e-2, log=True)}",
        f"window.lookback={trial.suggest_categorical('lookback', [168, 336])}",
        f"train.max_epochs={flag('max_epochs', [8, 12, 20])}",  # cosine schedule length
        "train.patience=4",
    ]


def make_objective(base: ExperimentConfig, data: PreparedData, start: str, end: str):
    import optuna

    inner = dataclasses.replace(base.backtest, test_start=start, test_end=end)
    folds = make_walk_forward_folds(data.index, inner)

    def objective(trial: optuna.Trial) -> float:
        cfg = config_from_dict(apply_overrides(to_dict(base), suggest(trial)))
        scores: list[float] = []
        for step, fold in enumerate(folds):
            result = run_backtest(cfg, data, folds=[fold], seeds=[base.seed])
            scores.append(score(result.predictions, cfg.model.quantiles, True)["mape"])
            mean = sum(scores) / len(scores)
            trial.report(mean, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_mape", scores)
        return sum(scores) / len(scores)

    return objective


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/gru_cnn.yaml", help="base experiment")
    add_override_argument(parser)
    parser.add_argument("--trials", type=int, default=30, help="number of Optuna trials")
    parser.add_argument("--timeout", type=int, help="stop after this many seconds")
    parser.add_argument("--jobs", type=int, default=1, help="trials run in parallel threads")
    parser.add_argument("--inner-start", default="2018-07-01 01:00", help="first inner target")
    parser.add_argument("--inner-end", default="2018-12-31 00:00", help="last inner target")
    parser.add_argument("--out", default="reports/tuning", help="output directory")
    parser.add_argument(
        "--study-name", default="gru_cnn", help="study name inside <out>/study.db (resumable)"
    )
    parser.add_argument("--log-file", help="also write the log to this file")
    args = parser.parse_args(argv)
    setup_logging(log_file=args.log_file)
    quiet_warnings()
    try:
        import optuna
    except ImportError as error:
        raise SystemExit("tuning needs optuna: pip install 'stlf-gru-cnn[tune]'") from error

    base, data = load_experiment(args.config, args.overrides)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if pd.Timestamp(base.backtest.test_start) <= pd.Timestamp(args.inner_end):
        raise SystemExit("the inner tuning period must end before backtest.test_start")
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{(out / 'study.db').as_posix()}",
        load_if_exists=True,  # re-running the command resumes an interrupted search
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    objective = make_objective(base, data, args.inner_start, args.inner_end)
    # A failed trial (e.g. a GPU driver reset) is recorded as failed instead of ending the study.
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.jobs,
        catch=(RuntimeError,),
    )
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    best = {"inner_mape": study.best_value, "params": study.best_params}
    (out / "best_params.yaml").write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    logger.info("Best inner MAPE %.4f%% with %s", study.best_value, study.best_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
