"""``train.py`` — train a deployable model on the full history and save it as an artifact.

The last ``--val-days`` of data are held out for early stopping; everything before is used for
training. The artifact folder holds the resolved config, weights, scaler statistics and
training metadata, which is all ``predict.py`` needs::

    python train.py --config configs/gru_cnn.yaml
    python predict.py --artifact artifacts/gru_cnn
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stlf import __version__
from stlf.cli.common import add_override_argument, load_experiment, quiet_warnings
from stlf.config import save_config
from stlf.data.splits import final_training_split
from stlf.data.windowing import select_origins
from stlf.evaluation.plots import plot_attention
from stlf.forecasters import NeuralForecaster, build_forecaster
from stlf.logging_utils import get_logger, setup_logging
from stlf.training.engine import TrainingHistory

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="experiment YAML file")
    add_override_argument(parser)
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument(
        "--val-days", type=int, help="early-stopping block (default: backtest.val_days)"
    )
    parser.add_argument("--out", help="artifact directory (default: <artifact_dir>/<name>)")
    parser.add_argument("--attention-plot", help="save the decoder attention-by-lag figure here")
    parser.add_argument("--log-file", help="also write the log to this file")
    args = parser.parse_args(argv)
    setup_logging(log_file=args.log_file)
    quiet_warnings()

    cfg, data = load_experiment(args.config, args.overrides)
    seed = cfg.seed if args.seed is None else args.seed
    forecaster = build_forecaster(cfg, seed)
    if not forecaster.persistable:
        raise SystemExit(f"model '{cfg.model.name}' has nothing to train or save")

    observed = data.actual.dropna().index
    split = final_training_split(observed, args.val_days or cfg.backtest.val_days)
    logger.info("Training on labels <= %s, early-stopping on %s -> %s",
                split.train_end, split.val_start, split.val_end)  # fmt: skip
    history = forecaster.fit(data, split)

    out = Path(args.out or Path(cfg.artifact_dir) / cfg.name)
    forecaster.save(out)
    save_config(cfg, out / "config.yaml")
    metadata = {
        "stlf_version": __version__,
        "model": cfg.model.name,
        "seed": seed,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_start": str(data.index[0]),
        "data_end": str(observed[-1]),
        "train_end": str(split.train_end),
        "val_start": str(split.val_start),
        "val_end": str(split.val_end),
    }
    if isinstance(history, TrainingHistory):
        metadata.update(best_epoch=history.best_epoch, best_val_loss=history.best_val_loss)
        history.to_frame().to_csv(out / "training_history.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Artifact saved to %s", out)

    if args.attention_plot and isinstance(forecaster, NeuralForecaster):
        origins = select_origins(
            data.index,
            lookback=cfg.window.lookback,
            horizon=cfg.window.horizon,
            first_target=split.val_start,
            last_target=split.val_end,
            hour=cfg.window.origin_hour,
        )
        weights = forecaster.attention_map(data, origins)
        plot_attention(np.asarray(weights), Path(args.attention_plot))
        logger.info("Attention figure saved to %s", args.attention_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
