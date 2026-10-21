"""``predict.py`` — produce a day-ahead forecast from a saved artifact.

The data file must contain the load history up to the origin and the known covariates
(holiday/school flags and, if the model uses future weather, the weather forecast) for the
forecast hours; their load cells may be empty::

    python predict.py --artifact artifacts/gru_cnn --origin "2020-06-26 00:00" --out forecast.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stlf.cli.common import load_prepared, quiet_warnings
from stlf.config import load_config
from stlf.evaluation.backtest import quantile_column
from stlf.forecasters import load_forecaster
from stlf.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def forecast_frame(
    artifact: Path, data_path: Path | None, origin: str | None
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Load ``artifact``, validate inputs around ``origin`` and return the forecast table."""
    cfg = load_config(artifact / "config.yaml")
    data = load_prepared(cfg, data_path)
    forecaster = load_forecaster(artifact, cfg)
    horizon = cfg.window.horizon

    if origin is None:
        observed = data.actual.dropna().index
        candidates = observed[observed.hour == cfg.window.origin_hour]
        candidates = candidates[candidates + pd.Timedelta(hours=horizon) <= data.index[-1]]
        if len(candidates) == 0:
            raise SystemExit("no origin has both observed history and a full horizon of covariates")
        stamp = candidates[-1]
    else:
        stamp = pd.Timestamp(origin)
    if stamp not in data.index:
        raise SystemExit(
            f"origin {stamp} is outside the data ({data.index[0]} .. {data.index[-1]})"
        )
    position = int(data.index.get_loc(stamp))
    if position + 1 < forecaster.required_history:
        raise SystemExit(f"need {forecaster.required_history} h of history before the origin")
    if position + horizon >= len(data.index):
        raise SystemExit(f"the data must extend {horizon} h past the origin for the covariates")
    history = data.frame[data.spec.target].iloc[
        position + 1 - forecaster.required_history : position + 1
    ]
    if history.isna().any():
        raise SystemExit("the load history before the origin contains gaps")

    forecast = forecaster.predict(data, np.array([position]))[0]
    targets = data.index[position + 1 : position + 1 + horizon]
    table = pd.DataFrame({"target_time": targets, "horizon": np.arange(1, horizon + 1)})
    for k, q in enumerate(cfg.model.quantiles):
        table[quantile_column(q)] = forecast[:, k]
    table["actual"] = data.actual.reindex(targets).to_numpy()
    return stamp, table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifact", required=True, help="directory written by train.py")
    parser.add_argument("--data", help="CSV with history + future covariates (default: config)")
    parser.add_argument(
        "--origin", help="forecast origin, e.g. '2020-06-26 00:00' (default: latest)"
    )
    parser.add_argument("--out", help="write the forecast to this CSV")
    args = parser.parse_args(argv)
    setup_logging()
    quiet_warnings()

    origin, table = forecast_frame(Path(args.artifact), Path(args.data) if args.data else None,
                                   args.origin)  # fmt: skip
    logger.info("Forecast issued at %s for %s -> %s", origin, table["target_time"].iloc[0],
                table["target_time"].iloc[-1])  # fmt: skip
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        logger.info("Saved to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
