"""Helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from stlf.config import ExperimentConfig, load_config
from stlf.data.loader import load_dataset
from stlf.features.engineering import PreparedData, prepare_data
from stlf.logging_utils import get_logger

logger = get_logger(__name__)


def add_override_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="KEY=VALUE",
        nargs="+",
        default=[],
        help="override config values, e.g. --set train.max_epochs=20 window.lookback=336",
    )


def quiet_warnings() -> None:
    """Silence third-party deprecation chatter so the structured log stays readable."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


def load_experiment(
    config_path: str | Path, overrides: list[str] | None = None
) -> tuple[ExperimentConfig, PreparedData]:
    """Load a config and the prepared dataset it describes."""
    cfg = load_config(config_path, overrides or [])
    return cfg, load_prepared(cfg)


def load_prepared(cfg: ExperimentConfig, data_path: str | Path | None = None) -> PreparedData:
    """Read the raw CSV named by ``cfg`` (or ``data_path``) and build its features."""
    path = Path(data_path) if data_path else Path(cfg.data.raw_dir) / cfg.data.load_file
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python download_data.py` first")
    raw = load_dataset(path)
    logger.info("Loaded %s: %d hourly rows, %s -> %s", path, len(raw), raw.index[0], raw.index[-1])
    return prepare_data(raw, cfg)
