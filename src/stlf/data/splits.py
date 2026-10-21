"""Expanding-window walk-forward splits.

Each fold has three label periods, strictly ordered in time::

    |---------------- train ----------------|--- val ---|--- test (one fold) ---|
                                    train_end  val_start/val_end  test_start/test_end

The model of fold *k* is trained on every hour before its validation block, early-stopped on
the validation block, and scored on the test block. Folds tile the test period, so every test
hour is forecast exactly once by a model that never saw it — the opposite of the shuffled
``train_test_split`` in the original notebook, which let the network interpolate between
neighbouring hours it had trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stlf.config import BacktestConfig


@dataclass(frozen=True)
class FoldSplit:
    """Inclusive label-time bounds of one walk-forward fold."""

    name: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def final_training_split(index: pd.DatetimeIndex, val_days: int) -> FoldSplit:
    """Split for a deployable model: train on everything except a trailing validation block."""
    val_end = index[-1]
    val_start = val_end - pd.Timedelta(days=val_days) + pd.Timedelta(hours=1)
    return FoldSplit(
        name="final",
        train_end=val_start - pd.Timedelta(hours=1),
        val_start=val_start,
        val_end=val_end,
        test_start=val_end,
        test_end=val_end,
    )


def make_walk_forward_folds(index: pd.DatetimeIndex, cfg: BacktestConfig) -> list[FoldSplit]:
    """Tile ``[test_start, test_end]`` into consecutive ``fold_months``-long test blocks."""
    start = pd.Timestamp(cfg.test_start)
    end = pd.Timestamp(cfg.test_end) if cfg.test_end is not None else index[-1]
    if not index[0] < start <= end <= index[-1]:
        raise ValueError(f"test period {start}..{end} is outside the data {index[0]}..{index[-1]}")
    folds: list[FoldSplit] = []
    block_start = start
    while block_start <= end:
        next_start = block_start + pd.DateOffset(months=cfg.fold_months)
        block_end = min(next_start - pd.Timedelta(hours=1), end)
        val_end = block_start - pd.Timedelta(hours=1)
        val_start = block_start - pd.Timedelta(days=cfg.val_days)
        folds.append(
            FoldSplit(
                name=f"{block_start:%Y-%m}",
                train_end=val_start - pd.Timedelta(hours=1),
                val_start=val_start,
                val_end=val_end,
                test_start=block_start,
                test_end=block_end,
            )
        )
        block_start = next_start
    return folds
