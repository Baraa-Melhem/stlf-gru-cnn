"""Loading and validating the hourly Panama national load dataset.

Timestamps are Panama local time (UTC-05:00, no daylight saving) and label the *end* of each
hourly interval: the first row, ``2015-01-03 01:00``, is the load of 00:00-01:00. A day-ahead
forecast issued at ``D 00:00`` therefore covers the 24 rows ``D 01:00 .. D+1 00:00``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stlf.config import WEATHER_COLUMNS

TARGET_COLUMN = "nat_demand"
CALENDAR_COLUMNS: tuple[str, ...] = ("Holiday_ID", "holiday", "school")
REQUIRED_COLUMNS: tuple[str, ...] = (TARGET_COLUMN, *WEATHER_COLUMNS, *CALENDAR_COLUMNS)

# The 14 one-week test windows proposed by the dataset authors (first target hour of each).
OFFICIAL_TEST_WEEKS: tuple[str, ...] = (
    "2019-04-13 01:00", "2019-05-25 01:00", "2019-06-15 01:00", "2019-07-20 01:00",
    "2019-08-17 01:00", "2019-09-14 01:00", "2019-10-12 01:00", "2019-11-02 01:00",
    "2019-12-21 01:00", "2020-01-04 01:00", "2020-02-08 01:00", "2020-03-07 01:00",
    "2020-05-16 01:00", "2020-06-13 01:00",
)  # fmt: skip


class DataValidationError(ValueError):
    """Raised when an input file violates the schema or the hourly-continuity contract."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the continuous dataset, enforce its schema and return an hourly-indexed frame.

    The target column may contain trailing missing values (future rows supplied only for
    their covariates at prediction time), but the index must be strictly hourly with no gaps
    or duplicates, because every window computation is positional.
    """
    frame = pd.read_csv(path, parse_dates=["datetime"])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{path}: missing required columns {missing}")
    frame = frame.set_index("datetime").sort_index()
    if frame.index.has_duplicates:
        duplicated = frame.index[frame.index.duplicated()].unique()
        raise DataValidationError(f"{path}: duplicated timestamps, e.g. {list(duplicated[:3])}")
    expected = pd.date_range(frame.index[0], frame.index[-1], freq="h")
    if len(expected) != len(frame):
        gaps = expected.difference(frame.index)
        raise DataValidationError(f"{path}: {len(gaps)} missing hours, first at {gaps[0]}")
    frame.index = expected
    frame.index.name = "datetime"
    covariates = [column for column in REQUIRED_COLUMNS if column != TARGET_COLUMN]
    if frame[covariates].isna().any().any():
        bad = frame[covariates].isna().sum()
        raise DataValidationError(f"{path}: covariates contain NaN: {bad[bad > 0].to_dict()}")
    return frame[list(REQUIRED_COLUMNS)].astype("float64")


def flag_load_anomalies(load: pd.Series, ratio: float, window: int) -> pd.Series:
    """Flag hours whose load falls below ``ratio`` x the trailing ``window``-hour median.

    The national load never legitimately halves within a week; such drops are blackouts or
    metering faults (8 hours in this dataset, e.g. 2019-01-20). The rolling median is
    trailing-only, so the flag at time *t* uses no information after *t*.
    """
    baseline = load.rolling(window, min_periods=max(1, window // 4)).median().shift(1)
    return (load < ratio * baseline).fillna(False).astype(bool)


def load_operator_forecast(path: str | Path) -> pd.Series:
    """Read the grid operator's weekly pre-dispatch forecast as an hourly series (MW)."""
    frame = pd.read_csv(path)
    index = pd.to_datetime(frame["datetime"], format="%m/%d/%Y %H:%M")
    series = pd.Series(frame["load_forecast"].to_numpy(dtype=np.float64), index=index)
    series.index.name = "datetime"
    return series[~series.index.duplicated()].sort_index().rename("operator_forecast")


def official_week_mask(timestamps: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Boolean mask: which target timestamps fall inside one of the 14 official test weeks."""
    values = pd.DatetimeIndex(timestamps)
    mask = np.zeros(len(values), dtype=bool)
    for start in OFFICIAL_TEST_WEEKS:
        first = pd.Timestamp(start)
        mask |= (values >= first) & (values <= first + pd.Timedelta(hours=167))
    return mask


def official_week_label(timestamps: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Label each timestamp with its official test week (``""`` outside them)."""
    values = pd.DatetimeIndex(timestamps)
    labels = np.full(len(values), "", dtype=object)
    for start in OFFICIAL_TEST_WEEKS:
        first = pd.Timestamp(start)
        inside = (values >= first) & (values <= first + pd.Timedelta(hours=167))
        labels[inside] = first.strftime("%Y-%m-%d")
    return labels
