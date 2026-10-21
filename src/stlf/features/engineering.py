"""Feature engineering: turn the raw frame into model-ready covariates plus a feature contract.

Covariates are split by *when they are known*, which is what makes a forecaster causal:

* ``known``     — known for the whole forecast horizon (calendar, holidays, school terms and,
  when ``future_weather`` is on, weather — i.e. a perfect-weather-forecast assumption).
* ``past_only`` — observed only up to the forecast origin (weather when ``future_weather``
  is off).
* the target    — the (anomaly-cleaned) load, observed only up to the origin.
* target lags   — ``y(t - k)`` for each forecast hour ``t`` with ``k >= horizon`` (same hour
  yesterday / last week). They are already observed at the origin, so they are legitimate
  decoder inputs; windowing gathers them from the target series on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stlf.config import ExperimentConfig
from stlf.data.loader import TARGET_COLUMN, flag_load_anomalies
from stlf.logging_utils import get_logger

logger = get_logger(__name__)

TARGET = "load"
HOLIDAY_ID = "holiday_id"
N_HOLIDAY_IDS = 23  # 0 = regular day, 1..22 = the holidays listed in the dataset metadata
CALENDAR_FEATURES: tuple[str, ...] = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend",
)  # fmt: skip
HOLIDAY_FEATURES: tuple[str, ...] = ("holiday", "school")


@dataclass(frozen=True)
class FeatureSpec:
    """Column contract between feature engineering, scaling, windowing and the models."""

    target: str
    past_only: tuple[str, ...]
    known: tuple[str, ...]
    scaled: tuple[str, ...]
    holiday_id: str | None
    target_lags: tuple[int, ...] = ()

    @property
    def past_inputs(self) -> tuple[str, ...]:
        """Columns of the encoder input; the target is always channel 0."""
        return (self.target, *self.past_only, *self.known)


@dataclass(frozen=True)
class PreparedData:
    """Model-ready features plus the raw observations used for scoring."""

    frame: pd.DataFrame
    spec: FeatureSpec
    actual: pd.Series
    anomaly: pd.Series

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index  # type: ignore[return-value]


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical hour/day-of-week/day-of-year encodings and a weekend flag.

    Sine/cosine pairs put 23:00 next to 00:00 and December next to January, which a raw
    integer (or the Unix timestamp used by the original notebook) cannot express.
    """
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy() - 1 + hour / 24.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_weekend": (dow >= 5).astype(np.float64),
        },
        index=index,
    )


def prepare_data(raw: pd.DataFrame, cfg: ExperimentConfig) -> PreparedData:
    """Build every feature the experiment asks for and return them with their contract."""
    feats, data_cfg = cfg.features, cfg.data
    actual = raw[TARGET_COLUMN].rename("actual")

    if data_cfg.clean_anomalies:
        anomaly = flag_load_anomalies(actual, data_cfg.anomaly_ratio, data_cfg.anomaly_window)
        cleaned = actual.mask(anomaly).interpolate(method="time", limit_area="inside")
        if anomaly.any():
            logger.info("Interpolated %d anomalous load hours (blackouts)", int(anomaly.sum()))
    else:
        anomaly = pd.Series(False, index=raw.index)
        cleaned = actual

    columns: dict[str, pd.Series | np.ndarray] = {TARGET: cleaned}
    known: list[str] = []
    past_only: list[str] = []
    scaled: list[str] = [TARGET]

    if feats.calendar:
        calendar = calendar_features(raw.index)
        columns.update({name: calendar[name] for name in CALENDAR_FEATURES})
        known.extend(CALENDAR_FEATURES)

    holiday_id: str | None = None
    if feats.holidays:
        columns["holiday"] = raw["holiday"]
        columns["school"] = raw["school"]
        columns[HOLIDAY_ID] = raw["Holiday_ID"].clip(0, N_HOLIDAY_IDS - 1)
        known.extend(HOLIDAY_FEATURES)
        holiday_id = HOLIDAY_ID
        # Was the same hour k hours earlier a holiday? Calendar facts, known in advance, that
        # tell a model when a same-hour load lag (yesterday / last week) is atypical.
        for lag in feats.target_lags:
            name = f"holiday_lag{lag}"
            columns[name] = raw["holiday"].shift(lag).fillna(0.0)
            known.append(name)

    weather: list[str] = list(feats.weather_columns)
    for name in weather:
        columns[name] = raw[name]
    if feats.mean_temperature:
        temperature_columns = [name for name in raw.columns if name.startswith("T2M_")]
        columns["t2m_mean"] = raw[temperature_columns].mean(axis=1)
        weather.append("t2m_mean")
    (known if feats.future_weather else past_only).extend(weather)
    scaled.extend(weather)

    if feats.timestamp:
        columns["timestamp"] = raw.index.asi8 / 1e9
        known.append("timestamp")
        scaled.append("timestamp")

    frame = pd.DataFrame(columns, index=raw.index)
    spec = FeatureSpec(
        target=TARGET,
        past_only=tuple(past_only),
        known=tuple(known),
        scaled=tuple(scaled),
        holiday_id=holiday_id,
        target_lags=feats.target_lags,
    )
    return PreparedData(frame=frame, spec=spec, actual=actual, anomaly=anomaly.rename("anomaly"))
