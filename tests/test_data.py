from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stlf.data.loader import (
    DataValidationError,
    flag_load_anomalies,
    load_dataset,
    official_week_mask,
)
from stlf.data.splits import final_training_split, make_walk_forward_folds
from stlf.features.engineering import calendar_features, prepare_data
from stlf.features.scaling import StandardScaler
from tests.conftest import make_raw, tiny_config


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.reset_index().to_csv(path, index=False)
    return path


def test_loader_round_trip(tmp_path: Path) -> None:
    raw = make_raw(days=10)
    loaded = load_dataset(_write(raw, tmp_path / "data.csv"))
    pd.testing.assert_frame_equal(loaded, raw[loaded.columns], check_freq=False)


def test_loader_rejects_gaps_and_duplicates(tmp_path: Path) -> None:
    raw = make_raw(days=5)
    with pytest.raises(DataValidationError, match="missing hours"):
        load_dataset(_write(raw.drop(raw.index[10]), tmp_path / "gap.csv"))
    with pytest.raises(DataValidationError, match="duplicated"):
        load_dataset(_write(pd.concat([raw, raw.iloc[[3]]]), tmp_path / "dup.csv"))


def test_anomaly_flag_detects_blackout_and_is_causal() -> None:
    load = pd.Series(1000.0 + np.zeros(400))
    load.iloc[300:303] = 90.0
    flags = flag_load_anomalies(load, ratio=0.5, window=168)
    assert flags.iloc[300:303].all() and flags.sum() == 3
    changed = load.copy()
    changed.iloc[350:] = 5.0  # rewriting the future must not change past flags
    assert flag_load_anomalies(changed, 0.5, 168).iloc[:350].equals(flags.iloc[:350])


def test_cleaning_interpolates_anomalies_but_keeps_raw_actuals() -> None:
    raw = make_raw(days=20)
    raw.iloc[300, 0] = 50.0
    data = prepare_data(raw, tiny_config())
    assert data.anomaly.iloc[300]
    assert data.actual.iloc[300] == 50.0
    assert data.frame["load"].iloc[300] > 500


def test_calendar_features_are_cyclical() -> None:
    index = pd.date_range("2020-01-01", periods=24 * 7, freq="h")
    calendar = calendar_features(index)
    assert np.allclose(calendar["hour_sin"] ** 2 + calendar["hour_cos"] ** 2, 1.0)
    assert calendar["is_weekend"].sum() == 48


def test_future_weather_switch_moves_weather_to_past_only(raw: pd.DataFrame) -> None:
    known = prepare_data(raw, tiny_config()).spec
    hidden = prepare_data(raw, tiny_config(features={"future_weather": False})).spec
    assert "T2M_toc" in known.known and "T2M_toc" not in known.past_only
    assert "T2M_toc" in hidden.past_only and "T2M_toc" not in hidden.known
    assert hidden.past_inputs[0] == "load"


def test_scaler_statistics_ignore_rows_after_the_training_cut(data) -> None:
    cut = data.index[1000]
    scaler = StandardScaler.fit(data.frame.loc[:cut], data.spec.scaled)
    tampered = data.frame.copy()
    tampered.loc[tampered.index > cut, "load"] = 1e6
    again = StandardScaler.fit(tampered.loc[:cut], data.spec.scaled)
    assert scaler.to_dict() == again.to_dict()
    restored = scaler.inverse(scaler.transform(data.frame)["load"].to_numpy(), "load")
    assert np.allclose(restored, data.frame["load"].to_numpy())


def test_walk_forward_folds_are_ordered_and_tile_the_test_period(data) -> None:
    cfg = tiny_config()
    folds = make_walk_forward_folds(data.index, cfg.backtest)
    assert folds[0].test_start == pd.Timestamp(cfg.backtest.test_start)
    assert folds[-1].test_end == data.index[-1]
    for fold in folds:
        assert fold.train_end < fold.val_start <= fold.val_end < fold.test_start <= fold.test_end
        assert fold.val_end + pd.Timedelta(hours=1) == fold.test_start
    for previous, current in itertools.pairwise(folds):
        assert previous.test_end + pd.Timedelta(hours=1) == current.test_start


def test_final_split_holds_out_the_tail(data) -> None:
    split = final_training_split(data.index, val_days=7)
    assert split.val_end == data.index[-1]
    assert split.val_end - split.val_start == pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    assert split.train_end + pd.Timedelta(hours=1) == split.val_start


def test_official_week_mask() -> None:
    stamps = pd.DatetimeIndex(["2019-04-13 00:00", "2019-04-13 01:00", "2019-04-20 00:00",
                               "2019-04-20 01:00"])  # fmt: skip
    assert official_week_mask(stamps).tolist() == [False, True, True, False]
