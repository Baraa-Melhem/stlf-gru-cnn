"""End-to-end checks on CPU: backtest, persistence, reproducibility, CLI round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stlf.config import save_config
from stlf.data.splits import final_training_split, make_walk_forward_folds
from stlf.evaluation.backtest import quantile_column, run_backtest
from stlf.evaluation.report import load_run, save_run, summarize
from stlf.forecasters import build_forecaster, load_forecaster
from tests.conftest import tiny_config


def test_neural_backtest_produces_sorted_finite_quantiles(data, cfg) -> None:
    folds = make_walk_forward_folds(data.index, cfg.backtest)[:1]
    result = run_backtest(cfg, data, folds=folds, seeds=[0])
    frame = result.predictions
    columns = [quantile_column(q) for q in cfg.model.quantiles]
    assert len(frame) % cfg.window.horizon == 0 and len(frame) > 0
    assert np.isfinite(frame[columns].to_numpy()).all()
    assert (np.diff(frame[columns].to_numpy(), axis=1) >= 0).all()
    assert (frame["target_time"] >= folds[0].test_start).all()
    assert (frame["target_time"] <= folds[0].test_end).all()
    assert (frame["origin"].dt.hour == 0).all()
    summary = summarize(result, data, cfg)
    assert summary["overall"]["mape"] < 20


def test_seasonal_naive_is_exact(data) -> None:
    cfg = tiny_config(model={"name": "seasonal_naive", "seasonal_period": 168})
    folds = make_walk_forward_folds(data.index, cfg.backtest)[:1]
    frame = run_backtest(cfg, data, folds=folds, seeds=[0]).predictions
    expected = data.frame["load"].reindex(frame["target_time"] - pd.Timedelta(hours=168))
    assert np.allclose(frame[quantile_column(0.5)].to_numpy(), expected.to_numpy())


@pytest.mark.parametrize("model", ["gru_cnn", "lightgbm", "legacy_gru_cnn"])
def test_save_load_round_trip_reproduces_predictions(data, tmp_path: Path, model: str) -> None:
    sections: dict = {"model": {"name": model}}
    if model == "lightgbm":
        sections["model"]["gbm"] = {"n_estimators": 30, "lags": [24, 48],
                                    "early_stopping_rounds": 10}  # fmt: skip
    if model == "legacy_gru_cnn":
        sections.update(model={"name": model, "quantiles": [0.5]}, train={
            "loss": "mse", "max_epochs": 2, "device": "cpu", "amp": False,
        }, features={"target_lags": []})  # fmt: skip
    cfg = tiny_config(**sections)
    forecaster = build_forecaster(cfg, seed=0)
    forecaster.fit(data, final_training_split(data.index, val_days=7))
    origins = np.array([400, 800, 1200])
    before = forecaster.predict(data, origins)
    forecaster.save(tmp_path)
    after = load_forecaster(tmp_path, cfg).predict(data, origins)
    assert np.allclose(before, after, atol=1e-4)


def test_same_seed_gives_identical_cpu_training(data) -> None:
    cfg = tiny_config(train={"max_epochs": 1, "device": "cpu", "amp": False,
                             "deterministic": True, "batch_size": 64})  # fmt: skip
    split = final_training_split(data.index, val_days=7)
    origins = np.array([500, 900])
    runs = []
    for _ in range(2):
        forecaster = build_forecaster(cfg, seed=7)
        forecaster.fit(data, split)
        runs.append(forecaster.predict(data, origins))
    assert np.array_equal(runs[0], runs[1])


def test_run_persistence(data, cfg, tmp_path: Path) -> None:
    folds = make_walk_forward_folds(data.index, cfg.backtest)[:1]
    result = run_backtest(cfg, data, folds=folds, seeds=[0, 1])
    summary = summarize(result, data, cfg)
    save_run(result, summary, cfg, tmp_path / "run")
    saved = load_run(tmp_path / "run")
    assert saved.metrics["seeds"] == [0, 1]
    assert len(saved.best_predictions()) == len(result.predictions) // 2
    assert saved.history is not None and set(saved.history["run"]) == {
        f"{folds[0].name}/seed0",
        f"{folds[0].name}/seed1",
    }


def test_train_and_predict_cli(raw, tmp_path: Path) -> None:
    from stlf.cli import predict, train

    data_path = tmp_path / "data.csv"
    raw.reset_index().to_csv(data_path, index=False)
    cfg = tiny_config(data={"raw_dir": str(tmp_path), "load_file": "data.csv"})
    config_path = tmp_path / "experiment.yaml"
    save_config(cfg, config_path)
    artifact = tmp_path / "artifact"
    assert train.main(["--config", str(config_path), "--out", str(artifact)]) == 0
    assert (artifact / "metadata.json").exists()
    out = tmp_path / "forecast.csv"
    origin = str(raw.index[-25])
    assert predict.main(["--artifact", str(artifact), "--origin", origin, "--out", str(out)]) == 0
    forecast = pd.read_csv(out)
    assert len(forecast) == 24
    assert forecast["actual"].notna().all()


def test_combine_averages_member_quantiles(raw, tmp_path: Path) -> None:
    from stlf.cli import evaluate
    from stlf.features.engineering import prepare_data

    raw.reset_index().to_csv(tmp_path / "data.csv", index=False)
    data_section = {"raw_dir": str(tmp_path), "load_file": "data.csv"}
    runs = tmp_path / "runs"
    for name, period in (("weekly", 168), ("daily", 24)):
        cfg = tiny_config(
            name=name,
            data=data_section,
            model={"name": "seasonal_naive", "seasonal_period": period},
        )
        data = prepare_data(raw, cfg)
        folds = make_walk_forward_folds(data.index, cfg.backtest)[:1]
        result = run_backtest(cfg, data, folds=folds, seeds=[0])
        save_run(result, summarize(result, data, cfg), cfg, runs / name)
    argv = ["--combine", "weekly", "daily", "--name", "both", "--runs-dir", str(runs)]
    assert evaluate.main(argv) == 0
    combined = load_run(runs / "both").predictions.set_index(["origin", "horizon"])
    weekly = load_run(runs / "weekly").predictions.set_index(["origin", "horizon"])
    daily = load_run(runs / "daily").predictions.set_index(["origin", "horizon"])
    median = quantile_column(0.5)
    expected = (weekly[median] + daily[median]) / 2
    assert np.allclose(combined[median], expected.loc[combined.index])
