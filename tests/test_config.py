from __future__ import annotations

from pathlib import Path

import pytest

from stlf.config import ConfigError, load_config, save_config, to_dict
from tests.conftest import tiny_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_shipped_config_is_valid(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.window.horizon == 24


def test_base_inheritance_and_overrides() -> None:
    cfg = load_config(CONFIG_DIR / "ablation_no_cnn.yaml", ["train.max_epochs=3"])
    assert cfg.model.gru_cnn.use_cnn is False
    assert cfg.model.gru_cnn.d_model == 64  # inherited from gru_cnn.yaml
    assert cfg.train.max_epochs == 3
    assert cfg.backtest.seeds == (0,)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  learnin_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="learnin_rate"):
        load_config(path)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("window:\n  lookback: twelve\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"window\.lookback"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("model", {"quantiles": [0.1, 0.9]}, "median"),
        ("features", {"target_lags": [12]}, "target_lags"),
        ("train", {"loss": "mse"}, "point losses"),
        ("model", {"name": "transformer"}, "model.name"),
    ],
)
def test_invariants_are_enforced(section: str, values: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        tiny_config(**{section: values})


def test_save_and_reload_round_trip(tmp_path: Path) -> None:
    cfg = tiny_config()
    save_config(cfg, tmp_path / "config.yaml")
    assert to_dict(load_config(tmp_path / "config.yaml")) == to_dict(cfg)
