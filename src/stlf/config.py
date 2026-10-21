"""Typed experiment configuration.

Every experiment is described by one frozen :class:`ExperimentConfig`. YAML files map onto it
one-to-one; a file may name a ``base`` file whose values it overrides, and command-line
``--set section.key=value`` overrides are applied last. Unknown keys raise immediately, so a
typo in a config can never be silently ignored.
"""

from __future__ import annotations

import copy
import dataclasses
import types
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

WEATHER_COLUMNS: tuple[str, ...] = (
    "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
    "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
    "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
)  # fmt: skip

NEURAL_MODELS = frozenset({"gru_cnn", "legacy_gru_cnn"})
ALL_MODELS = NEURAL_MODELS | {"lightgbm", "seasonal_naive", "operator"}


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class DataConfig:
    """Where the raw files live and how the load series is cleaned."""

    raw_dir: str = "data/raw"
    load_file: str = "continuous dataset.csv"
    operator_forecast_file: str = "weekly pre-dispatch forecast.csv"
    clean_anomalies: bool = True
    anomaly_ratio: float = 0.5
    anomaly_window: int = 168


@dataclass(frozen=True)
class FeatureConfig:
    """Which covariates are built and whether weather is treated as known in the future."""

    calendar: bool = True
    holidays: bool = True
    weather_columns: tuple[str, ...] = WEATHER_COLUMNS
    mean_temperature: bool = True
    future_weather: bool = True
    timestamp: bool = False
    target_lags: tuple[int, ...] = (24, 168)


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window geometry: hours of history seen and hours forecast per origin."""

    lookback: int = 168
    horizon: int = 24
    train_stride: int = 1
    origin_hour: int = 0


@dataclass(frozen=True)
class GRUCNNConfig:
    """Hyper-parameters of the hybrid CNN-GRU-attention network."""

    d_model: int = 64
    cnn_blocks: int = 3
    kernel_size: int = 3
    dilation_base: int = 2
    gru_hidden: int = 128
    gru_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    holiday_embedding_dim: int = 4
    use_cnn: bool = True
    use_gru: bool = True
    use_attention: bool = True
    use_revin: bool = True
    revin_exclude_holidays: bool = True
    origin_anchors: bool = False


@dataclass(frozen=True)
class GBMConfig:
    """Hyper-parameters of the LightGBM direct multi-horizon baseline."""

    n_estimators: int = 3000
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    lags: tuple[int, ...] = (24, 48, 72, 168, 336)
    origin_stride: int = 4
    early_stopping_rounds: int = 100


@dataclass(frozen=True)
class ModelConfig:
    """Model family selector plus the typed hyper-parameter block of each family."""

    name: str = "gru_cnn"
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    conformal: bool = True
    gru_cnn: GRUCNNConfig = field(default_factory=GRUCNNConfig)
    gbm: GBMConfig = field(default_factory=GBMConfig)
    seasonal_period: int = 168


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings for the neural models."""

    loss: str = "pinball"
    batch_size: int = 256
    max_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    schedule: str = "cosine"
    warmup_epochs: float = 1.0
    patience: int = 8
    grad_clip: float = 1.0
    ema_decay: float = 0.0
    amp: bool = True
    device: str = "auto"
    deterministic: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    """Expanding-window walk-forward evaluation protocol."""

    test_start: str = "2019-01-01 01:00"
    test_end: str | None = None
    fold_months: int = 3
    val_days: int = 56
    seeds: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration of one experiment."""

    name: str = "gru_cnn"
    seed: int = 0
    output_dir: str = "reports/runs"
    artifact_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @property
    def is_neural(self) -> bool:
        return self.model.name in NEURAL_MODELS


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"circular 'base' reference involving {path}")
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, Mapping):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    base_name = content.pop("base", None)
    if base_name is None:
        return dict(content)
    base = _read_yaml(path.parent / base_name, _seen | {path})
    return _deep_merge(base, content)


def _set_dotted(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    node = tree
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override '{dotted_key}' descends into a non-mapping value")
    node[leaf] = value


def apply_overrides(tree: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` strings; values are parsed as YAML scalars or lists."""
    result = copy.deepcopy(tree)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like section.key=value")
        key, raw_value = item.split("=", 1)
        _set_dotted(result, key.strip(), yaml.safe_load(raw_value))
    return result


def _coerce(value: Any, annotation: Any, location: str) -> Any:
    origin = typing.get_origin(annotation)
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{location} must be a mapping")
        return _build(annotation, value, location)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(annotation)
        if value is None and type(None) in options:
            return None
        non_null = [option for option in options if option is not type(None)]
        return _coerce(value, non_null[0], location)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{location} must be a list")
        (item_type, _ellipsis) = typing.get_args(annotation)
        return tuple(_coerce(item, item_type, f"{location}[]") for item in value)
    if annotation is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if annotation in (int, str, bool) and not isinstance(value, annotation):
        raise ConfigError(f"{location} must be {annotation.__name__}, got {value!r}")
    if annotation is int and isinstance(value, bool):
        raise ConfigError(f"{location} must be int, got {value!r}")
    return value


def _build(cls: type, values: Mapping[str, Any], location: str = "") -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        where = location or "top level"
        raise ConfigError(f"unknown key(s) {sorted(unknown)} at {where}")
    kwargs = {
        name: _coerce(value, hints[name], f"{location}.{name}".lstrip("."))
        for name, value in values.items()
    }
    return cls(**kwargs)


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    """Check cross-field invariants that the type system cannot express."""
    model, window = cfg.model, cfg.window
    if model.name not in ALL_MODELS:
        raise ConfigError(f"model.name must be one of {sorted(ALL_MODELS)}, got '{model.name}'")
    quantiles = model.quantiles
    if not quantiles or list(quantiles) != sorted(set(quantiles)):
        raise ConfigError("model.quantiles must be strictly increasing")
    if 0.5 not in quantiles or not all(0.0 < q < 1.0 for q in quantiles):
        raise ConfigError("model.quantiles must lie in (0, 1) and contain the median 0.5")
    if cfg.train.loss not in {"pinball", "mse", "huber"}:
        raise ConfigError("train.loss must be 'pinball', 'mse' or 'huber'")
    if not 0.0 <= cfg.train.ema_decay < 1.0:
        raise ConfigError("train.ema_decay must be in [0, 1)")
    if cfg.train.schedule not in {"cosine", "constant"}:
        raise ConfigError("train.schedule must be 'cosine' or 'constant'")
    if cfg.train.loss != "pinball" and quantiles != (0.5,):
        raise ConfigError("point losses (mse/huber) require model.quantiles: [0.5]")
    if window.lookback < 1 or window.horizon < 1 or window.train_stride < 1:
        raise ConfigError("window.lookback, horizon and train_stride must be positive")
    if not 0 <= window.origin_hour <= 23:
        raise ConfigError("window.origin_hour must be within [0, 23]")
    if model.gru_cnn.kernel_size % 2 == 0:
        raise ConfigError("model.gru_cnn.kernel_size must be odd so 'same' padding is symmetric")
    if model.gru_cnn.use_attention and model.gru_cnn.d_model % model.gru_cnn.attention_heads:
        raise ConfigError("model.gru_cnn.d_model must be divisible by attention_heads")
    if not (model.gru_cnn.use_cnn or model.gru_cnn.use_gru):
        raise ConfigError("the hybrid network needs at least one of use_cnn / use_gru")
    if any(lag < window.horizon for lag in cfg.features.target_lags):
        raise ConfigError("every features.target_lags entry must be >= window.horizon")
    if model.name == "lightgbm" and min(model.gbm.lags) < window.horizon:
        raise ConfigError("every model.gbm.lags entry must be >= window.horizon to stay causal")
    if model.name == "seasonal_naive" and model.seasonal_period < window.horizon:
        raise ConfigError("model.seasonal_period must be >= window.horizon to stay causal")
    if cfg.backtest.fold_months < 1 or cfg.backtest.val_days < 1 or not cfg.backtest.seeds:
        raise ConfigError("backtest needs fold_months >= 1, val_days >= 1 and at least one seed")
    return cfg


def config_from_dict(values: Mapping[str, Any]) -> ExperimentConfig:
    """Build and validate an :class:`ExperimentConfig` from a nested mapping."""
    return validate(_build(ExperimentConfig, values))


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    """Load a YAML experiment file, resolving ``base`` inheritance and CLI overrides."""
    tree = _read_yaml(Path(path))
    return config_from_dict(apply_overrides(tree, overrides))


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Serialise a config to plain YAML/JSON-friendly Python types."""

    def convert(value: Any) -> Any:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(dataclasses.asdict(cfg))


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write the fully-resolved config so every run is reproducible from its output folder."""
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(cfg), handle, sort_keys=False)
