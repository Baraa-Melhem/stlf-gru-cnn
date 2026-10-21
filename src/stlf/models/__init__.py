"""Neural network architectures and their factory."""

from __future__ import annotations

from torch import nn

from stlf.config import ExperimentConfig
from stlf.features.engineering import N_HOLIDAY_IDS
from stlf.models.gru_cnn import HybridGRUCNN
from stlf.models.legacy import LegacyGRUCNN

__all__ = ["HybridGRUCNN", "LegacyGRUCNN", "build_network", "count_parameters"]


def build_network(cfg: ExperimentConfig, *, n_past: int, n_known: int) -> nn.Module:
    """Instantiate the network named by ``cfg.model.name`` for the given input widths."""
    n_quantiles = len(cfg.model.quantiles)
    if cfg.model.name == "gru_cnn":
        return HybridGRUCNN(
            n_past=n_past,
            n_known=n_known,
            n_lags=len(cfg.features.target_lags),
            lookback=cfg.window.lookback,
            horizon=cfg.window.horizon,
            n_quantiles=n_quantiles,
            n_holidays=N_HOLIDAY_IDS,
            cfg=cfg.model.gru_cnn,
        )
    if cfg.model.name == "legacy_gru_cnn":
        return LegacyGRUCNN(n_known=n_known, n_quantiles=n_quantiles)
    raise ValueError(f"'{cfg.model.name}' is not a neural model")


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
