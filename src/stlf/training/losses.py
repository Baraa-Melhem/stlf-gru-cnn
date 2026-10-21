"""Training objectives."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class PinballLoss(nn.Module):
    """Mean quantile (pinball) loss over all horizons and quantile levels.

    For level ``q`` the loss is ``max(q·e, (q-1)·e)`` with ``e = y - ŷ_q``. Minimising it makes
    ``ŷ_q`` the conditional ``q``-quantile, so one network yields a median point forecast and a
    calibrated prediction interval. At ``q = 0.5`` it is half the MAE, which is far less
    sensitive to the dataset's blackout spikes than the squared error used originally.
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(list(quantiles), dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``prediction``: ``[B, H, Q]``; ``target``: ``[B, H]``."""
        error = target.unsqueeze(-1) - prediction
        q = self.quantiles.to(prediction.dtype)
        return torch.maximum(q * error, (q - 1) * error).mean()


class PointLoss(nn.Module):
    """Wrap an elementwise regression loss for single-output ``[B, H, 1]`` predictions."""

    def __init__(self, criterion: nn.Module) -> None:
        super().__init__()
        self.criterion = criterion

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(prediction.squeeze(-1), target)


def build_loss(name: str, quantiles: Sequence[float]) -> nn.Module:
    """Factory for the configured objective (``pinball``, ``mse`` or ``huber``)."""
    if name == "pinball":
        return PinballLoss(quantiles)
    if name == "mse":
        return PointLoss(nn.MSELoss())
    if name == "huber":
        return PointLoss(nn.HuberLoss(delta=1.0))
    raise ValueError(f"unknown loss '{name}'")
