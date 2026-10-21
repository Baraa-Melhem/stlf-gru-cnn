"""Reusable building blocks for the hybrid forecaster."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalisation of the target (Kim et al., ICLR 2022).

    Each window's load history is normalised by its own mean and standard deviation, and the
    forecast is mapped back with the same statistics. The network therefore learns load
    *shape* rather than absolute level, which is what lets it follow the multi-year growth
    trend and the ~15 % COVID-19 drop in 2020 without retraining.
    """

    def __init__(self, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def normalize(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalise ``x`` of shape ``[B, L]``; returns the result and ``(mean, std)``.

        ``weights`` (``[B, L]``, 0/1) restricts the statistics to selected time steps, e.g.
        non-holiday hours, so a holiday inside the window does not drag the level down and
        inflate the scale. Windows with no selected step fall back to plain statistics.
        """
        if weights is None:
            weights = torch.ones_like(x)
        weights = weights.to(x.dtype)
        empty = weights.sum(dim=1, keepdim=True) < 1
        weights = torch.where(empty, torch.ones_like(weights), weights)
        total = weights.sum(dim=1, keepdim=True)
        mean = (x * weights).sum(dim=1, keepdim=True) / total
        var = ((x - mean) ** 2 * weights).sum(dim=1, keepdim=True) / total
        std = torch.sqrt(var + self.eps)
        return self.normalize_with(x, mean, std), mean, std

    def normalize_with(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Normalise any tensor whose leading dim is the batch with given ``[B, 1]`` stats."""
        shape = (-1,) + (1,) * (x.dim() - 1)
        out = (x - mean.reshape(shape)) / std.reshape(shape)
        if self.affine:
            out = out * self.gamma + self.beta
        return out

    def denormalize(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize` for predictions ``y`` of shape ``[B, H, Q]``."""
        if self.affine:
            y = (y - self.beta) / (self.gamma + self.eps)
        return y * std.unsqueeze(-1) + mean.unsqueeze(-1)


class ResidualConvBlock(nn.Module):
    """Pre-norm residual block of two dilated, causal 1-D convolutions over the time axis.

    Stacking blocks with dilations 1, 2, 4, ... grows the receptive field exponentially, so
    three blocks with kernel 3 already see 29 consecutive hours — a full daily cycle — while
    the original notebook's ``kernel_size=1`` convolutions over a single timestep could not
    see any temporal neighbourhood at all.

    Padding is *causal* (left only). With symmetric "same" padding the last positions of the
    window — the hours just before the forecast origin, the most informative ones — would be
    convolved with zero padding; causal padding keeps every output a clean function of real
    observations at and before its own time step.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, L, C]`` -> ``[B, L, C]``."""
        h = self.norm(x).transpose(1, 2)
        h = self.dropout(F.gelu(self.conv1(F.pad(h, (self.left_pad, 0)))))
        h = self.dropout(self.conv2(F.pad(h, (self.left_pad, 0))))
        return x + h.transpose(1, 2)


class GatedResidualNetwork(nn.Module):
    """GRN from the Temporal Fusion Transformer (Lim et al., 2021).

    ``LayerNorm(skip(x) + GLU(W2 · ELU(W1 · x)))`` — the gate lets the block fall back to a
    near-identity mapping when the extra non-linearity does not help, which keeps a small
    model from over-fitting a few years of hourly data.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = F.glu(self.dropout(self.fc2(h)), dim=-1)
        return self.norm(self.skip(x) + h)
