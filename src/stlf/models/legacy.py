"""PyTorch port of the original Kaggle notebook's GRU-CNN model, kept as a reference baseline.

The topology mirrors the 2024 Keras model layer for layer: a CNN branch (two ``kernel_size=1``
convolutions with no-op ``pool_size=1`` pooling, then Dense-128) in parallel with a three-layer
GRU stack (64-128-256, dropout 0.5), concatenated into Dense-64 → Dense-32 → output. Each hour
is a one-step "sequence" of that hour's covariates: the model never sees past load.

The only unavoidable deviation is the recurrent activation — Keras allowed ``activation='relu'``
inside the GRU; PyTorch GRUs are tanh-only.
"""

from __future__ import annotations

import torch
from torch import nn


class LegacyGRUCNN(nn.Module):
    """Time-distributed re-implementation of the original hybrid model."""

    def __init__(self, *, n_known: int, n_quantiles: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_known, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.gru1 = nn.GRU(n_known, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 128, batch_first=True)
        self.gru3 = nn.GRU(128, 256, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(
            nn.Linear(128 + 256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, n_quantiles),
        )

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        holiday_past: torch.Tensor,
        holiday_future: torch.Tensor,
        future_lags: torch.Tensor,
    ) -> torch.Tensor:
        """Score each future hour independently from that hour's covariates only.

        ``past``, holiday and lag inputs are accepted for interface compatibility and ignored,
        exactly as the original model had no access to load history.
        """
        batch, horizon, channels = future.shape
        steps = future.reshape(batch * horizon, 1, channels)  # one-timestep sequences
        cnn = self.cnn(steps.transpose(1, 2))
        gru, _ = self.gru1(steps)
        gru, _ = self.gru2(self.dropout(gru))
        gru, _ = self.gru3(self.dropout(gru))
        gru = self.dropout(gru[:, -1])
        out = self.head(torch.cat([cnn, gru], dim=-1))
        return out.reshape(batch, horizon, -1)
