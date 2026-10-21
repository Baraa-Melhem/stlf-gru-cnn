"""Hybrid CNN-GRU-attention network for direct multi-horizon, multi-quantile load forecasting."""

from __future__ import annotations

import torch
from torch import nn

from stlf.config import GRUCNNConfig
from stlf.models.layers import GatedResidualNetwork, ResidualConvBlock, RevIN


class HybridGRUCNN(nn.Module):
    """Encoder-decoder forecaster built around the GRU-CNN idea of the original project.

    Data flow for one forecast origin::

        past [L, C_past] ─ RevIN(load) ─ Linear ─ causal residual CNN ─ GRU ─► memory [L, d]
                                                                         └──► context [d]
        future [H, C_known] ─ Linear + horizon embedding ─► queries [H, d]
        queries ⨯ memory ─ multi-head cross-attention ─► attended [H, d]
        [queries ‖ context ‖ attended] ─ GRN ─ Linear ─► quantiles [H, Q] ─ RevIN⁻¹

    * The CNN extracts local intra-day patterns *before* the GRU (sequential CNN→GRU) so the
      recurrent layer integrates de-noised local features rather than raw readings.
    * Cross-attention lets every target hour look up the most relevant history directly — in
      practice the same hour yesterday and one week ago — instead of squeezing a week into a
      single hidden state.
    * Each query also carries explicit anchors: the load at the same hour yesterday and last
      week, and the last observed load and last-24 h mean at the origin.
    * All ``H`` hours are produced in one shot (direct strategy): no recursive error build-up.
    """

    def __init__(
        self,
        *,
        n_past: int,
        n_known: int,
        n_lags: int,
        lookback: int,
        horizon: int,
        n_quantiles: int,
        n_holidays: int,
        cfg: GRUCNNConfig,
    ) -> None:
        super().__init__()
        d = cfg.d_model
        self.use_attention = cfg.use_attention
        self.revin = RevIN() if cfg.use_revin else None
        self.revin_skip_holidays = cfg.revin_exclude_holidays

        emb_dim = cfg.holiday_embedding_dim
        self.holiday_embedding = nn.Embedding(n_holidays, emb_dim) if emb_dim > 0 else None

        self.input_proj = nn.Linear(n_past + emb_dim, d)
        self.cnn = (
            nn.Sequential(
                *[
                    ResidualConvBlock(d, cfg.kernel_size, cfg.dilation_base**i, cfg.dropout)
                    for i in range(cfg.cnn_blocks)
                ]
            )
            if cfg.use_cnn
            else nn.Identity()
        )
        self.gru = (
            nn.GRU(
                d,
                cfg.gru_hidden,
                num_layers=cfg.gru_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
            )
            if cfg.use_gru
            else None
        )
        encoder_dim = cfg.gru_hidden if cfg.use_gru else d

        self.memory_proj = nn.Linear(encoder_dim, d)
        self.memory_norm = nn.LayerNorm(d)
        self.position = nn.Parameter(torch.zeros(1, lookback, d))
        self.context_proj = nn.Linear(encoder_dim, d)

        self.origin_anchors = cfg.origin_anchors
        n_anchors = 2 if cfg.origin_anchors else 0
        self.query_proj = nn.Linear(n_known + n_lags + n_anchors + emb_dim, d)
        self.horizon_embedding = nn.Parameter(torch.zeros(1, horizon, d))
        self.attention = (
            nn.MultiheadAttention(d, cfg.attention_heads, dropout=cfg.dropout, batch_first=True)
            if cfg.use_attention
            else None
        )
        decoder_in = d * (3 if cfg.use_attention else 2)
        self.decoder = GatedResidualNetwork(decoder_in, 2 * d, d, cfg.dropout)
        self.head = nn.Linear(d, n_quantiles)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.horizon_embedding, std=0.02)

    def _with_holidays(self, x: torch.Tensor, holidays: torch.Tensor) -> torch.Tensor:
        if self.holiday_embedding is None:
            return x
        return torch.cat([x, self.holiday_embedding(holidays)], dim=-1)

    def forward(
        self,
        past: torch.Tensor,
        future: torch.Tensor,
        holiday_past: torch.Tensor,
        holiday_future: torch.Tensor,
        future_lags: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Forecast all horizons.

        Args:
            past: ``[B, L, C_past]`` encoder inputs, scaled target in channel 0.
            future: ``[B, H, C_known]`` known covariates over the horizon.
            holiday_past: ``[B, L]`` holiday ids over the history.
            holiday_future: ``[B, H]`` holiday ids over the horizon.
            future_lags: ``[B, H, K]`` scaled load K lags before each target hour; normalised
                with the same RevIN statistics as the history so the decoder sees them on
                the same scale as the encoder.
            return_attention: Also return the head-averaged attention map ``[B, H, L]``.

        Returns:
            ``[B, H, Q]`` quantile forecasts in the scaled target space.
        """
        stats: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.revin is not None:
            weights = (holiday_past == 0) if self.revin_skip_holidays else None
            load, mean, std = self.revin.normalize(past[..., 0], weights)
            past = torch.cat([load.unsqueeze(-1), past[..., 1:]], dim=-1)
            future_lags = self.revin.normalize_with(future_lags, mean, std)
            stats = (mean, std)
        future = torch.cat([future, future_lags], dim=-1)
        if self.origin_anchors:
            # Persistence anchors: last observed load and the mean of the last 24 h, as seen
            # by every target hour (the horizon embedding lets their weight decay with lead).
            load = past[..., 0]
            anchors = torch.stack([load[:, -1], load[:, -24:].mean(dim=1)], dim=-1)
            future = torch.cat([future, anchors.unsqueeze(1).expand(-1, future.shape[1], -1)], -1)

        x = self.cnn(self.input_proj(self._with_holidays(past, holiday_past)))
        if self.gru is not None:
            sequence, hidden = self.gru(x)
            context = hidden[-1]
        else:
            sequence, context = x, x[:, -1]
        memory = self.memory_norm(self.memory_proj(sequence) + self.position)

        queries = self.query_proj(self._with_holidays(future, holiday_future))
        queries = queries + self.horizon_embedding
        horizon = queries.shape[1]
        parts = [queries, self.context_proj(context).unsqueeze(1).expand(-1, horizon, -1)]
        weights: torch.Tensor | None = None
        if self.attention is not None:
            attended, weights = self.attention(
                queries, memory, memory, need_weights=return_attention
            )
            parts.append(attended)

        out = self.head(self.decoder(torch.cat(parts, dim=-1)))
        if stats is not None and self.revin is not None:
            out = self.revin.denormalize(out, *stats)
        return (out, weights) if return_attention else out
