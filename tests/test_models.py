from __future__ import annotations

import itertools

import pytest
import torch

from stlf.config import GRUCNNConfig
from stlf.models import HybridGRUCNN, LegacyGRUCNN
from stlf.models.layers import ResidualConvBlock, RevIN
from stlf.training.losses import PinballLoss

B, L, H, C_PAST, C_KNOWN, N_LAGS = 4, 48, 24, 10, 7, 2


def _holidays(shape: tuple[int, int], generator: torch.Generator) -> torch.Tensor:
    """Holiday ids: about one hour in ten is a holiday (id 1-22), the rest regular (0)."""
    ids = torch.randint(1, 23, shape, generator=generator)
    return torch.where(torch.rand(shape, generator=generator) < 0.1, ids, torch.zeros_like(ids))


def _inputs(seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, L, C_PAST, generator=generator),
        torch.randn(B, H, C_KNOWN, generator=generator),
        _holidays((B, L), generator),
        _holidays((B, H), generator),
        torch.randn(B, H, N_LAGS, generator=generator),
    )


def _model(**overrides: bool) -> HybridGRUCNN:
    cfg = GRUCNNConfig(d_model=16, gru_hidden=16, cnn_blocks=2, attention_heads=2, **overrides)
    return HybridGRUCNN(n_past=C_PAST, n_known=C_KNOWN, n_lags=N_LAGS, lookback=L, horizon=H,
                        n_quantiles=3, n_holidays=23, cfg=cfg)  # fmt: skip


@pytest.mark.parametrize(
    ("use_cnn", "use_gru", "use_attention", "use_revin"),
    [c for c in itertools.product([True, False], repeat=4) if c[0] or c[1]],
)
def test_every_ablation_produces_the_right_shape(use_cnn, use_gru, use_attention, use_revin):
    model = _model(use_cnn=use_cnn, use_gru=use_gru, use_attention=use_attention,
                   use_revin=use_revin)  # fmt: skip
    out = model(*_inputs())
    assert out.shape == (B, H, 3)
    PinballLoss([0.05, 0.5, 0.95])(out, torch.randn(B, H)).backward()


def test_convolutions_are_causal() -> None:
    """Changing hour j of the history must not change conv features at hours before j."""
    block = ResidualConvBlock(channels=8, kernel_size=3, dilation=2, dropout=0.0).eval()
    x = torch.randn(2, 30, 8)
    changed = x.clone()
    changed[:, 20] += 5.0
    with torch.no_grad():
        before, after = block(x), block(changed)
    assert torch.allclose(before[:, :20], after[:, :20])
    assert not torch.allclose(before[:, 20:], after[:, 20:])


def test_attention_map_is_a_distribution_over_the_history() -> None:
    model = _model().eval()
    _, weights = model(*_inputs(), return_attention=True)
    assert weights.shape == (B, H, L)
    assert torch.allclose(weights.sum(-1), torch.ones(B, H), atol=1e-5)


def test_holiday_aware_revin_ignores_masked_steps() -> None:
    revin = RevIN(affine=False)
    x = torch.tensor([[10.0, 10.0, 10.0, -50.0]])
    weights = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    _, mean, _ = revin.normalize(x, weights)
    assert mean.item() == 10.0
    _, mean_all, _ = revin.normalize(x, torch.zeros_like(weights))  # nothing selected
    assert mean_all.item() == x.mean().item()


def test_revin_round_trip() -> None:
    revin = RevIN()
    x = torch.randn(B, L) * 5 + 3
    normalised, mean, std = revin.normalize(x)
    restored = revin.denormalize(normalised.unsqueeze(-1), mean, std).squeeze(-1)
    assert torch.allclose(restored, x, atol=1e-4)


def test_revin_makes_forecasts_level_and_scale_equivariant() -> None:
    """Shifting/scaling the load history (and its lags) shifts/scales the forecast identically."""
    model = _model().eval()
    past, future, hp, hf, lags = _inputs()
    with torch.no_grad():
        base = model(past, future, hp, hf, lags)
        moved_past = past.clone()
        moved_past[..., 0] = 2.5 * past[..., 0] + 4.0
        moved = model(moved_past, future, hp, hf, 2.5 * lags + 4.0)
    assert torch.allclose(moved, 2.5 * base + 4.0, atol=1e-3)


def test_legacy_model_ignores_history() -> None:
    model = LegacyGRUCNN(n_known=C_KNOWN, n_quantiles=1).eval()
    past, future, hp, hf, lags = _inputs()
    with torch.no_grad():
        out = model(past, future, hp, hf, lags)
        again = model(torch.zeros_like(past), future, hp, hf, torch.zeros_like(lags))
    assert out.shape == (B, H, 1)
    assert torch.equal(out, again)


def test_pinball_loss_matches_definition() -> None:
    prediction = torch.tensor([[[0.0, 1.0]]])
    target = torch.tensor([[2.0]])
    expected = (0.1 * 2.0 + 0.9 * 1.0) / 2
    assert PinballLoss([0.1, 0.9])(prediction, target).item() == pytest.approx(expected)
