from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from stlf.data.windowing import WindowBatcher, build_tensors, select_origins
from stlf.features.scaling import StandardScaler


def test_origins_respect_target_bounds_hour_and_history() -> None:
    index = pd.date_range("2020-01-01 01:00", periods=24 * 20, freq="h")
    first, last = pd.Timestamp("2020-01-05 01:00"), pd.Timestamp("2020-01-12 00:00")
    origins = select_origins(index, lookback=48, horizon=24, first_target=first,
                             last_target=last, hour=0)  # fmt: skip
    assert len(origins) == 7
    assert (index[origins].hour == 0).all()
    assert (index[origins + 1] >= first).all() and (index[origins + 24] <= last).all()
    assert (origins - 48 + 1 >= 0).all()


def test_batches_never_mix_past_and_future(data) -> None:
    scaler = StandardScaler.fit(data.frame, data.spec.scaled)
    tensors = build_tensors(scaler.transform(data.frame), data.spec, torch.device("cpu"))
    origins = np.array([200, 500, 900])
    batch = next(iter(WindowBatcher(tensors, origins, lookback=48, horizon=24, batch_size=8,
                                    target_lags=(24, 48))))  # fmt: skip
    target = tensors.target
    for b, origin in enumerate(origins):
        # Encoder: channel 0 is the target, ending exactly at the origin.
        assert torch.equal(batch["past"][b, :, 0], target[origin - 47 : origin + 1])
        # Labels: the 24 hours strictly after the origin.
        assert torch.equal(batch["target"][b], target[origin + 1 : origin + 25])
        # Lags: y(t-24) and y(t-48) for each target hour t, all at or before the origin.
        assert torch.equal(batch["future_lags"][b, :, 0], target[origin - 23 : origin + 1])
        assert torch.equal(batch["future_lags"][b, :, 1], target[origin - 47 : origin - 23])
    assert batch["future"].shape == (3, 24, tensors.n_known)
    assert batch["holiday_future"].dtype == torch.long


def test_lags_shorter_than_horizon_are_refused(data) -> None:
    scaler = StandardScaler.fit(data.frame, data.spec.scaled)
    tensors = build_tensors(scaler.transform(data.frame), data.spec, torch.device("cpu"))
    with pytest.raises(ValueError, match="leak"):
        WindowBatcher(tensors, np.array([200]), lookback=48, horizon=24, batch_size=1,
                      target_lags=(12,))  # fmt: skip


def test_shuffled_batches_cover_every_origin_once(data) -> None:
    scaler = StandardScaler.fit(data.frame, data.spec.scaled)
    tensors = build_tensors(scaler.transform(data.frame), data.spec, torch.device("cpu"))
    origins = np.arange(100, 400)
    batcher = WindowBatcher(tensors, origins, lookback=48, horizon=24, batch_size=64,
                            shuffle=True, seed=3)  # fmt: skip
    firsts = torch.cat([b["past"][:, -1, 0] for b in batcher])
    assert len(batcher) == 5
    assert torch.equal(firsts.sort().values, tensors.target[origins].sort().values)
