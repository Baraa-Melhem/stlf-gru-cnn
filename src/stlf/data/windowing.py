"""Sliding-window sampling for direct multi-horizon forecasting.

A *forecast origin* ``o`` is the integer position of the last observed hour. The encoder sees
rows ``o-L+1 .. o`` and the model predicts rows ``o+1 .. o+H``. Every window is gathered on the
fly from a few contiguous tensors with one fancy-indexing call per batch, so a 40k-origin
training set costs no extra memory and no Python-level per-sample work.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from stlf.features.engineering import FeatureSpec


@dataclass(frozen=True)
class WindowTensors:
    """Full-length, already-scaled series that windows are cut from.

    Attributes:
        past: ``[T, C_past]`` encoder inputs, target in channel 0.
        known: ``[T, C_known]`` covariates known over the horizon.
        target: ``[T]`` scaled target.
        holiday: ``[T]`` integer holiday ids (zeros if holidays are disabled).
    """

    past: torch.Tensor
    known: torch.Tensor
    target: torch.Tensor
    holiday: torch.Tensor

    @property
    def n_past(self) -> int:
        return int(self.past.shape[1])

    @property
    def n_known(self) -> int:
        return int(self.known.shape[1])


def build_tensors(scaled: pd.DataFrame, spec: FeatureSpec, device: torch.device) -> WindowTensors:
    """Lay out a scaled feature frame as the tensors :class:`WindowBatcher` slices from."""

    def as_tensor(columns: tuple[str, ...]) -> torch.Tensor:
        if not columns:
            return torch.zeros((len(scaled), 0), dtype=torch.float32, device=device)
        values = scaled[list(columns)].to_numpy(dtype=np.float32)
        return torch.from_numpy(values).to(device)

    holiday = (
        scaled[spec.holiday_id].to_numpy(dtype=np.int64)
        if spec.holiday_id is not None
        else np.zeros(len(scaled), dtype=np.int64)
    )
    return WindowTensors(
        past=as_tensor(spec.past_inputs),
        known=as_tensor(spec.known),
        target=torch.from_numpy(scaled[spec.target].to_numpy(dtype=np.float32)).to(device),
        holiday=torch.from_numpy(holiday).to(device),
    )


def select_origins(
    index: pd.DatetimeIndex,
    *,
    lookback: int,
    horizon: int,
    first_target: pd.Timestamp | None = None,
    last_target: pd.Timestamp | None = None,
    stride: int = 1,
    hour: int | None = None,
) -> np.ndarray:
    """Return origin positions whose full window fits the series and the target bounds.

    Args:
        index: Hourly timestamps of the series.
        lookback: Encoder length; origins need ``lookback - 1`` earlier rows.
        horizon: Forecast length; origins need ``horizon`` later rows.
        first_target: Earliest timestamp allowed as a *first* forecast target (inclusive).
        last_target: Latest timestamp allowed as a *last* forecast target (inclusive). Bounding
            targets (not inputs) is what keeps train/validation/test labels disjoint while
            still letting a test window read the history before it.
        stride: Keep every ``stride``-th admissible origin.
        hour: If given, keep only origins at this hour of day (e.g. 0 for day-ahead at midnight).
    """
    positions = np.arange(lookback - 1, len(index) - horizon)
    if positions.size == 0:
        return positions
    first = index[positions + 1]
    last = index[positions + horizon]
    keep = np.ones(positions.size, dtype=bool)
    if first_target is not None:
        keep &= first >= first_target
    if last_target is not None:
        keep &= last <= last_target
    if hour is not None:
        keep &= index[positions].hour == hour
    return positions[keep][::stride]


class WindowBatcher:
    """Iterate over ``(past, future, lags, holidays, target)`` batches for a set of origins.

    ``future_lags[b, h, k]`` is the scaled load ``target_lags[k]`` hours before target hour
    ``h``; since every lag is at least the horizon, it is always observed at the origin.
    """

    def __init__(
        self,
        tensors: WindowTensors,
        origins: np.ndarray,
        *,
        lookback: int,
        horizon: int,
        batch_size: int,
        target_lags: tuple[int, ...] = (),
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if len(origins) == 0:
            raise ValueError("WindowBatcher received no forecast origins")
        if any(lag < horizon for lag in target_lags):
            raise ValueError("target lags shorter than the horizon would leak future load")
        device = tensors.past.device
        self._lags = torch.as_tensor(target_lags, dtype=torch.long, device=device)
        self.tensors = tensors
        self.origins = torch.as_tensor(np.asarray(origins), dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._past_offsets = torch.arange(-lookback + 1, 1, device=device)
        self._future_offsets = torch.arange(1, horizon + 1, device=device)

    def __len__(self) -> int:
        return (len(self.origins) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        order = (
            torch.randperm(len(self.origins), generator=self._generator).to(self.origins.device)
            if self.shuffle
            else torch.arange(len(self.origins), device=self.origins.device)
        )
        for start in range(0, len(order), self.batch_size):
            origins = self.origins[order[start : start + self.batch_size]]
            past_idx = origins[:, None] + self._past_offsets
            future_idx = origins[:, None] + self._future_offsets
            yield {
                "past": self.tensors.past[past_idx],
                "future": self.tensors.known[future_idx],
                "future_lags": self.tensors.target[future_idx[..., None] - self._lags],
                "holiday_past": self.tensors.holiday[past_idx],
                "holiday_future": self.tensors.holiday[future_idx],
                "target": self.tensors.target[future_idx],
            }
