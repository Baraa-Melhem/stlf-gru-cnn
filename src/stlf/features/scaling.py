"""Leak-free standardisation: statistics come from the training rows of each fold only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StandardScaler:
    """Per-column z-scoring that remembers its statistics and serialises to JSON.

    Fitting on the training slice and *then* transforming the full frame is the whole point:
    the original notebook fitted on all 48k rows before splitting, so test-period means and
    variances leaked into training.
    """

    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str]) -> StandardScaler:
        """Compute mean/std of ``columns`` on ``frame`` (which must be the training slice)."""
        if frame.empty:
            raise ValueError("cannot fit a scaler on an empty frame")
        stats = frame[list(columns)].agg(["mean", "std"])
        means = {column: float(stats.at["mean", column]) for column in columns}
        stds = {column: float(stats.at["std", column]) for column in columns}
        return cls(mean_=means, std_={k: (v if v > 1e-12 else 1.0) for k, v in stds.items()})

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.mean_)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``frame`` with the fitted columns standardised."""
        out = frame.copy()
        for column in self.columns:
            out[column] = (frame[column] - self.mean_[column]) / self.std_[column]
        return out

    def inverse(self, values: np.ndarray, column: str) -> np.ndarray:
        """Map standardised ``values`` of ``column`` back to original units."""
        return values * self.std_[column] + self.mean_[column]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"mean": dict(self.mean_), "std": dict(self.std_)}

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, float]]) -> StandardScaler:
        return cls(mean_=dict(payload["mean"]), std_=dict(payload["std"]))
