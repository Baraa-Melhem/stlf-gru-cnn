"""LightGBM direct multi-horizon quantile baseline.

One gradient-boosted model per quantile covers all 24 horizons: each (origin, horizon) pair is a
row whose features are the horizon index, calendar/holiday/weather covariates of the *target*
hour, and load lags that are already observed at the origin (every lag >= horizon). Load
features and the target are divided by the mean load of the week before the origin — the
tree-model analogue of RevIN, without which trees cannot extrapolate beyond the training range
(e.g. the 2019 record highs or the 2020 lockdown lows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import lightgbm as lgb
import numpy as np
import pandas as pd

from stlf.config import ExperimentConfig
from stlf.data.splits import FoldSplit
from stlf.data.windowing import select_origins
from stlf.features.engineering import PreparedData
from stlf.forecasters.base import (
    Forecaster,
    apply_margin,
    conformal_margin,
    target_positions,
)
from stlf.logging_utils import get_logger

logger = get_logger(__name__)

_LEVEL_WINDOW = 168
_STATE = "lightgbm_state.json"


class LightGBMForecaster(Forecaster):
    """Quantile LightGBM on a level-normalised direct multi-horizon design matrix."""

    persistable: ClassVar[bool] = True

    def __init__(self, cfg: ExperimentConfig, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        self.boosters: list[lgb.Booster] = []
        self.margin = 0.0

    @property
    def required_history(self) -> int:
        return max(*self.cfg.model.gbm.lags, _LEVEL_WINDOW)

    def _design(self, data: PreparedData, origins: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the design matrix (rows = origin x horizon) and each origin's load level."""
        horizon = self.cfg.window.horizon
        load = data.frame[data.spec.target].to_numpy(dtype=np.float64)
        cumulative = np.concatenate([[0.0], np.cumsum(np.nan_to_num(load))])
        origins = np.asarray(origins)
        targets = target_positions(origins, horizon)
        level = (cumulative[origins + 1] - cumulative[origins + 1 - _LEVEL_WINDOW]) / _LEVEL_WINDOW
        mean_24 = (cumulative[origins + 1] - cumulative[origins + 1 - 24]) / 24

        def per_origin(values: np.ndarray) -> np.ndarray:
            return np.broadcast_to(values[:, None], targets.shape)

        columns: dict[str, np.ndarray] = {
            "horizon": np.broadcast_to(np.arange(1, horizon + 1)[None, :], targets.shape),
            "last_load": per_origin(load[origins] / level),
            "mean_24": per_origin(mean_24 / level),
        }
        for lag in self.cfg.model.gbm.lags:
            columns[f"lag_{lag}"] = load[targets - lag] / level[:, None]
        known = data.frame[list(data.spec.known)].to_numpy(dtype=np.float64)[targets]
        for k, name in enumerate(data.spec.known):
            columns[name] = known[..., k]
        if data.spec.holiday_id is not None:
            holiday = data.frame[data.spec.holiday_id].to_numpy(dtype=np.int64)
            columns[data.spec.holiday_id] = holiday[targets]
        design = pd.DataFrame({name: values.reshape(-1) for name, values in columns.items()})
        return design, level

    def _relative_target(
        self, data: PreparedData, origins: np.ndarray, level: np.ndarray
    ) -> np.ndarray:
        load = data.frame[data.spec.target].to_numpy(dtype=np.float64)
        targets = target_positions(origins, self.cfg.window.horizon)
        return (load[targets] / level[:, None]).reshape(-1)

    def fit(self, data: PreparedData, split: FoldSplit) -> None:
        gbm, window = self.cfg.model.gbm, self.cfg.window
        common = {"lookback": self.required_history, "horizon": window.horizon}
        train_origins = select_origins(
            data.index, **common, last_target=split.train_end, stride=gbm.origin_stride
        )
        val_origins = select_origins(
            data.index, **common, first_target=split.val_start, last_target=split.val_end
        )
        x_train, level_train = self._design(data, train_origins)
        x_val, level_val = self._design(data, val_origins)
        y_train = self._relative_target(data, train_origins, level_train)
        y_val = self._relative_target(data, val_origins, level_val)
        categorical = [data.spec.holiday_id] if data.spec.holiday_id is not None else "auto"

        self.boosters = []
        for quantile in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=gbm.n_estimators,
                learning_rate=gbm.learning_rate,
                num_leaves=gbm.num_leaves,
                min_child_samples=gbm.min_child_samples,
                subsample=gbm.subsample,
                subsample_freq=1,
                colsample_bytree=gbm.colsample_bytree,
                reg_lambda=gbm.reg_lambda,
                random_state=self.seed,
                deterministic=True,
                force_row_wise=True,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="quantile",
                categorical_feature=categorical,
                callbacks=[lgb.early_stopping(gbm.early_stopping_rounds, verbose=False)],
            )
            self.boosters.append(model.booster_)
            logger.info(
                "lightgbm q=%.2f | %d rows | best iteration %d",
                quantile,
                len(x_train),
                model.best_iteration_,
            )

        self.margin = 0.0
        if self.cfg.model.conformal and len(self.quantiles) >= 3:
            forecast = self.predict(data, val_origins)
            actual = y_val.reshape(forecast.shape[:2]) * level_val[:, None]
            nominal = self.quantiles[-1] - self.quantiles[0]
            self.margin = conformal_margin(actual, forecast[..., 0], forecast[..., -1], nominal)
            logger.info("conformal interval margin: %+.1f MW", self.margin)
        return None

    def predict(self, data: PreparedData, origins: np.ndarray) -> np.ndarray:
        if not self.boosters:
            raise RuntimeError("forecaster is not fitted")
        design, level = self._design(data, origins)
        shape = (len(origins), self.cfg.window.horizon)
        outputs = [
            booster.predict(design, num_iteration=booster.best_iteration or None).reshape(shape)
            * level[:, None]
            for booster in self.boosters
        ]
        return apply_margin(np.sort(np.stack(outputs, axis=-1), axis=-1), self.margin)

    def save(self, directory: Path) -> None:
        if not self.boosters:
            raise RuntimeError("cannot save an unfitted forecaster")
        directory.mkdir(parents=True, exist_ok=True)
        for quantile, booster in zip(self.quantiles, self.boosters, strict=True):
            booster.save_model(str(directory / f"lightgbm_q{quantile:.3f}.txt"))
        state = {"seed": self.seed, "conformal_margin": self.margin}
        (directory / _STATE).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, cfg: ExperimentConfig) -> LightGBMForecaster:
        state = json.loads((directory / _STATE).read_text(encoding="utf-8"))
        forecaster = cls(cfg, seed=int(state["seed"]))
        forecaster.margin = float(state["conformal_margin"])
        forecaster.boosters = [
            lgb.Booster(model_file=str(directory / f"lightgbm_q{quantile:.3f}.txt"))
            for quantile in cfg.model.quantiles
        ]
        return forecaster
