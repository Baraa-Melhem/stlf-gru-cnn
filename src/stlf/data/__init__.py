"""Raw data access: download, loading/validation, and sliding-window batching."""

from stlf.data.loader import (
    OFFICIAL_TEST_WEEKS,
    TARGET_COLUMN,
    DataValidationError,
    flag_load_anomalies,
    load_dataset,
    load_operator_forecast,
    official_week_mask,
)
from stlf.data.windowing import WindowBatcher, WindowTensors, build_tensors, select_origins

__all__ = [
    "OFFICIAL_TEST_WEEKS",
    "TARGET_COLUMN",
    "DataValidationError",
    "WindowBatcher",
    "WindowTensors",
    "build_tensors",
    "flag_load_anomalies",
    "load_dataset",
    "load_operator_forecast",
    "official_week_mask",
    "select_origins",
]
