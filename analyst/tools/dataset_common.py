"""
dataset_common.py
-----------------

What the analyst's DataFrame tools share: finding a loaded dataset and its
numeric columns by the names the agent gives.

Tools work on the DataFrames load_dataset put in the run's `datasets`.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from analyst.tools.sql_common import SqlToolError, split_names

# Shown to the model instead of a traceback when a computation fails.
DatasetToolError = SqlToolError


def get_dataset(datasets: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    frame = datasets.get(str(name or "").strip())
    if frame is None:
        known = ", ".join(datasets) or "none yet: load one with load_dataset"
        raise DatasetToolError(f"Unknown dataset {name!r}. Loaded datasets: {known}.", "UNKNOWN_DATASET")
    return frame


def numeric_columns(frame: pd.DataFrame, names: str | list[str]) -> list[str]:
    """The requested columns, all present and numeric."""
    wanted = split_names(names) if isinstance(names, str) else list(names)
    if not wanted:
        raise DatasetToolError("No column given.")
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise DatasetToolError(
            f"Unknown columns {missing}. Available: {', '.join(map(str, frame.columns))}.",
            "INVALID_COLUMN",
        )
    not_numeric = [c for c in wanted if not pd.api.types.is_numeric_dtype(frame[c])]
    if not_numeric:
        raise DatasetToolError(f"Columns {not_numeric} are not numeric.", "INVALID_COLUMN")
    return wanted


def tool_error(e: Exception, tool: str) -> dict[str, Any]:
    if isinstance(e, DatasetToolError):
        return {"kind": "error", "message": str(e), "code": e.code}
    return {"kind": "error", "message": f"{tool} failed: {e}", "code": "TOOL_FAILED"}
