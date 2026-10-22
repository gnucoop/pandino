"""
values.py
---------

Turning what the agents compute into plain JSON values, and back into short text.

Agents compute with pandas and numpy, whose scalars, arrays and frames are not
JSON. Everything an agent saves (outputs, chart data, table rows) goes through
to_native() first, so the run state can be persisted and read back as is.
preview() is the other direction: a bounded text rendering of a value, which is
what an agent reads instead of the value itself.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def to_native(value: Any) -> Any:
    """Recursively convert numpy/pandas values to JSON-compatible Python values."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, pd.DataFrame):
        return [to_native(r) for r in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        # A Series is usually a grouped result: its index carries the labels.
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, pd.Index):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_native(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        # NaN and infinities are not JSON: they become null.
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def preview(value: Any, max_chars: int) -> str:
    """The value as JSON, cut to max_chars with a marker saying how much is missing."""
    text = to_json(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}… [truncated, {len(text) - max_chars} more chars]"


_DATASET_REFERENCE_KEYS = frozenset({"dataset", "rows", "columns", "head"})


def dataset_reference(name: str, frame: pd.DataFrame, head: list[dict[str, Any]]) -> dict[str, Any]:
    """
    What an output holds for a table too large to save whole (save_sub_task_output):
    the name of the run dataset keeping the table, its shape, and its first rows.
    """
    return {"dataset": name, "rows": len(frame), "columns": [str(c) for c in frame.columns], "head": head}


def is_dataset_reference(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == _DATASET_REFERENCE_KEYS


def describe(value: Any) -> str:
    """A one-line shape description: what a variable holds, without its content."""
    if isinstance(value, pd.DataFrame):
        return f"DataFrame {value.shape[0]} rows x {value.shape[1]} cols: {', '.join(map(str, value.columns))}"
    if isinstance(value, dict):
        return f"dict with keys: {', '.join(map(str, value.keys()))}"
    if isinstance(value, list):
        return f"list of {len(value)}"
    return type(value).__name__
