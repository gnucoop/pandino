import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan

logger = logging.getLogger(__name__)


def _to_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np
        if isinstance(value, (np.generic,)):
            return value.item()
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


class DescribeTool(Tool):
    """
    Return a compact per-column summary of the dataset.

    Output fields per column (when available):
    - column, dtype, count, missing, unique
    - mean, std, min, max (numeric columns)
    - top, freq (categorical columns)
    """

    name = "describe"
    description = (
        "Return a compact per-column summary of the dataset. "
        "For each column it reports type, non-null count, missing count, "
        "number of unique values, and numeric statistics when applicable."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, describe will run on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "columns": {
            "type": "array",
            "description": "Optional list of columns to describe. If omitted, describe all columns.",
            "items": {"type": "string"},
            "nullable": True,
        },
        "n": {
            "type": "integer",
            "description": "Max number of columns to return (max 50).",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        data: list[dict[str, Any]] | None = None,
        columns: Optional[list[str]] = None,
        n: Optional[int] = 50,
    ) -> dict[str, Any]:
        
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")

                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "table", "data": []}

                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            if columns:
                # keep only existing columns, allow case-insensitive match (useful on tool-provided data)
                if data is not None:
                    lowered = {c.lower(): c for c in df.columns}
                    normalized: list[str] = []
                    for c in columns:
                        c_clean = (c or "").strip()
                        if not c_clean:
                            continue
                        if c_clean in df.columns:
                            normalized.append(c_clean)
                        else:
                            hit = lowered.get(c_clean.lower())
                            if hit:
                                normalized.append(hit)
                    cols = normalized
                else:
                    cols = [c for c in columns if c in df.columns]

                if cols:
                    df = df[cols]

            n_int = max(1, min(int(n or 50), 50))

            records: list[dict[str, Any]] = []
            for col in list(df.columns)[:n_int]:
                s = df[col]
                dtype = str(s.dtype)
                count = int(s.count())
                missing = int(s.isna().sum())
                unique = int(s.nunique(dropna=True))

                row: dict[str, Any] = {
                    "column": col,
                    "dtype": dtype,
                    "count": count,
                    "missing": missing,
                    "unique": unique,
                }

                non_null = s.dropna()
                is_bool = pd.api.types.is_bool_dtype(s)

                # Detect boolean-like object columns (e.g., True/False stored as object)
                if not is_bool and not non_null.empty:
                    unique_vals = set(non_null.unique().tolist())
                    boolish = {True, False, "True", "False", "true", "false", 1, 0, "1", "0"}
                    if unique_vals and unique_vals.issubset(boolish):
                        is_bool = True

                if is_bool:
                    vc = non_null.astype(str).value_counts()
                    if not vc.empty:
                        row["top"] = _to_json_scalar(vc.index[0])
                        row["freq"] = _to_json_scalar(int(vc.iloc[0]))
                else:
                    s_num = pd.to_numeric(s, errors="coerce")
                    if s_num.notna().any():
                        row.update(
                            {
                                "mean": _to_json_scalar(float(s_num.mean())),
                                "std": _to_json_scalar(float(s_num.std(ddof=0))),
                                "min": _to_json_scalar(float(s_num.min())),
                                "max": _to_json_scalar(float(s_num.max())),
                            }
                        )
                    else:
                        vc = non_null.astype(str).value_counts()
                        if not vc.empty:
                            row["top"] = _to_json_scalar(vc.index[0])
                            row["freq"] = _to_json_scalar(int(vc.iloc[0]))

                records.append({k: _to_json_scalar(v) for k, v in row.items()})

            records = replace_nan(records)

            logger.info("[datachat][describe_tool] cols=%s", len(records))
            return {"kind": "table", "data": records}

        except Exception as e:
            logger.exception("[datachat][describe_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
