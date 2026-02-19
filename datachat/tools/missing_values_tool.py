import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


def _to_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


class MissingValuesTool(Tool):
    """
    Return missing-value counts per column.
    """

    name = "missing_values"
    description = (
        "Return missing-value statistics per column, including total rows, "
        "number of missing values, and missing percentage."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, missing values will be computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "columns": {
            "type": "array",
            "description": "Optional list of columns to check. If omitted, checks all columns.",
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

            rows: list[dict[str, Any]] = []
            for col in list(df.columns)[:n_int]:
                s = df[col]
                missing = int(s.isna().sum())
                total = int(len(s))
                pct = float(missing / total) if total > 0 else 0.0
                rows.append(
                    {
                        "column": col,
                        "missing": _to_json_scalar(missing),
                        "total": _to_json_scalar(total),
                        "missing_pct": _to_json_scalar(round(pct, 4)),
                    }
                )

            rows = replace_nan(rows)

            logging.info("[datachat][missing_values_tool] cols=%s", len(rows))
            return {"kind": "table", "data": rows}

        except Exception as e:
            logging.exception("[datachat][missing_values_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
