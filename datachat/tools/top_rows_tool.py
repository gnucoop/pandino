import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


def _to_json_scalar(value: Any) -> Any:
    """
    Ensure any cell value becomes a JSON scalar: str | int | float | bool | None.
    - dict/list -> flattened string (no braces).
    - other non-serializable -> string.
    """
    if value is None:
        return None

    # pandas missing values
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)):
        return value

    # dict -> "k=v; k2=v2"
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k}={str(v).replace('{', '').replace('}', '')}")
        return "; ".join(parts)

    # list/tuple/set -> "v1; v2; v3"
    if isinstance(value, (list, tuple, set)):
        parts = [str(v).replace("{", "").replace("}", "") for v in value]
        return "; ".join(parts)

    # fallback
    return str(value)


def _truncate_cell(v: Any, max_chars: int) -> Any:
    """
    Prevent "infinite text" in cells by truncating long strings.
    Keep non-strings as-is.
    """
    if max_chars <= 0:
        return v
    if isinstance(v, str) and len(v) > max_chars:
        return v[:max_chars] + "…"
    return v


def _coerce_sort_key(series: pd.Series) -> pd.Series:
    """
    Build a robust sort key:
    1) try datetime
    2) else try numeric
    3) else string (case-insensitive)
    """
    # Datetime attempt
    dt = pd.to_datetime(series, errors="coerce", utc=False)
    if dt.notna().any():
        return dt

    # Numeric attempt
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num

    # String fallback (stable-ish)
    return series.astype(str).str.strip().str.lower()


class TopRowsTool(Tool):
    """
    Smolagents tool: return top/bottom rows by sorting on a column.

    The DataFrame is injected at instantiation time and cannot be changed by the LLM.
    """

    name = "top_rows"
    description = (
        "Return rows sorted by a specified column, with configurable order, "
        "offset pagination, and column selection. "
        "Returns a table."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "n": {
            "type": "integer",
            "description": "Number of rows to return (max 20).",
            "nullable": True,
        },
        "offset": {
            "type": "integer",
            "description": "Optional offset for pagination (default 0).",
            "nullable": True,
        },
        "sort_by": {
            "type": "string",
            "description": "Column name to sort by.",
            "nullable": True,
        },
        "metric": {
            "type": "string",
            "description": "Alias of sort_by (some users/LLMs call it metric).",
            "nullable": True,
        },
        "ascending": {
            "type": "boolean",
            "description": "True for lowest/oldest first, False for highest/most recent first.",
            "nullable": True,
        },
        "columns": {
            "type": "array",
            "description": "Optional list of columns to include. If omitted, a subset will be chosen.",
            "items": {"type": "string"},
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, sorting will be done on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "max_cell_chars": {
            "type": "integer",
            "description": "Max characters allowed per cell (default 200, max 2000).",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        sort_by: Optional[str] = None,
        n: Optional[int] = 5,
        offset: Optional[int] = 0,
        ascending: Optional[bool] = False,
        columns: Optional[list[str]] = None,
        data: list[dict[str, Any]] | None = None,
        max_cell_chars: Optional[int] = 200,
        metric: Optional[str] = None,  # alias for sort_by
    ) -> dict[str, Any]:
        try:
            # --- normalize alias ---
            if (not sort_by) and metric:
                sort_by = metric

            sort_by_clean = (sort_by or "").strip()
            if not sort_by_clean:
                return {"kind": "error", "message": "Missing sort_by column.", "code": "MISSING_SORT_COLUMN"}

            n_int = max(1, min(int(n or 5), 20))
            offset_int = max(0, int(offset or 0))
            asc = bool(ascending) if ascending is not None else False
            max_chars = int(max_cell_chars or 200)
            max_chars = max(0, min(max_chars, 2000))

            # --- datasource selection (session df vs tool-produced data) ---
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")

                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}

                if len(data) == 0:
                    return {"kind": "table", "data": [], "meta": {"offset": offset_int, "returned": 0, "total_matches": 0}}

                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            # --- column existence + small fallback for tool-provided data ---
            if sort_by_clean not in df.columns:
                if data is not None:
                    lowered = {c.lower(): c for c in df.columns}
                    hit = lowered.get(sort_by_clean.lower())
                    if hit:
                        sort_by_clean = hit

            if sort_by_clean not in df.columns:
                return {"kind": "error", "message": f"Invalid sort_by column: {sort_by_clean}", "code": "INVALID_SORT_COLUMN"}

            # --- choose columns to return ---
            if columns:
                chosen = [c for c in columns if c in df.columns]
                df_view = df[chosen] if chosen else df
            else:
                # If we're operating on tool-produced data, keep all columns (already "small").
                # If session dataset, cap to first 10 columns for safety.
                df_view = df if data is not None else df[list(df.columns)[:10]]

            # --- sorting (robust) ---
            sort_key = _coerce_sort_key(df[sort_by_clean])
            df_sorted = (
                df_view.assign(__sort_key=sort_key)
                .sort_values(by="__sort_key", ascending=asc, na_position="last")
                .drop(columns=["__sort_key"], errors="ignore")
            )

            total = int(len(df_sorted))
            page = df_sorted.iloc[offset_int : offset_int + n_int]

            records = page.to_dict(orient="records")

            # sanitize to JSON scalars + truncate long text + NaN -> None
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_row: dict[str, Any] = {}
                for k, v in row.items():
                    v2 = _to_json_scalar(v)
                    v2 = _truncate_cell(v2, max_chars)
                    safe_row[str(k)] = v2
                safe_records.append(safe_row)

            safe_records = replace_nan(safe_records)

            logging.info(
                "[datachat][top_rows_tool] sort_by=%s asc=%s n=%s offset=%s returned=%s total=%s cols=%s data_mode=%s",
                sort_by_clean,
                asc,
                n_int,
                offset_int,
                len(safe_records),
                total,
                len(page.columns),
                "tool_data" if data is not None else "session_df",
            )

            return {
                "kind": "table",
                "data": safe_records,
                "meta": {
                    "offset": offset_int,
                    "returned": len(safe_records),
                    "total_matches": total,
                    "sort_by": sort_by_clean,
                    "ascending": asc,
                },
            }

        except Exception as e:
            logging.exception("[datachat][top_rows_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
