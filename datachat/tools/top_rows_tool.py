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


class TopRowsTool(Tool):
    """
    Smolagents tool: return top/bottom rows by sorting on a column.

    The DataFrame is injected at instantiation time and cannot be changed by the LLM.
    """

    name = "top_rows"
    description = (
        "Return the top/bottom N rows sorted by a given column as a JSON table. "
        "Use this tool when the user asks for highest/lowest values, most recent/oldest, "
        "top N by a metric, etc. The dataset is fixed for the session."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "n": {
            "type": "integer",
            "description": "Number of rows to return (max 20).",
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
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        sort_by: Optional[str] = None,
        n: Optional[int] = 5,
        ascending: Optional[bool] = False,
        columns: Optional[list[str]] = None,
        metric: Optional[str] = None,  # alias for sort_by
    ) -> dict[str, Any]:
        try:
            # --- alias handling (robust against LLM param names) ---
            if (not sort_by) and metric:
                sort_by = metric

            if not sort_by:
                return {
                    "kind": "error",
                    "message": "Missing sort_by column.",
                    "code": "MISSING_SORT_COLUMN",
                }

            df = self._df

            if sort_by not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid sort_by column: {sort_by}",
                    "code": "INVALID_SORT_COLUMN",
                }

            n_int = max(1, min(int(n or 5), 20))
            asc = bool(ascending) if ascending is not None else False

            # Choose columns
            if columns:
                chosen = [c for c in columns if c in df.columns]
                if chosen:
                    df_view = df[chosen]
                else:
                    df_view = df
            else:
                df_view = df[list(df.columns)[:10]]  # keep small by default

            # Sorting: try numeric conversion first, fallback to string sort
            series = df[sort_by]
            series_num = pd.to_numeric(series, errors="coerce")
            if series_num.notna().any():
                sort_key = series_num
            else:
                sort_key = series.astype(str)

            df_sorted = df_view.assign(__sort_key=sort_key).sort_values(
                by="__sort_key", ascending=asc, na_position="last"
            )
            df_sorted = df_sorted.drop(columns=["__sort_key"], errors="ignore")

            sample = df_sorted.head(n_int)
            records = sample.to_dict(orient="records")

            # sanitize to JSON scalars only
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_row: dict[str, Any] = {
                    str(k): _to_json_scalar(v)
                    for k, v in row.items()
                }
                safe_records.append(safe_row)

            safe_records = replace_nan(safe_records)

            logging.info(
                "[datachat][top_rows_tool] sort_by=%s asc=%s n=%s cols=%s",
                sort_by,
                asc,
                n_int,
                len(sample.columns),
            )
            return {"kind": "table", "data": safe_records}

        except Exception as e:
            logging.exception("[datachat][top_rows_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}