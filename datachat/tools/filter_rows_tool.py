import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


def _to_json_scalar(v: Any) -> Any:
    # JSON scalars only: str, int/float, bool, None
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    # pandas / numpy scalars
    try:
        import numpy as np  # optional at runtime if installed
        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass
    return str(v)


def _coerce_filter_value(df: pd.DataFrame, col: str, raw: Any) -> Any:
    """
    Try to coerce the raw value coming from the LLM into the column's "natural" type.
    This fixes common issues like value="true" (string) vs column boolean True.
    """
    if raw is None:
        return None

    # If it's already not a string, keep it
    if not isinstance(raw, str):
        return raw

    s = raw.strip()

    # Bool coercion
    s_low = s.lower()
    if s_low in {"true", "false"}:
        # If the column is bool dtype, coerce confidently
        if pd.api.types.is_bool_dtype(df[col]):
            return s_low == "true"

        # Heuristic: if the column contains actual bools, coerce
        non_null = df[col].dropna()
        if not non_null.empty and non_null.map(lambda x: isinstance(x, bool)).any():
            return s_low == "true"

        # otherwise leave it as original string
        return raw

    # Numeric coercion if column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        num = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
        if pd.notna(num):
            # if it's an integer-like float, keep as float anyway (json-safe)
            return float(num) if isinstance(num, (float, int)) else num

    return raw


def _eq_mask(series: pd.Series, value: Any) -> pd.Series:
    """
    Build an equality mask in a type-aware way.
    """
    # If value is bool, compare to bool series where possible
    if isinstance(value, bool):
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False) == value
        # handle "True"/"False" strings stored in column
        return series.astype(str).str.strip().str.lower() == ("true" if value else "false")

    # If value is numeric, compare after numeric coercion
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        ser_num = pd.to_numeric(series, errors="coerce")
        return ser_num == float(value)

    # Default: string compare (case-insensitive, trimmed)
    return series.astype(str).str.strip().str.lower() == str(value).strip().lower()


class FilterRowsTool(Tool):
    """
    Smolagents tool: filter rows on a bound DataFrame.

    MVP: supports only equality filter (op='eq').
    """

    name = "filter_rows"
    description = (
        "Filter rows in the dataset by a simple condition and return a small table. "
        "Use this when the user asks for rows where a column equals a value "
        "(e.g., Problemi = 'Lavoro')."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "where_col": {
            "type": "string",
            "description": "Column name to filter on.",
        },
        "op": {
            "type": "string",
            "description": "Filter operation: 'eq' (default), 'lt', 'lte', 'gt', 'gte'.",
            "nullable": True,
        },
        "value": {
            "type": "string",
            "description": "Value to match (equality).",
        },
        "n": {
            "type": "integer",
            "description": "Max number of rows to return (max 20).",
            "nullable": True,
        },
        "columns": {
            "type": "array",
            "description": "Optional list of columns to include.",
            "items": {"type": "string"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        where_col: str,
        value: str,
        op: Optional[str] = None,
        n: Optional[int] = 5,
        columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        try:
            df = self._df

            if not where_col or where_col not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid where_col column: {where_col}",
                    "code": "INVALID_FILTER_COLUMN",
                }

            n_int = max(1, min(int(n or 5), 20))

            # Choose columns
            if columns:
                chosen = [c for c in columns if c in df.columns]
                df_view = df[chosen] if chosen else df
            else:
                df_view = df[list(df.columns)[:10]]  # keep small by default

            # Normalize operation
            op_clean = (op or "eq").strip().lower()

            allowed_ops = {"eq", "lt", "lte", "gt", "gte"}
            if op_clean not in allowed_ops:
                return {
                    "kind": "error",
                    "message": f"Invalid filter operation: {op_clean}",
                    "code": "INVALID_FILTER_OP",
                }

            series = df[where_col]

            # Numeric comparisons
            if op_clean in {"lt", "lte", "gt", "gte"}:
                # Ensure column is numeric
                series_num = pd.to_numeric(series, errors="coerce")

                try:
                    value_num = float(value)
                except Exception:
                    return {
                        "kind": "error",
                        "message": f"Value '{value}' is not numeric and cannot be used with '{op_clean}'.",
                        "code": "NON_NUMERIC_VALUE",
                    }

                if op_clean == "lt":
                    mask = series_num < value_num
                elif op_clean == "lte":
                    mask = series_num <= value_num
                elif op_clean == "gt":
                    mask = series_num > value_num
                else:  # gte
                    mask = series_num >= value_num

            else:
                # Equality filter (existing, type-aware)
                value_coerced = _coerce_filter_value(df, where_col, value)
                mask = _eq_mask(series, value_coerced)
            
            filtered = df_view[mask].head(n_int)

            records = filtered.to_dict(orient="records")

            # sanitize to JSON scalars only + NaN -> None
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_row: dict[str, Any] = {str(k): _to_json_scalar(v) for k, v in row.items()}
                safe_records.append(safe_row)

            safe_records = replace_nan(safe_records)

            logging.info(
                "[datachat][filter_rows_tool] where_col=%s value=%s n=%s rows=%s",
                where_col,
                value,
                n_int,
                len(safe_records),
            )
            return {"kind": "table", "data": safe_records}

        except Exception as e:
            logging.exception("[datachat][filter_rows_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}