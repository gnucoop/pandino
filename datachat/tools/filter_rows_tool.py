import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan

logger = logging.getLogger(__name__)


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

    MVP+: supports:
    - eq (default) for strings/bools/numbers (type-aware)
    - lt/lte/gt/gte for numeric comparisons
    - optional second condition (AND) via where_col2/op2/value2
    """

    name = "filter_rows"
    description = (
        "Return rows that satisfy one or two conditions on specified columns. "
        "Supports equality and numeric comparisons (lt, lte, gt, gte). "
        "Returns a table of matching rows."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "where_col": {
            "type": "string",
            "description": "Column name to filter on.",
        },
        "value": {
            "type": "any",
            "description": "Value to match.",
        },
        "op": {
            "type": "string",
            "description": "Filter operation: 'eq' (default), 'lt', 'lte', 'gt', 'gte'.",
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, filtering will be applied to this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        # NEW: optional second condition (AND)
        "where_col2": {
            "type": "string",
            "description": "Optional second column to filter on (AND).",
            "nullable": True,
        },
        "op2": {
            "type": "string",
            "description": "Optional second operation: 'eq' (default), 'lt', 'lte', 'gt', 'gte'.",
            "nullable": True,
        },
        "value2": {
            "type": "any",
            "description": "Optional second value to match (AND).",
            "nullable": True,
        },

        "n": {
            "type": "integer",
            "description": "Max number of rows to return (max 50).",
            "nullable": True,
        },
        "offset": {
            "type": "integer",
            "description": "Optional offset for pagination (default 0).",
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
        value: Any,
        op: Optional[str] = None,
        data: list[dict[str, Any]] | None = None,
        where_col2: Optional[str] = None,
        op2: Optional[str] = None,
        value2: Optional[Any] = None,
        n: Optional[int] = 5,
        offset: Optional[int] = None,
        columns: Optional[list[str]] = None,
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

            if not where_col or where_col not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid where_col column: {where_col}",
                    "code": "INVALID_FILTER_COLUMN",
                }

            n_int = max(1, min(int(n or 5), 50))
            offset_int = max(0, int(offset or 0))

            # Choose columns
            if columns:
                chosen = [c for c in columns if c in df.columns]
                df_view = df[chosen] if chosen else df
            else:
                df_view = df[list(df.columns)[:10]]  # keep small by default

            allowed_ops = {"eq", "lt", "lte", "gt", "gte"}

            # -----------------------------
            # Build mask #1
            # -----------------------------
            op_clean = (op or "eq").strip().lower()
            if op_clean not in allowed_ops:
                return {
                    "kind": "error",
                    "message": f"Invalid filter operation: {op_clean}",
                    "code": "INVALID_FILTER_OP",
                }

            series1 = df[where_col]

            if op_clean in {"lt", "lte", "gt", "gte"}:
                series1_num = pd.to_numeric(series1, errors="coerce")
                try:
                    value_num = float(value)
                except Exception:
                    return {
                        "kind": "error",
                        "message": f"Value '{value}' is not numeric and cannot be used with '{op_clean}'.",
                        "code": "NON_NUMERIC_VALUE",
                    }

                if op_clean == "lt":
                    mask1 = series1_num < value_num
                elif op_clean == "lte":
                    mask1 = series1_num <= value_num
                elif op_clean == "gt":
                    mask1 = series1_num > value_num
                else:  # gte
                    mask1 = series1_num >= value_num
            else:
                value_coerced = _coerce_filter_value(df, where_col, value)
                mask1 = _eq_mask(series1, value_coerced)

            # -----------------------------
            # Optional mask #2 (AND)
            # -----------------------------
            mask_final = mask1
            where_col2_clean = (where_col2 or "").strip()

            if where_col2_clean and value2 is not None:
                if where_col2_clean not in df.columns:
                    return {
                        "kind": "error",
                        "message": f"Invalid where_col2 column: {where_col2_clean}",
                        "code": "INVALID_FILTER_COLUMN_2",
                    }

                op2_clean = (op2 or "eq").strip().lower()
                if op2_clean not in allowed_ops:
                    return {
                        "kind": "error",
                        "message": f"Invalid filter operation (op2): {op2_clean}",
                        "code": "INVALID_FILTER_OP_2",
                    }

                series2 = df[where_col2_clean]

                if op2_clean in {"lt", "lte", "gt", "gte"}:
                    series2_num = pd.to_numeric(series2, errors="coerce")
                    try:
                        value2_num = float(str(value2))
                    except Exception:
                        return {
                            "kind": "error",
                            "message": f"Value2 '{value2}' is not numeric and cannot be used with '{op2_clean}'.",
                            "code": "NON_NUMERIC_VALUE_2",
                        }

                    if op2_clean == "lt":
                        mask2 = series2_num < value2_num
                    elif op2_clean == "lte":
                        mask2 = series2_num <= value2_num
                    elif op2_clean == "gt":
                        mask2 = series2_num > value2_num
                    else:  # gte
                        mask2 = series2_num >= value2_num
                else:
                    value2_coerced = _coerce_filter_value(df, where_col2_clean, value2)
                    mask2 = _eq_mask(series2, value2_coerced)

                mask_final = mask_final & mask2

            filtered_all = df_view[mask_final]
            total_matches = int(mask_final.sum())

            filtered = filtered_all.iloc[offset_int : offset_int + n_int]

            records = filtered.to_dict(orient="records")

            # sanitize to JSON scalars only + NaN -> None
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_row: dict[str, Any] = {str(k): _to_json_scalar(v) for k, v in row.items()}
                safe_records.append(safe_row)

            safe_records = replace_nan(safe_records)

            logger.info(
                "event=tool_call_result where_col=%s op=%s value=%s where_col2=%s op2=%s value2=%s n=%s rows=%s",
                where_col,
                op_clean,
                value,
                where_col2_clean or None,
                (op2 or "eq") if (where_col2_clean and value2 is not None) else None,
                value2 if (where_col2_clean and value2 is not None) else None,
                n_int,
                len(safe_records),
            )

            return {
                "kind": "table",
                "data": safe_records,
                "meta": {
                    "offset": offset_int,
                    "returned": len(safe_records),
                    "total_matches": total_matches,
                },
            }

        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}