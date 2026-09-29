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


# Distinguishes an omitted `value` from an explicit value=None (CURRENT eq semantics).
_OMITTED: Any = object()

_VALUELESS_OPS = {"is_empty", "is_not_empty"}
_TEXT_OPS = {"contains", "not_contains"}


def _empty_mask(series: pd.Series) -> pd.Series:
    """
    True where a cell holds no value: NaN/None, or (for non-numeric columns) a
    blank/whitespace-only string.
    """
    missing = series.isna()
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return missing
    return missing | (series.astype(str).str.strip() == "")


def _is_text_series(series: pd.Series) -> bool:
    """
    True when every non-missing cell is a string (or the column is all missing).
    Inspects values rather than dtype: on pandas 1.5 is_string_dtype() is True for
    any object column, including mixed or boolean-with-None ones.
    """
    return pd.api.types.infer_dtype(series, skipna=True) in {"string", "empty"}


def _contains_mask(series: pd.Series, needle: str) -> pd.Series:
    """
    Case-insensitive literal substring match (regex=False: the needle comes from an
    LLM and is meant as text, not a pattern). Textual columns only; missing cells
    never match.
    """
    text = series.astype(str).str.contains(needle, case=False, regex=False, na=False)
    return text & ~series.isna()


def _text_op_mask(series: pd.Series, op: str, value: Any) -> pd.Series:
    """Mask for is_empty/is_not_empty/contains/not_contains."""
    if op == "is_empty":
        return _empty_mask(series)
    if op == "is_not_empty":
        return ~_empty_mask(series)
    needle = str(value).strip()
    if op == "contains":
        return _contains_mask(series, needle)
    # not_contains: complement of contains over filled-in cells only
    return ~_contains_mask(series, needle) & ~_empty_mask(series)


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
    - is_empty/is_not_empty (no value needed)
    - contains/not_contains: case-insensitive literal substring
    - optional second condition (AND) via where_col2/op2/value2
    """

    name = "filter_rows"
    description = (
        "Return rows that satisfy one or two conditions on specified columns. "
        "Supports equality, numeric comparisons (lt, lte, gt, gte), "
        "'is_empty'/'is_not_empty' (column missing/blank or filled in; no value needed) and "
        "'contains'/'not_contains' (case-insensitive literal substring search in text). "
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
            "description": "Value to match. Not needed for 'is_empty'/'is_not_empty'.",
            "nullable": True,
        },
        "op": {
            "type": "string",
            "description": (
                "Filter operation: 'eq' (default), 'lt', 'lte', 'gt', 'gte', "
                "'is_empty', 'is_not_empty', 'contains', 'not_contains'."
            ),
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
            "description": (
                "Optional second operation: same values as 'op'. With 'is_empty'/'is_not_empty' "
                "the second condition applies as soon as where_col2 is set."
            ),
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
        value: Any = _OMITTED,
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

            allowed_ops = {"eq", "lt", "lte", "gt", "gte"} | _VALUELESS_OPS | _TEXT_OPS

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

            # Only is_empty/is_not_empty may omit the value. An explicit value=None
            # keeps its CURRENT meaning for eq.
            if value is _OMITTED:
                if op_clean not in _VALUELESS_OPS:
                    return {
                        "kind": "error",
                        "message": f"A value is required for '{op_clean}'.",
                        "code": "MISSING_FILTER_VALUE",
                    }
                value = None

            if op_clean in _TEXT_OPS and (value is None or not str(value).strip()):
                return {
                    "kind": "error",
                    "message": f"A non-empty value is required for '{op_clean}'.",
                    "code": "MISSING_FILTER_VALUE",
                }

            if op_clean in _TEXT_OPS and not _is_text_series(series1):
                return {
                    "kind": "error",
                    "message": f"'{op_clean}' requires a text column; '{where_col}' is not text.",
                    "code": "NON_TEXT_COLUMN",
                }

            if op_clean in _VALUELESS_OPS | _TEXT_OPS:
                mask1 = _text_op_mask(series1, op_clean, value)
            elif op_clean in {"lt", "lte", "gt", "gte"}:
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

            # is_empty/is_not_empty need no value, so naming the column is enough.
            op2_probe = (op2 or "eq").strip().lower()
            second_active = bool(where_col2_clean) and (
                value2 is not None or op2_probe in _VALUELESS_OPS
            )

            if second_active:
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

                if op2_clean in _TEXT_OPS and not str(value2).strip():
                    return {
                        "kind": "error",
                        "message": f"A non-empty value2 is required for '{op2_clean}'.",
                        "code": "MISSING_FILTER_VALUE_2",
                    }

                if op2_clean in _TEXT_OPS and not _is_text_series(series2):
                    return {
                        "kind": "error",
                        "message": f"'{op2_clean}' requires a text column; '{where_col2_clean}' is not text.",
                        "code": "NON_TEXT_COLUMN_2",
                    }

                if op2_clean in _VALUELESS_OPS | _TEXT_OPS:
                    mask2 = _text_op_mask(series2, op2_clean, value2)
                elif op2_clean in {"lt", "lte", "gt", "gte"}:
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
                (op2 or "eq") if second_active else None,
                value2 if second_active else None,
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