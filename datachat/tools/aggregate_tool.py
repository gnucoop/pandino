import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan
from datachat.tools.limits import (
    MIN_RELIABLE_SAMPLE,
    InvalidLimit,
    invalid_limit_error,
    optional_limit,
    sample_warning,
)

logger = logging.getLogger(__name__)


def _to_json_scalar(value: Any) -> Any:
    """
    Convert values to JSON scalars (string/number/bool/null).

    Notes:
    - NaN -> None is handled by replace_nan downstream
    - pd.Timestamp -> ISO string
    - Everything else -> str(value)
    """
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    # Keep bool explicit (bool is subclass of int)
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, str)):
        return value

    return str(value)


class _FilterError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _apply_single_filter(
    df: pd.DataFrame,
    *,
    where_col: str,
    op: str,
    value: Any,
) -> pd.DataFrame:
    """
    Apply a single filter to df and return the filtered df.

    Supported ops: eq, lt, lte, gt, gte
    - eq: case-insensitive string match; supports bool strings "true"/"false"
    - comparisons: numeric comparisons (value must be numeric-coercible)

    Raises _FilterError when the filter cannot be applied as requested.
    """
    where_col_clean = (where_col or "").strip()
    if not where_col_clean or where_col_clean not in df.columns:
        raise _FilterError(f"Invalid filter column: {where_col}", "INVALID_FILTER_COLUMN")

    op_clean = (op or "eq").strip().lower()
    allowed_ops = {"eq", "lt", "lte", "gt", "gte"}
    if op_clean not in allowed_ops:
        raise _FilterError(f"Invalid filter operation: {op_clean}", "INVALID_FILTER_OP")

    if value is None:
        raise _FilterError(f"A filter value is required for column '{where_col_clean}'.", "MISSING_FILTER_VALUE")

    s = df[where_col_clean]

    # Numeric comparisons
    if op_clean in {"lt", "lte", "gt", "gte"}:
        s_num = pd.to_numeric(s, errors="coerce")
        try:
            v_num = float(value)
        except Exception:
            raise _FilterError(
                f"Value '{value}' is not numeric and cannot be used with '{op_clean}'.",
                "NON_NUMERIC_VALUE",
            ) from None

        if op_clean == "lt":
            mask = s_num < v_num
        elif op_clean == "lte":
            mask = s_num <= v_num
        elif op_clean == "gt":
            mask = s_num > v_num
        else:  # gte
            mask = s_num >= v_num

        return df[mask]

    # Equality (type-aware-ish)
    # If value is bool, compare properly
    if isinstance(value, bool):
        if pd.api.types.is_bool_dtype(s):
            mask = s.fillna(False) == value
        else:
            mask = s.astype(str).str.strip().str.lower() == ("true" if value else "false")
        return df[mask]

    # If value is string "true"/"false", treat as bool intent
    if isinstance(value, str):
        v_str = value.strip()
        v_low = v_str.lower()
        if v_low in {"true", "false"}:
            v_bool = v_low == "true"
            if pd.api.types.is_bool_dtype(s):
                mask = s.fillna(False) == v_bool
            else:
                mask = s.astype(str).str.strip().str.lower() == ("true" if v_bool else "false")
            return df[mask]

    # Default: string compare (case-insensitive, trimmed)
    mask = s.astype(str).str.strip().str.lower() == str(value).strip().lower()
    return df[mask]


class AggregateTool(Tool):
    """
    Group-by aggregations on the session DataFrame.

    Supported ops:
    - count: rows per group
    - mean/sum/min/max: aggregation on a metric column

    NEW:
    - Optional pre-aggregation filtering via where_col/op_filter/value (+ optional AND condition)
      This enables one-shot queries like:
      "Tra i migranti, chi ha più visite?"
      -> aggregate(group_by="Nome e Cognome", op="count", where_col="MIgrante", op_filter="eq", value=True, ...)
    """

    name = "aggregate"
    description = (
        "Group rows by one or two columns and apply an aggregation (count, mean, sum, etc.). "
        "Works on the full dataset or on a subset passed via `data`. "
        "Returns all groups unless `n` is given; suitable for further filtering, sorting, plotting, or counting."
    )
    output_type = "object"

    # IMPORTANT: keys MUST match forward() params exactly (excluding self)
    inputs: ClassVar[dict[str, Any]] = {
        "group_by": {
            "type": "any",
            "description": (
                "Column name to group by (e.g., 'Problemi'), or a list of at most two "
                "column names for a two-dimensional breakdown (e.g., ['Problemi', 'MIgrante'])."
            ),
        },
        "op": {
            "type": "string",
            "description": "Aggregation operation: one of 'count', 'mean', 'sum', 'min', 'max'.",
        },
        "metric": {
            "type": "string",
            "description": "Metric column to aggregate (required for mean/sum/min/max).",
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the aggregation will be computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "n": {
            "type": "integer",
            "description": "Optional max number of groups to return (top-N by the sort order). If omitted, all groups are returned.",
            "nullable": True,
        },
        "ascending": {
            "type": "boolean",
            "description": "Sort ascending by the aggregated value (default False).",
            "nullable": True,
        },

        # --- NEW: optional pre-aggregation filter #1 ---
        "where_col": {
            "type": "string",
            "description": "Optional filter column applied before aggregation (requires `value`).",
            "nullable": True,
        },
        "op_filter": {
            "type": "string",
            "description": "Optional filter operation: one of 'eq', 'lt', 'lte', 'gt', 'gte'.",
            "nullable": True,
        },
        "value": {
            "type": "any",
            "description": "Optional filter value (applied before aggregation).",
            "nullable": True,
        },

        # --- NEW: optional pre-aggregation filter #2 (AND) ---
        "where_col2": {
            "type": "string",
            "description": "Optional second filter column (AND, requires `value2`).",
            "nullable": True,
        },
        "op2_filter": {
            "type": "string",
            "description": "Optional second filter operation: one of 'eq', 'lt', 'lte', 'gt', 'gte'.",
            "nullable": True,
        },
        "value2": {
            "type": "any",
            "description": "Optional second filter value (AND).",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        group_by: str,
        op: str,
        metric: Optional[str] = None,
        data: list[dict[str, Any]] | None = None,
        n: Optional[int] = None,
        ascending: Optional[bool] = False,
        where_col: Optional[str] = None,
        op_filter: Optional[str] = None,
        value: Any = None,
        where_col2: Optional[str] = None,
        op2_filter: Optional[str] = None,
        value2: Any = None,
    ) -> dict[str, Any]:
        
        try:
            # Data source selection:
            # - default: session dataset (self._df)
            # - if 'data' is provided: build a temporary dataframe from tool output records
            if data is not None:
                # allow passing full table payload {"kind":"table","data":[...]} by mistake
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")  # type: ignore[assignment]

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

            # --- group_by: one column, or a list of at most two ---
            # A list used to be reduced to its first entry, which silently turned a
            # two-dimensional request into a one-dimensional (wrong) answer.
            if isinstance(group_by, list):
                group_by_cols = [str(x).strip() for x in group_by if str(x).strip()]
            else:
                single = (group_by or "").strip()
                group_by_cols = [single] if single else []

            op_clean = (op or "").strip().lower()
            metric_clean = (metric or "").strip() if metric is not None else None

            if not group_by_cols:
                return {"kind": "error", "message": "Missing group_by column.", "code": "MISSING_GROUP_BY"}

            if len(group_by_cols) > 2:
                return {
                    "kind": "error",
                    "message": f"group_by accepts at most 2 columns, got {len(group_by_cols)}.",
                    "code": "TOO_MANY_GROUP_BY",
                }

            missing_cols = [c for c in group_by_cols if c not in df.columns]
            if missing_cols:
                return {
                    "kind": "error",
                    "message": f"Invalid group_by column: {', '.join(missing_cols)}",
                    "code": "INVALID_GROUP_BY",
                }

            # Label for the log line; the grouping itself uses the list.
            group_by_clean = " + ".join(group_by_cols)

            allowed_ops = {"count", "mean", "sum", "min", "max"}
            if op_clean not in allowed_ops:
                return {
                    "kind": "error",
                    "message": f"Invalid op '{op_clean}'. Allowed: {sorted(allowed_ops)}",
                    "code": "INVALID_OP",
                }

            limit = optional_limit(n)
            asc = bool(ascending) if ascending is not None else False

            # -----------------------------
            # NEW: apply optional filters BEFORE aggregation
            # -----------------------------
            df_work = df

            wc1 = (where_col or "").strip()
            wc2 = (where_col2 or "").strip()
            try:
                for col, op_f, val in ((wc1, op_filter, value), (wc2, op2_filter, value2)):
                    if col or val is not None:
                        df_work = _apply_single_filter(df_work, where_col=col, op=(op_f or "eq"), value=val)
            except _FilterError as e:
                return {"kind": "error", "message": str(e), "code": e.code}

            if df_work.empty:
                logger.info(
                    "event=tool_call_empty where_col=%s value=%s where_col2=%s value2=%s",
                    wc1 or None,
                    value,
                    wc2 or None,
                    value2,
                )
                return {"kind": "table", "data": []}

            # ---- compute aggregation ----
            if op_clean == "count":
                out = (
                    df_work.groupby(group_by_cols, dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                value_col = "count"
            else:
                if not metric_clean:
                    return {
                        "kind": "error",
                        "message": f"Missing metric column for op='{op_clean}'.",
                        "code": "MISSING_METRIC",
                    }

                if metric_clean not in df.columns:
                    return {
                        "kind": "error",
                        "message": f"Invalid metric column: {metric_clean}",
                        "code": "INVALID_METRIC",
                    }

                # Prefer numeric for mean/sum/min/max when possible
                metric_num = pd.to_numeric(df_work[metric_clean], errors="coerce")

                if op_clean in {"mean", "sum"}:
                    agg_series = metric_num
                else:
                    # min/max: fallback to string if numeric is entirely NaN
                    agg_series = metric_num if metric_num.notna().any() else df_work[metric_clean].astype(str)

                tmp = df_work[group_by_cols].copy()
                tmp[metric_clean] = agg_series
                out = (
                    tmp.groupby(group_by_cols, dropna=False)[metric_clean]
                    .agg(op_clean)
                    .reset_index()
                )

                value_col = f"{op_clean}_{metric_clean}"
                out = out.rename(columns={metric_clean: value_col})

            # ---- sort + trim ----
            out = out.sort_values(by=value_col, ascending=asc, na_position="last").iloc[:limit]

            # ---- small-sample caveat (non-count only), judged on the groups returned ----
            note = None
            if op_clean != "count" and not out.empty:
                sizes = (
                    df_work.groupby(group_by_cols, dropna=False)
                    .size()
                    .reset_index(name="__rows__")
                )
                shown = out[group_by_cols].merge(sizes, on=group_by_cols, how="left")["__rows__"]
                thin = shown[shown < MIN_RELIABLE_SAMPLE]
                if not thin.empty:
                    note = sample_warning(int(thin.min()), count=int(thin.shape[0]))

            # ---- sanitize to JSON-friendly records ----
            records = out.to_dict(orient="records")
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_row: dict[str, Any] = {str(k): _to_json_scalar(v) for k, v in row.items()}
                safe_records.append(safe_row)

            safe_records = replace_nan(safe_records)

            logger.info(
                "event=tool_call_result group_by=%s op=%s metric=%s n=%s asc=%s filter1=%s filter2=%s",
                group_by_clean,
                op_clean,
                metric_clean,
                limit,
                asc,
                f"{wc1}:{op_filter or 'eq'}:{value}" if (wc1 and value is not None) else None,
                f"{wc2}:{op2_filter or 'eq'}:{value2}" if (wc2 and value2 is not None) else None,
            )

            payload: dict[str, Any] = {"kind": "table", "data": safe_records}
            if note:
                payload["note"] = note
            return payload

        except InvalidLimit as e:
            return invalid_limit_error(e)
        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
