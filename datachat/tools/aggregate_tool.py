import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


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


class AggregateTool(Tool):
    """
    Group-by aggregations on the session DataFrame.

    Supported ops:
    - count: rows per group
    - mean/sum/min/max: aggregation on a metric column
    """

    name = "aggregate"
    description = (
        "Aggregate the dataset by grouping on one column and applying an operation. "
        "Use it for questions like 'how many per category', 'average X by group', "
        "'top groups by count', etc. Returns a small table."
    )
    output_type = "object"

    # IMPORTANT: keys MUST match forward() params exactly (excluding self)
    inputs: ClassVar[dict[str, Any]] = {
        "group_by": {
            "type": "string",
            "description": "Column name to group by (e.g., 'Problemi', 'MIgrante').",
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
        "n": {
            "type": "integer",
            "description": "Max number of result rows to return (max 50).",
            # smolagents may treat this as nullable depending on signature parsing;
            # making it nullable avoids agent-creation failures and is safe (we clamp anyway).
            "nullable": True,
        },
        "ascending": {
            "type": "boolean",
            "description": "Sort ascending by the aggregated value (default False).",
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
        n: Optional[int] = 10,
        ascending: Optional[bool] = False,
    ) -> dict[str, Any]:
        try:
            df = self._df

            # --- group_by hardening: sometimes the LLM passes ["col"] instead of "col" ---
            if isinstance(group_by, list):
                gb_list = [str(x).strip() for x in group_by if str(x).strip()]
                if not gb_list:
                    group_by_clean = ""
                else:
                    group_by_clean = gb_list[0]
            else:
                group_by_clean = (group_by or "").strip()

            op_clean = (op or "").strip().lower()
            metric_clean = (metric or "").strip() if metric is not None else None

            if not group_by_clean:
                return {"kind": "error", "message": "Missing group_by column.", "code": "MISSING_GROUP_BY"}

            if group_by_clean not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid group_by column: {group_by_clean}",
                    "code": "INVALID_GROUP_BY",
                }

            allowed_ops = {"count", "mean", "sum", "min", "max"}
            if op_clean not in allowed_ops:
                return {
                    "kind": "error",
                    "message": f"Invalid op '{op_clean}'. Allowed: {sorted(allowed_ops)}",
                    "code": "INVALID_OP",
                }

            n_default = 10
            n_int = max(1, min(int(n if n is not None else n_default), 50))
            asc = bool(ascending) if ascending is not None else False

            # ---- compute aggregation ----
            if op_clean == "count":
                out = (
                    df.groupby(group_by_clean, dropna=False)
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
                metric_num = pd.to_numeric(df[metric_clean], errors="coerce")

                if op_clean in {"mean", "sum"}:
                    agg_series = metric_num
                else:
                    # min/max: fallback to string if numeric is entirely NaN
                    agg_series = metric_num if metric_num.notna().any() else df[metric_clean].astype(str)

                tmp = pd.DataFrame({group_by_clean: df[group_by_clean], metric_clean: agg_series})
                out = (
                    tmp.groupby(group_by_clean, dropna=False)[metric_clean]
                    .agg(op_clean)
                    .reset_index()
                )

                value_col = f"{op_clean}_{metric_clean}"
                out = out.rename(columns={metric_clean: value_col})

            # ---- sort + trim ----
            out = out.sort_values(by=value_col, ascending=asc, na_position="last").head(n_int)

            # ---- sanitize to JSON-friendly records ----
            records = out.to_dict(orient="records")
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_row: dict[str, Any] = {str(k): _to_json_scalar(v) for k, v in row.items()}
                safe_records.append(safe_row)

            safe_records = replace_nan(safe_records)

            logging.info(
                "[datachat][aggregate_tool] group_by=%s op=%s metric=%s n=%s asc=%s",
                group_by_clean,
                op_clean,
                metric_clean,
                n_int,
                asc,
            )

            return {"kind": "table", "data": safe_records}

        except Exception as e:
            logging.exception("[datachat][aggregate_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}