import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


def _to_json_scalar(value: Any) -> Any:
    """
    Ensure JSON-scalar output for table cells.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _parse_iso_date(s: Any) -> pd.Timestamp | None:
    if s is None:
        return None
    try:
        txt = str(s).strip()
        if not txt:
            return None
        ts = pd.to_datetime(txt, errors="coerce", utc=False)
        if pd.isna(ts):
            return None
        return ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
    except Exception:
        return None


def _freq_to_pandas(freq: str) -> str | None:
    """
    Map tool freq to pandas resample period.
    """
    f = (freq or "").strip().lower()
    if f in {"day", "daily", "d"}:
        return "D"
    if f in {"week", "weekly", "w"}:
        return "W"
    if f in {"month", "monthly", "m"}:
        return "M"
    return None


def _format_period(ts: pd.Timestamp, freq: str) -> str:
    """
    Produce stable period labels.
    - day  -> YYYY-MM-DD
    - week -> YYYY-MM-DD→YYYY-MM-DD (Mon→Sun)
    - month-> YYYY-MM
    """
    f = (freq or "").strip().lower()
    if f in {"day", "daily", "d"}:
        return ts.strftime("%Y-%m-%d")

    if f in {"week", "weekly", "w"}:
        start = (ts - pd.Timedelta(days=ts.weekday())).normalize()
        end = start + pd.Timedelta(days=6)
        return f"{start.strftime('%Y-%m-%d')}→{end.strftime('%Y-%m-%d')}"

    return ts.strftime("%Y-%m")


class TrendTool(Tool):
    """
    Time trend tool: aggregates rows by a date column over a chosen frequency.

    Supported:
    - count: number of rows per period
    - mean/sum: numeric metric per period

    Optional:
    - start/end date range filters (inclusive)
    - include_empty:
        * False (default): return only periods that have data (preferred for "trend" intent)
        * True: keep empty buckets (may include zeros / NaNs depending on op)
    """

    name = "trend"
    description = (
        "Compute a time trend by grouping rows on a date column using day, week, or month buckets. "
        "Works on the full dataset or on a subset passed via `data`. "
        "Returns a small table usable for comparisons or plotting."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "date_col": {
            "type": "string",
            "description": "Name of the date/time column to use (e.g., 'created_at').",
        },
        "freq": {
            "type": "string",
            "description": "Grouping frequency: 'day', 'week', or 'month'.",
        },
        "op": {
            "type": "string",
            "description": "Aggregation: 'count' or 'mean' or 'sum'.",
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the trend will be computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "metric": {
            "type": "string",
            "description": "Numeric column for mean/sum (must be null for count).",
            "nullable": True,
        },
        "start": {
            "type": "any",
            "description": "Optional start date (ISO), inclusive. Example: '2025-09-01'.",
            "nullable": True,
        },
        "end": {
            "type": "any",
            "description": "Optional end date (ISO), inclusive. Example: '2025-12-31'.",
            "nullable": True,
        },
        "n": {
            "type": "integer",
            "description": "Max number of periods to return (max 50).",
            "nullable": True,
        },
        "ascending": {
            "type": "boolean",
            "description": "Sort ascending by period (default True).",
            "nullable": True,
        },
        "include_empty": {
            "type": "boolean",
            "description": (
                "If True, keep empty time buckets in the output. "
                "If False (default), return only periods that have data."
            ),
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        date_col: str,
        freq: str,
        op: str,
        data: list[dict[str, Any]] | None = None,
        metric: Optional[str] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        n: Optional[int] = 50,
        ascending: Optional[bool] = True,
        include_empty: Optional[bool] = False,
    ) -> dict[str, Any]:
        try:
            # Data source selection
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

            date_col_clean = (date_col or "").strip()
            if not date_col_clean:
                return {"kind": "error", "message": "Missing date_col column.", "code": "MISSING_DATE_COL"}

            if date_col_clean not in df.columns:
                # fallback candidates when operating on tool-provided data
                if data is not None:
                    for candidate in ("created_at", "date", "datetime", "timestamp"):
                        if candidate in df.columns:
                            date_col_clean = candidate
                            break

            if date_col_clean not in df.columns:
                return {"kind": "error", "message": f"Invalid date_col column: {date_col_clean}", "code": "INVALID_DATE_COL"}

            pandas_freq = _freq_to_pandas(freq)
            if not pandas_freq:
                return {"kind": "error", "message": f"Invalid freq '{freq}'. Allowed: day, week, month.", "code": "INVALID_FREQ"}

            op_clean = (op or "").strip().lower()
            allowed_ops = {"count", "mean", "sum"}
            if op_clean not in allowed_ops:
                return {"kind": "error", "message": f"Invalid op '{op_clean}'. Allowed: {sorted(allowed_ops)}", "code": "INVALID_OP"}

            metric_clean = (metric or "").strip() if metric is not None else None
            if op_clean == "count":
                if metric is not None and metric_clean:
                    return {"kind": "error", "message": "metric must be null/empty when op='count'.", "code": "METRIC_NOT_ALLOWED"}
            else:
                if not metric_clean:
                    return {"kind": "error", "message": f"Missing metric for op='{op_clean}'.", "code": "MISSING_METRIC"}
                if metric_clean not in df.columns:
                    return {"kind": "error", "message": f"Invalid metric column: {metric_clean}", "code": "INVALID_METRIC"}

            n_int = max(1, min(int(n if n is not None else 50), 50))
            asc = bool(ascending) if ascending is not None else True
            keep_empty = bool(include_empty) if include_empty is not None else False

            dt = pd.to_datetime(df[date_col_clean], errors="coerce")
            if dt.isna().all():
                return {"kind": "error", "message": f"Column '{date_col_clean}' has no parseable dates.", "code": "NO_PARSEABLE_DATES"}

            tmp = df.copy()
            tmp["__dt"] = dt

            start_ts = _parse_iso_date(start)
            end_ts = _parse_iso_date(end)

            tmp = tmp[tmp["__dt"].notna()]
            if start_ts is not None:
                tmp = tmp[tmp["__dt"] >= start_ts]
            if end_ts is not None:
                tmp = tmp[tmp["__dt"] <= end_ts]

            if tmp.empty:
                return {"kind": "error", "message": "No rows match the requested date range.", "code": "EMPTY_RESULT"}

            tmp = tmp.sort_values("__dt").set_index("__dt")

            # ---- aggregation ----
            if op_clean == "count":
                out = tmp.resample(pandas_freq).size().rename("count").reset_index()
                value_col = "count"

                # Default behavior: show only observed periods (avoid misleading zero-filled ranges)
                if not keep_empty:
                    out = out[out["count"] > 0]

            else:
                metric_num = pd.to_numeric(tmp[metric_clean], errors="coerce")  # type: ignore[arg-type]
                if metric_num.notna().sum() == 0:
                    return {
                        "kind": "error",
                        "message": f"Metric '{metric_clean}' has no numeric values for op='{op_clean}'.",
                        "code": "NO_NUMERIC_DATA",
                    }

                resampler = metric_num.resample(pandas_freq)

                if op_clean == "sum":
                    # pandas often returns 0 for empty bins; we want "empty" to be empty unless include_empty=True
                    agg_df = resampler.agg(["sum", "count"])
                    series = agg_df["sum"]
                    series = series.mask(agg_df["count"] == 0, other=pd.NA)
                else:
                    series = resampler.mean()

                value_col = f"{op_clean}_{metric_clean}"
                out = series.rename(value_col).reset_index()

                if not keep_empty:
                    out = out.dropna(subset=[value_col])

            if out.empty:
                return {"kind": "error", "message": "No periods contain data for the requested trend.", "code": "EMPTY_RESULT"}

            # Identify the period datetime column robustly
            period_src_col: Optional[str] = None
            if "__dt" in out.columns:
                period_src_col = "__dt"
            elif date_col_clean in out.columns:
                period_src_col = date_col_clean
            elif "index" in out.columns:
                period_src_col = "index"
            else:
                for c in out.columns:
                    if pd.api.types.is_datetime64_any_dtype(out[c]):
                        period_src_col = c
                        break

            if not period_src_col:
                return {"kind": "error", "message": "Internal error: missing period column after resample.", "code": "INTERNAL_PERIOD_MISSING"}

            out = out.rename(columns={period_src_col: "period_dt"})
            out["period"] = out["period_dt"].apply(lambda x: _format_period(pd.Timestamp(x), freq))
            out = out.drop(columns=["period_dt"], errors="ignore")

            out = out.sort_values(by="period", ascending=asc).head(n_int)

            records = out[["period", value_col]].to_dict(orient="records")
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_records.append({str(k): _to_json_scalar(v) for k, v in row.items()})

            safe_records = replace_nan(safe_records)

            logging.info(
                "[datachat][trend_tool] date_col=%s freq=%s op=%s metric=%s start=%s end=%s n=%s rows=%s include_empty=%s",
                date_col_clean,
                freq,
                op_clean,
                metric_clean,
                start_ts.isoformat() if start_ts is not None else None,
                end_ts.isoformat() if end_ts is not None else None,
                n_int,
                len(safe_records),
                keep_empty,
            )

            return {"kind": "table", "data": safe_records}

        except Exception as e:
            logging.exception("[datachat][trend_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
