import datetime as dt
import logging
import math
import numbers
from collections import Counter
from typing import Any, Callable, ClassVar, Optional

import numpy as np
import pandas as pd
from smolagents import Tool

from datachat.chart_registry import record_chart
from datachat.result_provenance import TrustedResult, lookup_trusted_result
from datachat.tools.crosstab_tool import (
    _DATETIME,
    _MISSING,
    _NUMBER,
    _STRING,
    _category_key,
    _json_value,
    _labels,
    _sort_categories,
    _text,
)

logger = logging.getLogger(__name__)

_KINDS = ("bar", "line", "pie", "doughnut", "scatter")
_SINGLE_SERIES_KINDS = {"pie", "doughnut"}

_Key = tuple[int, Any]


class _ChartError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _optional_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _display(key: _Key, labels: dict[_Key, str]) -> Any:
    """The native JSON value, unless its text had to be disambiguated (as in crosstab)."""
    return _json_value(key) if labels[key] == _text(key) else labels[key]


def _value(value: Any) -> Optional[float]:
    """
    A precomputed value as supplied: a finite number, or None for a missing one.

    Missing stays None (a gap), never 0. Anything else is refused rather than
    guessed at, because the chart only renders what an upstream tool computed.
    """
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Number):
        raise _ChartError("Values to chart must be numbers or null.", "NON_NUMERIC_VALUE")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise _ChartError("Values to chart must be finite numbers or null.", "NON_FINITE_VALUE")
    return number


def _coordinate(value: Any) -> Optional[float]:
    """
    A scatter coordinate, or None when the value cannot be placed on an axis.

    Numbers and numeric strings qualify when finite. Missing values, booleans,
    dates and other text do not.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    elif not isinstance(value, numbers.Number):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int):
        return value
    number = float(value)
    return number if math.isfinite(number) else None


def _iso_timestamp(value: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _line_axis(domain: list[_Key]) -> list[_Key]:
    """
    Order a line chart's x values along their axis when that axis is
    unambiguous: all numbers, all dates, or all ISO date strings (as the
    aggregate tool writes dates). Otherwise the upstream order is kept.
    Missing x values go last.
    """
    present = [k for k in domain if k[0] != _MISSING]
    missing = [k for k in domain if k[0] == _MISSING]
    classes = {k[0] for k in present}
    try:
        if classes in ({_NUMBER}, {_DATETIME}):
            return sorted(present, key=lambda k: k[1]) + missing
        if classes == {_STRING}:
            stamps = {k: _iso_timestamp(k[1]) for k in present}
            if all(s is not None for s in stamps.values()):
                return sorted(present, key=lambda k: stamps[k]) + missing
    except TypeError:
        # e.g. tz-aware next to naive timestamps
        pass
    return domain


class ChartTool(Tool):
    """
    Structured chart data for the client to draw.

    The chart is recorded for the current request and attached to the
    response by the route; the agent receives a short confirmation only.
    Counts per value of one column and raw scatter points are computed here;
    everything else (metrics, counts split by a second column) is rendered
    from records an upstream tool already computed.
    """

    name = "chart"
    description = (
        "Draw a chart (bar, line, pie, doughnut, scatter) that is shown to the user with your answer. "
        "Three uses: "
        "(1) without `data`, kind bar/line/pie/doughnut: counts the rows for each value of `x` in the dataset; "
        "(2) kind scatter: plots the numeric columns `x` and `y` as points (rows where x or y is not a finite number are skipped); "
        "(3) with `data` (records from another tool, e.g. aggregate): draws `y` against `x` exactly as supplied, "
        "optionally one series per value of `series_by`. "
        "This tool never aggregates: to chart a mean, sum or a count split by a second column, "
        "call aggregate first and pass its data here."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "kind": {
            "type": "string",
            "description": "Chart type: 'bar', 'line', 'pie', 'doughnut' or 'scatter'.",
            "enum": list(_KINDS),
        },
        "x": {
            "type": "string",
            "description": "Column for the categories (or the x axis).",
        },
        "y": {
            "type": "string",
            "description": (
                "Column holding the values to draw. Required with `data` and for scatter; "
                "not allowed for a plain count without `data`."
            ),
            "nullable": True,
        },
        "series_by": {
            "type": "string",
            "description": (
                "Optional column whose values split the chart into one series each "
                "(bar, line, scatter only)."
            ),
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, they are charted as they are, instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "title": {
            "type": "string",
            "description": "Optional chart title.",
            "nullable": True,
        },
        "horizontal": {
            "type": "boolean",
            "description": "Draw horizontal bars (bar charts only). Default false.",
            "nullable": True,
        },
    }

    def __init__(self, df: Optional[pd.DataFrame]) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        kind: str,
        x: str,
        y: Optional[str] = None,
        series_by: Optional[str] = None,
        data: list[dict[str, Any]] | None = None,
        title: Optional[str] = None,
        horizontal: Optional[bool] = None,
    ) -> dict[str, Any]:
        try:
            spec, trusted, stats = self._build(kind, x, y, series_by, data, title, horizontal)
        except _ChartError as e:
            logger.info("event=tool_call_rejected code=%s", e.code)
            return {"kind": "error", "message": str(e), "code": e.code}
        except Exception as e:
            # The message may quote dataset values; only its type is logged.
            logger.warning("event=tool_call_failed error_type=%s", type(e).__name__)
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

        note = trusted.note if trusted is not None else None
        record_chart(
            spec,
            note=note,
            more_rows_available=trusted is not None and trusted.more_rows_available,
        )
        logger.info(
            "event=tool_call_result kind=%s mode=%s row_count=%s label_count=%s "
            "dataset_count=%s point_count=%s skipped_rows=%s trusted_note=%s",
            spec["type"],
            stats["mode"],
            stats["row_count"],
            len(spec["labels"]) if spec["labels"] is not None else 0,
            len(spec["datasets"]),
            sum(len(d["data"]) for d in spec["datasets"]),
            stats["skipped_rows"],
            note is not None,
        )

        text = (
            f"Chart recorded ({spec['type']}, {len(spec['datasets'])} dataset(s)). "
            "It is attached to your final answer automatically: do not put chart data in the "
            "final answer, just answer the user in words."
        )
        if stats["skipped_rows"]:
            text += f" {stats['skipped_rows']} row(s) were skipped because x or y is not a finite number."
        return {"kind": "text", "text": text}

    # --- building ---

    def _build(
        self,
        kind: Any,
        x: Any,
        y: Any,
        series_by: Any,
        data: Any,
        title: Any,
        horizontal: Any,
    ) -> tuple[dict[str, Any], Optional[TrustedResult], dict[str, Any]]:
        chart_kind = str(kind or "").strip().lower()
        if chart_kind not in _KINDS:
            raise _ChartError(
                f"Invalid kind '{kind}'. Allowed: {list(_KINDS)}. "
                "Box and hexbin plots are drawn by the 'plot' tool.",
                "INVALID_KIND",
            )

        x_col = _optional_name(x)
        y_col = _optional_name(y)
        series_col = _optional_name(series_by)
        if x_col is None:
            raise _ChartError("Missing x column.", "MISSING_X")

        if horizontal is None:
            horizontal = False
        if not isinstance(horizontal, bool):
            raise _ChartError("horizontal must be true or false.", "INVALID_OPTION")
        if horizontal and chart_kind != "bar":
            raise _ChartError("horizontal applies to bar charts only.", "INVALID_OPTION")

        if series_col is not None and chart_kind in _SINGLE_SERIES_KINDS:
            raise _ChartError(f"A {chart_kind} chart has a single series: series_by is not allowed.", "SINGLE_SERIES_ONLY")
        if series_col is not None and series_col in (x_col, y_col):
            raise _ChartError("series_by must be a different column from x and y.", "SAME_DIMENSION")

        trusted: Optional[TrustedResult] = None
        if data is not None:
            if isinstance(data, dict) and "data" in data:
                data = data.get("data")
            if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
                raise _ChartError("Invalid data: expected a list of records.", "INVALID_DATA")
            records: list[dict[str, Any]] = data
            columns: set[Any] = set().union(*(row.keys() for row in records))
            values: Callable[[str], list[Any]] = lambda col: [row.get(col) for row in records]  # noqa: E731
            row_count = len(records)
            # Only the tool-produced list itself carries its trusted facts.
            trusted = lookup_trusted_result({"data": data})
        elif self._df is not None:
            df = self._df
            columns = set(df.columns)
            values = lambda col: df[col].tolist()  # noqa: E731
            row_count = len(df)
        else:
            raise _ChartError("No data to chart: there is no dataset and no data was passed.", "INVALID_DATA")

        if row_count == 0:
            raise _ChartError("No rows to chart.", "EMPTY_DATA")

        unknown = [c for c in (x_col, y_col, series_col) if c is not None and c not in columns]
        if unknown:
            raise _ChartError(f"Invalid column: {', '.join(unknown)}", "INVALID_COLUMN")

        if chart_kind == "scatter":
            if y_col is None:
                raise _ChartError("A scatter chart needs both x and y.", "MISSING_Y")
            labels, datasets, skipped = self._points(values, x_col, y_col, series_col)
            mode = "points"
            y_label = y_col
        elif data is not None:
            if y_col is None:
                raise _ChartError("With data, y must name the column holding the values to draw.", "MISSING_Y")
            labels, datasets = self._records(values, chart_kind, x_col, y_col, series_col)
            skipped, mode, y_label = 0, "records", y_col
        else:
            if y_col is not None or series_col is not None:
                raise _ChartError(
                    "Without data the chart counts rows per value of x; y and series_by need data. "
                    "Compute a metric or a split count with aggregate first and pass its data.",
                    "DATA_REQUIRED",
                )
            labels, datasets = self._counts(values(x_col))
            skipped, mode, y_label = 0, "count", "count"

        spec = {
            "type": chart_kind,
            "labels": labels,
            "datasets": datasets,
            "title": _optional_name(title),
            "x_label": x_col,
            "y_label": y_label,
            "stacked": False,
            "horizontal": horizontal,
        }
        return spec, trusted, {"mode": mode, "row_count": row_count, "skipped_rows": skipped}

    @staticmethod
    def _counts(column: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
        """Rows per observed value, typed like crosstab's categories, missing last."""
        counts = Counter(_category_key(v) for v in column)
        keys = _sort_categories(set(counts))
        labels = _labels(keys)
        return (
            [_display(k, labels) for k in keys],
            [{"label": "count", "data": [counts[k] for k in keys]}],
        )

    @staticmethod
    def _records(
        values: Callable[[str], list[Any]],
        kind: str,
        x_col: str,
        y_col: str,
        series_col: Optional[str],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Precomputed long-form records, pivoted onto one shared labels domain."""
        xs = [_category_key(v) for v in values(x_col)]
        ys = [_value(v) for v in values(y_col)]
        series = [_category_key(v) for v in values(series_col)] if series_col else [None] * len(xs)

        cells: dict[tuple[Any, _Key], Optional[float]] = {}
        for xk, sk, yv in zip(xs, series, ys):
            if (sk, xk) in cells:
                raise _ChartError(
                    "More than one row for the same x"
                    + (" and series" if series_col else "")
                    + ". The chart draws values as supplied and does not aggregate: aggregate first.",
                    "DUPLICATE_POINT",
                )
            cells[(sk, xk)] = yv

        domain = list(dict.fromkeys(xs))
        if kind == "line":
            domain = _line_axis(domain)
        x_labels = _labels(domain)

        if series_col is None:
            datasets = [{"label": y_col, "data": [cells[(None, xk)] for xk in domain]}]
        else:
            series_keys = _sort_categories(set(series))
            series_labels = _labels(series_keys)
            # An absent combination is unknown, not zero.
            datasets = [
                {"label": series_labels[sk], "data": [cells.get((sk, xk)) for xk in domain]}
                for sk in series_keys
            ]
        return [_display(k, x_labels) for k in domain], datasets

    @staticmethod
    def _points(
        values: Callable[[str], list[Any]],
        x_col: str,
        y_col: str,
        series_col: Optional[str],
    ) -> tuple[None, list[dict[str, Any]], int]:
        """Raw x/y points, one dataset per series value; unplaceable rows are counted."""
        xs = values(x_col)
        ys = values(y_col)
        series = [_category_key(v) for v in values(series_col)] if series_col else [None] * len(xs)

        points: dict[Any, list[dict[str, float]]] = {}
        skipped = 0
        for xv, yv, sk in zip(xs, ys, series):
            px, py = _coordinate(xv), _coordinate(yv)
            if px is None or py is None:
                skipped += 1
                continue
            points.setdefault(sk, []).append({"x": px, "y": py})

        if not points:
            raise _ChartError("No row has a finite number in both x and y.", "NO_POINTS")

        if series_col is None:
            return None, [{"label": f"{x_col} / {y_col}", "data": points[None]}], skipped
        series_keys = _sort_categories(set(points))
        series_labels = _labels(series_keys)
        return None, [{"label": series_labels[sk], "data": points[sk]} for sk in series_keys], skipped
