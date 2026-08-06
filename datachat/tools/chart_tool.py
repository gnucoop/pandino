import logging
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from smolagents import Tool

from datachat.tools.limits import resolve_limit, truncation_note

# Chart kinds that reduce to values a client-side chart library can draw directly.
# 'box' and 'hexbin' have no such reduction and stay with the image-producing plot tool.
_DATA_KINDS = {"bar", "line", "area", "pie", "doughnut", "scatter", "hist", "kde"}
_IMAGE_ONLY_KINDS = {"box", "hexbin"}

# Aggregations that need a y column. 'count' is handled separately: it needs none, and is
# accepted so the vocabulary matches the aggregate tool's 'op'.
_AGG_OPS = {"mean", "sum", "min", "max"}

_DEFAULT_BINS = 10
_KDE_POINTS = 100

# A dataset beyond this is pointless to draw and expensive to ship.
_MAX_POINTS = 2000

# Reserved sentinel for missing categories. Deliberately language-neutral: the backend
# must not choose a display language, and the client localises it (DINO_CLIENT_SPEC.md).
_BLANK_LABEL = "(empty)"

# What pandas produces when a missing value has already been stringified upstream.
_NULL_TEXTS = {"", "nan", "none", "<na>", "nat", "null"}

# Beyond these, vertical bars either truncate their labels or rotate them to unreadability,
# so a horizontal bar chart is the better default. Survey column values are often whole
# phrases -- programme names here run to 29 characters, edition names to 92.
_HORIZONTAL_LABEL_CHARS = 20
_HORIZONTAL_CATEGORY_COUNT = 10


def _to_number(value: Any) -> Optional[float]:
    """
    JSON-safe number, or None for anything non-finite.

    Integral values stay integers so counts serialise as 20 rather than 20.0 -- otherwise
    axis ticks and tooltips read as "20.0 risposte".
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    if num.is_integer():
        return int(num)
    return round(num, 6)


def _labels_of(index: Any) -> list[str]:
    """
    Category labels as strings, with blanks made visible rather than dropped.

    Covers every flavour of missing, including values already stringified upstream:
    astype(str) turns None into the literal "None", which would otherwise become a category.
    """
    labels: list[str] = []
    for value in index:
        text = "" if value is None else str(value).strip()
        labels.append(_BLANK_LABEL if text.lower() in _NULL_TEXTS else text)
    return labels


def _prefers_horizontal(
    kind: str,
    labels: Optional[list[str]],
    explicit: Optional[bool],
) -> bool:
    """
    Whether a bar chart should be drawn with horizontal bars, largest at the top.

    Only bars have a meaningful orientation. A histogram stays vertical whatever its labels:
    the bins belong on the x axis, which is what makes it read as a distribution.

    With horizontal bars the first entry renders at the top, and the series is already sorted
    largest-first, so descending order and "largest at top" come out the same thing.
    """
    if explicit is not None:
        return bool(explicit)
    if kind != "bar":
        return False
    if not labels:
        return False

    if len(labels) > _HORIZONTAL_CATEGORY_COUNT:
        return True

    # The 80th percentile rather than the max: one freak long value should not flip an
    # otherwise short axis. A single outlier can be truncated; a whole axis of long labels
    # cannot.
    lengths = [len(str(label)) for label in labels]
    return float(np.percentile(lengths, 80)) > _HORIZONTAL_LABEL_CHARS


class ChartTool(Tool):
    """
    Produce a chart *specification* for the client to render, rather than an image.

    The payload is a few KB instead of a few hundred, the result is crisp at any size, and
    the client owns colours and light/dark theming. Several charts can therefore accompany a
    written answer, which a base64 image could not afford.

    See DINO_CLIENT_SPEC.md for the contract.
    """

    name = "chart"
    description = (
        "Build a chart for the user to see, as data the client renders. "
        "Kinds: 'bar', 'line', 'area', 'pie', 'doughnut', 'scatter', 'hist', 'kde'. "
        "Prefer this over 'plot' for every visualisation; 'plot' is only for 'box' and "
        "'hexbin'. "
        "To comment on charts in the same answer, call this once per chart and pass the "
        "returned 'chart' objects in a 'charts' list on your final text payload."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "kind": {
            "type": "string",
            "description": (
                "Chart kind: 'bar' (counts or an aggregate per category), 'line', 'area', "
                "'pie', 'doughnut', 'scatter', 'hist' (distribution of one numeric column), "
                "'kde' (smoothed distribution)."
            ),
            "enum": ["bar", "line", "area", "pie", "doughnut", "scatter", "hist", "kde"],
        },
        "x": {
            "type": "string",
            "description": (
                "Column for the categories or the horizontal axis. For 'hist' and 'kde' this "
                "is the numeric column whose distribution is drawn."
            ),
        },
        "y": {
            "type": "string",
            "description": (
                "Optional numeric column to aggregate per category. Omit for counts. "
                "Required for 'scatter'."
            ),
            "nullable": True,
        },
        "agg": {
            "type": "string",
            "description": (
                "How to combine 'y' per category: 'mean' (default), 'sum', 'min', 'max'. "
                "Use 'count' to count rows per category, which is also what happens when 'y' "
                "is omitted. Same vocabulary as the 'aggregate' tool's 'op'."
            ),
            "enum": ["count", "mean", "sum", "min", "max"],
            "nullable": True,
        },
        "series_by": {
            "type": "string",
            "description": (
                "Optional column producing one series per distinct value -- use for grouped "
                "bars or multi-line charts, e.g. satisfaction per programme split by role."
            ),
            "nullable": True,
        },
        "bins": {
            "type": "integer",
            "description": "Number of bins for 'hist' (default 10).",
            "nullable": True,
        },
        "n": {
            "type": "integer",
            "description": (
                "Optional cap on how many categories to show, largest first. Leave unset to "
                "show them all."
            ),
            "nullable": True,
        },
        "sort_by_value": {
            "type": "boolean",
            "description": (
                "Order categories by value, largest first (default True). Set False to keep "
                "the natural order of the column -- right for dates and rating scales."
            ),
            "nullable": True,
        },
        "horizontal": {
            "type": "boolean",
            "description": (
                "Draw a bar chart with horizontal bars, largest at the top. Leave unset to "
                "decide automatically: horizontal when the category labels are long or "
                "numerous, which is usually the readable choice. Ignored for other kinds."
            ),
            "nullable": True,
        },
        "title": {
            "type": "string",
            "description": "Optional chart title.",
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the chart is built from this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame, collector: Any = None) -> None:
        super().__init__()
        self._df = df
        # Optional sink (the engine) notified of every chart built, so the final answer can
        # carry them even if the model does not attach them itself.
        self._collector = collector

    def _report(self, spec: dict[str, Any]) -> None:
        recorder = getattr(self._collector, "record_chart", None)
        if not callable(recorder):
            return
        try:
            recorder(spec)
        except Exception:
            # Bookkeeping must never cost the user their chart.
            logging.warning("[datachat][chart_tool] could not record chart", exc_info=True)

    # ------------------------------------------------------------------

    def forward(
        self,
        kind: str,
        x: str,
        y: Optional[str] = None,
        agg: Optional[str] = None,
        series_by: Optional[str] = None,
        bins: Optional[int] = None,
        n: Optional[int] = None,
        sort_by_value: Optional[bool] = None,
        horizontal: Optional[bool] = None,
        title: Optional[str] = None,
        data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")
                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "error", "message": "No data to chart.", "code": "EMPTY_DATA"}
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            kind_clean = (kind or "").strip().lower()
            if kind_clean == "density":
                kind_clean = "kde"

            if kind_clean in _IMAGE_ONLY_KINDS:
                return {
                    "kind": "error",
                    "message": (
                        f"'{kind_clean}' cannot be expressed as chart data. "
                        f"Use the 'plot' tool for it."
                    ),
                    "code": "KIND_NEEDS_PLOT",
                }
            if kind_clean not in _DATA_KINDS:
                return {
                    "kind": "error",
                    "message": f"Invalid chart kind '{kind_clean}'. Allowed: {sorted(_DATA_KINDS)}",
                    "code": "INVALID_KIND",
                }

            x_clean = (x or "").strip()
            if not x_clean:
                return {"kind": "error", "message": "Missing x column.", "code": "MISSING_X"}
            if x_clean not in df.columns:
                return {"kind": "error", "message": f"Column not found: {x_clean}", "code": "INVALID_COLUMN"}

            y_clean = (y or "").strip() or None
            if y_clean and y_clean not in df.columns:
                return {"kind": "error", "message": f"Column not found: {y_clean}", "code": "INVALID_COLUMN"}

            series_clean = (series_by or "").strip() or None
            if series_clean and series_clean not in df.columns:
                return {"kind": "error", "message": f"Column not found: {series_clean}", "code": "INVALID_COLUMN"}

            agg_clean = (agg or "mean").strip().lower()

            # 'count' counts rows per category and ignores y, exactly as aggregate's
            # op="count" ignores metric. Asking to count a column is a request for the
            # default behaviour, so it must not be an error.
            if agg_clean == "count":
                y_clean = None

            # Only police agg when it can actually affect the result. Without a y there is
            # nothing to aggregate, so an unused value must not fail the call -- rejecting
            # agg="count" on a counts chart is what broke a whole request.
            if y_clean and agg_clean not in _AGG_OPS:
                return {
                    "kind": "error",
                    "message": f"Invalid agg '{agg_clean}'. Allowed: {', '.join(sorted(_AGG_OPS | {'count'}))}",
                    "code": "INVALID_AGG",
                }

            if kind_clean == "scatter" and not y_clean:
                return {"kind": "error", "message": "'scatter' requires a y column.", "code": "MISSING_Y"}

            limit = resolve_limit(n)
            # Ordering by value suits rankings; the natural order suits dates and 1-4 scales.
            sort_desc = True if sort_by_value is None else bool(sort_by_value)

            if kind_clean in {"hist", "kde"}:
                spec, note = self._distribution(df, x_clean, kind_clean, bins)
            elif kind_clean == "scatter":
                spec, note = self._scatter(df, x_clean, y_clean, series_clean)
            else:
                spec, note = self._categorical(
                    df, x_clean, y_clean, agg_clean, series_clean, kind_clean, limit, sort_desc
                )

            if spec is None:
                return {"kind": "error", "message": note or "Could not build the chart.", "code": "NO_CHART_DATA"}

            spec["title"] = (title or "").strip() or None
            # 'stacked' is only a hint, and only meaningful with more than one series.
            spec["stacked"] = bool(series_clean) and kind_clean == "bar" and len(spec["datasets"]) > 1
            spec["horizontal"] = _prefers_horizontal(kind_clean, spec["labels"], horizontal)

            logging.info(
                "[datachat][chart_tool] kind=%s x=%s y=%s agg=%s series_by=%s datasets=%s points=%s",
                kind_clean, x_clean, y_clean, agg_clean if y_clean else None, series_clean,
                len(spec["datasets"]), len(spec["datasets"][0]["data"]) if spec["datasets"] else 0,
            )

            self._report(spec)

            payload: dict[str, Any] = {
                "kind": "chart",
                "chart": spec,
                "export_name": f"chart_{kind_clean}_{x_clean}",
            }
            if note:
                payload["note"] = note
            return payload

        except Exception as e:
            logging.exception("[datachat][chart_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    # ------------------------------------------------------------------
    # Categorical: bar / line / area / pie / doughnut
    # ------------------------------------------------------------------

    def _categorical(
        self,
        df: pd.DataFrame,
        x: str,
        y: Optional[str],
        agg: str,
        series_by: Optional[str],
        kind: str,
        limit: Optional[int],
        sort_desc: bool,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """
        Counts per category, or an aggregate of `y` per category.

        Kept deliberately equivalent to the tools that answer the same question in table
        form -- counts match `unique_values`, aggregates match `aggregate` -- so a chart can
        never disagree with the numbers printed beside it.
        """
        if series_by and kind in {"pie", "doughnut"}:
            return None, "A pie chart cannot show multiple series; drop series_by or use a bar chart."

        categories = df[x]
        y_num = pd.to_numeric(df[y], errors="coerce") if y else None

        if y is not None and y_num is not None and not y_num.notna().any():
            return None, f"Column '{y}' has no numeric values to aggregate."

        # Category order is decided once, on the totals, so every series shares one x axis.
        if y is None:
            totals = categories.value_counts(dropna=False)
            y_label = "count"
        else:
            frame = pd.DataFrame({x: categories, "_v": y_num}).dropna(subset=["_v"])
            if frame.empty:
                return None, f"No rows have both '{x}' and a numeric '{y}'."
            totals = frame.groupby(x, dropna=False)["_v"].agg(agg)
            y_label = f"{agg}({y})"

        totals = totals.sort_values(ascending=False) if sort_desc else totals.sort_index()

        total_categories = int(totals.shape[0])
        if limit is not None:
            totals = totals.head(limit)
        elif total_categories > _MAX_POINTS:
            totals = totals.head(_MAX_POINTS)

        keys = list(totals.index)
        labels = _labels_of(keys)
        note = truncation_note(len(labels), total_categories, unit="categories")

        datasets: list[dict[str, Any]] = []
        if not series_by:
            datasets.append(
                {
                    "label": y_label,
                    "data": [_to_number(v) for v in totals.to_numpy()],
                }
            )
        else:
            # One dataset per series value, aligned to the shared category order. Missing
            # combinations stay null so the client draws a gap rather than a zero -- on a
            # 1-4 scale a zero is not a possible answer.
            if y is None:
                grid = df.groupby([x, series_by], dropna=False).size()
            else:
                frame = pd.DataFrame(
                    {x: categories, series_by: df[series_by], "_v": y_num}
                ).dropna(subset=["_v"])
                grid = frame.groupby([x, series_by], dropna=False)["_v"].agg(agg)

            series_values = [s for s in pd.Series(df[series_by]).dropna().unique()]
            for series_value in series_values:
                row: list[Optional[float]] = []
                for key in keys:
                    try:
                        row.append(_to_number(grid.loc[(key, series_value)]))
                    except (KeyError, TypeError):
                        row.append(None)
                datasets.append({"label": str(series_value), "data": row})

        if not datasets or not labels:
            return None, "No categories to chart."

        # Chart.js has no "area" type: an area chart is a line with fill enabled, so emit
        # exactly that rather than asking the client to translate.
        chart_type = "line" if kind == "area" else kind
        if kind == "area":
            for dataset in datasets:
                dataset["fill"] = True

        return (
            {
                "type": chart_type,
                "labels": labels,
                "datasets": datasets,
                "x_label": x,
                "y_label": y_label,
            },
            note,
        )

    # ------------------------------------------------------------------
    # Distribution: hist / kde
    # ------------------------------------------------------------------

    def _distribution(
        self,
        df: pd.DataFrame,
        x: str,
        kind: str,
        bins: Optional[int],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        values = pd.to_numeric(df[x], errors="coerce").dropna()
        if values.empty:
            return None, f"Column '{x}' has no numeric values."

        if kind == "hist":
            bin_count = max(2, int(bins)) if bins else _DEFAULT_BINS
            counts, edges = np.histogram(values.to_numpy(dtype=float), bins=bin_count)
            labels = [f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(counts))]
            return (
                {
                    "type": "bar",
                    "labels": labels,
                    "datasets": [{"label": "count", "data": [int(c) for c in counts]}],
                    "x_label": x,
                    "y_label": "count",
                },
                None,
            )

        # kde: a smoothed curve, evaluated on an even grid so the client just draws a line.
        if values.nunique() < 2:
            return None, f"Column '{x}' has no variation, so a density curve is undefined."
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            return None, "scipy is required for a density chart. Use kind='hist' instead."

        kde = gaussian_kde(values.to_numpy(dtype=float))
        grid = np.linspace(float(values.min()), float(values.max()), _KDE_POINTS)
        density = kde(grid)
        return (
            {
                "type": "line",
                "labels": [f"{g:.3f}" for g in grid],
                "datasets": [{"label": f"density({x})", "data": [_to_number(d) for d in density]}],
                "x_label": x,
                "y_label": "density",
            },
            None,
        )

    # ------------------------------------------------------------------
    # Scatter
    # ------------------------------------------------------------------

    def _scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        series_by: Optional[str],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        frame = pd.DataFrame(
            {
                "_x": pd.to_numeric(df[x], errors="coerce"),
                "_y": pd.to_numeric(df[y], errors="coerce"),
            }
        )
        if series_by:
            frame["_s"] = df[series_by].astype(str)
        frame = frame.dropna(subset=["_x", "_y"])

        if frame.empty:
            return None, f"No rows have numeric values in both '{x}' and '{y}'."

        total_points = int(frame.shape[0])
        notes: list[str] = []
        datasets: list[dict[str, Any]] = []

        groups = frame.groupby("_s") if series_by else [(f"{x} / {y}", frame)]
        for label, group in groups:
            if group.shape[0] > _MAX_POINTS:
                group = group.head(_MAX_POINTS)
                notes.append(
                    f"Series '{label}' was limited to {_MAX_POINTS} of {total_points} points."
                )
            datasets.append(
                {
                    "label": str(label),
                    "data": [
                        {"x": _to_number(px), "y": _to_number(py)}
                        for px, py in zip(group["_x"], group["_y"])
                    ],
                }
            )

        return (
            {
                "type": "scatter",
                "labels": None,
                "datasets": datasets,
                "x_label": x,
                "y_label": y,
            },
            " ".join(notes) if notes else None,
        )
