import os
import uuid
import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

# Headless-safe matplotlib
import matplotlib
matplotlib.use("Agg")  # must be set BEFORE importing pyplot
import matplotlib.pyplot as plt


class PlotTool(Tool):
    """
    Smolagents tool: generate a simple plot from a bound DataFrame and save it to a PNG file.

    Supported kinds:
    - bar: counts by x (if y is empty) OR mean(y) by x (if y is provided)
    - hist: histogram of a numeric column x
    - line: line plot of y vs x (tries numeric conversion; sorts by x)
    - pie: composition chart as a derived view of:
        * count by category (y is empty)
        * sum(y) by category (y is provided, agg must be 'sum')
    - box:
        * single-column mode: boxplot of a numeric column x (y must be empty)
        * grouped mode: boxplots of numeric y grouped by categorical x

    Returns:
      {"kind":"image_path","path":"...png"} on success
      {"kind":"error","message":"...","code":"..."} on failure
    """

    name = "plot"
    description = (
        "Generate a chart from tabular data using a specified chart type "
        "(bar, line, histogram, pie, box). "
        "Returns the file path of the generated image."
    )
    output_type = "object"

    # IMPORTANT: inputs keys must match forward() parameters exactly
    inputs: ClassVar[dict[str, Any]] = {
        "kind": {
            "type": "string",
            "description": "Plot kind: one of 'bar', 'line', 'hist', 'pie', 'box'.",
        },
        "x": {
            "type": "string",
            "description": (
                "X column name. "
                "For hist: numeric column to plot. "
                "For box: either a numeric column (single-column mode) OR a categorical column (grouped mode when y is provided)."
            ),
        },
        "y": {
            "type": "string",
            "description": (
                "Y column name. "
                "Used for line; optional for bar; optional for pie. "
                "For box (grouped mode): numeric column to plot per category of x."
            ),
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the plot will be built from this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "agg": {
            "type": "string",
            "description": (
                "Aggregation for bar when y is provided: 'mean' (default) or 'sum'. "
                "For pie with y, only 'sum' is allowed. "
                "Ignored for hist/line/box."
            ),
            "nullable": True,
        },
        "n": {
            "type": "integer",
            "description": "Max number of categories for bar/pie/box-grouped (max 50).",
            "nullable": True,
        },
        "bins": {
            "type": "integer",
            "description": "Bins for histogram (max 100).",
            "nullable": True,
        },
        "title": {
            "type": "string",
            "description": "Optional plot title.",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        super().__init__()
        self._df = df
        self._output_dir = output_dir

    # NOTE: we deliberately keep params "nullable" (Optional-like) and mark them nullable=True in inputs
    def forward(
        self,
        kind: str,
        x: str,
        y: str | None = None,
        data: list[dict[str, Any]] | None = None,
        agg: str | None = None,
        n: int | None = 20,
        bins: int | None = 20,
        title: str | None = None,
    ) -> dict[str, Any]:
        try:
            # Data source selection:
            # - default: session dataset (self._df)
            # - if 'data' is provided: build a temporary dataframe from tool output records
            if data is not None:
                # Allow passing either raw records OR a full table payload {"kind":"table","data":[...]}
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")

                if not isinstance(data, list):
                    return {
                        "kind": "error",
                        "message": "Invalid data: expected a list of records.",
                        "code": "INVALID_DATA",
                    }
                if len(data) == 0:
                    return {
                        "kind": "error",
                        "message": "Invalid data: empty list of records.",
                        "code": "EMPTY_DATA",
                    }
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {
                        "kind": "error",
                        "message": "Invalid data: could not build a table from records.",
                        "code": "INVALID_DATA",
                    }
            else:
                df = self._df

            kind_clean = (kind or "").strip().lower()
            x_clean = (x or "").strip()
            y_clean = (y or "").strip() if y is not None else None
            agg_clean = (agg or "mean").strip().lower() if agg is not None else "mean"

            if kind_clean not in {"bar", "line", "hist", "pie", "box"}:
                return {
                    "kind": "error",
                    "message": f"Invalid plot kind: {kind_clean}. Allowed: bar, line, hist, pie, box.",
                    "code": "INVALID_KIND",
                }

            if not x_clean or x_clean not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid x column: {x_clean}",
                    "code": "INVALID_X",
                }

            if kind_clean in {"line"}:
                if not y_clean:
                    return {
                        "kind": "error",
                        "message": "Missing y column for line plot.",
                        "code": "MISSING_Y",
                    }
                if y_clean not in df.columns:
                    return {
                        "kind": "error",
                        "message": f"Invalid y column: {y_clean}",
                        "code": "INVALID_Y",
                    }

            if kind_clean == "bar" and y_clean:
                if y_clean not in df.columns:
                    return {
                        "kind": "error",
                        "message": f"Invalid y column: {y_clean}",
                        "code": "INVALID_Y",
                    }
                if agg_clean not in {"mean", "sum"}:
                    return {
                        "kind": "error",
                        "message": f"Invalid agg: {agg_clean}. Allowed: mean, sum.",
                        "code": "INVALID_AGG",
                    }

            if kind_clean == "pie":
                if y_clean:
                    if y_clean not in df.columns:
                        return {
                            "kind": "error",
                            "message": f"Invalid y column: {y_clean}",
                            "code": "INVALID_Y",
                        }
                    if agg_clean not in {"sum"}:
                        return {
                            "kind": "error",
                            "message": f"Invalid agg for pie: {agg_clean}. Allowed: sum.",
                            "code": "INVALID_AGG",
                        }
                else:
                    # y is None/empty => pie is counts by category; agg is ignored
                    pass

            if kind_clean == "box" and y_clean:
                # grouped mode requires y to exist
                if y_clean not in df.columns:
                    return {
                        "kind": "error",
                        "message": f"Invalid y column: {y_clean}",
                        "code": "INVALID_Y",
                    }

            n_int = int(n) if n is not None else 20
            n_int = max(1, min(n_int, 50))

            bins_int = int(bins) if bins is not None else 20
            bins_int = max(5, min(bins_int, 100))

            os.makedirs(self._output_dir, exist_ok=True)
            filename = f"plot_{uuid.uuid4().hex}.png"
            out_path = os.path.join(self._output_dir, filename)

            # Build plot
            plt.figure()  # do NOT set a specific style/color

            if kind_clean == "hist":
                series = pd.to_numeric(df[x_clean], errors="coerce").dropna()
                if series.empty:
                    return {
                        "kind": "error",
                        "message": f"Column '{x_clean}' has no numeric values for histogram.",
                        "code": "NO_NUMERIC_DATA",
                    }
                plt.hist(series, bins=bins_int)
                plt.xlabel(x_clean)
                plt.ylabel("count")

            elif kind_clean == "box":
                # Mode A: grouped boxplot (x = category, y = numeric)
                if y_clean:
                    y_num = pd.to_numeric(df[y_clean], errors="coerce")
                    tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num}).dropna(subset=[y_clean])

                    if tmp.empty or tmp[y_clean].notna().sum() == 0:
                        return {
                            "kind": "error",
                            "message": f"Column '{y_clean}' has no numeric values for box plot.",
                            "code": "NO_NUMERIC_DATA",
                        }

                    grouped = tmp.groupby(x_clean, dropna=False)[y_clean]

                    # Rank groups by *valid numeric* values (not raw row count),
                    # so we don't pick big groups that become empty after numeric coercion.
                    valid_counts = grouped.apply(lambda s: pd.to_numeric(s, errors="coerce").notna().sum())
                    valid_counts = valid_counts[valid_counts > 0].sort_values(ascending=False).head(n_int)

                    pairs: list[tuple[str, Any]] = []
                    for k in valid_counts.index.tolist():
                        arr = pd.to_numeric(grouped.get_group(k), errors="coerce").dropna().to_numpy(dtype=float, copy=False)
                        if len(arr) > 0:
                            pairs.append((str(k), arr))

                    if not pairs:
                        return {
                            "kind": "error",
                            "message": "Not enough numeric data to build grouped box plots.",
                            "code": "EMPTY_RESULT",
                        }

                    labels = [p[0] for p in pairs]
                    data_lists = [p[1] for p in pairs]

                    if not data_lists:
                        return {
                            "kind": "error",
                            "message": "Not enough numeric data to build grouped box plots.",
                            "code": "EMPTY_RESULT",
                        }

                    ax = plt.gca()
                    ax.boxplot(data_lists)
                    
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    
                    ax.set_xlabel(x_clean)
                    ax.set_ylabel(y_clean)


                # Mode B: single-column boxplot (x = numeric)
                else:
                    series = pd.to_numeric(df[x_clean], errors="coerce").dropna()

                    if series.empty:
                        return {
                            "kind": "error",
                            "message": f"Column '{x_clean}' has no numeric values for box plot.",
                            "code": "NO_NUMERIC_DATA",
                        }

                    plt.boxplot(series.to_numpy(dtype=float, copy=False))
                    plt.xlabel(x_clean)

            elif kind_clean == "bar":
                if not y_clean:
                    # counts by x
                    counts = (
                        df.groupby(x_clean, dropna=False)
                        .size()
                        .sort_values(ascending=False)
                        .head(n_int)
                    )
                    # Convert index to str for labels (handle NaN)
                    x_labels = [str(v) for v in counts.index.tolist()]
                    y_vals = counts.values.tolist()
                    plt.bar(x_labels, y_vals)
                    plt.xlabel(x_clean)
                    plt.ylabel("count")
                    plt.xticks(rotation=45, ha="right")
                else:
                    # mean/sum(y) by x
                    y_num = pd.to_numeric(df[y_clean], errors="coerce")
                    tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num})
                    grouped = tmp.groupby(x_clean, dropna=False)[y_clean]
                    if agg_clean == "sum":
                        agg_series = grouped.sum()
                        y_label = f"sum({y_clean})"
                    else:
                        agg_series = grouped.mean()
                        y_label = f"mean({y_clean})"

                    agg_series = agg_series.sort_values(ascending=False).head(n_int)

                    x_labels = [str(v) for v in agg_series.index.tolist()]
                    y_vals = agg_series.values.tolist()
                    plt.bar(x_labels, y_vals)
                    plt.xlabel(x_clean)
                    plt.ylabel(y_label)
                    plt.xticks(rotation=45, ha="right")

            elif kind_clean == "line":
                if not y_clean:
                    return {
                        "kind": "error",
                        "message": "Missing y column for line plot.",
                        "code": "MISSING_Y",
                    }
                assert y_clean is not None  # for type checkers

                # Always coerce Y to numeric
                y_num = pd.to_numeric(df[y_clean], errors="coerce")

                # Try numeric X first
                x_num = pd.to_numeric(df[x_clean], errors="coerce")

                # ---------------------------------
                # Mode A: numeric X
                # ---------------------------------
                if x_num.notna().any():
                    tmp = pd.DataFrame({x_clean: x_num, y_clean: y_num}).dropna()

                    if tmp.empty:
                        return {
                            "kind": "error",
                            "message": f"Not enough numeric data to plot '{y_clean}' vs '{x_clean}'.",
                            "code": "NO_NUMERIC_DATA",
                        }

                    tmp = tmp.sort_values(by=x_clean)

                    x_arr = tmp[x_clean].to_numpy(dtype=float, copy=False)
                    y_arr = tmp[y_clean].to_numpy(dtype=float, copy=False)

                    plt.plot(x_arr, y_arr)
                    plt.xlabel(x_clean)
                    plt.ylabel(y_clean)

                # ---------------------------------
                # Mode B: categorical X (e.g. period)
                # ---------------------------------
                else:
                    tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num}).dropna()

                    if tmp.empty:
                        return {
                            "kind": "error",
                            "message": f"Not enough numeric data to plot '{y_clean}'.",
                            "code": "NO_NUMERIC_DATA",
                        }

                    # Preserve order as given (important for trend tool)
                    x_vals = tmp[x_clean].astype(str).tolist()
                    y_vals = tmp[y_clean].astype(float).tolist()

                    plt.plot(range(len(x_vals)), y_vals)

                    ax = plt.gca()
                    ax.set_xticks(range(len(x_vals)))
                    ax.set_xticklabels(x_vals, rotation=45, ha="right")

                    plt.xlabel(x_clean)
                    plt.ylabel(y_clean)

            elif kind_clean == "pie":
                if not y_clean:
                    # composition as counts by category (agg ignored)
                    counts = (
                        df.groupby(x_clean, dropna=False)
                        .size()
                        .sort_values(ascending=False)
                        .head(n_int)
                    )
                    labels = [str(v) for v in counts.index.tolist()]
                    values = counts.values.tolist()

                    if not values or sum(values) == 0:
                        return {
                            "kind": "error",
                            "message": f"Column '{x_clean}' has no values to compute pie counts.",
                            "code": "EMPTY_RESULT",
                        }

                    plt.pie(values, labels=labels, autopct="%1.1f%%")
                else:
                    # composition as sum(y) by category
                    y_num = pd.to_numeric(df[y_clean], errors="coerce")
                    tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num}).dropna(subset=[y_clean])

                    if tmp.empty or tmp[y_clean].notna().sum() == 0:
                        return {
                            "kind": "error",
                            "message": f"Column '{y_clean}' has no numeric values for pie sum.",
                            "code": "NO_NUMERIC_DATA",
                        }

                    sums = (
                        tmp.groupby(x_clean, dropna=False)[y_clean]
                        .sum()
                        .sort_values(ascending=False)
                        .head(n_int)
                    )

                    labels = [str(v) for v in sums.index.tolist()]
                    values = sums.values.tolist()

                    if not values or sum(values) == 0:
                        return {
                            "kind": "error",
                            "message": f"Sum of '{y_clean}' by '{x_clean}' is empty or zero.",
                            "code": "EMPTY_RESULT",
                        }

                    plt.pie(values, labels=labels, autopct="%1.1f%%")

            if title:
                plt.title(str(title))

            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

            logging.info("[datachat][plot_tool] saved plot=%s", out_path)
            return {"kind": "image_path", "path": out_path}

        except Exception as e:
            logging.exception("[datachat][plot_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
