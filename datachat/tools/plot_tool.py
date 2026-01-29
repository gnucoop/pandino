
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

    Returns:
      {"kind":"image_path","path":"...png"} on success
      {"kind":"error","message":"...","code":"..."} on failure
    """

    name = "plot"
    description = (
        "Generate a plot from the dataset and return an image path. "
        "Use for chart requests (bar/line/hist)."
    )
    output_type = "object"

    # IMPORTANT: inputs keys must match forward() parameters exactly
    inputs: ClassVar[dict[str, Any]] = {
        "kind": {
            "type": "string",
            "description": "Plot kind: one of 'bar', 'line', 'hist'.",
        },
        "x": {
            "type": "string",
            "description": "X column name (for hist, this is the numeric column to plot).",
        },
        "y": {
            "type": "string",
            "description": "Y column name (used for line; optional for bar).",
            "nullable": True,
        },
        "agg": {
            "type": "string",
            "description": "Aggregation for bar when y is provided: 'mean' (default) or 'sum'.",
            "nullable": True,
        },
        "n": {
            "type": "integer",
            "description": "Max number of categories for bar (max 50).",
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
        agg: str | None = None,
        n: int | None = 20,
        bins: int | None = 20,
        title: str | None = None,
    ) -> dict[str, Any]:
        try:
            df = self._df

            kind_clean = (kind or "").strip().lower()
            x_clean = (x or "").strip()
            y_clean = (y or "").strip() if y is not None else None
            agg_clean = (agg or "mean").strip().lower() if agg is not None else "mean"

            if kind_clean not in {"bar", "line", "hist"}:
                return {
                    "kind": "error",
                    "message": f"Invalid plot kind: {kind_clean}. Allowed: bar, line, hist.",
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

                x_num = pd.to_numeric(df[x_clean], errors="coerce")
                y_num = pd.to_numeric(df[y_clean], errors="coerce")
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