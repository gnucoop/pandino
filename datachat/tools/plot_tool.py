import os
import uuid
import logging
from typing import Any, ClassVar

import pandas as pd
import numpy as np
from smolagents import Tool

# Headless-safe matplotlib
import matplotlib
matplotlib.use("Agg")  # must be set BEFORE importing pyplot
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class PlotTool(Tool):
    """
    Smolagents tool: generate a simple plot from a bound DataFrame and save it to a PNG file.

    Supported kinds:
    - bar: counts by x (if y is empty) OR mean(y) by x (if y is provided)
    - hist: histogram of a numeric column x
    - line: line plot of y vs x (tries numeric conversion; sorts by x)
    - scatter: scatter plot of numeric y vs numeric x
    - hexbin: hexagonal-bin plot of numeric y vs numeric x
    - kde/density:
        * univariate density on numeric x (when y is empty)
        * bivariate density on numeric x/y (when y is provided)
    - area: area chart built from y aggregated by x (mean/sum)
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
        "(bar, line, histogram, pie, box, scatter, hexbin, kde/density, area). "
        "Returns the file path of the generated image."
    )
    output_type = "object"

    # IMPORTANT: inputs keys must match forward() parameters exactly
    inputs: ClassVar[dict[str, Any]] = {
        "kind": {
            "type": "string",
            "description": (
                "Plot kind: one of 'bar', 'line', 'hist', 'pie', 'box', "
                "'scatter', 'hexbin', 'kde', 'density', 'area'."
            ),
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
                "Used for line/scatter/hexbin/area; optional for bar/pie/kde. "
                "For box (grouped mode): numeric column to plot per category of x."
            ),
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "You may pass either a raw list of records OR a full table payload "
                "like {'kind':'table','data':[...]}."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "agg": {
            "type": "string",
            "description": (
                "Aggregation for bar when y is provided: 'mean' (default) or 'sum'. "
                "Aggregation for area: 'mean' (default) or 'sum'. "
                "For pie with y, only 'sum' is allowed. "
                "Ignored for hist/line/box/scatter/hexbin/kde."
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
        "gridsize": {
            "type": "integer",
            "description": "Hexbin grid size (default 30, min 5, max 200).",
            "nullable": True,
        },
        "alpha": {
            "type": "number",
            "description": "Transparency for scatter/area/kde (default 0.8, min 0.05, max 1.0).",
            "nullable": True,
        },
        "fill": {
            "type": "boolean",
            "description": "If true, use filled rendering for area and kde (default true).",
            "nullable": True,
        },
        "bw_method": {
            "type": "any",
            "description": "KDE bandwidth method: 'scott', 'silverman', or positive numeric value.",
            "nullable": True,
        },
        "sort_x": {
            "type": "boolean",
            "description": "Sort x axis for area and scatter (default true).",
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
        gridsize: int | None = 30,
        alpha: float | None = 0.8,
        fill: bool | None = True,
        bw_method: str | float | None = None,
        sort_x: bool | None = True,
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
                    using_intermediate_data = True
                except Exception:
                    return {
                        "kind": "error",
                        "message": "Invalid data: could not build a table from records.",
                        "code": "INVALID_DATA",
                    }
            else:
                df = self._df
                using_intermediate_data = False

            kind_clean = (kind or "").strip().lower()
            x_clean = (x or "").strip()
            y_clean = (y or "").strip() if y is not None else None
            agg_clean = (agg or "mean").strip().lower() if agg is not None else "mean"
            if kind_clean == "density":
                kind_clean = "kde"

            if kind_clean not in {"bar", "line", "hist", "pie", "box", "scatter", "hexbin", "kde", "area"}:
                return {
                    "kind": "error",
                    "message": (
                        f"Invalid plot kind: {kind_clean}. "
                        "Allowed: bar, line, hist, pie, box, scatter, hexbin, kde, density, area."
                    ),
                    "code": "INVALID_KIND",
                }

            if not x_clean or x_clean not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid x column: {x_clean}",
                    "code": "INVALID_X",
                }

            if y_clean and y_clean not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid y column: {y_clean}",
                    "code": "INVALID_Y",
                }

            if kind_clean in {"line", "scatter", "hexbin", "area"}:
                if not y_clean:
                    return {
                        "kind": "error",
                        "message": f"Missing y column for {kind_clean} plot.",
                        "code": "MISSING_Y",
                    }

            if kind_clean in {"bar", "area"} and y_clean:
                if agg_clean not in {"mean", "sum"}:
                    return {
                        "kind": "error",
                        "message": f"Invalid agg: {agg_clean}. Allowed: mean, sum.",
                        "code": "INVALID_AGG",
                    }

            if kind_clean == "pie":
                if y_clean:
                    if agg_clean not in {"sum"}:
                        return {
                            "kind": "error",
                            "message": f"Invalid agg for pie: {agg_clean}. Allowed: sum.",
                            "code": "INVALID_AGG",
                        }
                else:
                    # y is None/empty => pie is counts by category; agg is ignored
                    pass

            n_int = int(n) if n is not None else 20
            n_int = max(1, min(n_int, 50))

            bins_int = int(bins) if bins is not None else 20
            bins_int = max(5, min(bins_int, 100))

            gridsize_int = int(gridsize) if gridsize is not None else 30
            gridsize_int = max(5, min(gridsize_int, 200))

            alpha_float = float(alpha) if alpha is not None else 0.8
            alpha_float = max(0.05, min(alpha_float, 1.0))

            fill_bool = bool(fill) if fill is not None else True
            sort_x_bool = bool(sort_x) if sort_x is not None else True

            bw: str | float | None
            if bw_method is None:
                bw = None
            elif isinstance(bw_method, (int, float)):
                bw = float(bw_method)
                if bw <= 0:
                    return {
                        "kind": "error",
                        "message": "Invalid bw_method numeric value. Must be > 0.",
                        "code": "INVALID_BW_METHOD",
                    }
            else:
                bw_candidate = str(bw_method).strip().lower()
                if bw_candidate in {"scott", "silverman"}:
                    bw = bw_candidate
                else:
                    try:
                        bw_num = float(bw_candidate)
                        if bw_num <= 0:
                            raise ValueError()
                        bw = bw_num
                    except Exception:
                        return {
                            "kind": "error",
                            "message": "Invalid bw_method. Allowed: 'scott', 'silverman', or positive number.",
                            "code": "INVALID_BW_METHOD",
                        }

            os.makedirs(self._output_dir, exist_ok=True)
            filename = f"plot_{uuid.uuid4().hex}.png"
            out_path = os.path.join(self._output_dir, filename)

            # Build plot
            plt.figure()  # do NOT set a specific style/color

            def _err(message: str, code: str) -> dict[str, Any]:
                plt.close()
                return {"kind": "error", "message": message, "code": code}

            def _best_effort_sort_key(series: pd.Series) -> pd.Series:
                dt = pd.to_datetime(series, errors="coerce", utc=False)
                if dt.notna().any():
                    return dt
                num = pd.to_numeric(series, errors="coerce")
                if num.notna().any():
                    return num
                return series.astype(str).str.strip().str.lower()

            if kind_clean == "hist":
                series = pd.to_numeric(df[x_clean], errors="coerce").dropna()
                if series.empty:
                    return _err(f"Column '{x_clean}' has no numeric values for histogram.", "NO_NUMERIC_DATA")
                plt.hist(series, bins=bins_int)
                plt.xlabel(x_clean)
                plt.ylabel("count")

            elif kind_clean == "box":
                # Mode A: grouped boxplot (x = category, y = numeric)
                if y_clean:
                    y_num = pd.to_numeric(df[y_clean], errors="coerce")
                    tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num}).dropna(subset=[y_clean])

                    if tmp.empty or tmp[y_clean].notna().sum() == 0:
                        return _err(f"Column '{y_clean}' has no numeric values for box plot.", "NO_NUMERIC_DATA")

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
                        return _err("Not enough numeric data to build grouped box plots.", "EMPTY_RESULT")

                    labels = [p[0] for p in pairs]
                    data_lists = [p[1] for p in pairs]

                    if not data_lists:
                        return _err("Not enough numeric data to build grouped box plots.", "EMPTY_RESULT")

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
                        return _err(f"Column '{x_clean}' has no numeric values for box plot.", "NO_NUMERIC_DATA")

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
                    y_vals = counts.to_numpy(dtype=float, copy=False)
                    plt.bar(x_labels, y_vals)
                    plt.xlabel(x_clean)
                    plt.ylabel("count")
                    plt.xticks(rotation=45, ha="right")
                else:
                    y_num = pd.to_numeric(df[y_clean], errors="coerce")
                    tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num}).dropna(subset=[y_clean])

                    if tmp.empty:
                        return _err(
                            f"Column '{y_clean}' has no numeric values for bar plot.",
                            "NO_NUMERIC_DATA",
                        )

                    if using_intermediate_data:
                        x_labels = tmp[x_clean].astype(str).tolist()
                        y_vals = tmp[y_clean].to_numpy(dtype=float, copy=False)
                        y_label = y_clean
                    else:
                        grouped = tmp.groupby(x_clean, dropna=False)[y_clean]
                        if agg_clean == "sum":
                            agg_series = grouped.sum()
                            y_label = f"sum({y_clean})"
                        else:
                            agg_series = grouped.mean()
                            y_label = f"mean({y_clean})"

                        agg_series = agg_series.sort_values(ascending=False).head(n_int)
                        x_labels = [str(v) for v in agg_series.index.tolist()]
                        y_vals = agg_series.to_numpy(dtype=float, copy=False)

                    plt.bar(x_labels, y_vals)
                    plt.xlabel(x_clean)
                    plt.ylabel(y_label)
                    plt.xticks(rotation=45, ha="right")

            elif kind_clean == "line":
                if not y_clean:
                    return _err("Missing y column for line plot.", "MISSING_Y")
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
                        return _err(
                            f"Not enough numeric data to plot '{y_clean}' vs '{x_clean}'.",
                            "NO_NUMERIC_DATA",
                        )

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
                        return _err(f"Not enough numeric data to plot '{y_clean}'.", "NO_NUMERIC_DATA")

                    # Preserve order as given (important for trend tool)
                    x_vals = tmp[x_clean].astype(str).tolist()
                    y_vals = tmp[y_clean].astype(float).tolist()

                    plt.plot(range(len(x_vals)), y_vals)

                    ax = plt.gca()
                    ax.set_xticks(range(len(x_vals)))
                    ax.set_xticklabels(x_vals, rotation=45, ha="right")

                    plt.xlabel(x_clean)
                    plt.ylabel(y_clean)

            elif kind_clean == "scatter":
                assert y_clean is not None
                x_num = pd.to_numeric(df[x_clean], errors="coerce")
                y_num = pd.to_numeric(df[y_clean], errors="coerce")
                tmp = pd.DataFrame({x_clean: x_num, y_clean: y_num}).dropna()

                if tmp.empty or len(tmp) < 2:
                    return _err(
                        f"Not enough numeric data to plot scatter '{y_clean}' vs '{x_clean}'.",
                        "NO_NUMERIC_DATA",
                    )

                if sort_x_bool:
                    tmp = tmp.sort_values(by=x_clean)

                plt.scatter(
                    tmp[x_clean].to_numpy(dtype=float, copy=False),
                    tmp[y_clean].to_numpy(dtype=float, copy=False),
                    alpha=alpha_float,
                )
                plt.xlabel(x_clean)
                plt.ylabel(y_clean)

            elif kind_clean == "hexbin":
                assert y_clean is not None
                x_num = pd.to_numeric(df[x_clean], errors="coerce")
                y_num = pd.to_numeric(df[y_clean], errors="coerce")
                tmp = pd.DataFrame({x_clean: x_num, y_clean: y_num}).dropna()

                if tmp.empty or len(tmp) < 2:
                    return _err(
                        f"Not enough numeric data to plot hexbin '{y_clean}' vs '{x_clean}'.",
                        "NO_NUMERIC_DATA",
                    )

                hb = plt.hexbin(
                    tmp[x_clean].to_numpy(dtype=float, copy=False),
                    tmp[y_clean].to_numpy(dtype=float, copy=False),
                    gridsize=gridsize_int,
                    mincnt=1,
                )
                plt.colorbar(hb, label="count")
                plt.xlabel(x_clean)
                plt.ylabel(y_clean)

            elif kind_clean == "kde":
                from scipy.stats import gaussian_kde

                x_num = pd.to_numeric(df[x_clean], errors="coerce")

                if not y_clean:
                    series = x_num.dropna()
                    if len(series) < 2:
                        return _err(
                            f"Not enough numeric data to compute density for '{x_clean}'.",
                            "NO_NUMERIC_DATA",
                        )
                    var = pd.to_numeric(series.var(), errors="coerce")
                    if pd.isna(var) or float(var) < 1e-12:
                        return _err(
                            f"Column '{x_clean}' has zero variance for density plot.",
                            "ZERO_VARIANCE",
                        )

                    density = gaussian_kde(series.to_numpy(dtype=float, copy=False), bw_method=bw)
                    x_min = float(series.min())
                    x_max = float(series.max())
                    x_grid = np.linspace(x_min, x_max, 200)
                    y_grid = density(x_grid)

                    if fill_bool:
                        plt.fill_between(x_grid, y_grid, alpha=alpha_float)
                    plt.plot(x_grid, y_grid, alpha=max(alpha_float, 0.2))
                    plt.xlabel(x_clean)
                    plt.ylabel("density")
                else:
                    y_num = pd.to_numeric(df[y_clean], errors="coerce")
                    tmp = pd.DataFrame({x_clean: x_num, y_clean: y_num}).dropna()

                    if len(tmp) < 2:
                        return _err(
                            f"Not enough numeric data to compute density for '{x_clean}' and '{y_clean}'.",
                            "NO_NUMERIC_DATA",
                        )
                    var_x = pd.to_numeric(tmp[x_clean].var(), errors="coerce")
                    var_y = pd.to_numeric(tmp[y_clean].var(), errors="coerce")

                    if (
                        pd.isna(var_x)
                        or pd.isna(var_y)
                        or var_x < 1e-12
                        or var_y < 1e-12
                    ):
                        return _err(
                            "At least one column has zero variance for bivariate density plot.",
                            "ZERO_VARIANCE",
                        )

                    points = np.vstack(
                        [
                            tmp[x_clean].to_numpy(dtype=float, copy=False),
                            tmp[y_clean].to_numpy(dtype=float, copy=False),
                        ]
                    )
                    kde = gaussian_kde(points, bw_method=bw)

                    x_min = float(tmp[x_clean].min())
                    x_max = float(tmp[x_clean].max())
                    y_min = float(tmp[y_clean].min())
                    y_max = float(tmp[y_clean].max())
                    grid_x, grid_y = np.mgrid[x_min:x_max:120j, y_min:y_max:120j]
                    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
                    z = np.reshape(kde(grid_coords), grid_x.shape)

                    if fill_bool:
                        plt.contourf(grid_x, grid_y, z, levels=12, alpha=alpha_float)
                    else:
                        plt.contour(grid_x, grid_y, z, levels=12, alpha=alpha_float)

                    plt.xlabel(x_clean)
                    plt.ylabel(y_clean)

            elif kind_clean == "area":
                assert y_clean is not None
                y_num = pd.to_numeric(df[y_clean], errors="coerce")
                tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num}).dropna(subset=[y_clean])

                if tmp.empty:
                    return _err(
                        f"Column '{y_clean}' has no numeric values for area plot.",
                        "NO_NUMERIC_DATA",
                    )

                grouped = tmp.groupby(x_clean, dropna=False)[y_clean]
                series = grouped.sum() if agg_clean == "sum" else grouped.mean()
                if series.empty:
                    return _err("Not enough data to build area plot.", "EMPTY_RESULT")

                ordered = series.reset_index(name=y_clean)
                if sort_x_bool:
                    ordered = ordered.assign(__sort_key=_best_effort_sort_key(ordered[x_clean])).sort_values(
                        by="__sort_key", ascending=True, na_position="last"
                    )

                x_labels = ordered[x_clean].astype(str).tolist()
                y_vals = ordered[y_clean].astype(float).tolist()
                x_idx = list(range(len(x_labels)))

                if fill_bool:
                    plt.fill_between(x_idx, y_vals, alpha=alpha_float)
                plt.plot(x_idx, y_vals, alpha=max(alpha_float, 0.2))
                ax = plt.gca()
                ax.set_xticks(x_idx)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
                plt.xlabel(x_clean)
                plt.ylabel(f"{agg_clean}({y_clean})")

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
                    
                    values = counts.to_numpy(dtype=float, copy=False)

                    if values.size == 0 or float(np.sum(values)) == 0.0:
                        return _err(f"Column '{x_clean}' has no values to compute pie counts.", "EMPTY_RESULT")

                    plt.pie(values, labels=labels, autopct="%1.1f%%")
                else:
                    # composition as sum(y) by category
                    y_num = pd.to_numeric(df[y_clean], errors="coerce")
                    tmp = pd.DataFrame({x_clean: df[x_clean], y_clean: y_num}).dropna(subset=[y_clean])

                    if tmp.empty or tmp[y_clean].notna().sum() == 0:
                        return _err(f"Column '{y_clean}' has no numeric values for pie sum.", "NO_NUMERIC_DATA")

                    sums = (
                        tmp.groupby(x_clean, dropna=False)[y_clean]
                        .sum()
                        .sort_values(ascending=False)
                        .head(n_int)
                    )

                    labels = [str(v) for v in sums.index.tolist()]
                    values = sums.values.tolist()

                    if not values or sum(values) == 0:
                        return _err(f"Sum of '{y_clean}' by '{x_clean}' is empty or zero.", "EMPTY_RESULT")

                    plt.pie(values, labels=labels, autopct="%1.1f%%")

            if title:
                plt.title(str(title))

            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

            logger.info("[datachat][plot_tool] saved plot=%s", out_path)
            return {"kind": "image_path", "path": out_path}

        except Exception as e:
            logger.exception("[datachat][plot_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
