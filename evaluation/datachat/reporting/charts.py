"""Deterministic drawing of recorded ``charts[]`` specifications (reporting design §6.3).

``render_chart_png(spec)`` draws exactly the persisted values of one chart
spec (type, labels, datasets, title, axis labels, orientation, stacking) as
PNG bytes. Nothing is added, inferred, sorted or corrected: a spec that
cannot be drawn faithfully (unknown type, non-numeric values, mismatched
lengths, more series than fixed colours) yields ``None`` and the report shows
its values as data only.

Matplotlib's object API is used with the Agg canvas (no pyplot global state)
and PNG metadata is stripped, so identical specs give identical bytes.
"""

import io
import math
from numbers import Real
from typing import Any

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

CHART_DIR = "charts"

# Fixed categorical order (never cycled): a spec needing more colours is not drawn.
_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SURFACE, _INK, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_RC = {"font.family": "DejaVu Sans", "font.size": 9}


def chart_filename(scenario_id: str, index: int) -> str:
    """Stable file name of the ``index``-th (1-based) chart of a scenario, under ``reports/charts/``."""
    return f"{scenario_id}_chart_{index}.png"


def _number(value: Any) -> float | None:
    """A recorded numeric value; None for a recorded null. Raises for anything else."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"not a drawable number: {value!r}")
    return float(value)


def _series(spec: dict[str, Any]) -> tuple[list[str], list[str], list[list[float | None]]]:
    labels, datasets = spec.get("labels"), spec.get("datasets")
    if not isinstance(labels, list) or not isinstance(datasets, list) or not datasets:
        raise ValueError("labels/datasets missing")
    if len(datasets) > len(_SERIES_COLORS):
        raise ValueError("more series than fixed colours")
    names, values = [], []
    for i, d in enumerate(datasets, 1):
        if not isinstance(d, dict) or not isinstance(d.get("data"), list) or len(d["data"]) != len(labels):
            raise ValueError("dataset does not match labels")
        names.append(str(d["label"]) if d.get("label") is not None else f"serie {i}")
        values.append([_number(v) for v in d["data"]])
    return [str(x) for x in labels], names, values


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar(ax, spec, labels, names, values) -> None:
    horizontal, stacked = spec.get("horizontal") is True, spec.get("stacked") is True
    n = len(values)
    width = 0.8 if stacked or n == 1 else 0.8 / n
    base = [0.0] * len(labels)
    for i, (name, data) in enumerate(zip(names, values)):
        heights = [0.0 if v is None else v for v in data]
        offset = 0.0 if stacked else (i - (n - 1) / 2) * width
        pos = [k + offset for k in range(len(labels))]
        draw = ax.barh if horizontal else ax.bar
        kwargs = {"left": base} if horizontal else {"bottom": base}
        draw(pos, heights, width, color=_SERIES_COLORS[i], edgecolor=_SURFACE, linewidth=1, label=name,
             **(kwargs if stacked else {}))
        if stacked:
            base = [b + h for b, h in zip(base, heights)]
    ticks = range(len(labels))
    if horizontal:
        ax.set_yticks(list(ticks), labels)
        ax.invert_yaxis()  # first recorded label on top, as read
        ax.grid(False, axis="y")
    else:
        ax.set_xticks(list(ticks), labels, rotation=45 if len(labels) > 6 else 0,
                      ha="right" if len(labels) > 6 else "center")
        ax.grid(False, axis="x")


def _line(ax, labels, names, values) -> None:
    for i, (name, data) in enumerate(zip(names, values)):
        ax.plot(range(len(labels)), [math.nan if v is None else v for v in data], color=_SERIES_COLORS[i],
                linewidth=2, marker="o", markersize=4, label=name)
    ax.set_xticks(list(range(len(labels))), labels, rotation=45 if len(labels) > 6 else 0,
                  ha="right" if len(labels) > 6 else "center")


def _pie(ax, spec, labels, values) -> None:
    if len(values) != 1 or len(labels) > len(_SERIES_COLORS) or any(v is None or v < 0 for v in values[0]):
        raise ValueError("pie needs one non-negative series within the fixed colours")
    wedge = {"edgecolor": _SURFACE, "linewidth": 2}
    if spec.get("type") == "doughnut":
        wedge["width"] = 0.4
    ax.pie(values[0], labels=labels, colors=_SERIES_COLORS[:len(labels)], wedgeprops=wedge, startangle=90,
           counterclock=False, textprops={"color": _INK})
    ax.set_aspect("equal")


def _scatter(ax, spec) -> list[str]:
    datasets = spec.get("datasets")
    if not isinstance(datasets, list) or not datasets or len(datasets) > len(_SERIES_COLORS):
        raise ValueError("scatter datasets missing")
    names = []
    for i, d in enumerate(datasets, 1):
        points = d.get("data") if isinstance(d, dict) else None
        if not isinstance(points, list) or not all(isinstance(p, dict) for p in points):
            raise ValueError("scatter points malformed")
        xs, ys = [_number(p.get("x")) for p in points], [_number(p.get("y")) for p in points]
        if None in xs or None in ys:
            raise ValueError("scatter point without coordinates")
        name = str(d["label"]) if d.get("label") is not None else f"serie {i}"
        ax.scatter(xs, ys, s=18, color=_SERIES_COLORS[i - 1], edgecolors=_SURFACE, linewidths=0.5, label=name)
        names.append(name)
    return names


def render_chart_png(spec: Any) -> bytes | None:
    """PNG bytes of one recorded chart spec, or None when it cannot be drawn faithfully."""
    if not isinstance(spec, dict):
        return None
    kind = spec.get("type")
    with rc_context(_RC):
        fig = Figure(figsize=(7, 4), dpi=100, facecolor=_SURFACE)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        try:
            if kind == "scatter":
                _style(ax)
                names = _scatter(ax, spec)
            else:
                labels, names, values = _series(spec)
                if kind in ("pie", "doughnut"):
                    _pie(ax, spec, labels, values)
                    names = []
                elif kind == "bar":
                    _style(ax)
                    _bar(ax, spec, labels, names, values)
                elif kind == "line":
                    _style(ax)
                    _line(ax, labels, names, values)
                else:
                    return None
        except (ValueError, TypeError):
            return None
        if spec.get("title") is not None:
            ax.set_title(str(spec["title"]), color=_INK, loc="left", fontsize=11)
        if kind not in ("pie", "doughnut"):
            x_label, y_label = spec.get("x_label"), spec.get("y_label")
            if kind == "bar" and spec.get("horizontal") is True:
                x_label, y_label = y_label, x_label  # values run along x when bars are horizontal
            if x_label is not None:
                ax.set_xlabel(str(x_label), color=_MUTED)
            if y_label is not None:
                ax.set_ylabel(str(y_label), color=_MUTED)
        if len(names) > 1:
            ax.legend(frameon=False, labelcolor=_INK)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=_SURFACE, metadata={"Software": None})
        return buffer.getvalue()
