import re
from typing import Any, ClassVar, Optional

from smolagents import Tool

from analyst.state import RunState
from analyst.values import to_native

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,60}$")
CHART_TYPES = ("bar", "line", "pie", "doughnut", "scatter", "bubble", "radar", "polarArea")


def _chart_problem(spec: Any, max_points: int) -> Optional[str]:
    """Why `spec` is not a usable Chart.js configuration, or None."""
    if not isinstance(spec, dict):
        return "spec must be a dict."
    if spec.get("type") not in CHART_TYPES:
        return f"spec.type must be one of: {', '.join(CHART_TYPES)}."
    data = spec.get("data")
    if not isinstance(data, dict):
        return "spec.data must be a dict with 'labels' and 'datasets'."
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return "spec.data.datasets must be a non-empty list."
    points = 0
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, dict) or not isinstance(dataset.get("data"), list):
            return f"spec.data.datasets[{i}] must be a dict with a 'data' list."
        points += len(dataset["data"])
    labels = data.get("labels")
    if labels is not None and not isinstance(labels, list):
        return "spec.data.labels must be a list."
    if points > max_points:
        return f"The chart has {points} points, above the limit of {max_points}: aggregate first."
    options = spec.get("options")
    if options is not None and not isinstance(options, dict):
        return "spec.options must be a dict."
    return None


class SaveChartTool(Tool):
    """
    Save a chart for the report, as a Chart.js configuration.

    The report here is interactive HTML, so a chart is the
    configuration Chart.js renders, with its data already computed.
    It is stored in the run state and placed into the report by name.
    """

    name = "save_chart"
    description = (
        "Save a chart for the report. spec is a Chart.js configuration dict: "
        "{'type': 'bar'|'line'|'pie'|'doughnut'|'scatter'|'bubble'|'radar'|'polarArea', "
        "'data': {'labels': [...], 'datasets': [{'label': '...', 'data': [...]}]}, "
        "'options': {...}} (optional, e.g. stacked scales or an indexAxis 'y' for "
        "horizontal bars). A line dataset may carry 'fill' for areas. Data must be "
        "computed values; colors are assigned by the report."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "name": {"type": "string", "description": "snake_case chart name, unique in the run."},
        "title": {"type": "string", "description": "The chart title shown in the report."},
        "spec": {"type": "object", "description": "The Chart.js configuration."},
    }

    def __init__(self, state: RunState, task_id: str, *, max_charts: int, max_points: int) -> None:
        super().__init__()
        self._state = state
        self._task_id = task_id
        self._max_charts = max_charts
        self._max_points = max_points

    def forward(self, name: str, title: str, spec: dict[str, Any]) -> dict[str, Any]:
        try:
            name = str(name or "").strip()
            if not _NAME_RE.match(name):
                return {"kind": "error", "message": "name must be snake_case.", "code": "INVALID_NAME"}
            if name not in self._state.charts and len(self._state.charts) >= self._max_charts:
                return {
                    "kind": "error",
                    "message": f"The limit of {self._max_charts} charts is reached: keep only the essential ones.",
                    "code": "CHARTS_FULL",
                }

            native = to_native(spec)
            problem = _chart_problem(native, self._max_points)
            if problem is not None:
                return {"kind": "error", "message": problem, "code": "INVALID_CHART"}

            self._state.charts[name] = {
                "title": str(title or name).strip(),
                "spec": native,
                "task_id": self._task_id,
            }
            return {"kind": "text", "text": f"Chart {name!r} saved ({len(self._state.charts)}/{self._max_charts})."}
        except Exception as e:
            return {"kind": "error", "message": f"save_chart failed: {e}", "code": "TOOL_FAILED"}
