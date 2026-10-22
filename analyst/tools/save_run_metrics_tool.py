from typing import Any, ClassVar, Optional

from smolagents import Tool

from analyst.metrics import fill_metrics
from analyst.state import RunState
from analyst.values import to_native


class SaveRunMetricsTool(Tool):
    """
    Fix the run's tracked metrics, the KPI cards of the report.

    Every metric and category of the brief's metrics_config is read from
    the sub-task outputs by key, so the agent normally passes nothing and
    never relays a number. It may pass values only for keys the outputs lack.
    """

    name = "save_run_metrics"
    description = (
        "Fill the report's KPI cards: every metric and category of the brief's "
        "metrics_config is taken from the sub-task outputs by key. Call it with no "
        "argument; pass `values` ({key: value}) only for a key reported missing, with a "
        "value taken from the outputs. Call it before get_delta_section and get_html_template."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "values": {
            "type": "object",
            "description": "Optional {key: value} for metrics the outputs lack.",
            "nullable": True,
        }
    }

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self._state = state

    def forward(self, values: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        try:
            if self._state.metrics_config is None:
                self._state.metrics = {"metrics": {}, "categories": {}}
                return {"kind": "text", "text": "The brief tracks no metrics: nothing to save."}

            filled = fill_metrics(
                self._state.metrics_config,
                self._state.merged_outputs(),
                to_native(values) if isinstance(values, dict) else None,
                datasets=self._state.datasets,
            )
            missing = filled.pop("missing")
            self._state.metrics = filled
            return {
                "kind": "metrics",
                "metrics": filled["metrics"],
                "categories": {k: f"{len(v)} items" for k, v in filled["categories"].items()},
                "missing": missing,
            }
        except Exception as e:
            return {"kind": "error", "message": f"save_run_metrics failed: {e}", "code": "TOOL_FAILED"}
