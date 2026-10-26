from typing import Any, ClassVar

from smolagents import Tool

from analyst.metrics import compute_delta
from analyst.state import RunState
from services.analysis_report_service import previous_report_metadata


class GetDeltaSectionTool(Tool):
    """
    How this run's metrics changed since the previous run of the same brief.

    A brief is run as is, so the previous completed run of the same brief_id is the baseline.
    The comparison is kept in the run state, where the report shows it on the KPI cards;
    the agent gets the changes to comment on, not HTML.
    """

    name = "get_delta_section"
    description = (
        "Compare this run's metrics (call save_run_metrics first) with the previous "
        "completed run of the same brief: metric changes and category items added or "
        "removed. The report shows the changes on the KPI cards; mention the relevant "
        "ones in the text. Returns nothing to compare when this is the first run."
    )
    output_type = "object"
    inputs: ClassVar[dict[str, Any]] = {}

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self._state = state

    def forward(self) -> dict[str, Any]:
        try:
            config = self._state.metrics_config
            if config is None or self._state.metrics is None:
                return {
                    "kind": "text",
                    "text": "No metrics to compare (none tracked, or save_run_metrics not called yet).",
                }

            previous = previous_report_metadata(self._state.brief_id, self._state.run_id)
            if previous is None or not isinstance(previous.get("metrics"), dict):
                return {"kind": "text", "text": "This is the first run of this brief: nothing to compare."}

            delta = compute_delta(config, self._state.metrics, previous["metrics"])
            delta["previous_run_at"] = str(previous.get("finished_at") or "")[:10]
            self._state.delta = delta
            return {"kind": "delta", **delta}
        except Exception as e:
            return {"kind": "error", "message": f"get_delta_section failed: {e}", "code": "TOOL_FAILED"}
