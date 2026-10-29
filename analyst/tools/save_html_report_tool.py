from typing import Any, ClassVar

from smolagents import Tool

from analyst.state import RunState


class SaveHtmlReportTool(Tool):
    """
    Save the rendered report as the run's report.

    The HTML is the one get_html_template kept in the run state: the agent
    passes no content. The orchestrator stores it with the run (analysis_reports).
    """

    name = "save_html_report"
    description = (
        "Save the report rendered by get_html_template as this run's report. Takes no "
        "argument. Your final answer may follow only after it succeeded."
    )
    output_type = "object"
    inputs: ClassVar[dict[str, Any]] = {}

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self._state = state

    def forward(self) -> dict[str, Any]:
        try:
            if not self._state.html:
                return {
                    "kind": "error",
                    "message": "Nothing to save: call get_html_template first.",
                    "code": "NOT_RENDERED",
                }
            self._state.report_saved = True
            return {"kind": "text", "text": f"Report saved ({len(self._state.html)} characters)."}
        except Exception as e:
            return {"kind": "error", "message": f"save_html_report failed: {e}", "code": "TOOL_FAILED"}
