from typing import Any, ClassVar

from smolagents import Tool

from analyst.state import RunState

REPORT_FILE = "report.html"


class SaveHtmlReportTool(Tool):
    """
    Write the rendered report into the run folder.

    The HTML is the one get_html_template kept in the run state, and the file name is fixed: 
    the agent chooses neither content nor path.
    """

    name = "save_html_report"
    description = (
        "Write the report rendered by get_html_template to the run folder. Takes no "
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
            path = self._state.run_dir / REPORT_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._state.html, encoding="utf-8")
            self._state.report_path = path
            return {"kind": "text", "text": f"Report saved ({len(self._state.html)} characters)."}
        except Exception as e:
            return {"kind": "error", "message": f"save_html_report failed: {e}", "code": "TOOL_FAILED"}
