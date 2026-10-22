from typing import Any, Callable, ClassVar, Optional

from smolagents import Tool

from analyst.state import RunState
from analyst.values import preview, to_native

MAX_INSIGHTS_PER_TASK = 3
_MAX_TEXT_CHARS = 400
_MAX_EVIDENCE_CHARS = 1000


class RecordInsightTool(Tool):
    """
    Record a notable fact found in the data, beyond the task's required outputs.

    The required outputs are what the brief asked for; an insight is what the
    agent came across on the way: an outlier client, a sudden drop, a strong
    correlation. It must carry the computed evidence it rests on, so that the
    report can present it as a finding and not as a guess. Insights land in the
    report's additional-findings section, kept apart from what the brief asked.
    """

    name = "record_insight"
    description = (
        "Record a notable fact you found in the data that the report should mention, "
        "beyond your required outputs: an outlier, a break in a trend, a strong "
        "correlation, a concentration, a data-quality issue. text is one or two "
        "sentences naming the numbers; evidence is the computed value(s) it rests on "
        f"(a number, dict or small list, never typed by hand). At most "
        f"{MAX_INSIGHTS_PER_TASK} per task: keep the ones that matter."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "text": {"type": "string", "description": "The finding, in one or two sentences."},
        "evidence": {"type": "any", "description": "The computed values that support it."},
    }

    def __init__(
        self,
        state: RunState,
        task_id: str,
        *,
        max_insights: int,
        notify: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__()
        self._state = state
        self._task_id = task_id
        self._max_insights = max_insights
        self._notify = notify

    def forward(self, text: str, evidence: Any) -> dict[str, Any]:
        try:
            text = str(text or "").strip()
            if not text:
                return {"kind": "error", "message": "The insight text is empty.", "code": "EMPTY_INSIGHT"}
            native = to_native(evidence)
            if native in (None, "", [], {}):
                return {
                    "kind": "error",
                    "message": "An insight needs the computed evidence it rests on.",
                    "code": "NO_EVIDENCE",
                }

            mine = sum(1 for i in self._state.insights if i["task_id"] == self._task_id)
            if mine >= MAX_INSIGHTS_PER_TASK or len(self._state.insights) >= self._max_insights:
                return {
                    "kind": "error",
                    "message": "The insight limit is reached: keep only the most important ones.",
                    "code": "INSIGHTS_FULL",
                }

            self._state.insights.append({
                "task_id": self._task_id,
                "text": text[:_MAX_TEXT_CHARS],
                "evidence": preview(native, _MAX_EVIDENCE_CHARS),
            })
            if self._notify is not None:
                self._notify(text[:_MAX_TEXT_CHARS])
            return {"kind": "text", "text": f"Insight recorded ({mine + 1}/{MAX_INSIGHTS_PER_TASK} for this task)."}
        except Exception as e:
            return {"kind": "error", "message": f"record_insight failed: {e}", "code": "TOOL_FAILED"}
