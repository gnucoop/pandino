from typing import Any, ClassVar, Optional

from smolagents import Tool

from analyst.state import RunState
from analyst.tools.sql_common import split_names
from analyst.values import preview


class ReadSubTaskStateTool(Tool):
    """
    The results of the sub-tasks, as the report agent reads them.

    Each output value comes back as a JSON preview cut to ANALYST_PREVIEW_MAX_CHARS,
    so that a large output (a monthly series, a matrix) cannot flood the report agent's context:
    the numbers it needs to quote are in the previews, and charts and tables
    hold the rest.
    """

    name = "read_sub_task_state"
    description = (
        "Read the results of the tasks: status, the agent's summary, every saved output "
        "(long values are cut) and the insights recorded beyond the outputs. task_ids is "
        "'all' (default) or comma-separated ids."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "task_ids": {
            "type": "string",
            "description": "'all', or comma-separated sub-task ids.",
            "nullable": True,
        }
    }

    def __init__(self, state: RunState, *, preview_chars: int) -> None:
        super().__init__()
        self._state = state
        self._preview_chars = preview_chars

    def forward(self, task_ids: Optional[str] = None) -> dict[str, Any]:
        try:
            wanted = split_names(task_ids or "all")
            if wanted in ([], ["all"]):
                wanted = list(self._state.tasks)
            unknown = [t for t in wanted if t not in self._state.tasks]
            if unknown:
                return {
                    "kind": "error",
                    "message": f"Unknown sub-tasks {unknown}. Known: {', '.join(self._state.tasks)}.",
                    "code": "UNKNOWN_TASK",
                }

            tasks = {}
            for task_id in wanted:
                task = self._state.tasks[task_id]
                tasks[task_id] = {
                    "name": task.name,
                    "status": task.status,
                    "summary": task.summary,
                    "outputs": {k: preview(v, self._preview_chars) for k, v in task.outputs.items()},
                }
                if task.error:
                    tasks[task_id]["error"] = task.error
                insights = [
                    {"text": i["text"], "evidence": i["evidence"]}
                    for i in self._state.insights
                    if i["task_id"] == task_id
                ]
                if insights:
                    tasks[task_id]["insights"] = insights
                if task.exploratory:
                    tasks[task_id]["exploratory"] = True
            return {"kind": "sub_task_state", "tasks": tasks}
        except Exception as e:
            return {"kind": "error", "message": f"read_sub_task_state failed: {e}", "code": "TOOL_FAILED"}
