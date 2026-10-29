from typing import Any, ClassVar

from smolagents import Tool

from analyst.state import COMPLETED, FAILED, RunState


class GetSubTaskStatusTool(Tool):
    """Which sub-tasks completed and which failed."""

    name = "get_sub_task_status"
    description = (
        "Status of every sub-task (completed / failed), without their outputs. A failed "
        "sub-task has no results: say so in the report instead of inventing them."
    )
    output_type = "object"
    inputs: ClassVar[dict[str, Any]] = {}

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self._state = state

    def forward(self) -> dict[str, Any]:
        tasks = self._state.ordered_tasks()
        return {
            "kind": "sub_task_status",
            "tasks": {t.id: t.status for t in tasks},
            "completed": [t.id for t in tasks if t.status == COMPLETED],
            "failed": [t.id for t in tasks if t.status == FAILED],
        }
