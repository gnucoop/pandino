from typing import Any, ClassVar

from smolagents import Tool

from analyst.state import RunState


class GetAllSavedChartsTool(Tool):
    """
    The charts and tables the sub-tasks saved, by name.

    Only names, titles and shapes come back: the report places charts and tables by name,
    so their data never passes through the report agent.
    """

    name = "get_all_saved_charts"
    description = (
        "List the charts and tables saved by the sub-tasks: name, title, the sub-task "
        "that made it, and its type (charts) or columns and row count (tables). Place "
        "them in report sections by name."
    )
    output_type = "object"
    inputs: ClassVar[dict[str, Any]] = {}

    def __init__(self, state: RunState) -> None:
        super().__init__()
        self._state = state

    def forward(self) -> dict[str, Any]:
        return {
            "kind": "saved_items",
            "charts": [
                {"name": n, "title": c["title"], "task_id": c["task_id"], "type": c["spec"].get("type")}
                for n, c in self._state.charts.items()
            ],
            "tables": [
                {"name": n, "title": t["title"], "task_id": t["task_id"],
                 "columns": t["columns"], "rows": len(t["rows"])}
                for n, t in self._state.tables.items()
            ],
        }
