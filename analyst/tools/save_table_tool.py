import re
from typing import Any, ClassVar

from smolagents import Tool

from analyst.state import RunState
from analyst.values import to_native

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,60}$")
MAX_TABLE_ROWS = 50
MAX_TABLE_COLUMNS = 12


class SaveTableTool(Tool):
    """
    Save a small table for the report.

    What the report shows as a table: rows already computed, few and narrow.
    The report renders them escaped, by name.
    """

    name = "save_table"
    description = (
        f"Save a small table for the report (at most {MAX_TABLE_ROWS} rows and "
        f"{MAX_TABLE_COLUMNS} columns). rows is a list of dicts with the same keys, or a "
        "DataFrame; the keys are the column headers, so name them for the reader."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "name": {"type": "string", "description": "snake_case table name, unique in the run."},
        "title": {"type": "string", "description": "The table title shown in the report."},
        "rows": {"type": "any", "description": "List of dicts, or a DataFrame."},
    }

    def __init__(self, state: RunState, task_id: str) -> None:
        super().__init__()
        self._state = state
        self._task_id = task_id

    def forward(self, name: str, title: str, rows: Any) -> dict[str, Any]:
        try:
            name = str(name or "").strip()
            if not _NAME_RE.match(name):
                return {"kind": "error", "message": "name must be snake_case.", "code": "INVALID_NAME"}

            records = to_native(rows)
            if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
                return {"kind": "error", "message": "rows must be a non-empty list of dicts.", "code": "INVALID_TABLE"}
            if len(records) > MAX_TABLE_ROWS:
                return {
                    "kind": "error",
                    "message": f"{len(records)} rows, above the limit of {MAX_TABLE_ROWS}: keep the top rows only.",
                    "code": "TABLE_TOO_LARGE",
                }

            columns = list(records[0])
            if len(columns) > MAX_TABLE_COLUMNS:
                return {
                    "kind": "error",
                    "message": f"{len(columns)} columns, above the limit of {MAX_TABLE_COLUMNS}.",
                    "code": "TABLE_TOO_LARGE",
                }

            self._state.tables[name] = {
                "title": str(title or name).strip(),
                "columns": columns,
                "rows": [[r.get(c) for c in columns] for r in records],
                "task_id": self._task_id,
            }
            return {"kind": "text", "text": f"Table {name!r} saved ({len(records)} rows)."}
        except Exception as e:
            return {"kind": "error", "message": f"save_table failed: {e}", "code": "TOOL_FAILED"}
