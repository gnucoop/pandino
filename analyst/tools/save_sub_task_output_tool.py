import re
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

from analyst.state import TaskRecord
from analyst.values import dataset_reference, to_json, to_native

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_records(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, dict) for row in value)


def _head_that_fits(rows: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """The longest prefix of rows whose JSON fits in max_chars (estimated, then shrunk)."""
    n = max(1, len(rows) * max_chars // max(1, len(to_json(rows))))
    while n > 0 and len(to_json(rows[:n])) > max_chars:
        n = n * 9 // 10
    return rows[:n]


class SaveSubTaskOutputTool(Tool):
    """
    Save one output of the current sub-task.

    The task is bound when the tool is built, so an output can never land on the wrong task.
    The value is converted to plain JSON and capped in size: an output is a result, not
    a dataset. Outputs are what later sub-tasks receive as `inputs` and what
    the report is written from.

    A table above the cap is not refused: fighting the cap cost the agent its
    steps (renamed columns, ever smaller copies) and the task its retries. The
    whole table goes to the run's datasets, where the chart agent reads it, and
    the output keeps a reference to it plus the first rows that fit. A table
    above max_rows is refused: that is raw data, not a result.

    Whatever the size of the value, at most max_chars // 2 items of it are ever
    converted: every item takes at least two characters of JSON, so more could
    never fit.
    """

    name = "save_sub_task_output"
    description = (
        "Save one output of the current sub-task under its key. Every required output "
        "key must be saved before your final answer; optional ones when available. The "
        "value may be a number, string, list or dict (DataFrames and Series are converted), "
        "and must be a computed result, not raw rows. A table too large to keep whole is "
        "stored as a dataset and its first rows are saved: sort it first. Saving a key "
        "again replaces it."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "key": {"type": "string", "description": "The output key, exactly as listed in the sub-task."},
        "value": {"type": "any", "description": "The computed value."},
    }

    def __init__(
        self, task: TaskRecord, datasets: dict[str, pd.DataFrame], *, max_chars: int, max_rows: int
    ) -> None:
        super().__init__()
        self._task = task
        self._datasets = datasets
        self._max_chars = max_chars
        self._max_rows = max_rows

    def _too_large(self, key: str, what: str) -> dict[str, Any]:
        return {
            "kind": "error",
            "message": f"The value of {key!r} is {what}: save an aggregate or a summary, not raw rows.",
            "code": "OUTPUT_TOO_LARGE",
        }

    def forward(self, key: str, value: Any) -> dict[str, Any]:
        try:
            key = str(key or "").strip()
            if not _KEY_RE.match(key):
                return {"kind": "error", "message": "key must be snake_case.", "code": "INVALID_KEY"}

            if isinstance(value, pd.DataFrame):
                return self._save_table(key, value)

            # A list of records that fits is saved as it is: going through a
            # DataFrame would change it (missing keys, ints turned into floats).
            item_limit = self._max_chars // 2
            if not isinstance(value, str) and hasattr(value, "__len__") and len(value) > item_limit:
                if _is_records(value):
                    return self._save_table(key, pd.DataFrame(value))
                return self._too_large(key, f"{len(value)} items, more than {self._max_chars} characters can hold")
            native = to_native(value)
            size = len(to_json(native))
            if size <= self._max_chars:
                return self._store(key, native)
            if _is_records(native):
                return self._save_table(key, pd.DataFrame(native))
            return self._too_large(key, f"{size} characters, above the limit of {self._max_chars}")
        except Exception as e:
            return {"kind": "error", "message": f"save_sub_task_output failed: {e}", "code": "TOOL_FAILED"}

    def _save_table(self, key: str, frame: pd.DataFrame) -> dict[str, Any]:
        if len(frame) > self._max_rows:
            return self._too_large(key, f"a table of {len(frame)} rows, above the limit of {self._max_rows}")

        rows = to_native(frame.head(self._max_chars // 2))
        if len(rows) == len(frame) and len(to_json(rows)) <= self._max_chars:
            return self._store(key, rows)

        name = f"{self._task.id}.{key}"
        # A copy: the agent may go on reshaping its frame after saving it.
        self._datasets[name] = frame.copy()
        # Half the cap for the rows: the reference around them must fit too.
        head = _head_that_fits(rows, self._max_chars // 2)
        reference = dataset_reference(name, frame, head)
        return self._store(
            key, reference, note=f" as dataset datasets[{name!r}] ({len(head)} of {len(frame)} rows kept in the output)"
        )

    def _store(self, key: str, native: Any, note: str = "") -> dict[str, Any]:
        self._task.outputs[key] = native
        missing = self._task.missing_outputs()
        return {
            "kind": "text",
            "text": f"Saved {key!r}{note}. "
            + (f"Required outputs still missing: {', '.join(missing)}." if missing else "All required outputs are saved."),
        }
