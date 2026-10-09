from typing import Any, ClassVar

from smolagents import Tool

from interviewer.state import InterviewState


class RecordFindingTool(Tool):
    """
    Keeps a fact about the data across interview turns.

    The interviewer's memory is reset on every turn, so a query result is gone
    by the next one. A finding recorded here is shown to the agent at the start
    of every later turn and ends up in the brief's data_notes.
    """

    name = "record_finding"
    description = (
        "Record a fact you learned about the data so that it is available in later turns "
        "(query results are not). Use one short, self-contained sentence per call, naming "
        "real tables, columns and values, e.g. '\"Ordini\".\"Stato\" holds codes P, S, C' "
        "or 'orders span 2019-03 to 2025-11'."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "finding": {
            "type": "string",
            "description": "The fact to record, in one sentence.",
        }
    }

    def __init__(self, state: InterviewState) -> None:
        super().__init__()
        self._state = state

    def forward(self, finding: str) -> dict[str, Any]:
        try:
            if not str(finding or "").strip():
                return {"kind": "error", "message": "The finding is empty.", "code": "EMPTY_FINDING"}

            if not self._state.add_note(finding):
                return {
                    "kind": "error",
                    "message": (
                        f"The limit of {self._state.max_notes} findings is reached. "
                        "Keep only what the brief needs."
                    ),
                    "code": "NOTES_FULL",
                }

            return {
                "kind": "text",
                "text": f"Finding recorded ({len(self._state.notes)}/{self._state.max_notes}).",
            }
        except Exception as e:
            return {"kind": "error", "message": f"record_finding failed: {e}", "code": "TOOL_FAILED"}
