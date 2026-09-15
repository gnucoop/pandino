import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

logger = logging.getLogger(__name__)


class RowCountTool(Tool):
    """
    Returns the total number of records (rows) in the session DataFrame.

    This is a primitive operation used to answer questions like:
    - "Quante sono le visite?"
    - "Quanti record ci sono in totale?"
    """

    name = "row_count"
    description = (
    "Return the total number of rows in the dataset or in the provided table data."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, row_count will be computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        }
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self, 
        data: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")

                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}

                n_rows = int(len(data))
            else:
                n_rows = int(len(self._df))

            logger.info("event=tool_call_result n_rows=%s", n_rows)

            return {"kind": "table", "data": [{"row_count": n_rows}]}

        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
        