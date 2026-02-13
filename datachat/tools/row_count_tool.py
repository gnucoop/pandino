import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool


class RowCountTool(Tool):
    """
    Returns the total number of records (rows) in the session DataFrame.

    This is a primitive operation used to answer questions like:
    - "Quante sono le visite?"
    - "Quanti record ci sono in totale?"
    """

    name = "row_count"
    description = "Return the total number of rows (records) in the dataset."
    output_type = "object"

    # No inputs: the operation is unambiguous and always available.
    inputs: ClassVar[dict[str, Any]] = {}

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(self) -> dict[str, Any]:
        try:
            n_rows = int(len(self._df))

            logging.info("[datachat][row_count_tool] n_rows=%s", n_rows)

            return {
                "kind": "text",
                "text": f"Totale record nel dataset: {n_rows}",
                "format": "plain",
            }

        except Exception as e:
            logging.exception("[datachat][row_count_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
        