import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


def _to_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


class UniqueValuesTool(Tool):
    """
    Return unique values (and counts) for a given column.
    """

    name = "unique_values"
    description = (
        "Return unique values for a column, with their counts. Use this when the user "
        "asks for distinct values or categories."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": "Column name to list unique values for.",
        },
        "n": {
            "type": "integer",
            "description": "Max number of unique values to return (max 50).",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        column: str,
        n: Optional[int] = 20,
    ) -> dict[str, Any]:
        try:
            df = self._df

            col = (column or "").strip()
            if not col or col not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid column: {col}",
                    "code": "INVALID_COLUMN",
                }

            n_int = max(1, min(int(n or 20), 50))

            vc = df[col].dropna().astype(str).value_counts().head(n_int)
            records = [{"value": _to_json_scalar(idx), "count": _to_json_scalar(int(cnt))} for idx, cnt in vc.items()]

            records = replace_nan(records)

            logging.info("[datachat][unique_values_tool] col=%s n=%s", col, n_int)
            return {"kind": "table", "data": records}

        except Exception as e:
            logging.exception("[datachat][unique_values_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
