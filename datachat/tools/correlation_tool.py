import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


class CorrelationTool(Tool):
    """
    Compute correlation between two numeric columns.
    """

    name = "correlation"
    description = (
        "Compute correlation between two numeric columns. Use when the user asks "
        "if there is a correlation or relationship between two numeric variables."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "col_x": {
            "type": "string",
            "description": "First numeric column.",
        },
        "col_y": {
            "type": "string",
            "description": "Second numeric column.",
        },
        "method": {
            "type": "string",
            "description": "Correlation method (only 'pearson' supported).",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(self, col_x: str, col_y: str, method: str | None = None) -> dict[str, Any]:
        try:
            df = self._df

            x = (col_x or "").strip()
            y = (col_y or "").strip()
            if not x or x not in df.columns:
                return {"kind": "error", "message": f"Invalid col_x: {x}", "code": "INVALID_COLUMN"}
            if not y or y not in df.columns:
                return {"kind": "error", "message": f"Invalid col_y: {y}", "code": "INVALID_COLUMN"}

            method_clean = (method or "pearson").strip().lower()
            if method_clean != "pearson":
                return {"kind": "error", "message": f"Invalid method: {method_clean}", "code": "INVALID_METHOD"}

            s_x = pd.to_numeric(df[x], errors="coerce")
            s_y = pd.to_numeric(df[y], errors="coerce")
            tmp = pd.DataFrame({x: s_x, y: s_y}).dropna()

            if len(tmp) < 2:
                return {
                    "kind": "error",
                    "message": "Not enough valid numeric pairs to compute correlation.",
                    "code": "INSUFFICIENT_DATA",
                }

            if tmp[x].nunique(dropna=True) < 2 or tmp[y].nunique(dropna=True) < 2:
                return {
                    "kind": "error",
                    "message": "One of the columns has zero variance; correlation is undefined.",
                    "code": "ZERO_VARIANCE",
                }

            corr = float(tmp[x].corr(tmp[y], method="pearson"))
            row = {
                "col_x": x,
                "col_y": y,
                "method": method_clean,
                "correlation": corr,
                "n": int(len(tmp)),
            }
            records = replace_nan([row])

            logging.info("[datachat][correlation_tool] x=%s y=%s n=%s corr=%.6f", x, y, len(tmp), corr)
            return {"kind": "table", "data": records}

        except Exception as e:
            logging.exception("[datachat][correlation_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
