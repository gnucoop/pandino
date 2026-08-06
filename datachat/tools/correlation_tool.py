import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan

logger = logging.getLogger(__name__)


class CorrelationTool(Tool):
    """
    Compute correlation between two numeric columns.
    """

    name = "correlation"
    description = (
        "Compute the Pearson correlation coefficient between two numeric columns. "
        "Both columns must contain numeric values. "
        "Returns a single-row table with correlation and number of valid pairs."
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
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, correlation will be computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
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

    def forward(
        self, 
        col_x: str, 
        col_y: str, 
        data: list[dict[str, Any]] | None = None,
        method: str | None = None
    ) -> dict[str, Any]:
        
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")

                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "error", "message": "Not enough data to compute correlation.", "code": "INSUFFICIENT_DATA"}

                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            x = (col_x or "").strip()
            y = (col_y or "").strip()

            if not x or not y:
                return {"kind": "error", "message": "Missing col_x or col_y.", "code": "MISSING_COLUMNS"}

            if x not in df.columns or y not in df.columns:
                if data is not None:
                    lowered = {c.lower(): c for c in df.columns}
                    if x not in df.columns:
                        hit = lowered.get(x.lower())
                        if hit:
                            x = hit
                    if y not in df.columns:
                        hit = lowered.get(y.lower())
                        if hit:
                            y = hit

            if x not in df.columns:
                return {"kind": "error", "message": f"Invalid col_x: {x}", "code": "INVALID_COLUMN"}
            if y not in df.columns:
                return {"kind": "error", "message": f"Invalid col_y: {y}", "code": "INVALID_COLUMN"}

            method_clean = (method or "pearson").strip().lower()
            if method_clean != "pearson":
                return {"kind": "error", "message": f"Invalid method: {method_clean}", "code": "INVALID_METHOD"}

            s_x = pd.to_numeric(df[x], errors="coerce")
            s_y = pd.to_numeric(df[y], errors="coerce")

            x_valid = int(s_x.notna().sum())
            y_valid = int(s_y.notna().sum())

            # If one column has (almost) no numeric values, correlation is not the right operation
            if x_valid < 2:
                return {
                    "kind": "error",
                    "message": f"Column '{x}' has not enough numeric values to compute correlation.",
                    "code": "NO_NUMERIC_DATA",
                }
            if y_valid < 2:
                return {
                    "kind": "error",
                    "message": f"Column '{y}' has not enough numeric values to compute correlation.",
                    "code": "NO_NUMERIC_DATA",
                }

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

            logger.info("x=%s y=%s n=%s corr=%.6f", x, y, len(tmp), corr)
            return {"kind": "table", "data": records}

        except Exception as e:
            logger.exception("failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
