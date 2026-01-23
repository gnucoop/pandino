import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


def _to_json_scalar(v: Any) -> Any:
    # JSON scalars only: str, int/float, bool, None
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    # pandas / numpy scalars
    try:
        import numpy as np  # optional at runtime if installed
        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass
    return str(v)


class FilterRowsTool(Tool):
    """
    Smolagents tool: filter rows on a bound DataFrame.

    MVP: supports only equality filter (op='eq').
    """

    name = "filter_rows"
    description = (
        "Filter rows in the dataset by a simple condition and return a small table. "
        "Use this when the user asks for rows where a column equals a value "
        "(e.g., Problemi = 'Lavoro')."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "where_col": {
            "type": "string",
            "description": "Column name to filter on.",
        },
        "value": {
            "type": "string",
            "description": "Value to match (equality).",
        },
        "n": {
            "type": "integer",
            "description": "Max number of rows to return (max 20).",
            "nullable": True,
        },
        "columns": {
            "type": "array",
            "description": "Optional list of columns to include.",
            "items": {"type": "string"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        where_col: str,
        value: str,
        n: Optional[int] = 5,
        columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        try:
            df = self._df

            if not where_col or where_col not in df.columns:
                return {
                    "kind": "error",
                    "message": f"Invalid where_col column: {where_col}",
                    "code": "INVALID_FILTER_COLUMN",
                }

            n_int = max(1, min(int(n or 5), 20))

            # Choose columns
            if columns:
                chosen = [c for c in columns if c in df.columns]
                df_view = df[chosen] if chosen else df
            else:
                df_view = df[list(df.columns)[:10]]  # keep small by default

            # Equality filter (string compare, safe)
            series = df[where_col].astype(str)
            mask = series == str(value)
            filtered = df_view[mask].head(n_int)

            records = filtered.to_dict(orient="records")

            # sanitize to JSON scalars only + NaN -> None
            safe_records: list[dict[str, Any]] = []
            for row in records:
                safe_row: dict[str, Any] = {str(k): _to_json_scalar(v) for k, v in row.items()}
                safe_records.append(safe_row)

            safe_records = replace_nan(safe_records)

            logging.info(
                "[datachat][filter_rows_tool] where_col=%s value=%s n=%s rows=%s",
                where_col,
                value,
                n_int,
                len(safe_records),
            )
            return {"kind": "table", "data": safe_records}

        except Exception as e:
            logging.exception("[datachat][filter_rows_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}