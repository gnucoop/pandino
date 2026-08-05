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
        "Return distinct values of a specified column with their counts. "
        "Supports execution on the session dataset or on provided table data. "
        "Returns a table."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": "Column name to list unique values for.",
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, unique values will be computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },        
        "n": {
            "type": "integer",
            "description": (
                "Optional cap on the number of distinct values returned, most frequent "
                "first. Leave unset to return all of them: the result is previewed and "
                "exported automatically, so capping here only shrinks the user's download."
            ),
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        column: str,
        data: list[dict[str, Any]] | None = None,
        n: Optional[int] = None,
    ) -> dict[str, Any]:
        
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")

                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "table", "data": []}

                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            col = (column or "").strip()
            if not col:
                return {"kind": "error", "message": "Missing column.", "code": "MISSING_COLUMN"}

            if col not in df.columns:
                # If we are operating on tool-provided data, try a small fallback:
                # - case-insensitive match
                if data is not None:
                    lowered = {c.lower(): c for c in df.columns}
                    if col.lower() in lowered:
                        col = lowered[col.lower()]

            if col not in df.columns:
                return {"kind": "error", "message": f"Invalid column: {col}", "code": "INVALID_COLUMN"}

            # No implicit cap: the transport layer previews the result and exports the rest,
            # so truncating here would silently shrink both the answer and the download.
            n_int = max(1, int(n)) if n else None

            s = df[col].dropna()
            vc = s.value_counts()
            total_distinct = int(vc.shape[0])
            if n_int is not None:
                vc = vc.head(n_int)
            records = [{"value": _to_json_scalar(idx), "count": _to_json_scalar(int(cnt))} for idx, cnt in vc.items()]

            records = replace_nan(records)

            logging.info(
                "[datachat][unique_values_tool] col=%s returned=%s total_distinct=%s n=%s",
                col, len(records), total_distinct, n_int if n_int is not None else "all",
            )

            payload: dict[str, Any] = {
                "kind": "table",
                "data": records,
                "export_name": f"unique_{col}",
            }
            if len(records) < total_distinct:
                payload["note"] = (
                    f"{total_distinct} distinct values exist; the {len(records)} most "
                    f"frequent were returned because a limit was requested."
                )
            return payload

        except Exception as e:
            logging.exception("[datachat][unique_values_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
