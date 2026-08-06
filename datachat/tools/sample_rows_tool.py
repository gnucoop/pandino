import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan

logger = logging.getLogger(__name__)


def _truncate_cell(value: Any, max_chars: int) -> Any:
    """
    Prevent "infinite text" payloads.
    - Only truncates strings (including stringified non-JSON objects).
    - Leaves numbers/bools/None untouched.
    """
    if max_chars <= 0:
        return value

    if value is None:
        return None

    # Keep simple scalars as-is
    if isinstance(value, (int, float, bool)):
        return value

    # Convert to string if it's not already (safer for weird objects)
    s = value if isinstance(value, str) else str(value)

    if len(s) <= max_chars:
        return s

    # add an ellipsis to make truncation obvious
    if max_chars <= 1:
        return "…"
    return s[: max_chars - 1] + "…"


def _truncate_records(records: list[dict[str, Any]], max_cell_chars: int) -> list[dict[str, Any]]:
    if max_cell_chars is None:
        return records
    max_chars = int(max_cell_chars)
    max_chars = max(0, min(max_chars, 10000))  # hard cap for sanity
    if max_chars == 0:
        return records

    out: list[dict[str, Any]] = []
    for row in records:
        safe_row: dict[str, Any] = {}
        for k, v in row.items():
            safe_row[str(k)] = _truncate_cell(v, max_chars)
        out.append(safe_row)
    return out


class SampleRowsTool(Tool):
    """
    Return a small sample of rows from a DataFrame.

    Composable design:
    - Default: samples from the session dataset (self._df)
    - If 'data' is provided: samples from upstream tool output (list[dict])

    Additions:
    - offset: micro-pagination (slice before taking n rows)
    - max_cell_chars: limits cell text size to avoid giant payloads
    """

    name = "sample_rows"
    description = (
        "Return a small subset of rows from the dataset or from provided table data. "
        "Supports column selection, offset pagination, and cell-size limiting. "
        "Returns a table."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "n": {
            "type": "integer",
            "description": "Number of rows to return (max 20).",
            "nullable": True,
        },
        "offset": {
            "type": "integer",
            "description": "Optional offset for pagination (default 0).",
            "nullable": True,
        },
        "columns": {
            "type": "array",
            "description": (
                "Optional list of columns to include. If omitted, a subset will be chosen."
            ),
            "items": {"type": "string"},
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, sampling will be done on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "max_cell_chars": {
            "type": "integer",
            "description": "Max characters per cell (default 3000, max 10000). Truncates long strings.",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df  # bound dataframe

    def forward(
        self,
        n: int = 5,
        offset: int | None = None,
        columns: list[str] | None = None,
        data: list[dict[str, Any]] | None = None,
        max_cell_chars: int | None = 3000,
    ) -> dict[str, Any]:
        try:
            n_int = max(1, min(int(n or 5), 20))
            offset_int = max(0, int(offset or 0))

            # -----------------------------
            # Data source selection
            # -----------------------------
            is_upstream = data is not None
            if data is not None:
                # allow passing whole tool payload {"kind":"table","data":[...]}
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")  # type: ignore[assignment]

                if not isinstance(data, list):
                    return {
                        "kind": "error",
                        "message": "Invalid data: expected a list of records.",
                        "code": "INVALID_DATA",
                    }
                if len(data) == 0:
                    return {"kind": "table", "data": [], "meta": {"offset": offset_int, "returned": 0}}

                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {
                        "kind": "error",
                        "message": "Invalid data: could not build a table from records.",
                        "code": "INVALID_DATA",
                    }
            else:
                df = self._df

            # -----------------------------
            # Column selection
            # -----------------------------
            if columns:
                cols = [c for c in columns if c in df.columns]
                if cols:
                    df_view = df[cols]
                else:
                    df_view = df
            else:
                # If we're sampling from upstream tool output, it's usually already small/curated,
                # so keep all columns. If sampling from the session dataset, keep it compact.
                df_view = df if is_upstream else df[list(df.columns)[:10]]

            # -----------------------------
            # Pagination then sample
            # -----------------------------
            # slice by offset before taking n
            sample = df_view.iloc[offset_int : offset_int + n_int]

            records = sample.to_dict(orient="records")
            records = replace_nan(records)

            # -----------------------------
            # Anti “testo infinito”
            # -----------------------------
            records = _truncate_records(records, max_cell_chars if max_cell_chars is not None else 3000)
            records = replace_nan(records)  # in case truncation introduced weird values

            logger.info(
                "event=tool_call_result n=%s offset=%s cols=%s source=%s returned=%s",
                n_int,
                offset_int,
                len(sample.columns),
                "upstream" if is_upstream else "session_df",
                len(records),
            )

            return {
                "kind": "table",
                "data": records,
                "meta": {
                    "offset": offset_int,
                    "returned": len(records),
                    # total is easy only for DataFrame; we can still provide it
                    "total_rows": int(len(df_view)),
                },
            }

        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
