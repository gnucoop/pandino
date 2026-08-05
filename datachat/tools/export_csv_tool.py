import logging
import os
import uuid
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


class ExportCsvTool(Tool):
    """
    Save a column (or tool output records) to a CSV file on disk.

    Useful when the user wants to download or inspect the complete dataset,
    the full sentiment analysis, or any tabular result too large for the chat.
    """

    name = "export_csv"
    description = (
        "Save a column from the dataset (or records from another tool) to a CSV file. "
        "Useful when the user requests the full data export, "
        "the complete sentiment analysis results, "
        "or any large table that cannot be displayed entirely in chat. "
        "Returns the file path where the CSV was saved."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": (
                "Column name to export. Required unless 'data' is provided. "
                "When 'data' is also given, 'column' is used as the filename hint."
            ),
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) from another tool. "
                "If provided, saves these records instead of the session column. "
                "Example: export_csv(data=sentiment_analysis_result, column='commenti')"
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "filename": {
            "type": "string",
            "description": (
                "Optional custom filename. "
                "If omitted, a name is generated from the column name and a unique ID."
            ),
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        super().__init__()
        self._df = df
        self._output_dir = output_dir

    def forward(
        self,
        column: str,
        data: list[dict[str, Any]] | None = None,
        filename: Optional[str] = None,
    ) -> dict[str, Any]:
        try:
            col = (column or "").strip()
            export_dir = self._output_dir
            os.makedirs(export_dir, exist_ok=True)

            # Generate filename
            name_part = col.replace("/", "_").replace(" ", "_") if col else "export"
            if name_part.lower() in {"", "export"}:
                name_part = "export"
            fname = f"{name_part}_{uuid.uuid4().hex[:8]}.csv" if not filename else filename
            out_path = os.path.join(export_dir, fname)

            if data is not None:
                # Save provided records (e.g. from another tool's output)
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")
                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "error", "message": "No data to export.", "code": "EMPTY_DATA"}
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                # Save column from session dataset
                if not col:
                    return {"kind": "error", "message": "Missing column name.", "code": "MISSING_COLUMN"}
                if col not in self._df.columns:
                    return {"kind": "error", "message": f"Column not found: {col}", "code": "INVALID_COLUMN"}
                df = self._df[[col]]

            df.to_csv(out_path, index=False)

            logging.info("[datachat][export_csv_tool] saved=%s rows=%d cols=%s", out_path, len(df), list(df.columns))
            return {"kind": "text", "text": f"File CSV salvato: {out_path}"}

        except Exception as e:
            logging.exception("[datachat][export_csv_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
