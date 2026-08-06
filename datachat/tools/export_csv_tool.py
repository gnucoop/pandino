import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool


class ExportCsvTool(Tool):
    """
    Export raw dataset columns to a downloadable CSV.

    This is only needed for data the user never asked to see rendered -- e.g. "give me
    the whole dataset as a file". Any table returned by another tool is exported
    automatically when it exceeds the preview limits (see
    datachat.output_normalizer._build_table_response), so there is no need to call this
    tool to obtain a download link for a result that was just displayed.
    """

    name = "export_csv"
    description = (
        "Export raw dataset columns to a CSV file the user can download. "
        "Use ONLY when the user asks for a file of raw data that has not been analyzed, "
        "e.g. 'download the whole dataset' or 'give me column X as a file'. "
        "Do NOT call this to get a download link for a table another tool already "
        "returned -- large tables are exported automatically. "
        "Returns a download link."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "columns": {
            "type": "array",
            "description": (
                "Column names to export. Omit or pass an empty list to export the "
                "entire dataset."
            ),
            "items": {"type": "string"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame, exporter: Any = None) -> None:
        super().__init__()
        self._df = df
        self._exporter = exporter

    def forward(self, columns: Optional[list[str]] = None) -> dict[str, Any]:
        try:
            if self._exporter is None or not hasattr(self._exporter, "register_export"):
                return {
                    "kind": "error",
                    "message": "Export is not available in this session.",
                    "code": "EXPORT_UNAVAILABLE",
                }

            cols = [str(c).strip() for c in (columns or []) if str(c).strip()]

            missing = [c for c in cols if c not in self._df.columns]
            if missing:
                return {
                    "kind": "error",
                    "message": f"Column not found: {', '.join(missing)}",
                    "code": "INVALID_COLUMN",
                }

            out_df = self._df[cols] if cols else self._df

            if out_df.empty:
                return {"kind": "error", "message": "No data to export.", "code": "EMPTY_DATA"}

            hint = "_".join(cols) if cols else "dataset"
            token, download_filename = self._exporter.register_export(
                out_df.to_dict(orient="records"), hint=hint
            )

            logging.info(
                "[datachat][export_csv_tool] rows=%d cols=%s token=%s",
                len(out_df),
                list(out_df.columns),
                token,
            )
            return {
                "kind": "text",
                "text": f"CSV ready: {len(out_df)} rows.",
                "download_url": f"/datachat/export/{token}",
                "download_filename": download_filename,
            }

        except Exception as e:
            logging.exception("[datachat][export_csv_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
