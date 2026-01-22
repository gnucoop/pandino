import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


class SampleRowsTool(Tool):
    """
    Return a small sample of real rows from a bound DataFrame.

    The DataFrame is injected at instantiation time and cannot be changed by the LLM.
    """

    name = "sample_rows"
    description = (
        "Return a small sample of rows from the dataset as a JSON table. "
        "Use this tool when the user asks to see example rows, a preview, "
        "or wants a small table. The dataset is fixed for the session."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "n": {
            "type": "integer",
            "description": "Number of rows to return (max 20).",
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
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df  # bound dataframe

    def forward(self, n: int = 5, columns: list[str] | None = None) -> list[dict[str, Any]]:
        try:
            n_int = max(1, min(int(n or 5), 20))

            df = self._df

            if columns:
                # keep only existing columns, preserve order
                cols = [c for c in columns if c in df.columns]
                if cols:
                    df = df[cols]
            else:
                # default subset: keep response small and readable
                df = df[list(df.columns)[:10]]

            sample = df.head(n_int)
            records = sample.to_dict(orient="records")

            logging.info(
                "[datachat][sample_rows_tool] returning n=%s cols=%s",
                n_int,
                len(sample.columns),
            )
            return replace_nan(records)

        except Exception as e:
            logging.exception("[datachat][sample_rows_tool] failed")
            return []