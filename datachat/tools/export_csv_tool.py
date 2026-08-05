import json
import logging
import os
import uuid
from collections import Counter
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import LiteLLMModel, Tool

from datachat.output_normalizer import replace_nan

_SENTIMENT_SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Analyze the emotional tone of each text and return ONLY a valid JSON object "
    "mapping each input text to its sentiment result. "
    "No explanations, no extra text, no markdown."
)

_MAX_UNIQUE_VALUES = 500


class ExportCsvTool(Tool):
    """
    Save a column (or tool output records) to a CSV file on disk.

    When include_sentiment=True, runs full sentiment analysis on the column
    and exports ALL rows with sentiment labels (no row limit).
    Useful for complete data export of analysis results.
    """

    name = "export_csv"
    description = (
        "Save records or a dataset column to a CSV file on disk. "
        "When include_sentiment=True, runs sentiment analysis on the column "
        "and exports ALL rows with sentiment labels (no limit). "
        "When 'data' is provided, saves those records as-is. "
        "When only 'column' is given, exports the raw column values. "
        "Useful for full data export of large results. "
        "Returns the file path where the CSV was saved."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": (
                "Column name to export. Required unless 'data' is provided."
            ),
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records from another tool. "
                "Saves these records as-is (subject to caller's row limit)."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "filename": {
            "type": "string",
            "description": "Optional custom filename.",
            "nullable": True,
        },
        "include_sentiment": {
            "type": "boolean",
            "description": (
                "If True, runs full sentiment analysis on 'column' and exports "
                "ALL rows with sentiment labels and scores (no row limit). "
                "The CSV will contain the original column plus 'sentiment' and 'score' columns."
            ),
            "nullable": True,
        },
        "sentiment_labels": {
            "type": "array",
            "description": (
                "Custom sentiment labels when include_sentiment=True "
                "(default: ['positive','negative','neutral'])."
            ),
            "items": {"type": "string"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame, output_dir: str, model: Optional[LiteLLMModel] = None) -> None:
        super().__init__()
        self._df = df
        self._output_dir = output_dir
        self._model = model

    def forward(
        self,
        column: str,
        data: list[dict[str, Any]] | None = None,
        filename: Optional[str] = None,
        include_sentiment: Optional[bool] = False,
        sentiment_labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        try:
            col = (column or "").strip()
            export_dir = self._output_dir
            os.makedirs(export_dir, exist_ok=True)

            name_part = col.replace("/", "_").replace(" ", "_") if col else "export"
            if name_part.lower() in {"", "export"}:
                name_part = "export"
            fname = f"{name_part}_{uuid.uuid4().hex[:8]}.csv" if not filename else filename
            out_path = os.path.join(export_dir, fname)

            inc_sent = bool(include_sentiment) if include_sentiment is not None else False

            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")
                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "error", "message": "No data to export.", "code": "EMPTY_DATA"}
                try:
                    out_df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            elif inc_sent:
                if not self._model:
                    return {"kind": "error", "message": "LLM model not available for sentiment export.", "code": "MODEL_UNAVAILABLE"}
                if not col:
                    return {"kind": "error", "message": "Missing column name.", "code": "MISSING_COLUMN"}
                if col not in self._df.columns:
                    return {"kind": "error", "message": f"Column not found: {col}", "code": "INVALID_COLUMN"}
                out_df = self._export_sentiment(col, sentiment_labels)
            else:
                if not col:
                    return {"kind": "error", "message": "Missing column name.", "code": "MISSING_COLUMN"}
                if col not in self._df.columns:
                    return {"kind": "error", "message": f"Column not found: {col}", "code": "INVALID_COLUMN"}
                out_df = self._df[[col]]

            out_df.to_csv(out_path, index=False)
            logging.info("[datachat][export_csv_tool] saved=%s rows=%d cols=%s", out_path, len(out_df), list(out_df.columns))
            return {"kind": "text", "text": f"File CSV salvato: {out_path}"}

        except Exception as e:
            logging.exception("[datachat][export_csv_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    # ------------------------------------------------------------------
    # Internal: run full sentiment analysis (no row limit)
    # ------------------------------------------------------------------

    def _export_sentiment(self, col: str, sentiment_labels: Optional[list[str]]) -> pd.DataFrame:
        labels = sentiment_labels or ["positive", "negative", "neutral"]
        labels_str = ", ".join(f"'{l}'" for l in labels)

        s = self._df[col].dropna().astype(str)
        s = s[s.str.strip() != ""]
        unique_vals = s.unique().tolist()
        if len(unique_vals) > _MAX_UNIQUE_VALUES:
            unique_vals = unique_vals[:_MAX_UNIQUE_VALUES]

        if not unique_vals:
            return self._df[[col]]

        items_str = "\n".join(f'  "{i}": "{v}"' for i, v in enumerate(unique_vals))
        user_prompt = (
            f"Assign a sentiment label (one of: {labels_str}) and a confidence score (0-1) "
            f"to each of the following texts.\n\n"
            f"Texts:\n{{\n{items_str}\n}}\n\n"
            f"Respond with a JSON object mapping each index to "
            f'{{"sentiment": "<label>", "score": <float>}}. '
            f"Return ONLY valid JSON, no other text."
        )

        messages = [
            {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = self._model(messages)
        raw_text = response.content.strip()

        parsed: dict[str, Any] = {}
        try:
            cleaned = raw_text
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            parsed = json.loads(cleaned)
        except Exception:
            pass

        lookup: dict[str, dict[str, Any]] = {}
        if isinstance(parsed, dict):
            cat_lower = {l.lower(): l for l in labels}
            for key, val in parsed.items():
                if isinstance(val, dict) and "sentiment" in val:
                    raw_sentiment = str(val.get("sentiment", "neutral")).strip().lower()
                    sentiment = cat_lower.get(raw_sentiment, labels[0])
                    if sentiment not in labels:
                        sentiment = labels[0]
                    try:
                        score = float(val.get("score", 0.5))
                    except (ValueError, TypeError):
                        score = 0.5
                    lookup[str(key).strip()] = {"sentiment": sentiment, "score": round(score, 4)}

        records: list[dict[str, Any]] = []
        for _, row in self._df.iterrows():
            raw = row.get(col)
            if pd.isna(raw) or not str(raw).strip():
                continue
            text_val = str(raw).strip()
            matched = None
            if text_val in unique_vals:
                val_idx = str(unique_vals.index(text_val))
                matched = lookup.get(val_idx)
            if matched is None:
                for k, v in lookup.items():
                    try:
                        lu_idx = int(k)
                        if lu_idx < len(unique_vals) and unique_vals[lu_idx] == text_val:
                            matched = v
                            break
                    except (ValueError, IndexError):
                        continue
            if matched is None:
                matched = {"sentiment": labels[0], "score": 0.5}
            records.append({
                str(col): text_val,
                "sentiment": matched["sentiment"],
                "score": matched["score"],
            })

        return pd.DataFrame(records)
