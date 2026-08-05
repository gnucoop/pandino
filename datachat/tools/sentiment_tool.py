import json
import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import LiteLLMModel, Tool

from datachat.output_normalizer import replace_nan

_MAX_UNIQUE_VALUES = 500


def _to_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


_SENTIMENT_SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Analyze the emotional tone of each text and return ONLY a valid JSON object "
    "mapping each input text to its sentiment result. "
    "No explanations, no extra text, no markdown."
)


class SentimentAnalysisTool(Tool):
    """
    Analyze the emotional tone (sentiment) of a text column using the configured LLM.

    Processes unique text values in a single batch and returns per-row
    sentiment labels with confidence scores.
    """

    name = "sentiment_analysis"
    description = (
        "Analyze the emotional tone (sentiment) of a text column. "
        "Assigns each text value a label ('positive', 'negative', 'neutral') "
        "with a numeric confidence score (0-1). "
        "By default (aggregate=True) returns complete summary counts per sentiment. "
        "Use aggregate=False with max_rows for a limited per-row sample. "
        "For the full per-row results, use export_csv on this column. "
        "Do NOT use for topic classification -- use 'classify' for that."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": "Name of the text column to analyze for sentiment.",
        },
        "aggregate": {
            "type": "boolean",
            "description": (
                "If True (default), return aggregate counts per sentiment label. "
                "If False, return per-row results (limited to max_rows). "
                "For the full per-row export, use export_csv."
            ),
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the analysis will run on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "labels": {
            "type": "array",
            "description": (
                "Custom sentiment labels (default: ['positive','negative','neutral']). "
                "Provide as a list of strings, e.g. ['positivo','negativo','neutro']."
            ),
            "items": {"type": "string"},
            "nullable": True,
        },
        "max_rows": {
            "type": "integer",
            "description": (
                "Max rows to return when aggregate=False (default 50). "
                "Increase this or use export_csv for the complete per-row export."
            ),
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame, model: LiteLLMModel) -> None:
        super().__init__()
        self._df = df
        self._model = model

    def forward(
        self,
        column: str,
        aggregate: Optional[bool] = True,
        data: list[dict[str, Any]] | None = None,
        labels: Optional[list[str]] = None,
        max_rows: Optional[int] = 50,
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
                return {"kind": "error", "message": "Missing column name.", "code": "MISSING_COLUMN"}
            if col not in df.columns:
                return {"kind": "error", "message": f"Column not found: {col}", "code": "INVALID_COLUMN"}

            agg = bool(aggregate) if aggregate is not None else True
            sentiment_labels = labels or ["positive", "negative", "neutral"]
            max_n = max(1, int(max_rows or 50))

            # Collect unique non-empty text values (limit to avoid huge prompts)
            s = df[col].dropna().astype(str)
            s = s[s.str.strip() != ""]
            unique_vals = s.unique().tolist()
            if len(unique_vals) > _MAX_UNIQUE_VALUES:
                unique_vals = unique_vals[:_MAX_UNIQUE_VALUES]

            if not unique_vals:
                return {"kind": "error", "message": "No non-empty text values found in column.", "code": "EMPTY_COLUMN"}

            # Build prompt
            labels_str = ", ".join(f"'{l}'" for l in sentiment_labels)
            items_str = "\n".join(f'  "{i}": "{v}"' for i, v in enumerate(unique_vals))
            user_prompt = (
                f"Assign a sentiment label (one of: {labels_str}) and a confidence score (0-1) "
                f"to each of the following texts.\n\n"
                f"Texts:\n{{\n{items_str}\n}}\n\n"
                f"Respond with a JSON object mapping each index to "
                f'{{"sentiment": "<label>", "score": <float>}}. '
                f"Example: {{\"0\": {{\"sentiment\": \"positive\", \"score\": 0.95}}, ...}}\n"
                f"Return ONLY valid JSON, no other text."
            )

            # Call LLM
            messages = [
                {"role": "system", "content": _SENTIMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                response = self._model(messages)
                raw_text = response.content.strip()
            except Exception as e:
                logging.exception("[datachat][sentiment_tool] LLM call failed")
                return {"kind": "error", "message": f"LLM call failed: {e}", "code": "LLM_FAILED"}

            # Parse JSON response (handle potential markdown fences)
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

            if not isinstance(parsed, dict):
                return {"kind": "error", "message": "Failed to parse LLM response as JSON.", "code": "PARSE_FAILED"}

            # Build lookup: value -> sentiment/score
            lookup: dict[str, dict[str, Any]] = {}
            for key, val in parsed.items():
                if isinstance(val, dict) and "sentiment" in val:
                    idx_str = str(key).strip()
                    sentiment = str(val.get("sentiment", "neutral")).strip().lower()
                    if sentiment not in {l.lower() for l in sentiment_labels}:
                        sentiment = "neutral"
                    try:
                        score = float(val.get("score", 0.5))
                    except (ValueError, TypeError):
                        score = 0.5
                    lookup[idx_str] = {"sentiment": sentiment, "score": round(score, 4)}

            # Map back to original rows (skip empty/NaN rows)
            records: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                raw = row.get(col)
                if pd.isna(raw) or not str(raw).strip():
                    continue
                text_val = str(raw).strip()
                matched = None

                # Try full string match first
                if text_val in unique_vals:
                    val_idx = str(unique_vals.index(text_val))
                    matched = lookup.get(val_idx)

                # Fallback: iterate lookup
                if matched is None:
                    for key, val in lookup.items():
                        try:
                            lu_idx = int(key)
                            if lu_idx < len(unique_vals) and unique_vals[lu_idx] == text_val:
                                matched = val
                                break
                        except (ValueError, IndexError):
                            continue

                if matched is None:
                    matched = {"sentiment": "neutral", "score": 0.5}

                records.append({
                    str(col): text_val,
                    "sentiment": matched["sentiment"],
                    "score": matched["score"],
                })

            if agg:
                from collections import Counter
                counts = Counter(r["sentiment"] for r in records)
                agg_records = [
                    {"sentiment": label, "count": count}
                    for label, count in counts.most_common()
                ]
                agg_records = replace_nan(agg_records)
                logging.info("[datachat][sentiment_tool] col=%s rows=%d agg=%s", col, len(records), len(agg_records))
                return {"kind": "table", "data": agg_records}

            records = replace_nan(records[:max_n])
            logging.info("[datachat][sentiment_tool] col=%s rows=%d", col, len(records))
            return {"kind": "table", "data": records}

        except Exception as e:
            logging.exception("[datachat][sentiment_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
