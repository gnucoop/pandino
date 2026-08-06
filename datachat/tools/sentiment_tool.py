import json
import logging
from collections import Counter
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import LiteLLMModel, Tool

from datachat.output_normalizer import replace_nan

_MAX_UNIQUE_VALUES = 500

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
        "By default (aggregate=True) returns summary counts per sentiment. "
        "Use aggregate=False to return the per-row results; large results are shown as a "
        "preview and the complete set is offered as a download automatically. "
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
                "If False, return the per-row results."
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

            # Collect unique non-empty text values (limit to avoid huge prompts)
            s = df[col].dropna().astype(str).str.strip()
            s = s[s != ""]
            all_unique = s.unique().tolist()
            unique_vals = all_unique[:_MAX_UNIQUE_VALUES]
            skipped_unique = len(all_unique) - len(unique_vals)

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

            if not isinstance(parsed, dict) or not parsed:
                return {"kind": "error", "message": "Failed to parse LLM response as JSON.", "code": "PARSE_FAILED"}

            # Build lookup: text value -> sentiment/score.
            # The LLM keys its answer by the index we sent, so resolve those indexes to
            # the text they stood for and key on the text itself: that makes the per-row
            # mapping below a dict hit instead of a scan over unique_vals.
            by_text: dict[str, dict[str, Any]] = {}
            label_set = {l.lower() for l in sentiment_labels}
            for key, val in parsed.items():
                if not isinstance(val, dict) or "sentiment" not in val:
                    continue
                try:
                    idx = int(str(key).strip())
                except (ValueError, TypeError):
                    continue
                if not 0 <= idx < len(unique_vals):
                    continue

                sentiment = str(val.get("sentiment", "")).strip().lower()
                if sentiment not in label_set:
                    # An off-menu label is not a signal we can use; treat it as unscored
                    # rather than silently rounding it to a real category.
                    continue
                try:
                    score = round(float(val.get("score", 0.5)), 4)
                except (ValueError, TypeError):
                    score = None

                by_text[unique_vals[idx]] = {"sentiment": sentiment, "score": score}

            if not by_text:
                # The model answered, but nothing in it was usable. Returning a table of
                # empty labels here would look like a successful analysis of nothing.
                return {
                    "kind": "error",
                    "message": "The model returned no usable sentiment labels.",
                    "code": "PARSE_FAILED",
                }

            # Map back to original rows (skip empty/NaN rows).
            # Rows the LLM never scored -- because the column had more distinct values
            # than we could send, or because it omitted an index -- are left null. They
            # must NOT be defaulted to a real label with a confidence score: that would
            # be indistinguishable from a genuine result, in the UI and in the CSV.
            records: list[dict[str, Any]] = []
            unscored = 0
            for _, row in df.iterrows():
                raw = row.get(col)
                if pd.isna(raw) or not str(raw).strip():
                    continue
                text_val = str(raw).strip()

                matched = by_text.get(text_val)
                if matched is None:
                    unscored += 1

                records.append({
                    str(col): text_val,
                    "sentiment": matched["sentiment"] if matched else None,
                    "score": matched["score"] if matched else None,
                })

            logging.info(
                "[datachat][sentiment_tool] col=%s rows=%d scored=%d unscored=%d skipped_unique=%d",
                col, len(records), len(records) - unscored, unscored, skipped_unique,
            )

            if agg:
                counts = Counter(r["sentiment"] for r in records)
                agg_records = [
                    {"sentiment": label if label is not None else "(not analyzed)", "count": count}
                    for label, count in counts.most_common()
                ]
                return {
                    "kind": "table",
                    "data": replace_nan(agg_records),
                    "export_name": f"sentiment_{col}",
                    "note": self._coverage_note(unscored, skipped_unique),
                }

            return {
                "kind": "table",
                "data": replace_nan(records),
                "export_name": f"sentiment_{col}",
                "note": self._coverage_note(unscored, skipped_unique),
            }

        except Exception as e:
            logging.exception("[datachat][sentiment_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    @staticmethod
    def _coverage_note(unscored: int, skipped_unique: int) -> Optional[str]:
        """Describe incomplete coverage so it is never presented as a full result."""
        if not unscored:
            return None
        if skipped_unique:
            return (
                f"{unscored} rows could not be analyzed: the column has "
                f"{skipped_unique} more distinct values than the {_MAX_UNIQUE_VALUES}-value "
                f"analysis limit. Their sentiment is empty, not neutral."
            )
        return (
            f"{unscored} rows were not scored by the model. "
            f"Their sentiment is empty, not neutral."
        )
