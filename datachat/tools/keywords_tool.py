import html
import logging
import re
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan
from datachat.tools.limits import resolve_limit, truncation_note
from datachat.tools.stopwords import get_stopwords

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(value: str) -> str:
    """
    Strip the markup that survey exports carry into free-text answers.

    Without this, "&nbsp;" survives as the token "nbsp" and ranks as the single most
    common "word" in the column -- this dataset contains 795 of them.
    """
    text = _TAG_RE.sub(" ", str(value))
    text = html.unescape(text)
    # &nbsp; unescapes to U+00A0, which is not whitespace to the default tokenizer.
    text = text.replace("\xa0", " ")
    return _WS_RE.sub(" ", text).strip()


class KeywordsTool(Tool):
    """
    Most frequent words or two-word phrases in a free-text column.

    Fills the gap between the two LLM text tools: `sentiment_analysis` reports tone and
    `classify` assigns rows to groups, but neither answers "what do people actually talk
    about, and how often" -- the standard first pass over open-ended answers. Cheap and
    deterministic: no LLM call, so it costs nothing against the token budget.
    """

    name = "keywords"
    description = (
        "List the most frequent words or two-word phrases in a free-text column, with how "
        "many answers mention each. Use as the first look at open-ended answers, before "
        "'classify' or 'sentiment_analysis'. Stopwords are removed automatically. "
        "Does not interpret meaning or tone -- use 'sentiment_analysis' for tone."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": "Name of the free-text column to analyze.",
        },
        "ngram": {
            "type": "integer",
            "description": (
                "1 for single words (default), 2 for two-word phrases. "
                "Phrases are often more informative on short survey answers."
            ),
            "nullable": True,
        },
        "top_n": {
            "type": "integer",
            "description": (
                "Optional cap on how many terms to return, most frequent first. "
                "Leave unset to return all of them above min_count."
            ),
            "nullable": True,
        },
        "min_count": {
            "type": "integer",
            "description": "Ignore terms mentioned fewer than this many times (default 2).",
            "nullable": True,
        },
        "language": {
            "type": "string",
            "description": (
                "Stopword language: 'italian', 'english', or 'both'. "
                "Omit to detect it from the text."
            ),
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, keywords are computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        column: str,
        ngram: Optional[int] = None,
        top_n: Optional[int] = None,
        min_count: Optional[int] = None,
        language: Optional[str] = None,
        data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            try:
                from sklearn.feature_extraction.text import CountVectorizer
            except ImportError:
                return {
                    "kind": "error",
                    "message": "scikit-learn is required for keyword extraction. Install it: pip install scikit-learn",
                    "code": "MISSING_SKLEARN",
                }

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

            n_gram = 2 if str(ngram or 1).strip() == "2" else 1
            min_n = max(1, int(min_count)) if min_count else 2
            limit = resolve_limit(top_n)

            texts = df[col].dropna().astype(str).map(_clean_text)
            texts = texts[texts != ""].tolist()
            if not texts:
                return {
                    "kind": "error",
                    "message": "No non-empty text values found in column.",
                    "code": "EMPTY_COLUMN",
                }

            stop_words = get_stopwords(lang=language, texts=texts)

            try:
                vectorizer = CountVectorizer(
                    ngram_range=(n_gram, n_gram),
                    stop_words=stop_words,
                    lowercase=True,
                    # Two-plus letters: drops stray initials and digits that carry no theme.
                    token_pattern=r"(?u)\b[a-zA-ZàèéìòùÀÈÉÌÒÙ][a-zA-ZàèéìòùÀÈÉÌÒÙ]+\b",
                )
                matrix = vectorizer.fit_transform(texts)
            except ValueError as e:
                # Raised when every token was a stopword.
                return {
                    "kind": "error",
                    "message": f"No usable terms found in column: {e}",
                    "code": "NO_TERMS",
                }

            terms = vectorizer.get_feature_names_out()
            total_counts = matrix.sum(axis=0).A1  # occurrences overall
            doc_counts = (matrix > 0).sum(axis=0).A1  # answers containing the term
            n_answers = len(texts)

            rows = [
                {
                    "term": str(terms[i]),
                    "count": int(total_counts[i]),
                    "answers": int(doc_counts[i]),
                    "share_of_answers": round(float(doc_counts[i]) / n_answers, 4),
                }
                for i in range(len(terms))
                if int(total_counts[i]) >= min_n
            ]
            rows.sort(key=lambda r: (-r["count"], r["term"]))

            total_terms = len(rows)
            if limit is not None:
                rows = rows[:limit]

            logging.info(
                "[datachat][keywords_tool] col=%s ngram=%s answers=%s terms=%s returned=%s min_count=%s",
                col, n_gram, n_answers, total_terms, len(rows), min_n,
            )

            payload: dict[str, Any] = {
                "kind": "table",
                "data": replace_nan(rows),
                "export_name": f"keywords_{col}",
                "meta": {
                    "column": col,
                    "ngram": n_gram,
                    "answers_analyzed": n_answers,
                    "terms_found": total_terms,
                    "min_count": min_n,
                },
            }
            note = truncation_note(len(rows), total_terms, unit="terms")
            if note:
                payload["note"] = note
            return payload

        except Exception as e:
            logging.exception("[datachat][keywords_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
