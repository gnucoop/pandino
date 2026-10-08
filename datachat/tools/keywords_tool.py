import html
import logging
import numbers
import re
import unicodedata
from collections import Counter
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.result_provenance import inherits_more_rows_available, record_trusted_result
from datachat.tools.keywords_stopwords import LANGUAGES, STOPWORDS
from datachat.tools.limits import InvalidLimit, invalid_limit_error, optional_limit

logger = logging.getLogger(__name__)

DEFAULT_MIN_ANSWERS = 2

_TAG_RE = re.compile(r"<[^>]*>")
_WS_RE = re.compile(r"\s+")
# Runs of Unicode letters and digits; apostrophes, hyphens, underscores,
# punctuation and emoji separate them.
_RUN_RE = re.compile(r"[^\W_]+")


class _ToolError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _is_missing(value: Any) -> bool:
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _is_integer(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _clean(text: str) -> str:
    """Analytical text: tags removed, entities unescaped, whitespace collapsed, lowercased."""
    text = html.unescape(_TAG_RE.sub(" ", text))
    text = unicodedata.normalize("NFC", text)
    return _WS_RE.sub(" ", text.replace("\xa0", " ")).strip().lower()


def _tokens(text: str) -> list[str]:
    """Alphabetic unigrams of at least two letters; runs containing digits are dropped."""
    return [run for run in _RUN_RE.findall(text) if len(run) >= 2 and run.isalpha()]


class KeywordsTool(Tool):
    """
    Recurring words in a free-text column, ranked by how many responses mention
    them. Local and deterministic: no provider call.
    """

    name = "keywords"
    description = (
        "Find the words that recur across free-text responses in one column. For each word: "
        "answers = number of responses that contain it, count = total occurrences, "
        "share_of_answers = answers / analyzed responses. Ranked by answers, then count, then "
        "word. The stopword language is required: stopwords of that language are removed ('all' removes those of every supported language, which can drop real words of another language). "
        "Only text values are analyzed. Computed locally by counting words; it does not "
        "interpret meaning or tone."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "column": {
            "type": "string",
            "description": "Name of the free-text column to analyze.",
        },
        "language": {
            "type": "string",
            "description": (
                "Required stopword language: 'italian', 'english', 'french', 'spanish' or 'all'. "
                "There is no default; choose the language of the responses."
            ),
            "nullable": True,
        },
        "n": {
            "type": "integer",
            "description": "Optional max number of words to return. If omitted, all qualifying words are returned.",
            "nullable": True,
        },
        "min_answers": {
            "type": "integer",
            "description": f"Minimum number of responses a word must appear in (default {DEFAULT_MIN_ANSWERS}).",
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
        language: Optional[str] = None,
        n: Optional[int] = None,
        min_answers: Optional[int] = None,
        data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            col = column.strip() if isinstance(column, str) else ""
            if not col:
                raise _ToolError("Missing column name.", "MISSING_COLUMN")
            if language is None or (isinstance(language, str) and not language.strip()):
                raise _ToolError(
                    f"Missing language. Expected one of: {', '.join(LANGUAGES)}.",
                    "MISSING_LANGUAGE",
                )
            lang = language
            if lang not in LANGUAGES:
                raise _ToolError(
                    f"Invalid language: {language!r}. Expected one of: {', '.join(LANGUAGES)}.",
                    "INVALID_LANGUAGE",
                )
            threshold = DEFAULT_MIN_ANSWERS if min_answers is None else min_answers
            if not _is_integer(threshold) or threshold < 1:
                raise _ToolError(
                    f"Invalid min_answers: {min_answers!r}. Expected a positive integer.",
                    "INVALID_MIN_ANSWERS",
                )
            if n is not None and not _is_integer(n):
                raise InvalidLimit(f"Invalid n: {n!r}. Expected a positive integer.")
            limit = optional_limit(n)

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

            if col not in df.columns:
                raise _ToolError(f"Invalid column: {col}", "INVALID_COLUMN")

            return self._keywords(
                df[col].tolist(), col, STOPWORDS[lang], threshold, limit, lang, inherits_more_rows_available(data)
            )

        except _ToolError as e:
            return {"kind": "error", "message": e.message, "code": e.code}
        except InvalidLimit as e:
            return invalid_limit_error(e)
        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    def _keywords(
        self,
        values: list[Any],
        col: str,
        stopwords: frozenset[str],
        threshold: int,
        limit: Optional[int],
        lang: str,
        more_rows_available: bool,
    ) -> dict[str, Any]:
        analyzed = 0
        non_text = 0
        answers: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        for value in values:
            if _is_missing(value):
                continue
            if not isinstance(value, str):
                non_text += 1
                continue
            text = _clean(value)
            if not text:
                continue
            analyzed += 1
            terms = [t for t in _tokens(text) if t not in stopwords]
            counts.update(terms)
            answers.update(set(terms))

        ranked = sorted(
            (term for term, a in answers.items() if a >= threshold),
            key=lambda term: (-answers[term], -counts[term], term),
        )
        qualifying = len(ranked)
        if limit is not None:
            ranked = ranked[:limit]
        rows = [
            {
                "term": term,
                "answers": answers[term],
                "count": counts[term],
                "share_of_answers": answers[term] / analyzed,
            }
            for term in ranked
        ]

        logger.info(
            "event=tool_call_result col=%s language=%s analyzed=%s non_text=%s vocabulary=%s "
            "qualifying=%s returned=%s min_answers=%s",
            col,
            lang,
            analyzed,
            non_text,
            len(answers),
            qualifying,
            len(rows),
            threshold,
        )

        payload: dict[str, Any] = {"kind": "table", "data": rows}
        note = f"Excluded {non_text} non-text row(s) from keyword analysis." if non_text else None
        if note:
            payload["note"] = note
        return record_trusted_result(payload, more_rows_available=more_rows_available, note=note)
