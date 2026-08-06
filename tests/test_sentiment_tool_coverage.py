"""Tests that the sentiment tool never invents a label it did not get from the model.

Unscored rows must stay empty. A fabricated 'neutral'/0.5 is indistinguishable from a
real result once it reaches the UI or the exported CSV.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from datachat.tools import sentiment_tool
from datachat.tools.sentiment_tool import SentimentAnalysisTool


class FakeModel:
    """Returns a canned JSON body, like the LLM would."""

    def __init__(self, payload, raw: str | None = None) -> None:
        self._raw = raw if raw is not None else json.dumps(payload)
        self.prompts: list[str] = []

    def __call__(self, messages):
        self.prompts.append(messages[-1]["content"])
        return SimpleNamespace(content=self._raw)


def _tool(df, model):
    return SentimentAnalysisTool(df, model=model)


def test_scored_rows_carry_the_model_label():
    df = pd.DataFrame({"txt": ["great", "awful"]})
    model = FakeModel(
        {
            "0": {"sentiment": "positive", "score": 0.9},
            "1": {"sentiment": "negative", "score": 0.8},
        }
    )

    result = _tool(df, model).forward("txt", aggregate=False)

    assert result["kind"] == "table"
    assert result["data"] == [
        {"txt": "great", "sentiment": "positive", "score": 0.9},
        {"txt": "awful", "sentiment": "negative", "score": 0.8},
    ]
    assert result["note"] is None


def test_rows_the_model_skipped_are_left_empty():
    df = pd.DataFrame({"txt": ["great", "awful"]})
    model = FakeModel({"0": {"sentiment": "positive", "score": 0.9}})

    result = _tool(df, model).forward("txt", aggregate=False)

    unscored = [r for r in result["data"] if r["txt"] == "awful"][0]
    assert unscored["sentiment"] is None
    assert unscored["score"] is None
    assert "not scored" in result["note"]


def test_off_menu_label_is_not_rounded_to_a_real_category():
    """An unexpected label is not evidence for any category, least of all the first one."""
    df = pd.DataFrame({"txt": ["great", "hmm"]})
    model = FakeModel(
        {
            "0": {"sentiment": "positive", "score": 0.9},
            "1": {"sentiment": "ambivalent", "score": 0.7},
        }
    )

    result = _tool(df, model).forward("txt", aggregate=False)

    by_text = {r["txt"]: r for r in result["data"]}
    assert by_text["great"]["sentiment"] == "positive"
    assert by_text["hmm"]["sentiment"] is None


def test_entirely_off_menu_response_is_an_error():
    """If nothing the model said was usable, that is a failure, not an empty analysis."""
    df = pd.DataFrame({"txt": ["hmm"]})
    model = FakeModel({"0": {"sentiment": "ambivalent", "score": 0.7}})

    result = _tool(df, model).forward("txt", aggregate=False)

    assert result["kind"] == "error"
    assert result["code"] == "PARSE_FAILED"


def test_values_beyond_the_unique_limit_are_reported_not_defaulted(monkeypatch):
    monkeypatch.setattr(sentiment_tool, "_MAX_UNIQUE_VALUES", 2)
    df = pd.DataFrame({"txt": ["a", "b", "c", "d"]})
    model = FakeModel(
        {
            "0": {"sentiment": "positive", "score": 0.9},
            "1": {"sentiment": "negative", "score": 0.9},
        }
    )

    result = _tool(df, model).forward("txt", aggregate=False)

    by_text = {r["txt"]: r for r in result["data"]}
    assert by_text["a"]["sentiment"] == "positive"
    assert by_text["c"]["sentiment"] is None
    assert by_text["d"]["sentiment"] is None
    assert "analysis limit" in result["note"]
    # Only the values that fit the limit were ever sent to the model.
    assert '"c"' not in model.prompts[0]


def test_aggregate_counts_unscored_rows_separately():
    df = pd.DataFrame({"txt": ["great", "awful"]})
    model = FakeModel({"0": {"sentiment": "positive", "score": 0.9}})

    result = _tool(df, model).forward("txt", aggregate=True)

    counts = {r["sentiment"]: r["count"] for r in result["data"]}
    assert counts["positive"] == 1
    assert counts["(not analyzed)"] == 1
    assert "neutral" not in counts


def test_empty_and_nan_rows_are_skipped():
    df = pd.DataFrame({"txt": ["great", None, "   ", "awful"]})
    model = FakeModel(
        {
            "0": {"sentiment": "positive", "score": 0.9},
            "1": {"sentiment": "negative", "score": 0.9},
        }
    )

    result = _tool(df, model).forward("txt", aggregate=False)

    assert [r["txt"] for r in result["data"]] == ["great", "awful"]


def test_duplicate_texts_all_receive_the_label():
    df = pd.DataFrame({"txt": ["great", "great", "awful"]})
    model = FakeModel(
        {
            "0": {"sentiment": "positive", "score": 0.9},
            "1": {"sentiment": "negative", "score": 0.8},
        }
    )

    result = _tool(df, model).forward("txt", aggregate=False)

    assert [r["sentiment"] for r in result["data"]] == [
        "positive",
        "positive",
        "negative",
    ]


def test_every_row_is_returned():
    """The transport layer previews and exports; the tool must not cut rows at all."""
    df = pd.DataFrame({"txt": [f"t{i}" for i in range(120)]})
    model = FakeModel(
        {str(i): {"sentiment": "positive", "score": 0.5} for i in range(120)}
    )

    result = _tool(df, model).forward("txt", aggregate=False)

    assert len(result["data"]) == 120


def test_the_tool_exposes_no_row_cap_parameter():
    """A `max_rows` input let the model silently shrink the CSV export. It is gone."""
    assert "max_rows" not in SentimentAnalysisTool.inputs

    import inspect

    params = inspect.signature(SentimentAnalysisTool.forward).parameters
    assert "max_rows" not in params


def test_result_carries_an_export_name():
    df = pd.DataFrame({"reviews": ["great"]})
    model = FakeModel({"0": {"sentiment": "positive", "score": 0.9}})

    result = _tool(df, model).forward("reviews", aggregate=False)

    assert result["export_name"] == "sentiment_reviews"


def test_markdown_fenced_response_is_parsed():
    df = pd.DataFrame({"txt": ["great"]})
    raw = '```json\n{"0": {"sentiment": "positive", "score": 0.9}}\n```'
    model = FakeModel(None, raw=raw)

    result = _tool(df, model).forward("txt", aggregate=False)

    assert result["data"][0]["sentiment"] == "positive"


def test_unparseable_response_is_an_error_not_a_table_of_neutrals():
    df = pd.DataFrame({"txt": ["great"]})
    model = FakeModel(None, raw="I cannot help with that.")

    result = _tool(df, model).forward("txt", aggregate=False)

    assert result["kind"] == "error"
    assert result["code"] == "PARSE_FAILED"


def test_missing_column_is_an_error():
    df = pd.DataFrame({"txt": ["great"]})
    result = _tool(df, FakeModel({})).forward("nope")

    assert result["kind"] == "error"
    assert result["code"] == "INVALID_COLUMN"
