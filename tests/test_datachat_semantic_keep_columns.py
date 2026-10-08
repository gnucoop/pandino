"""
I1: row-level semantic results (sentiment_analysis, classify_match) carry explicitly
requested source columns, so they chain directly into aggregate / crosstab by segment.
"""

import json

import numpy as np
import pandas as pd
import pytest
from flask import Flask
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import TokenUsage

from datachat.provider_contributions import get_provider_contributions
from datachat.result_provenance import lookup_trusted_result
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.classify_match_tool import ClassifyMatchTool
from datachat.tools.crosstab_tool import CrosstabTool
from datachat.tools.sentiment_tool import SentimentAnalysisTool

_app = Flask(__name__)

_SENTIMENT = {
    "Great service": {"sentiment": "positive", "score": 0.9},
    "Awful wait": {"sentiment": "negative", "score": 0.8},
    "It was ok": {"sentiment": "neutral", "score": 0.5},
}
_CATEGORY = {"Great service": 0, "Awful wait": 1, "It was ok": 0}
_CATEGORIES = ["Staff", "Waiting"]


class FakeModel:
    """Replies per item text for either tool's prompt; records the sent items."""

    def __init__(self, kind):
        self.kind = kind
        self.calls = []

    def generate(self, messages):
        user_text = messages[1].content[0]["text"]
        items = json.loads(user_text.split("Items:\n", 1)[1])
        self.calls.append(items)
        table = _SENTIMENT if self.kind == "sentiment" else {k: {"category_id": v} for k, v in _CATEGORY.items()}
        reply = {str(i["id"]): table[i["text"]] for i in items if i["text"] in table}
        return ChatMessage(role=MessageRole.ASSISTANT, content=json.dumps(reply), token_usage=TokenUsage(3, 2))


def _source():
    return pd.DataFrame(
        {
            "comment": pd.Series(
                ["Great service", "Awful wait", None, "Great service", "   ", "It was ok", "Awful wait"],
                dtype=object,
            ),
            "service": ["A", "B", "A", "B", "B", "A", "A"],
            "region": ["North", "South", "North", "South", "North", "South", "North"],
        }
    )


def _sentiment(df, **kwargs):
    model = FakeModel("sentiment")
    with _app.test_request_context("/datachat"):
        tool = SentimentAnalysisTool(df, model=model, provider="Deepinfra", model_name="m-1")
        out = tool.forward(**kwargs)
        facts = lookup_trusted_result(out)
        contributions = get_provider_contributions()
    return out, facts, contributions, model


def _classify(df, **kwargs):
    model = FakeModel("classify")
    kwargs.setdefault("categories", _CATEGORIES)
    with _app.test_request_context("/datachat"):
        tool = ClassifyMatchTool(df, model=model, provider="Deepinfra", model_name="m-1")
        out = tool.forward(**kwargs)
        facts = lookup_trusted_result(out)
        contributions = get_provider_contributions()
    return out, facts, contributions, model


# --- passthrough shape ---------------------------------------------------------------


def test_sentiment_keeps_only_requested_columns_in_order():
    out, *_ = _sentiment(_source(), col="comment", keep_columns=["service"])
    assert all(list(r) == ["comment", "service", "sentiment", "score"] for r in out["data"])
    assert all("region" not in r for r in out["data"])


def test_classify_keeps_only_requested_columns_in_order():
    out, *_ = _classify(_source(), column="comment", keep_columns=["region"])
    assert all(list(r) == ["comment", "region", "category", "classification_status"] for r in out["data"])


@pytest.mark.parametrize("keep", [["region", "service"], ["service", "region"]])
def test_multi_column_order_follows_the_caller(keep):
    s, *_ = _sentiment(_source(), col="comment", keep_columns=keep)
    c, *_ = _classify(_source(), column="comment", keep_columns=keep)
    assert list(s["data"][0]) == ["comment", *keep, "sentiment", "score"]
    assert list(c["data"][0]) == ["comment", *keep, "category", "classification_status"]


@pytest.mark.parametrize("keep", [None, []])
def test_omitted_or_empty_keep_columns_is_the_compact_output(keep):
    kwargs = {} if keep is None else {"keep_columns": keep}
    s, *_ = _sentiment(_source(), col="comment", **kwargs)
    c, *_ = _classify(_source(), column="comment", **kwargs)
    assert list(s["data"][0]) == ["comment", "sentiment", "score"]
    assert list(c["data"][0]) == ["comment", "category", "classification_status"]


def test_keep_columns_via_data_records():
    records = _source().to_dict(orient="records")
    out, *_ = _sentiment(pd.DataFrame({"x": [1]}), col="comment", data=records, keep_columns=["region"])
    assert [r["region"] for r in out["data"]] == list(_source()["region"])


# --- sentiment 1:1 row population ----------------------------------------------------


def test_sentiment_row_level_is_one_row_per_source_row_aligned():
    df = _source()
    out, _, _, model = _sentiment(df, col="comment", keep_columns=["service", "region"])
    rows = out["data"]
    assert len(rows) == len(df)
    assert [r["service"] for r in rows] == list(df["service"])
    assert [r["region"] for r in rows] == list(df["region"])
    assert [r["comment"] for r in rows] == list(df["comment"])
    assert [r["sentiment"] for r in rows] == [
        "positive", "negative", None, "positive", None, "neutral", "negative",
    ]
    assert rows[2]["score"] is None and rows[4]["score"] is None
    # Provider work is unchanged: distinct non-blank values, sent once.
    assert [i["text"] for i in model.calls[0]] == ["Great service", "Awful wait", "It was ok"]


def test_nan_text_is_kept_with_no_sentiment():
    df = pd.DataFrame({"comment": ["Great service", np.nan], "service": ["A", "B"]})
    out, *_ = _sentiment(df, col="comment", keep_columns=["service"])
    assert [(r["service"], r["sentiment"], r["score"]) for r in out["data"]] == [
        ("A", "positive", 0.9),
        ("B", None, None),
    ]


def test_duplicate_texts_keep_their_own_segment():
    df = pd.DataFrame({"comment": ["Great service", "Great service"], "region": ["North", "South"]})
    s, _, _, sm = _sentiment(df, col="comment", keep_columns=["region"])
    c, _, _, cm = _classify(df, column="comment", keep_columns=["region"])
    assert [(r["region"], r["sentiment"]) for r in s["data"]] == [("North", "positive"), ("South", "positive")]
    assert [(r["region"], r["category"]) for r in c["data"]] == [("North", "Staff"), ("South", "Staff")]
    assert len(sm.calls[0]) == 1 and len(cm.calls[0]) == 1


def test_sentiment_coverage_note_counts_only_the_non_empty_population():
    df = pd.DataFrame(
        {"comment": pd.Series(["Great service", None, "Unknown text", "", "Unknown text"], dtype=object)}
    )
    out, facts, *_ = _sentiment(df, col="comment")
    assert len(out["data"]) == 5
    assert "2 of 3 non-empty rows have no sentiment" in facts.note
    assert "2 not analyzed because the provider returned no usable sentiment" in facts.note


def test_sentiment_full_coverage_with_blank_rows_has_no_note():
    out, facts, *_ = _sentiment(_source(), col="comment", keep_columns=["service"])
    assert len(out["data"]) == 7 and facts is None


# --- aggregate mode ------------------------------------------------------------------


@pytest.mark.parametrize("keep", [["service"], []])
def test_aggregate_rejects_keep_columns_before_provider_work(keep):
    out, _, contributions, model = _sentiment(_source(), col="comment", aggregate=True, keep_columns=keep)
    assert out["kind"] == "error" and out["code"] == "KEEP_COLUMNS_WITH_AGGREGATE"
    assert model.calls == [] and contributions == []


def test_aggregate_without_keep_columns_is_unchanged():
    out, facts, *_ = _sentiment(_source(), col="comment", aggregate=True)
    assert out == {
        "kind": "table",
        "data": [
            {"sentiment": "positive", "count": 2},
            {"sentiment": "negative", "count": 2},
            {"sentiment": "neutral", "count": 1},
        ],
    }
    assert facts is None


# --- segmented downstream workflows ----------------------------------------------------


def test_classify_then_crosstab_category_by_region():
    out, *_ = _classify(_source(), column="comment", keep_columns=["region"])
    table = CrosstabTool(pd.DataFrame()).forward(rows="category", columns="region", data=out["data"])
    assert table["kind"] == "table"
    by_category = {r["category"]: (r["North"], r["South"]) for r in table["data"]}
    # Staff: Great service (N, S), It was ok (S); Waiting: Awful wait (S, N); no text: None (N), blank (N).
    assert by_category["Staff"] == (1, 2)
    assert by_category["Waiting"] == (1, 1)
    assert by_category["(empty)"] == (2, 0)


def test_sentiment_then_crosstab_sentiment_by_service():
    out, *_ = _sentiment(_source(), col="comment", keep_columns=["service"])
    table = CrosstabTool(pd.DataFrame()).forward(rows="sentiment", columns="service", data=out["data"])
    assert table["kind"] == "table"
    by_sentiment = {r["sentiment"]: (r["A"], r["B"]) for r in table["data"]}
    assert by_sentiment["positive"] == (1, 1)
    assert by_sentiment["negative"] == (1, 1)
    assert by_sentiment["neutral"] == (1, 0)
    assert by_sentiment["(empty)"] == (1, 1)


def test_sentiment_then_aggregate_by_service():
    out, *_ = _sentiment(_source(), col="comment", keep_columns=["service"])
    positive = [r for r in out["data"] if r["sentiment"] == "positive"]
    table = AggregateTool(pd.DataFrame()).forward(group_by="service", op="count", data=positive)
    assert sorted((r["service"], r["count"]) for r in table["data"]) == [("A", 1), ("B", 1)]


# --- collisions ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sentiment", "score"])
def test_sentiment_analyzed_column_colliding_with_result_fields_is_rejected(name):
    df = pd.DataFrame({name: ["Great service"], "service": ["A"]})
    out, _, contributions, model = _sentiment(df, col=name)
    assert out["kind"] == "error" and out["code"] == "INVALID_COLUMN"
    assert model.calls == [] and contributions == []


@pytest.mark.parametrize("name", ["sentiment", "score"])
def test_sentiment_keep_column_colliding_with_result_fields_is_rejected(name):
    df = pd.DataFrame({"comment": ["Great service"], name: ["x"]})
    out, _, contributions, model = _sentiment(df, col="comment", keep_columns=[name])
    assert out["code"] == "INVALID_KEEP_COLUMNS"
    assert model.calls == [] and contributions == []


@pytest.mark.parametrize("name", ["category", "classification_status"])
def test_classify_collisions_are_rejected(name):
    df = pd.DataFrame({"comment": ["Great service"], name: ["x"]})
    out, _, contributions, model = _classify(df, column=name)
    assert out["code"] == "INVALID_COLUMN"
    out, _, contributions, model = _classify(df, column="comment", keep_columns=[name])
    assert out["code"] == "INVALID_KEEP_COLUMNS"
    assert model.calls == [] and contributions == []


# --- keep_columns validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "keep",
    [
        "service",
        ("service",),
        ["service", 1],
        ["service", None],
        [""],
        ["   "],
        ["service", "service"],
        ["comment"],
        ["unknown"],
        ["service", "unknown"],
    ],
)
def test_invalid_keep_columns_are_rejected_before_provider_work(keep):
    for run, kwargs in ((_sentiment, {"col": "comment"}), (_classify, {"column": "comment"})):
        out, facts, contributions, model = run(_source(), keep_columns=keep, **kwargs)
        assert out == {
            "kind": "error",
            "code": "INVALID_KEEP_COLUMNS",
            "message": out["message"],
        }
        assert "keep_columns" in out["message"]
        assert model.calls == [] and contributions == [] and facts is None
