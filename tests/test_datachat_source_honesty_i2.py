"""
I2: ``more_rows_available`` is a source-level fact, so a table a tool derives from
a trusted partial result inherits it (transitively). Prose notes describe their
own table and are never inherited. Semantic coverage notes are visible to the
agent on the direct tool result.
"""

import copy

import pandas as pd
import pytest
from flask import Flask

from datachat.chart_registry import get_recorded_charts
from datachat.result_provenance import (
    TrustedResult,
    inherits_more_rows_available,
    lookup_trusted_result,
    record_trusted_result,
)
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.chart_tool import ChartTool
from datachat.tools.classify_match_tool import ClassifyMatchTool
from datachat.tools.compare_groups_tool import CompareGroupsTool
from datachat.tools.correlation_tool import CorrelationTool
from datachat.tools.crosstab_tool import CrosstabTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.keywords_tool import KeywordsTool
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.row_count_tool import RowCountTool
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.sentiment_tool import SentimentAnalysisTool
from datachat.tools.sql_engine_tool import SqlEngineTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.trend_tool import TrendTool
from datachat.tools.unique_values_tool import UniqueValuesTool
from tests.fake_sql_datasource import FakeDatasource
from tests.test_datachat_chart_tool import _chat, _ToolEngine
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)
from tests.test_datachat_semantic_keep_columns import FakeModel

_app = Flask(__name__)
_SESSION = pd.DataFrame({"unused": [1]})

_THIN_NOTE = "1 group(s) are based on fewer than 15 rows (the smallest has 3)."


def _rows():
    return [
        {"region": "North" if i % 2 else "South", "service": "A" if i % 3 else "B", "amount": i}
        for i in range(40)
    ]


def _partial_source():
    """What the SQL tool records for a result cut at its row cap."""
    return record_trusted_result({"kind": "table", "data": _rows()}, more_rows_available=True)


def _facts(payload):
    facts = lookup_trusted_result(payload)
    return facts if facts is not None else TrustedResult()


# --- A. source honesty ----------------------------------------------------------


def test_filter_rows_of_a_partial_source_stays_partial():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        out = FilterRowsTool(_SESSION).forward(where_col="region", value="North", data=src["data"])
        assert out["kind"] == "table" and 0 < len(out["data"]) < len(src["data"])
        assert lookup_trusted_result(out) == TrustedResult(more_rows_available=True)


def test_aggregate_of_a_partial_source_stays_partial():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        out = AggregateTool(_SESSION).forward(group_by="region", op="mean", metric="amount", data=src["data"])
        assert out["kind"] == "table"
        assert _facts(out).more_rows_available is True
        assert "note" not in out


def test_crosstab_of_a_partial_source_stays_partial():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        out = CrosstabTool(_SESSION).forward(rows="region", columns="service", data=src["data"])
        assert out["kind"] == "table"
        assert lookup_trusted_result(out) == TrustedResult(more_rows_available=True)


def test_accepts_the_wrapped_payload_as_data():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        out = FilterRowsTool(_SESSION).forward(where_col="region", value="North", data=src)
        assert _facts(out).more_rows_available is True


def test_multi_hop_filter_then_aggregate_stays_partial():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        filtered = FilterRowsTool(_SESSION).forward(where_col="service", value="A", data=src["data"])
        agg = AggregateTool(_SESSION).forward(group_by="region", op="count", data=filtered["data"])
        tab = CrosstabTool(_SESSION).forward(rows="region", columns="service", data=filtered["data"])
        assert _facts(agg).more_rows_available is True
        assert _facts(tab).more_rows_available is True


def test_untrusted_input_asserts_nothing():
    with _app.test_request_context("/datachat"):
        rows = _rows()
        assert lookup_trusted_result(FilterRowsTool(_SESSION).forward(where_col="region", value="North", data=rows)) is None
        assert lookup_trusted_result(CrosstabTool(_SESSION).forward(rows="region", columns="service", data=rows)) is None
        # Session DataFrame input: nothing to inherit either.
        df = pd.DataFrame(rows)
        assert lookup_trusted_result(FilterRowsTool(df).forward(where_col="region", value="North")) is None


def test_trusted_input_without_the_flag_does_not_invent_one():
    with _app.test_request_context("/datachat"):
        agg = AggregateTool(pd.DataFrame(_rows()[:23])).forward(group_by="service", op="mean", metric="amount")
        assert lookup_trusted_result(agg).more_rows_available is False
        out = FilterRowsTool(_SESSION).forward(where_col="service", value="A", data=agg["data"])
        assert lookup_trusted_result(out) is None


def test_copied_partial_rows_do_not_recover_trust():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        copied = copy.deepcopy(src["data"])
        assert inherits_more_rows_available(copied) is False
        out = FilterRowsTool(_SESSION).forward(where_col="region", value="North", data=copied)
        assert lookup_trusted_result(out) is None
        # A list rebuilt by the agent from a derived result is equally unknown.
        derived = FilterRowsTool(_SESSION).forward(where_col="region", value="North", data=src["data"])
        rebuilt = list(derived["data"])
        assert lookup_trusted_result({"kind": "table", "data": rebuilt}) is None


def test_chart_of_a_partial_source_is_recorded_as_partial():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        agg = AggregateTool(_SESSION).forward(group_by="region", op="count", data=src["data"])
        ChartTool(_SESSION).forward(kind="bar", x="region", y="count", data=agg["data"])
        ChartTool(_SESSION).forward(kind="bar", x="region", y="count", data=copy.deepcopy(agg["data"]))
        charts = get_recorded_charts()
        assert [c.more_rows_available for c in charts] == [True, False]
        assert all("more_rows_available" not in c.spec for c in charts)


def test_chart_of_a_partial_source_flags_the_response(monkeypatch):
    def tools():
        src = _partial_source()
        agg = AggregateTool(_SESSION).forward(group_by="region", op="count", data=src["data"])
        ChartTool(_SESSION).forward(kind="bar", x="region", y="count", data=agg["data"])

    body = _chat(monkeypatch, _ToolEngine(tools, lambda _: {"kind": "text", "text": "see chart"}))
    assert body["more_rows_available"] is True
    assert "note" not in body
    assert "more_rows_available" not in body["charts"][0]


def test_chart_of_a_complete_table_does_not_flag_the_response(monkeypatch):
    def tools():
        ChartTool(_SESSION).forward(kind="bar", x="region", y="count", data=[{"region": "a", "count": 1}])

    body = _chat(monkeypatch, _ToolEngine(tools, lambda _: {"kind": "text", "text": "x", "more_rows_available": True}))
    assert "more_rows_available" not in body


def test_derived_primary_table_reaches_the_client_as_partial(monkeypatch):
    def tools():
        src = _partial_source()
        filtered = FilterRowsTool(_SESSION).forward(where_col="service", value="A", data=src["data"])
        return AggregateTool(_SESSION).forward(group_by="region", op="count", data=filtered["data"])

    body = _chat(monkeypatch, _ToolEngine(tools, lambda agg: agg))
    assert body["type"] == "dataframe"
    assert body["more_rows_available"] is True


# --- B. prose notes -------------------------------------------------------------


def test_aggregate_note_is_not_inherited_by_a_filter():
    df = pd.DataFrame({"region": ["a"] * 3 + ["b"] * 20, "amount": range(23)})
    with _app.test_request_context("/datachat"):
        agg = AggregateTool(df).forward(group_by="region", op="mean", metric="amount")
        assert agg["note"] == _THIN_NOTE
        out = FilterRowsTool(_SESSION).forward(where_col="region", value="b", data=agg["data"])
        assert "note" not in out
        assert lookup_trusted_result(out) is None


def test_downstream_note_is_its_own_and_not_merged():
    rows = [{"region": "a", "amount": i} for i in range(3)] + [{"region": "b", "amount": i} for i in range(20)]
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": rows}, more_rows_available=True, note="upstream")
        agg = AggregateTool(_SESSION).forward(group_by="region", op="mean", metric="amount", data=src["data"])
        assert agg["note"] == _THIN_NOTE
        assert lookup_trusted_result(agg) == TrustedResult(more_rows_available=True, note=_THIN_NOTE)


def test_semantic_note_is_not_inherited_but_the_flag_is():
    rows = [{"comment": t, "region": r} for t, r in [("Great service", "N"), ("Unknown text", "S"), ("Awful wait", "N")]]
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": rows}, more_rows_available=True)
        tool = SentimentAnalysisTool(_SESSION, model=FakeModel("sentiment"), provider="Deepinfra", model_name="m-1")
        sent = tool.forward(col="comment", data=src["data"], keep_columns=["region"])
        assert sent["note"].startswith("Partial sentiment coverage")
        assert lookup_trusted_result(sent) == TrustedResult(more_rows_available=True, note=sent["note"])

        out = FilterRowsTool(_SESSION).forward(where_col="region", value="N", data=sent["data"])
        assert "note" not in out
        assert lookup_trusted_result(out) == TrustedResult(more_rows_available=True)


# --- C. semantic direct visibility ------------------------------------------------


def _sentiment(df):
    with _app.test_request_context("/datachat"):
        tool = SentimentAnalysisTool(df, model=FakeModel("sentiment"), provider="Deepinfra", model_name="m-1")
        out = tool.forward(col="comment")
        return out, lookup_trusted_result(out)


def _classify(df):
    with _app.test_request_context("/datachat"):
        tool = ClassifyMatchTool(df, model=FakeModel("classify"), provider="Deepinfra", model_name="m-1")
        out = tool.forward(column="comment", categories=["Staff", "Waiting"])
        return out, lookup_trusted_result(out)


def test_sentiment_partial_coverage_note_is_on_the_direct_result():
    out, facts = _sentiment(pd.DataFrame({"comment": ["Great service", "Unknown text"]}))
    assert out["note"] == facts.note
    assert out["note"].startswith("Partial sentiment coverage: 1 of 2 non-empty rows")
    assert facts.more_rows_available is False


def test_classify_unavailable_coverage_note_is_on_the_direct_result():
    out, facts = _classify(pd.DataFrame({"comment": pd.Series(["Great service", None], dtype=object)}))
    assert out["note"] == facts.note
    assert facts.note


def test_complete_semantic_coverage_has_no_note():
    df = pd.DataFrame({"comment": ["Great service", "Awful wait"]})
    for out, facts in (_sentiment(df), _classify(df)):
        assert "note" not in out
        assert facts is None


def test_semantic_note_is_forwarded_to_the_client_once(monkeypatch):
    df = pd.DataFrame({"comment": ["Great service", "Unknown text"]})

    def tools():
        return SentimentAnalysisTool(df, model=FakeModel("sentiment"), provider="Deepinfra", model_name="m-1").forward(
            col="comment"
        )

    body = _chat(monkeypatch, _ToolEngine(tools, lambda out: out))
    assert body["note"].startswith("Partial sentiment coverage: 1 of 2 non-empty rows")
    assert "more_rows_available" not in body


def test_reconstructed_payload_with_a_note_is_not_trusted(monkeypatch):
    df = pd.DataFrame({"comment": ["Great service", "Unknown text"]})

    def tools():
        return SentimentAnalysisTool(df, model=FakeModel("sentiment"), provider="Deepinfra", model_name="m-1").forward(
            col="comment"
        )

    body = _chat(
        monkeypatch,
        _ToolEngine(tools, lambda out: {"kind": "table", "data": copy.deepcopy(out["data"]), "note": out["note"]}),
    )
    assert "note" not in body


# --- A'. every reachable data= transformation --------------------------------------


def _mixed_rows(n=40):
    words = ["late delivery", "friendly staff", "late refund", "friendly support"]
    return [
        {
            "g": "a" if i % 2 else "b",
            "v": i,
            "w": (i * 7) % 11,
            "d": f"2024-{i % 12 + 1:02d}-01",
            "t": words[i % 4],
        }
        for i in range(n)
    ]


_DERIVED = {
    "row_count": lambda data: RowCountTool(_SESSION).forward(data=data),
    "compare_groups": lambda data: CompareGroupsTool(_SESSION).forward(
        metric="v", group_col="g", group_a="a", group_b="b", data=data
    ),
    "trend": lambda data: TrendTool(_SESSION).forward(date_col="d", freq="month", op="count", data=data),
    "describe": lambda data: DescribeTool(_SESSION).forward(data=data),
    "unique_values": lambda data: UniqueValuesTool(_SESSION).forward(column="g", data=data),
    "missing_values": lambda data: MissingValuesTool(_SESSION).forward(data=data),
    "correlation_pair": lambda data: CorrelationTool(_SESSION).forward(col_x="v", col_y="w", data=data),
    "correlation_ranking": lambda data: CorrelationTool(_SESSION).forward(col_x="v", data=data),
    "keywords": lambda data: KeywordsTool(_SESSION).forward(column="t", language="english", min_answers=1, data=data),
    "top_rows": lambda data: TopRowsTool(_SESSION).forward(sort_by="v", n=3, data=data),
    "sample_rows": lambda data: SampleRowsTool(_SESSION).forward(n=3, data=data),
}


@pytest.mark.parametrize("tool", sorted(_DERIVED))
def test_every_derived_result_of_a_partial_source_stays_partial(tool):
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": _mixed_rows()}, more_rows_available=True)
        out = _DERIVED[tool](src["data"])
        assert out["kind"] == "table" and out["data"], out
        assert _facts(out).more_rows_available is True


@pytest.mark.parametrize("tool", sorted(_DERIVED))
def test_every_derived_result_of_unknown_or_copied_input_asserts_nothing(tool):
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": _mixed_rows()}, more_rows_available=True)
        for data in (_mixed_rows(), copy.deepcopy(src["data"])):
            assert _facts(_DERIVED[tool](data)).more_rows_available is False


def test_row_count_of_a_partial_source_keeps_the_materialized_count():
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": _mixed_rows(60)}, more_rows_available=True)
        out = RowCountTool(_SESSION).forward(data=src["data"])
        assert out["data"] == [{"row_count": 60}]
        assert lookup_trusted_result(out) == TrustedResult(more_rows_available=True)


def test_own_notes_coexist_with_the_inherited_flag():
    rows = _mixed_rows(10) + [{"g": "a", "v": None, "w": 1, "d": "not a date", "t": 42}]
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": rows}, more_rows_available=True, note="upstream")
        for out in (_DERIVED["compare_groups"](src["data"]), _DERIVED["trend"](src["data"]), _DERIVED["keywords"](src["data"])):
            assert out["note"] and out["note"] != "upstream", out
            assert lookup_trusted_result(out) == TrustedResult(more_rows_available=True, note=out["note"])


# --- C'. semantic tools carry both facts ------------------------------------------


def test_classify_inherits_the_flag_and_keeps_its_own_note():
    rows = [{"comment": "Great service"}, {"comment": None}, {"comment": "Awful wait"}]
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": rows}, more_rows_available=True)
        tool = ClassifyMatchTool(_SESSION, model=FakeModel("classify"), provider="Deepinfra", model_name="m-1")
        out = tool.forward(column="comment", categories=["Staff", "Waiting"], data=src["data"])
        assert out["note"]
        assert lookup_trusted_result(out) == TrustedResult(more_rows_available=True, note=out["note"])


def test_semantic_result_of_a_partial_source_reaches_the_client_with_both_facts(monkeypatch):
    rows = [{"comment": "Great service"}, {"comment": "Unknown text"}]

    def tools():
        src = record_trusted_result({"kind": "table", "data": rows}, more_rows_available=True)
        tool = SentimentAnalysisTool(_SESSION, model=FakeModel("sentiment"), provider="Deepinfra", model_name="m-1")
        return tool.forward(col="comment", data=src["data"])

    body = _chat(monkeypatch, _ToolEngine(tools, lambda out: out))
    assert body["more_rows_available"] is True
    assert body["note"].startswith("Partial sentiment coverage: 1 of 2 non-empty rows")


# --- D. the inherited flag is visible to the agent ----------------------------------
#
# The payload key is advisory: it lets the agent carry the caveat into a text
# answer. Trust still comes only from the registry.


def _sql(rows, truncated):
    datasource = FakeDatasource(
        result_columns=list(rows[0]), rows=[tuple(r.values()) for r in rows], truncated=truncated, max_rows=len(rows)
    )
    return SqlEngineTool(datasource).forward("SELECT * FROM t")


def test_truncated_sql_payload_shows_the_flag_and_a_complete_one_does_not():
    with _app.test_request_context("/datachat"):
        partial = _sql(_rows(), truncated=True)
        complete = _sql(_rows(), truncated=False)
        assert partial["more_rows_available"] is True and partial["meta"]["truncated"] is True
        assert "more_rows_available" not in complete


def test_sql_filter_aggregate_chain_shows_the_flag_on_every_hop():
    with _app.test_request_context("/datachat"):
        src = _sql(_rows(), truncated=True)
        filtered = FilterRowsTool(_SESSION).forward(where_col="service", value="A", data=src["data"])
        agg = AggregateTool(_SESSION).forward(group_by="region", op="count", data=filtered["data"])
        assert filtered["more_rows_available"] is True
        assert agg["more_rows_available"] is True


def test_a_complete_query_after_a_truncated_one_is_not_flagged():
    with _app.test_request_context("/datachat"):
        _sql(_rows(), truncated=True)
        complete = _sql(_rows(), truncated=False)
        agg = AggregateTool(_SESSION).forward(group_by="region", op="count", data=complete["data"])
        assert "more_rows_available" not in complete
        assert "more_rows_available" not in agg


@pytest.mark.parametrize("tool", sorted(_DERIVED))
def test_every_derived_result_of_a_partial_source_shows_the_flag(tool):
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": _mixed_rows()}, more_rows_available=True)
        assert _DERIVED[tool](src["data"])["more_rows_available"] is True


@pytest.mark.parametrize("tool", sorted(_DERIVED))
def test_no_flag_is_shown_for_unknown_copied_or_dataframe_input(tool):
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": _mixed_rows()}, more_rows_available=True)
        for data in (_mixed_rows(), copy.deepcopy(src["data"])):
            assert "more_rows_available" not in _DERIVED[tool](data)


def test_dataframe_only_results_never_show_the_flag():
    df = pd.DataFrame(_rows())
    with _app.test_request_context("/datachat"):
        agg = AggregateTool(df).forward(group_by="region", op="mean", metric="amount")
        filtered = FilterRowsTool(df).forward(where_col="region", value="North")
        assert "more_rows_available" not in agg
        assert "more_rows_available" not in filtered


def test_semantic_results_show_the_flag_and_their_own_note_separately():
    rows = [{"comment": "Great service"}, {"comment": "Unknown text"}]
    with _app.test_request_context("/datachat"):
        src = record_trusted_result({"kind": "table", "data": rows}, more_rows_available=True)
        sent = SentimentAnalysisTool(_SESSION, model=FakeModel("sentiment"), provider="Deepinfra", model_name="m-1").forward(
            col="comment", data=src["data"]
        )
        cls = ClassifyMatchTool(_SESSION, model=FakeModel("classify"), provider="Deepinfra", model_name="m-1").forward(
            column="comment", categories=["Staff", "Waiting"], data=src["data"]
        )
        for out in (sent, cls):
            assert out["more_rows_available"] is True
            assert out["note"] and "more_rows_available" not in out["note"]


def test_a_forged_flag_is_not_trusted_and_not_inherited():
    with _app.test_request_context("/datachat"):
        forged = {"kind": "table", "data": _rows(), "more_rows_available": True}
        assert lookup_trusted_result(forged) is None
        assert inherits_more_rows_available(forged["data"]) is False
        out = AggregateTool(_SESSION).forward(group_by="region", op="count", data=forged["data"])
        assert "more_rows_available" not in out
        assert lookup_trusted_result(out) is None


def test_a_forged_flag_does_not_reach_the_client(monkeypatch):
    forged = {"kind": "table", "data": [{"region": "a", "count": 1}], "more_rows_available": True}
    body = _chat(monkeypatch, _ToolEngine(lambda: None, lambda _: forged))
    assert body["type"] == "dataframe"
    assert "more_rows_available" not in body


def test_client_table_shape_is_unchanged_by_the_agent_visible_flag(monkeypatch):
    def tools():
        src = _partial_source()
        return AggregateTool(_SESSION).forward(group_by="region", op="count", data=src["data"])

    body = _chat(monkeypatch, _ToolEngine(tools, lambda agg: agg))
    assert set(body) == {
        "type", "value", "result_rows", "result_columns", "preview_rows",
        "preview_columns", "truncated", "more_rows_available",
    }
    assert all("more_rows_available" not in row for row in body["value"])


def test_text_answer_after_a_partial_chain_is_unchanged_for_the_client(monkeypatch):
    def tools():
        src = _partial_source()
        return AggregateTool(_SESSION).forward(group_by="region", op="count", data=src["data"])

    body = _chat(monkeypatch, _ToolEngine(tools, lambda _: {"kind": "text", "text": "partial"}))
    assert body == {"type": "str", "value": "partial"}


def test_empty_prefiltered_aggregate_of_a_partial_source_stays_partial():
    with _app.test_request_context("/datachat"):
        src = _partial_source()
        out = AggregateTool(_SESSION).forward(group_by="region", op="count", where_col="region", value="Nowhere", data=src["data"])
        assert out == {"kind": "table", "data": [], "more_rows_available": True}
        assert _facts(out).more_rows_available is True


def test_empty_prefiltered_aggregate_of_complete_data_is_unchanged():
    with _app.test_request_context("/datachat"):
        out = AggregateTool(pd.DataFrame(_rows())).forward(group_by="region", op="count", where_col="region", value="Nowhere")
        assert out == {"kind": "table", "data": []}
        assert lookup_trusted_result(out) is None
