"""Bounded, honest /datachat table responses.

The HTTP value is a 50x10 preview; the response says so through result/preview
counts and ``truncated``. Source-level claims (``more_rows_available``) and
backend caveats (``note``) reach the client only for tables a tool produced in
the same request, never from what the agent wrote into the payload.
"""

import copy
import json

import pandas as pd
import pytest
from flask import Flask

from datachat.output_normalizer import normalize_datachat_response
from datachat.result_provenance import (
    TrustedResult,
    lookup_trusted_result,
    record_trusted_result,
)
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.sql_engine_tool import SqlEngineTool
from tests.fake_sql_datasource import FakeDatasource
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    _make_app,
    _patch_success_dependencies,
    _post_chat,
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)


def _table(rows, cols):
    return [{f"c{j}": i * cols + j for j in range(cols)} for i in range(rows)]


def _meta(body):
    return {k: body[k] for k in ("result_rows", "result_columns", "preview_rows", "preview_columns", "truncated")}


# --- presentation metadata --------------------------------------------------


@pytest.mark.parametrize(
    "rows, cols, expected",
    [
        (12, 5, (12, 5, 12, 5, False)),
        (237, 4, (237, 4, 50, 4, True)),
        (3, 14, (3, 14, 3, 10, True)),
        (237, 14, (237, 14, 50, 10, True)),
        (50, 10, (50, 10, 50, 10, False)),
        (0, 0, (0, 0, 0, 0, False)),
    ],
)
def test_table_preview_metadata(rows, cols, expected):
    out = normalize_datachat_response({"kind": "table", "data": _table(rows, cols)})

    assert out["type"] == "dataframe"
    assert len(out["value"]) == expected[2]
    assert all(len(row) <= 10 for row in out["value"])
    assert tuple(_meta(out).values()) == expected
    assert "more_rows_available" not in out
    assert "note" not in out


def test_value_is_unchanged_for_a_table_within_the_preview():
    data = [{"a": 1, "b": None}, {"a": float("nan"), "b": "x"}]
    out = normalize_datachat_response({"kind": "table", "data": data})
    assert out["value"] == [{"a": 1, "b": None}, {"a": None, "b": "x"}]


def test_columns_are_counted_over_every_row_not_the_first_preview():
    data = [{"a": 1}, {"b": 2}, {"c": 3}]
    out = normalize_datachat_response({"kind": "table", "data": data})
    assert out["result_columns"] == 3
    assert out["preview_columns"] == 3
    assert out["truncated"] is False


def test_non_dict_rows_are_neither_shown_nor_counted():
    data = ["junk"] * 60 + _table(3, 2)
    out = normalize_datachat_response({"kind": "table", "data": data})
    assert out["result_rows"] == 3
    assert out["preview_rows"] == 3
    assert out["truncated"] is False


def test_dataframe_payload_is_previewed_with_its_full_shape():
    df = pd.DataFrame(_table(60, 12))
    out = normalize_datachat_response({"kind": "table", "data": df})
    assert _meta(out) == {
        "result_rows": 60,
        "result_columns": 12,
        "preview_rows": 50,
        "preview_columns": 10,
        "truncated": True,
    }


def test_empty_dataframe_keeps_its_column_count():
    out = normalize_datachat_response({"kind": "table", "data": pd.DataFrame(columns=["a", "b"])})
    assert out["value"] == []
    assert _meta(out) == {
        "result_rows": 0,
        "result_columns": 2,
        "preview_rows": 0,
        "preview_columns": 0,
        "truncated": False,
    }


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"kind": "text", "text": "hi"}, {"type": "str", "value": "hi"}),
        ({"kind": "error", "message": "no"}, {"type": "str", "value": "no"}),
        ({"kind": "table", "data": {"x": 1}}, {"x": 1, "type": "dict"}),
    ],
)
def test_non_table_responses_are_unaffected(payload, expected):
    trusted = TrustedResult(more_rows_available=True, note="n")
    assert normalize_datachat_response(payload, trusted=trusted) == expected


def test_trusted_facts_are_added_only_when_given():
    data = _table(2, 2)
    out = normalize_datachat_response(
        {"kind": "table", "data": data, "meta": {"truncated": True}, "note": "llm"},
        trusted=TrustedResult(more_rows_available=True, note="backend"),
    )
    assert out["more_rows_available"] is True
    assert out["note"] == "backend"

    untrusted = normalize_datachat_response(
        {"kind": "table", "data": data, "meta": {"truncated": True}, "note": "llm"}
    )
    assert "more_rows_available" not in untrusted
    assert "note" not in untrusted


# --- registry ---------------------------------------------------------------


def test_registry_trusts_only_the_recorded_list_with_its_recorded_shape():
    app = Flask(__name__)
    with app.app_context():
        payload = record_trusted_result({"kind": "table", "data": _table(3, 2)}, more_rows_available=True)
        facts = TrustedResult(more_rows_available=True)

        assert lookup_trusted_result(payload) == facts
        # Rewrapped envelope around the same, untouched rows.
        assert lookup_trusted_result({"kind": "table", "data": payload["data"]}) == facts
        # Copies and reshaped tables are unknown.
        assert lookup_trusted_result(copy.deepcopy(payload)) is None
        assert lookup_trusted_result({"kind": "table", "data": payload["data"][:2]}) is None

        payload["data"].append({"c0": 0, "c1": 0})
        assert lookup_trusted_result(payload) is None


def test_registry_ignores_column_projection_in_place():
    app = Flask(__name__)
    with app.app_context():
        payload = record_trusted_result({"kind": "table", "data": _table(3, 2)}, note="n")
        payload["data"][0].pop("c1")
        assert lookup_trusted_result(payload) is None


def test_registry_is_inert_outside_an_app_context():
    payload = record_trusted_result({"kind": "table", "data": _table(1, 1)}, more_rows_available=True)
    assert lookup_trusted_result(payload) is None


def test_registry_does_not_leak_across_requests():
    app = Flask(__name__)
    data = _table(2, 2)
    with app.test_request_context("/datachat"):
        record_trusted_result({"kind": "table", "data": data}, more_rows_available=True)
        assert lookup_trusted_result({"kind": "table", "data": data}) is not None
    with app.test_request_context("/datachat"):
        assert lookup_trusted_result({"kind": "table", "data": data}) is None


# --- through POST /datachat -------------------------------------------------


class _ToolEngine:
    """Runs a real tool inside the request and returns what ``finish`` makes of it."""

    def __init__(self, run_tool, finish=lambda result: result):
        self._run_tool = run_tool
        self._finish = finish

    def chat(self, message):
        return self._finish(self._run_tool())

    def get_last_trace(self):
        return None


def _sql_tool(rows, truncated, cols=18):
    columns = [f"c{j}" for j in range(cols)]
    datasource = FakeDatasource(
        result_columns=columns,
        rows=[tuple(range(cols))] * rows,
        truncated=truncated,
    )
    return lambda: SqlEngineTool(datasource).forward("SELECT * FROM t")


def _chat(monkeypatch, engine):
    app, _stream, _agent_runs = _make_app()
    _patch_success_dependencies(monkeypatch, engine)
    response = _post_chat(app.test_client())
    assert response.status_code == 200
    return response.get_json()["response"]


def test_sql_result_capped_upstream_reports_more_rows_available(monkeypatch):
    body = _chat(monkeypatch, _ToolEngine(_sql_tool(200, truncated=True)))

    assert body["type"] == "dataframe"
    assert _meta(body) == {
        "result_rows": 200,
        "result_columns": 18,
        "preview_rows": 50,
        "preview_columns": 10,
        "truncated": True,
    }
    assert body["more_rows_available"] is True
    assert "meta" not in body


def test_sql_result_not_capped_omits_more_rows_available(monkeypatch):
    body = _chat(monkeypatch, _ToolEngine(_sql_tool(12, truncated=False, cols=5)))

    assert _meta(body) == {
        "result_rows": 12,
        "result_columns": 5,
        "preview_rows": 12,
        "preview_columns": 5,
        "truncated": False,
    }
    assert "more_rows_available" not in body


def test_sql_rows_rewrapped_by_the_agent_keep_their_provenance(monkeypatch):
    engine = _ToolEngine(
        _sql_tool(3, truncated=True),
        finish=lambda r: {"kind": "table", "data": r["data"]},
    )
    assert _chat(monkeypatch, engine)["more_rows_available"] is True


def test_rebuilt_sql_table_loses_source_honesty(monkeypatch):
    engine = _ToolEngine(_sql_tool(3, truncated=True), finish=copy.deepcopy)
    body = _chat(monkeypatch, engine)

    assert body["result_rows"] == 3
    assert "more_rows_available" not in body


def test_agent_authored_meta_is_not_trusted(monkeypatch):
    engine = _ToolEngine(
        lambda: None,
        finish=lambda _: {"kind": "table", "data": _table(3, 2), "meta": {"truncated": True}},
    )
    assert "more_rows_available" not in _chat(monkeypatch, engine)


def test_agent_output_parsed_from_json_text_is_not_trusted(monkeypatch):
    engine = _ToolEngine(
        _sql_tool(3, truncated=True),
        finish=lambda r: json.dumps(r),
    )
    body = _chat(monkeypatch, engine)
    assert body["result_rows"] == 3
    assert "more_rows_available" not in body


def _aggregate_with_note():
    df = pd.DataFrame({"region": ["a"] * 3 + ["b"] * 20, "amount": range(23)})
    return lambda: AggregateTool(df).forward(group_by="region", op="mean", metric="amount")


def test_trusted_aggregate_note_reaches_the_client(monkeypatch):
    tool = _aggregate_with_note()
    body = _chat(monkeypatch, _ToolEngine(tool))

    assert body["note"] == "1 group(s) are based on fewer than 15 rows (the smallest has 3)."
    assert body["result_rows"] == 2
    assert body["truncated"] is False


def test_note_on_a_rebuilt_table_is_not_forwarded(monkeypatch):
    engine = _ToolEngine(_aggregate_with_note(), finish=copy.deepcopy)
    body = _chat(monkeypatch, engine)

    assert "note" not in body
    assert body["result_rows"] == 2


def test_text_response_envelope_is_unchanged(monkeypatch):
    engine = _ToolEngine(lambda: None, finish=lambda _: {"kind": "text", "text": "hello"})
    assert _chat(monkeypatch, engine) == {"type": "str", "value": "hello"}


def test_provenance_from_one_request_does_not_reach_the_next(monkeypatch):
    held = {}

    def first(result):
        held["data"] = result["data"]
        return result

    assert _chat(monkeypatch, _ToolEngine(_sql_tool(3, truncated=True), finish=first))["more_rows_available"]

    replay = _ToolEngine(lambda: None, finish=lambda _: {"kind": "table", "data": held["data"]})
    assert "more_rows_available" not in _chat(monkeypatch, replay)
