"""Temporary CSV exports of truncated /datachat tables and their download."""

import csv
import io
import os

import pandas as pd
import pytest
from werkzeug.exceptions import Forbidden

import routes.datachat as datachat_route
from datachat.output_normalizer import normalize_datachat_response
from datachat.tools.sql_engine_tool import SqlEngineTool
from infrastructure import datachat_export_store as store
from tests.fake_sql_datasource import FakeDatasource
from tests.test_datachat_result_preview import _ToolEngine, _chat, _sql_tool, _table
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    _make_app,
    _patch_success_dependencies,
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)

KEY = "test-key"
OTHER_KEY = "other-key"
HEADERS = {"X-API-KEY": KEY, "X-USER-EMAIL": "user@example.com"}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def _token(body):
    return body["download_url"].rsplit("/", 1)[-1]


class _TableEngine:
    def __init__(self, data):
        self._data = data

    def chat(self, message):
        return {"kind": "table", "data": self._data}

    def get_last_trace(self):
        return None


# --- store ------------------------------------------------------------------


def test_register_writes_a_csv_under_an_opaque_token(isolated_datachat_exports):
    entry = store.register_export(KEY, ["a", "b"], [{"a": 1, "b": "x"}])

    assert entry.filename == "datachat-result.csv"
    assert len(entry.token) >= 32
    assert KEY not in entry.token
    assert os.path.dirname(entry.path) == str(isolated_datachat_exports)
    assert os.path.basename(entry.path) == f"{entry.token}.csv"
    assert _read_csv(entry.path) == [["a", "b"], ["1", "x"]]


def test_tokens_are_unique():
    tokens = {store.register_export(KEY, ["a"], []).token for _ in range(5)}
    assert len(tokens) == 5


def test_resolve_returns_the_entry_and_unknown_tokens_resolve_to_none():
    entry = store.register_export(KEY, ["a"], [{"a": 1}])
    assert store.resolve_export(entry.token) == entry
    assert store.resolve_export("nope") is None
    assert store.resolve_export("") is None


def test_expired_exports_are_removed_on_resolve(monkeypatch):
    entry = store.register_export(KEY, ["a"], [{"a": 1}])
    now = entry.created_at + store.EXPORT_TTL_S
    monkeypatch.setattr(store.time, "time", lambda: now)

    assert store.resolve_export(entry.token) is None
    assert entry.token not in store._exports
    assert not os.path.exists(entry.path)


def test_expired_exports_are_removed_on_register(monkeypatch):
    old = store.register_export(OTHER_KEY, ["a"], [{"a": 1}])
    monkeypatch.setattr(store.time, "time", lambda: old.created_at + store.EXPORT_TTL_S + 1)

    store.register_export(KEY, ["a"], [{"a": 2}])

    assert old.token not in store._exports
    assert not os.path.exists(old.path)


def test_retention_keeps_the_newest_exports_per_api_key(monkeypatch):
    monkeypatch.setattr(store, "MAX_EXPORTS_PER_API_KEY", 3)
    other = store.register_export(OTHER_KEY, ["a"], [])
    entries = [store.register_export(KEY, ["a"], [{"a": i}]) for i in range(5)]

    kept = [e.token for e in store._exports.values() if e.api_key == KEY]
    assert kept == [e.token for e in entries[2:]]
    assert all(not os.path.exists(e.path) for e in entries[:2])
    assert store.resolve_export(other.token) == other


def test_purge_removes_only_the_owners_exports():
    mine = [store.register_export(KEY, ["a"], []) for _ in range(2)]
    theirs = store.register_export(OTHER_KEY, ["a"], [])

    assert store.purge_exports(KEY) == 2

    assert all(store.resolve_export(e.token) is None for e in mine)
    assert all(not os.path.exists(e.path) for e in mine)
    assert store.resolve_export(theirs.token) == theirs


def test_failed_write_leaves_no_file_or_entry(isolated_datachat_exports, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_csv", boom)
    with pytest.raises(OSError):
        store.register_export(KEY, ["a"], [{"a": 1}])
    assert store._exports == {}
    assert os.listdir(isolated_datachat_exports) == []


def test_csv_cells_are_serialized_stably():
    row = {
        "none": None,
        "nan": float("nan"),
        "nat": pd.NaT,
        "flag": True,
        "int": 3,
        "float": 1.5,
        "text": "a,\"b\"",
        "ts": pd.Timestamp("2024-01-02T03:04:05"),
        "nested": {"k": 1},
    }
    entry = store.register_export(KEY, list(row), [row])

    assert _read_csv(entry.path)[1] == [
        "", "", "", "true", "3", "1.5", "a,\"b\"", "2024-01-02T03:04:05", "{'k': 1}",
    ]


# --- automatic export trigger -----------------------------------------------


def _exporter(calls):
    def export(columns, rows):
        calls.append((columns, list(rows)))
        return {"download_url": "/u", "download_filename": "f.csv"}

    return export


@pytest.mark.parametrize("rows, cols", [(12, 5), (50, 10), (0, 0)])
def test_tables_within_the_preview_are_not_exported(rows, cols):
    calls = []
    out = normalize_datachat_response(
        {"kind": "table", "data": _table(rows, cols)}, exporter=_exporter(calls)
    )
    assert calls == []
    assert "download_url" not in out
    assert "download_filename" not in out


@pytest.mark.parametrize("rows, cols", [(51, 3), (3, 11), (237, 14)])
def test_truncated_tables_are_exported_in_full(rows, cols):
    data = _table(rows, cols)
    calls = []
    out = normalize_datachat_response({"kind": "table", "data": data}, exporter=_exporter(calls))

    assert out["truncated"] is True
    assert out["download_url"] == "/u"
    assert out["download_filename"] == "f.csv"
    columns, exported = calls[0]
    assert columns == [f"c{j}" for j in range(cols)]
    assert exported == data
    assert len(out["value"]) == min(rows, 50)


def test_heterogeneous_records_export_every_column_in_first_seen_order():
    data = [{"a": i} for i in range(60)] + [{"b": 1, "a": 0}, {"c": 2}]
    calls = []
    out = normalize_datachat_response({"kind": "table", "data": data}, exporter=_exporter(calls))

    columns, exported = calls[0]
    assert columns == ["a", "b", "c"]
    assert len(exported) == out["result_rows"] == 62
    assert len(columns) == out["result_columns"]


def test_dataframe_export_keeps_its_column_order_and_non_string_names():
    df = pd.DataFrame({"z": range(60), 1: range(60)})
    calls = []
    normalize_datachat_response({"kind": "table", "data": df}, exporter=_exporter(calls))

    columns, exported = calls[0]
    assert columns == ["z", "1"]
    assert exported[59] == {"z": 59, "1": 59}


def test_dict_tables_are_not_exported():
    calls = []
    out = normalize_datachat_response(
        {"kind": "table", "data": {str(i): i for i in range(100)}}, exporter=_exporter(calls)
    )
    assert out["type"] == "dict"
    assert calls == []


# --- /datachat --------------------------------------------------------------


def test_datachat_attaches_a_download_for_truncated_tables(monkeypatch):
    data = _table(120, 12)
    body = _chat(monkeypatch, _TableEngine(data))

    assert body["download_filename"] == "datachat-result.csv"
    assert body["download_url"].startswith("/datachat/export/")
    rows = _read_csv(store.resolve_export(_token(body)).path)
    assert rows[0] == [f"c{j}" for j in range(12)]
    assert len(rows) == 121
    assert rows[120] == [str(119 * 12 + j) for j in range(12)]
    assert len(body["value"]) == 50
    assert store.resolve_export(_token(body)).api_key == KEY


def test_datachat_small_table_has_no_download(monkeypatch):
    body = _chat(monkeypatch, _TableEngine(_table(5, 3)))
    assert "download_url" not in body
    assert "download_filename" not in body
    assert store._exports == {}


def test_sql_export_contains_the_materialized_rows_only(monkeypatch):
    run_tool = _sql_tool(200, truncated=True)
    body = _chat(monkeypatch, _ToolEngine(run_tool))

    assert body["more_rows_available"] is True
    assert body["preview_rows"] == 50
    rows = _read_csv(store.resolve_export(_token(body)).path)
    assert len(rows) == 201
    assert len(rows[0]) == 18


def test_sql_export_does_not_query_again(monkeypatch):
    datasource = FakeDatasource(
        result_columns=["a"], rows=[(i,) for i in range(80)], truncated=True
    )
    body = _chat(monkeypatch, _ToolEngine(lambda: SqlEngineTool(datasource).forward("SELECT a FROM t")))

    assert len(datasource.run_select_calls) == 1
    assert len(_read_csv(store.resolve_export(_token(body)).path)) == 81


def test_export_failure_keeps_the_preview_response(monkeypatch, caplog):
    def boom(*_a, **_k):
        raise OSError("/secret/path is full")

    monkeypatch.setattr(datachat_route, "register_export", boom)
    body = _chat(monkeypatch, _TableEngine(_table(120, 3)))

    assert body["truncated"] is True
    assert len(body["value"]) == 50
    assert "download_url" not in body
    assert "download_filename" not in body
    assert "/secret/path" not in str(body)
    assert "/secret/path" not in caplog.text


# --- download ---------------------------------------------------------------


def _download(monkeypatch, token, headers=HEADERS, valid=True):
    app, _s, _a = _make_app()

    def fake_assert(api_key, user_email):
        if not valid:
            raise Forbidden(description="Invalid API key")

    monkeypatch.setattr(datachat_route, "assert_valid_api_key", fake_assert)
    return app.test_client().get(f"/datachat/export/{token}", headers=headers)


def test_owner_downloads_the_csv_as_an_attachment(monkeypatch):
    entry = store.register_export(KEY, ["a"], [{"a": 1}])
    response = _download(monkeypatch, entry.token)

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    disposition = response.headers["Content-Disposition"]
    assert disposition.startswith("attachment")
    assert "datachat-result.csv" in disposition
    assert list(csv.reader(io.StringIO(response.get_data(as_text=True)))) == [["a"], ["1"]]
    assert entry.path not in response.get_data(as_text=True)


def test_download_rejects_invalid_credentials(monkeypatch):
    entry = store.register_export(KEY, ["a"], [])
    assert _download(monkeypatch, entry.token, valid=False).status_code == 403


@pytest.mark.parametrize("missing", ["X-API-KEY", "X-USER-EMAIL"])
def test_download_requires_both_headers(monkeypatch, missing):
    entry = store.register_export(KEY, ["a"], [])
    headers = {k: v for k, v in HEADERS.items() if k != missing}
    assert _download(monkeypatch, entry.token, headers=headers).status_code == 400


def test_download_of_another_keys_export_is_forbidden(monkeypatch):
    entry = store.register_export(OTHER_KEY, ["a"], [])
    response = _download(monkeypatch, entry.token)
    assert response.status_code == 403
    assert entry.path not in response.get_data(as_text=True)


def test_download_of_unknown_or_expired_token_is_not_found(monkeypatch):
    assert _download(monkeypatch, "unknown").status_code == 404

    entry = store.register_export(KEY, ["a"], [])
    monkeypatch.setattr(store.time, "time", lambda: entry.created_at + store.EXPORT_TTL_S)
    assert _download(monkeypatch, entry.token).status_code == 404


def test_download_does_not_need_an_active_session(monkeypatch):
    entry = store.register_export(KEY, ["a"], [])
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: None)
    assert _download(monkeypatch, entry.token).status_code == 200


# --- /enddatachat -----------------------------------------------------------


def test_enddatachat_purges_only_that_keys_exports(monkeypatch):
    app, _s, _a = _make_app()
    monkeypatch.setattr(datachat_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(datachat_route, "deleteAgent", lambda api_key, user_name: None)
    mine = store.register_export(KEY, ["a"], [])
    theirs = store.register_export(OTHER_KEY, ["a"], [])

    response = app.test_client().post(
        "/enddatachat", headers={**HEADERS, "X-USER-NAME": "user"}
    )

    assert response.status_code == 200
    assert store.resolve_export(mine.token) is None
    assert store.resolve_export(theirs.token) == theirs


def test_enddatachat_invalidates_previously_issued_download_urls(monkeypatch):
    engine = _TableEngine(_table(120, 3))
    app, _s, _a = _make_app()
    _patch_success_dependencies(monkeypatch, engine)
    active = {KEY: engine}
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: active.get(api_key))
    monkeypatch.setattr(
        datachat_route, "deleteAgent", lambda api_key, user_name: active.pop(api_key, None)
    )
    client = app.test_client()

    body = client.post("/datachat", json={"chat": "hello"}, headers=HEADERS).get_json()["response"]
    download_url = body["download_url"]
    assert client.get(download_url, headers=HEADERS).status_code == 200

    ended = client.post("/enddatachat", headers={**HEADERS, "X-USER-NAME": "user"})
    assert ended.status_code == 200
    assert ended.get_json() == {"Agent deleted succesfully": "active"}

    response = client.get(download_url, headers=HEADERS)
    assert response.status_code == 404
    assert response.get_json() == {"error": "Export not found"}
