"""Foundation Intervention I2: schema and write primitive for Operational events.

Covers:
- build_insert_operational_event_query() SQL shape and parameter order.
- _insert_operational_event() (low-level, cursor-scoped primitive).
- insert_operational_event() (public, connection-owning primitive).
- Zero log emission from the new writer functions.
- Zero production callers of insert_operational_event after this intervention.

No live PostgreSQL: all coverage uses fake cursor/connection objects, same
style as tests/test_database_schema_fresh.py and
tests/test_database_usage_service_writer.py.
"""

import ast
import logging
import os
from datetime import datetime, timezone

import pytest

from infrastructure import database_pg
from infrastructure.database_methods import build_insert_operational_event_query


# --- 1/2. Query builder: SQL shape and params ---


def _sample_params():
    return dict(
        event_time=datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc),
        level="INFO",
        logger_name="services.compare_docs",
        event="compare_docs_started",
        request_id="req-123",
        app_id="app-456",
        provider="openai",
        model="gpt-4",
        duration_ms=42,
        error_type=None,
        details_json='{"page_number": 3}',
        message="Comparison started",
    )


def test_build_insert_operational_event_query_targets_operational_events_table():
    query, _params = build_insert_operational_event_query(**_sample_params())

    query_str = query.as_string(None)
    assert "INSERT INTO" in query_str
    assert "operational_events" in query_str


def test_build_insert_operational_event_query_includes_all_expected_columns():
    query, _params = build_insert_operational_event_query(**_sample_params())

    query_str = query.as_string(None)
    for column in (
        "event_time",
        "level",
        "logger",
        "event",
        "request_id",
        "app_id",
        "provider",
        "model",
        "duration_ms",
        "error_type",
        "details",
        "message",
    ):
        assert column in query_str


def test_build_insert_operational_event_query_casts_details_to_jsonb():
    query, _params = build_insert_operational_event_query(**_sample_params())

    assert "%s::jsonb" in query.as_string(None)


def test_build_insert_operational_event_query_has_no_returning_clause():
    query, _params = build_insert_operational_event_query(**_sample_params())

    assert "RETURNING" not in query.as_string(None)


def test_build_insert_operational_event_query_param_order_matches_writer_signature():
    kwargs = _sample_params()
    _query, params = build_insert_operational_event_query(**kwargs)

    assert params == (
        kwargs["event_time"],
        kwargs["level"],
        kwargs["logger_name"],
        kwargs["event"],
        kwargs["request_id"],
        kwargs["app_id"],
        kwargs["provider"],
        kwargs["model"],
        kwargs["duration_ms"],
        kwargs["error_type"],
        kwargs["details_json"],
        kwargs["message"],
    )


def test_build_insert_operational_event_query_preserves_none_for_nullable_fields():
    kwargs = _sample_params()
    kwargs.update(
        request_id=None,
        app_id=None,
        provider=None,
        model=None,
        duration_ms=None,
        error_type=None,
        details_json=None,
        message=None,
    )
    _query, params = build_insert_operational_event_query(**kwargs)

    assert params[4] is None  # request_id
    assert params[5] is None  # app_id
    assert params[6] is None  # provider
    assert params[7] is None  # model
    assert params[8] is None  # duration_ms
    assert params[9] is None  # error_type
    assert params[10] is None  # details_json
    assert params[11] is None  # message


# --- 3. Low-level primitive ---


class _RecordingCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))


def test_insert_operational_event_low_level_executes_exactly_one_insert():
    cursor = _RecordingCursor()

    result = database_pg._insert_operational_event(cursor, **_sample_params())

    assert len(cursor.executed) == 1
    assert result is None


def test_insert_operational_event_low_level_uses_query_builder_output():
    cursor = _RecordingCursor()
    kwargs = _sample_params()

    database_pg._insert_operational_event(cursor, **kwargs)

    expected_query, expected_params = build_insert_operational_event_query(**kwargs)
    executed_query, executed_params = cursor.executed[0]
    assert executed_query.as_string(None) == expected_query.as_string(None)
    assert executed_params == expected_params


def test_insert_operational_event_low_level_does_not_commit():
    class _CursorWithCommit(_RecordingCursor):
        def commit(self):
            raise AssertionError("low-level primitive must not commit")

    cursor = _CursorWithCommit()
    database_pg._insert_operational_event(cursor, **_sample_params())


# --- 4. Public writer success ---


class _TrackingConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commit_calls = 0
        self.close_calls = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commit_calls += 1

    def close(self):
        self.close_calls += 1


def test_insert_operational_event_public_success_path(monkeypatch):
    cursor = _RecordingCursor()
    conn = _TrackingConnection(cursor)
    connect_calls = {"n": 0}

    def _fake_connect():
        connect_calls["n"] += 1
        return conn

    monkeypatch.setattr(database_pg, "connect", _fake_connect)

    result = database_pg.insert_operational_event(**_sample_params())

    assert result is None
    assert connect_calls["n"] == 1
    assert len(cursor.executed) == 1
    assert conn.commit_calls == 1
    assert conn.close_calls == 1


def test_insert_operational_event_public_success_makes_no_extra_db_calls(monkeypatch):
    cursor = _RecordingCursor()
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    database_pg.insert_operational_event(**_sample_params())

    # Exactly one INSERT; no retry, no additional round trip.
    assert len(cursor.executed) == 1


# --- 5. Public writer failure ---


def test_insert_operational_event_propagates_cursor_failure(monkeypatch):
    class _RaisingCursor:
        def execute(self, query, params=None):
            raise psycopg_error()

    def psycopg_error():
        return RuntimeError("insert failed")

    conn = _TrackingConnection(_RaisingCursor())
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError, match="insert failed"):
        database_pg.insert_operational_event(**_sample_params())

    assert conn.close_calls == 1
    assert conn.commit_calls == 0


def test_insert_operational_event_does_not_retry_on_failure(monkeypatch):
    class _CountingRaisingCursor:
        def __init__(self):
            self.calls = 0

        def execute(self, query, params=None):
            self.calls += 1
            raise RuntimeError("boom")

    cursor = _CountingRaisingCursor()
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError):
        database_pg.insert_operational_event(**_sample_params())

    assert cursor.calls == 1


# --- 6. Logging-free writer ---


def test_insert_operational_event_public_emits_zero_log_records(monkeypatch, caplog):
    cursor = _RecordingCursor()
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with caplog.at_level(logging.DEBUG):
        database_pg.insert_operational_event(**_sample_params())

    assert caplog.records == []


def test_insert_operational_event_low_level_emits_zero_log_records(caplog):
    cursor = _RecordingCursor()

    with caplog.at_level(logging.DEBUG):
        database_pg._insert_operational_event(cursor, **_sample_params())

    assert caplog.records == []


def test_insert_operational_event_public_emits_zero_log_records_on_failure(monkeypatch, caplog):
    class _RaisingCursor:
        def execute(self, query, params=None):
            raise RuntimeError("boom")

    conn = _TrackingConnection(_RaisingCursor())
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            database_pg.insert_operational_event(**_sample_params())

    assert caplog.records == []


# --- 8. No production caller ---


def _iter_production_python_files():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excluded_dirs = {"tests", "venv", ".venv", "__pycache__", ".git", "docs"}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _references_insert_operational_event(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    if "insert_operational_event" not in source:
        return []
    tree = ast.parse(source, filename=file_path)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "insert_operational_event":
            hits.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == "insert_operational_event":
            hits.append(node.lineno)
    return hits


def test_insert_operational_event_has_zero_production_callers():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    database_pg_path = os.path.join(repo_root, "infrastructure", "database_pg.py")

    for file_path in _iter_production_python_files():
        if file_path == database_pg_path:
            # Its own definition site (_insert_operational_event and
            # insert_operational_event) is expected here.
            continue
        hits = _references_insert_operational_event(file_path)
        assert hits == [], f"unexpected production caller in {file_path}: lines {hits}"
