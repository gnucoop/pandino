"""Usage → Operational Admin drill-down: Operational read path.

Covers:
- build_get_operational_events_by_request_id_query() SQL shape/parameters.
- get_operational_events_by_request_id() row mapping, ordering pass-through,
  empty result, NULL preservation, details pass-through, millisecond
  event_time formatting, and failure propagation.

No live PostgreSQL: all coverage uses fake cursor/connection objects, same
style as tests/test_database_operational_events_writer.py.
"""

from datetime import datetime, timezone

import pytest

from infrastructure import database_pg
from infrastructure.database_methods import (
    build_get_operational_events_by_request_id_query,
)


# --- 1. Query builder ---


def test_query_targets_operational_events_table():
    query, _params = build_get_operational_events_by_request_id_query("req-1")

    query_str = query.as_string(None)
    assert "SELECT" in query_str
    assert "operational_events" in query_str


def test_query_projects_the_expected_columns_explicitly():
    query, _params = build_get_operational_events_by_request_id_query("req-1")

    query_str = query.as_string(None)
    for column in (
        "event_time",
        "level",
        "logger",
        "event",
        "app_id",
        "provider",
        "model",
        "duration_ms",
        "error_type",
        "details",
        "message",
    ):
        assert column in query_str


def test_query_does_not_use_select_star():
    query, _params = build_get_operational_events_by_request_id_query("req-1")

    assert "*" not in query.as_string(None)


def test_query_filters_on_parameterized_request_id():
    query, params = build_get_operational_events_by_request_id_query("req-1")

    query_str = query.as_string(None)
    assert "WHERE" in query_str
    assert "request_id" in query_str
    assert "%s" in query_str
    assert params == ("req-1",)


def test_query_never_interpolates_the_request_id_value():
    query, _params = build_get_operational_events_by_request_id_query(
        "9bf218009db0127d"
    )

    assert "9bf218009db0127d" not in query.as_string(None)


def test_query_orders_by_event_time_then_id_ascending():
    query, _params = build_get_operational_events_by_request_id_query("req-1")

    query_str = query.as_string(None)
    order_index = query_str.index("ORDER BY")
    order_clause = query_str[order_index:]

    assert order_clause.index("event_time") < order_clause.index("id")
    assert order_clause.count("ASC") == 2
    assert "DESC" not in order_clause


def test_query_has_no_pagination():
    query, _params = build_get_operational_events_by_request_id_query("req-1")

    query_str = query.as_string(None)
    assert "LIMIT" not in query_str
    assert "OFFSET" not in query_str


# --- 2. Reader fakes ---


class _FakeCursor:
    def __init__(self, rows=(), raise_on_execute=None):
        self.rows = list(rows)
        self.raise_on_execute = raise_on_execute
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if self.raise_on_execute is not None:
            raise self.raise_on_execute

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.close_calls = 0

    def cursor(self):
        return self._cursor

    def close(self):
        self.close_calls += 1


def _install(monkeypatch, cursor):
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    return conn


def _row(
    event_time=datetime(2026, 8, 27, 14, 22, 31, 482913, tzinfo=timezone.utc),
    level="INFO",
    logger_name="routes.multimodal",
    event="transcribe_started",
    app_id="app-1",
    provider="openai",
    model="whisper-1",
    duration_ms=842,
    error_type=None,
    details=None,
    message=None,
):
    return (
        event_time,
        level,
        logger_name,
        event,
        app_id,
        provider,
        model,
        duration_ms,
        error_type,
        details,
        message,
    )


# --- 3. Reader mapping ---


def test_reader_maps_row_to_the_expected_dict_keys(monkeypatch):
    cursor = _FakeCursor(rows=[_row()])
    _install(monkeypatch, cursor)

    events = database_pg.get_operational_events_by_request_id("req-1")

    assert len(events) == 1
    assert set(events[0]) == {
        "event_time",
        "level",
        "logger",
        "event",
        "app_id",
        "provider",
        "model",
        "duration_ms",
        "error_type",
        "details",
        "message",
    }


def test_reader_maps_values_to_the_correct_keys(monkeypatch):
    cursor = _FakeCursor(rows=[_row()])
    _install(monkeypatch, cursor)

    event = database_pg.get_operational_events_by_request_id("req-1")[0]

    assert event["level"] == "INFO"
    assert event["logger"] == "routes.multimodal"
    assert event["event"] == "transcribe_started"
    assert event["app_id"] == "app-1"
    assert event["provider"] == "openai"
    assert event["model"] == "whisper-1"
    assert event["duration_ms"] == 842


def test_reader_passes_the_request_id_through_to_the_query(monkeypatch):
    cursor = _FakeCursor(rows=[])
    _install(monkeypatch, cursor)

    database_pg.get_operational_events_by_request_id("req-xyz")

    _query, params = cursor.executed[0]
    assert params == ("req-xyz",)


def test_reader_preserves_the_order_supplied_by_the_query(monkeypatch):
    cursor = _FakeCursor(
        rows=[
            _row(event="first"),
            _row(event="second"),
            _row(event="third"),
        ]
    )
    _install(monkeypatch, cursor)

    events = database_pg.get_operational_events_by_request_id("req-1")

    assert [e["event"] for e in events] == ["first", "second", "third"]


def test_reader_returns_empty_list_for_zero_rows(monkeypatch):
    cursor = _FakeCursor(rows=[])
    _install(monkeypatch, cursor)

    assert database_pg.get_operational_events_by_request_id("req-1") == []


def test_reader_formats_event_time_with_millisecond_precision(monkeypatch):
    cursor = _FakeCursor(rows=[_row()])
    _install(monkeypatch, cursor)

    event = database_pg.get_operational_events_by_request_id("req-1")[0]

    assert event["event_time"] == "2026-08-27 14:22:31.482"


def test_reader_preserves_sql_nulls_as_none_without_display_sentinels(monkeypatch):
    cursor = _FakeCursor(
        rows=[
            _row(
                app_id=None,
                provider=None,
                model=None,
                duration_ms=None,
                error_type=None,
                details=None,
                message=None,
            )
        ]
    )
    _install(monkeypatch, cursor)

    event = database_pg.get_operational_events_by_request_id("req-1")[0]

    for key in (
        "app_id",
        "provider",
        "model",
        "duration_ms",
        "error_type",
        "details",
        "message",
    ):
        assert event[key] is None, key
    assert "N/A" not in [event[key] for key in event if key != "event_time"]


def test_reader_preserves_zero_duration_as_zero_not_none(monkeypatch):
    cursor = _FakeCursor(rows=[_row(duration_ms=0)])
    _install(monkeypatch, cursor)

    event = database_pg.get_operational_events_by_request_id("req-1")[0]

    assert event["duration_ms"] == 0
    assert event["duration_ms"] is not None


def test_reader_passes_details_through_unchanged(monkeypatch):
    details = {"branch": "audio", "reason": "missing_model", "extracted_chars": 12}
    cursor = _FakeCursor(rows=[_row(details=details)])
    _install(monkeypatch, cursor)

    event = database_pg.get_operational_events_by_request_id("req-1")[0]

    assert event["details"] == details
    assert isinstance(event["details"], dict)


def test_reader_does_not_truncate_or_reserialize_details(monkeypatch):
    long_value = "x" * 200
    cursor = _FakeCursor(rows=[_row(details={"reason": long_value})])
    _install(monkeypatch, cursor)

    event = database_pg.get_operational_events_by_request_id("req-1")[0]

    assert event["details"]["reason"] == long_value
    assert not isinstance(event["details"], str)


# --- 4. Reader failure ---


def test_reader_propagates_query_failure(monkeypatch):
    cursor = _FakeCursor(raise_on_execute=RuntimeError("select failed"))
    _install(monkeypatch, cursor)

    with pytest.raises(RuntimeError, match="select failed"):
        database_pg.get_operational_events_by_request_id("req-1")


def test_reader_does_not_convert_failure_into_empty_list(monkeypatch):
    cursor = _FakeCursor(raise_on_execute=RuntimeError("boom"))
    _install(monkeypatch, cursor)

    with pytest.raises(RuntimeError):
        database_pg.get_operational_events_by_request_id("req-1")


def test_reader_closes_the_connection_on_failure(monkeypatch):
    cursor = _FakeCursor(raise_on_execute=RuntimeError("boom"))
    conn = _install(monkeypatch, cursor)

    with pytest.raises(RuntimeError):
        database_pg.get_operational_events_by_request_id("req-1")

    assert conn.close_calls == 1


def test_reader_closes_the_connection_on_success(monkeypatch):
    cursor = _FakeCursor(rows=[_row()])
    conn = _install(monkeypatch, cursor)

    database_pg.get_operational_events_by_request_id("req-1")

    assert conn.close_calls == 1
