"""Usage Service Slice B: writer, insert query builder, and admin query/mapping.

Covers:
- build_insert_token_log_query() includes `service` in the explicit column
  list and parameter tuple, and preserves RETURNING id.
- log_token_usage() requires `service` (no default) and passes it through
  unchanged to the INSERT parameters.
- build_get_logs_for_admin_query() selects l.service.
- get_logs_for_admin() maps a persisted service value through unchanged and
  maps historical NULL rows to "N/A", following the existing model/provider
  `value or "N/A"` convention.

No live PostgreSQL: all coverage uses fake cursor/connection objects, same
style as tests/test_database_schema_fresh.py.
"""

import inspect

import pytest

from infrastructure import database_pg
from infrastructure.database_methods import (
    build_insert_token_log_query,
    build_get_logs_for_admin_query,
)


def test_build_insert_token_log_query_includes_service_column_and_param():
    query, params = build_insert_token_log_query(
        "2026-08-12 00:00:00", 1, 10, 5, 0.01, "gpt-4", "openai", "/datachat", "abc123"
    )

    query_str = query.as_string(None)
    assert "service" in query_str
    assert "request_id" in query_str
    assert "RETURNING id" in query_str
    assert params == (
        "2026-08-12 00:00:00",
        1,
        10,
        5,
        0.01,
        "gpt-4",
        "openai",
        "/datachat",
        "abc123",
    )


def test_log_token_usage_requires_service_argument():
    signature = inspect.signature(database_pg.log_token_usage)
    assert "service" in signature.parameters
    assert signature.parameters["service"].default is inspect.Parameter.empty


def test_log_token_usage_requires_request_id_argument():
    signature = inspect.signature(database_pg.log_token_usage)
    assert "request_id" in signature.parameters
    assert signature.parameters["request_id"].default is inspect.Parameter.empty


class _FakeCursor:
    def __init__(self, cost_row, log_id_row):
        self._cost_row = cost_row
        self._log_id_row = log_id_row
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        if len(self.executed) == 1:
            return self._cost_row
        return self._log_id_row

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def close(self):
        pass


def test_log_token_usage_persists_provided_service_as_insert_param(monkeypatch):
    cursor = _FakeCursor(cost_row=(0.001, 0.002), log_id_row=(77,))
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeConnection(cursor))

    log_id = database_pg.log_token_usage(
        user_id=1,
        token_input=10,
        token_output=5,
        model="gpt-4",
        provider="openai",
        service="/datachat",
        request_id="abc123",
    )

    assert log_id == 77
    insert_query, insert_params = cursor.executed[1]
    assert "service" in insert_query.as_string(None)
    assert "request_id" in insert_query.as_string(None)
    assert insert_params[-2] == "/datachat"
    assert insert_params[-1] == "abc123"


def test_build_get_logs_for_admin_query_selects_service():
    query, _params = build_get_logs_for_admin_query(limit=50)

    assert "l.service" in query.as_string(None)


class _FakeAdminCursor:
    def __init__(self, rows, total):
        self._rows = rows
        self._total = total
        self.execute_count = 0

    def execute(self, query, params=None):
        self.execute_count += 1

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return (self._total,)


class _FakeAdminConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def test_get_logs_for_admin_maps_non_null_service_through_unchanged(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", "/datachat"),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["service"] == "/datachat"


def test_get_logs_for_admin_maps_historical_null_service_to_n_a(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", None),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["service"] == "N/A"
    assert result["logs"][0]["model"] == "gpt-4"
    assert result["logs"][0]["provider"] == "openai"
