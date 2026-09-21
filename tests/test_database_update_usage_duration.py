"""Usage Duration Slice B1: fixed-intent runtime UPDATE capability.

Covers:
- build_update_usage_duration_query() targets logs.duration_ms by logs.id,
  uses placeholders for values, and preserves parameter ordering
  (duration_ms, log_id).
- update_usage_duration() opens its own connection, commits on success,
  returns True/False based on cursor.rowcount, and rolls back and
  re-raises on exception.

No live PostgreSQL: all coverage uses fake cursor/connection objects, same
style as tests/test_database_usage_service_writer.py.
"""

import pytest

from infrastructure import database_pg
from infrastructure.database_methods import build_update_usage_duration_query


def test_build_update_usage_duration_query_targets_logs_duration_ms_by_id():
    query, params = build_update_usage_duration_query(log_id=42, duration_ms=1234)

    query_str = query.as_string(None)
    assert "UPDATE" in query_str
    assert '"logs"' in query_str
    assert '"duration_ms"' in query_str
    assert '"id"' in query_str
    assert "RETURNING" not in query_str
    assert params == (1234, 42)


def test_build_update_usage_duration_query_uses_placeholders_not_interpolation():
    query, params = build_update_usage_duration_query(log_id=42, duration_ms=1234)

    query_str = query.as_string(None)
    assert "42" not in query_str
    assert "1234" not in query_str
    assert query_str.count("%s") == 2


def test_build_update_usage_duration_query_no_schema_qualification_introduced():
    query, _params = build_update_usage_duration_query(log_id=42, duration_ms=1234)

    query_str = query.as_string(None)
    assert '"logs"."logs"' not in query_str
    assert query_str.count(".") == 0


class _FakeCursor:
    def __init__(self, rowcount, raise_on_execute=None, raise_on_commit=None):
        self.rowcount = rowcount
        self.executed = []
        self._raise_on_execute = raise_on_execute
        self._raise_on_commit_flag = raise_on_commit is not None
        self._raise_on_commit = raise_on_commit

    def execute(self, query, params=None):
        if self._raise_on_execute is not None:
            raise self._raise_on_execute
        self.executed.append((query, params))

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, cursor, raise_on_commit=None):
        self._cursor = cursor
        self._raise_on_commit = raise_on_commit
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        if self._raise_on_commit is not None:
            raise self._raise_on_commit
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_update_usage_duration_row_updated_returns_true(monkeypatch):
    cursor = _FakeCursor(rowcount=1)
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    result = database_pg.update_usage_duration(log_id=42, duration_ms=1234)

    assert result is True
    assert len(cursor.executed) == 1
    query, params = cursor.executed[0]
    assert params == (1234, 42)
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.closed is True


def test_update_usage_duration_missing_row_returns_false(monkeypatch):
    cursor = _FakeCursor(rowcount=0)
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    result = database_pg.update_usage_duration(log_id=999, duration_ms=1234)

    assert result is False
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.closed is True


def test_update_usage_duration_execute_failure_rolls_back_and_propagates(monkeypatch):
    cursor = _FakeCursor(rowcount=0, raise_on_execute=RuntimeError("db down"))
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError):
        database_pg.update_usage_duration(log_id=42, duration_ms=1234)

    assert conn.rolled_back is True
    assert conn.committed is False
    assert conn.closed is True


def test_update_usage_duration_commit_failure_rolls_back_and_propagates(monkeypatch):
    cursor = _FakeCursor(rowcount=1)
    conn = _FakeConnection(cursor, raise_on_commit=RuntimeError("commit failed"))
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError):
        database_pg.update_usage_duration(log_id=42, duration_ms=1234)

    assert conn.rolled_back is True
    assert conn.closed is True
