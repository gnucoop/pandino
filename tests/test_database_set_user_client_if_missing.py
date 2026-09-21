"""Source Slice B1: atomic fill-if-empty persistence primitive for users.client.

Covers:
- build_set_user_client_if_missing_query() targets users.client by
  username, uses placeholders for values, and encodes the fill-if-empty
  invariant (client IS NULL) in the WHERE clause itself.
- set_user_client_if_missing() opens its own connection, commits on
  success, returns True/False based on cursor.rowcount, and rolls back
  and re-raises on exception.

No live PostgreSQL: all coverage uses fake cursor/connection objects, same
style as tests/test_database_update_usage_duration.py.
"""

import pytest

from infrastructure import database_pg
from infrastructure.database_methods import build_set_user_client_if_missing_query


def test_build_set_user_client_if_missing_query_targets_users_client_by_username():
    query, params = build_set_user_client_if_missing_query(
        username="alice", client="coopi"
    )

    query_str = query.as_string(None)
    assert "UPDATE" in query_str
    assert '"users"' in query_str
    assert '"client"' in query_str
    assert '"username"' in query_str
    assert params == ("coopi", "alice")


def test_build_set_user_client_if_missing_query_enforces_client_is_null():
    query, _params = build_set_user_client_if_missing_query(
        username="alice", client="coopi"
    )

    query_str = query.as_string(None)
    assert "IS NULL" in query_str
    assert '"client" IS NULL' in query_str


def test_build_set_user_client_if_missing_query_uses_placeholders_not_interpolation():
    query, params = build_set_user_client_if_missing_query(
        username="alice", client="coopi"
    )

    query_str = query.as_string(None)
    assert "alice" not in query_str
    assert "coopi" not in query_str
    assert query_str.count("%s") == 2


class _FakeCursor:
    def __init__(self, rowcount, raise_on_execute=None):
        self.rowcount = rowcount
        self.executed = []
        self._raise_on_execute = raise_on_execute

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


def test_set_user_client_if_missing_row_updated_returns_true(monkeypatch):
    cursor = _FakeCursor(rowcount=1)
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    result = database_pg.set_user_client_if_missing(username="alice", client="coopi")

    assert result is True
    assert len(cursor.executed) == 1
    query, params = cursor.executed[0]
    assert params == ("coopi", "alice")
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.closed is True


def test_set_user_client_if_missing_already_set_returns_false(monkeypatch):
    cursor = _FakeCursor(rowcount=0)
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    result = database_pg.set_user_client_if_missing(username="alice", client="dino")

    assert result is False
    assert len(cursor.executed) == 1
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.closed is True


def test_set_user_client_if_missing_unknown_username_returns_false(monkeypatch):
    cursor = _FakeCursor(rowcount=0)
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    result = database_pg.set_user_client_if_missing(username="ghost", client="coopi")

    assert result is False
    assert conn.committed is True
    assert conn.rolled_back is False
    assert conn.closed is True


def test_set_user_client_if_missing_execute_failure_rolls_back_and_propagates(
    monkeypatch,
):
    cursor = _FakeCursor(rowcount=0, raise_on_execute=RuntimeError("db down"))
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError):
        database_pg.set_user_client_if_missing(username="alice", client="coopi")

    assert conn.rolled_back is True
    assert conn.committed is False
    assert conn.closed is True


def test_set_user_client_if_missing_commit_failure_rolls_back_and_propagates(
    monkeypatch,
):
    cursor = _FakeCursor(rowcount=1)
    conn = _FakeConnection(cursor, raise_on_commit=RuntimeError("commit failed"))
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError):
        database_pg.set_user_client_if_missing(username="alice", client="coopi")

    assert conn.rolled_back is True
    assert conn.closed is True
