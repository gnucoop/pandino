import pytest

from infrastructure import database_pg
from infrastructure.database_pg import SchemaChangeResult


class FakeCursor:
    def __init__(self, *, fetchone_results=None, raise_map=None, events=None):
        self.calls = []
        self.fetchone_results = list(fetchone_results or [])
        self.raise_map = raise_map or {}
        self.events = events if events is not None else []

    def execute(self, query, params=()):
        idx = len(self.calls)
        self.calls.append((query, params))
        if idx in self.raise_map:
            self.events.append(f"execute_failed:{idx}")
            raise self.raise_map[idx]
        self.events.append(f"execute:{idx}")

    def fetchone(self):
        return self.fetchone_results.pop(0)

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor, events=None):
        self._cursor = cursor
        self.closed = False
        self.commit_count = 0
        self.rollback_count = 0
        self.events = events if events is not None else []

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commit_count += 1
        self.events.append("commit")

    def rollback(self):
        self.rollback_count += 1
        self.events.append("rollback")

    def close(self):
        self.closed = True


def _patch_connect(monkeypatch, cursor, events=None):
    conn = FakeConnection(cursor, events=events)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    return conn


def test_add_column_if_missing_rejects_unlisted_column_type(monkeypatch):
    connect_called = False

    def _fail_connect():
        nonlocal connect_called
        connect_called = True
        raise AssertionError("connect() must not be called for an invalid column type")

    monkeypatch.setattr(database_pg, "connect", _fail_connect)

    with pytest.raises(ValueError):
        database_pg.add_column_if_missing("public", "users", "client", "TEXT; DROP TABLE users;--")

    assert connect_called is False


def test_add_column_if_missing_column_already_exists(monkeypatch):
    cursor = FakeCursor(fetchone_results=[(1,)])
    conn = _patch_connect(monkeypatch, cursor)

    result = database_pg.add_column_if_missing("public", "users", "client", "TEXT")

    assert result == SchemaChangeResult.UNCHANGED
    assert len(cursor.calls) == 1
    assert conn.commit_count == 0
    assert conn.rollback_count == 0
    assert conn.closed is True


def test_add_column_if_missing_column_confirmed_absent(monkeypatch):
    events = []
    cursor = FakeCursor(fetchone_results=[None, (1,)], events=events)
    conn = _patch_connect(monkeypatch, cursor, events=events)

    result = database_pg.add_column_if_missing("public", "users", "client", "TEXT")

    assert result == SchemaChangeResult.CHANGED
    assert len(cursor.calls) == 3
    alter_query, _ = cursor.calls[1]
    assert "ALTER TABLE" in alter_query.as_string(None)
    assert "ADD COLUMN" in alter_query.as_string(None)
    assert conn.commit_count == 1
    assert conn.rollback_count == 0
    assert conn.closed is True
    # DDL -> post-DDL verification -> commit, in that order, commit only after
    # verification succeeds (Neon-verified: uncommitted DDL is already
    # visible to same-transaction schema inspection).
    assert events == ["execute:0", "execute:1", "execute:2", "commit"]


def test_add_column_if_missing_inspection_failure_never_executes_ddl(monkeypatch):
    cursor = FakeCursor(raise_map={0: RuntimeError("db unavailable")})
    conn = _patch_connect(monkeypatch, cursor)

    result = database_pg.add_column_if_missing("public", "users", "client", "TEXT")

    assert result == SchemaChangeResult.FAILED
    assert len(cursor.calls) == 1
    for query, _ in cursor.calls:
        assert "ALTER TABLE" not in query.as_string(None)
    assert conn.commit_count == 0
    assert conn.rollback_count == 0
    assert conn.closed is True


def test_add_column_if_missing_ddl_failure_rolls_back(monkeypatch):
    events = []
    cursor = FakeCursor(fetchone_results=[None], raise_map={1: RuntimeError("ddl failed")}, events=events)
    conn = _patch_connect(monkeypatch, cursor, events=events)

    result = database_pg.add_column_if_missing("public", "users", "client", "TEXT")

    assert result == SchemaChangeResult.FAILED
    assert len(cursor.calls) == 2
    assert conn.commit_count == 0
    assert conn.rollback_count == 1
    assert conn.closed is True
    assert events == ["execute:0", "execute_failed:1", "rollback"]


def test_add_column_if_missing_post_ddl_verification_failure_rolls_back(monkeypatch):
    events = []
    cursor = FakeCursor(fetchone_results=[None, None], events=events)
    conn = _patch_connect(monkeypatch, cursor, events=events)

    result = database_pg.add_column_if_missing("public", "users", "client", "TEXT")

    assert result == SchemaChangeResult.FAILED
    assert len(cursor.calls) == 3
    assert conn.commit_count == 0
    assert conn.rollback_count == 1
    assert conn.closed is True
    # DDL executed and post-DDL verification ran, but reported the column
    # still absent: no commit must ever happen on this path.
    assert events == ["execute:0", "execute:1", "execute:2", "rollback"]


def test_add_column_if_missing_post_ddl_verification_error_rolls_back(monkeypatch):
    events = []
    cursor = FakeCursor(fetchone_results=[None], raise_map={2: RuntimeError("db unavailable")}, events=events)
    conn = _patch_connect(monkeypatch, cursor, events=events)

    result = database_pg.add_column_if_missing("public", "users", "client", "TEXT")

    assert result == SchemaChangeResult.FAILED
    assert len(cursor.calls) == 3
    assert conn.commit_count == 0
    assert conn.rollback_count == 1
    assert conn.closed is True
    assert events == ["execute:0", "execute:1", "execute_failed:2", "rollback"]
