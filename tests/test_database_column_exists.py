import pytest

from infrastructure import database_methods, database_pg


class FakeCursor:
    def __init__(self, *, fetchone_result=None, raise_on_execute=False):
        self.calls = []
        self.fetchone_result = fetchone_result
        self.raise_on_execute = raise_on_execute

    def execute(self, query, params=()):
        self.calls.append((query, params))
        if self.raise_on_execute:
            raise RuntimeError("db unavailable")

    def fetchone(self):
        return self.fetchone_result


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def _patch_connect(monkeypatch, cursor):
    conn = FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    return conn


def test_build_check_column_exists_query_passes_schema_table_column():
    query, params = database_methods.build_check_column_exists_query(
        "public", "logs", "cost"
    )
    assert params == ("public", "logs", "cost")
    assert "information_schema.columns" in query.as_string(None)


def test_column_exists_returns_true_when_column_found(monkeypatch):
    cursor = FakeCursor(fetchone_result=(1,))
    conn = _patch_connect(monkeypatch, cursor)

    result = database_pg.column_exists("public", "logs", "cost")

    assert result is True
    assert conn.closed is True
    [(query, params)] = cursor.calls
    assert params == ("public", "logs", "cost")


def test_column_exists_returns_false_when_column_missing(monkeypatch):
    cursor = FakeCursor(fetchone_result=None)
    conn = _patch_connect(monkeypatch, cursor)

    result = database_pg.column_exists("public", "logs", "service")

    assert result is False
    assert conn.closed is True
    [(query, params)] = cursor.calls
    assert params == ("public", "logs", "service")


def test_column_exists_returns_false_on_database_error(monkeypatch):
    cursor = FakeCursor(raise_on_execute=True)
    conn = _patch_connect(monkeypatch, cursor)

    result = database_pg.column_exists("public", "logs", "cost")

    assert result is False
    assert conn.closed is True
