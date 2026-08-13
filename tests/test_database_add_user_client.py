"""Source Slice B2: new-user client persistence in the initial INSERT.

Covers:
- build_add_user_query() includes the client column/parameter in the same
  INSERT as username/api_key/date_valid_until (one INSERT, no separate
  UPDATE), and preserves the pre-existing columns/parameter order.
- add_user() forwards its new client parameter through to the query
  builder, defaulting to None so the pre-existing CLI/two-argument caller
  (infrastructure/database_pg.py::_resolve_cli_command) keeps working
  unchanged.

No live PostgreSQL: fake cursor/connection objects, same style as
tests/test_database_set_user_client_if_missing.py.
"""

from unittest.mock import patch

from infrastructure import database_pg
from infrastructure.database_methods import build_add_user_query


def test_build_add_user_query_includes_client_column_and_parameter():
    query, params = build_add_user_query(
        "alice", "encrypted-key", "2030-01-01 00:00:00", "coopi"
    )

    query_str = query.as_string(None)
    assert "INSERT INTO" in query_str
    assert '"users"' in query_str
    assert '"username"' in query_str
    assert '"api_key"' in query_str
    assert '"date_valid_until"' in query_str
    assert '"client"' in query_str
    assert params == ("alice", "encrypted-key", "2030-01-01 00:00:00", "coopi")


def test_build_add_user_query_defaults_client_to_none():
    query, params = build_add_user_query("alice", "encrypted-key", "2030-01-01 00:00:00")

    assert params == ("alice", "encrypted-key", "2030-01-01 00:00:00", None)


class _FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def test_add_user_forwards_client_into_insert(monkeypatch):
    cursor = _FakeCursor()
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with patch.object(
        database_pg, "get_cipher_suite"
    ) as fake_cipher_suite_factory:
        fake_cipher_suite_factory.return_value.encrypt.return_value = b"encrypted"

        result = database_pg.add_user(
            "alice", "plain-key", "2030-01-01 00:00:00", "coopi"
        )

    assert result is None
    assert len(cursor.executed) == 1
    _query, params = cursor.executed[0]
    assert params == ("alice", "encrypted", "2030-01-01 00:00:00", "coopi")
    assert conn.committed is True
    assert conn.closed is True


def test_add_user_without_client_argument_persists_null(monkeypatch):
    cursor = _FakeCursor()
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with patch.object(
        database_pg, "get_cipher_suite"
    ) as fake_cipher_suite_factory:
        fake_cipher_suite_factory.return_value.encrypt.return_value = b"encrypted"

        result = database_pg.add_user("alice", "plain-key")

    assert result is None
    _query, params = cursor.executed[0]
    assert params[-1] is None
