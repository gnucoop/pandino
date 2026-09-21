"""Source Slice C: get_user_by_username() exposes the persisted `client` column.

Covers:
- get_user_by_username() maps the physical `client` column (already
  retrieved by the existing `SELECT *` query, no new query added) into the
  returned dict under the "client" key.
- A NULL `client` column maps to `mapping["client"] is None`, unchanged.

No live PostgreSQL: fake cursor/connection objects, same style as
tests/test_database_add_user_client.py.
"""

from unittest.mock import patch

from infrastructure import database_pg


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def execute(self, query, params=None):
        pass

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def test_get_user_by_username_maps_client_column(monkeypatch):
    row = (1, "alice", b"encrypted", "2030-01-01", 10, "dino")
    cursor = _FakeCursor(row)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeConnection(cursor))

    with patch.object(database_pg, "get_cipher_suite") as fake_cipher_suite_factory:
        fake_cipher_suite_factory.return_value.decrypt.return_value = b"decrypted"

        result = database_pg.get_user_by_username("alice")

    assert result["client"] == "dino"


def test_get_user_by_username_maps_null_client_column_to_none(monkeypatch):
    row = (1, "alice", b"encrypted", "2030-01-01", 10, None)
    cursor = _FakeCursor(row)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeConnection(cursor))

    with patch.object(database_pg, "get_cipher_suite") as fake_cipher_suite_factory:
        fake_cipher_suite_factory.return_value.decrypt.return_value = b"decrypted"

        result = database_pg.get_user_by_username("alice")

    assert result["client"] is None
