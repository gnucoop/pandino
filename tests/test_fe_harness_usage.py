"""Tests for evaluation.datachat.collectors.usage, the read-only Usage collector.

psycopg.connect is monkeypatched with a recording fake connection, so no
database is ever contacted. Pins the read-only connection protocol
(startup options, read_only before the first statement, rollback, never
commit, close on every path), row mapping and password hygiene.
"""

import subprocess
import sys

import psycopg
import pytest

from evaluation.datachat.collectors import usage
from evaluation.datachat.collectors.usage import (
    USAGE_COLUMNS,
    USER_STATE_SQL,
    DbSettings,
    UserTokenState,
    check_read_access,
    collect_usage,
    fetch_user_token_state,
)
from evaluation.datachat.contract import USAGE_QUERY_SQL
from evaluation.datachat.model import Availability

RID = "0123456789abcdef"
PASSWORD = "super-secret-pw"


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.conn.events.append(("execute", query, params, self.conn.read_only))
        if self.conn.execute_error is not None:
            raise self.conn.execute_error

    def fetchall(self):
        self.conn.events.append(("fetchall",))
        return list(self.conn.rows)

    def fetchone(self):
        self.conn.events.append(("fetchone",))
        return self.conn.rows[0] if self.conn.rows else None


class FakeConnection:
    def __init__(self, rows=(), execute_error=None):
        self.rows = rows
        self.execute_error = execute_error
        self.events = []
        self.autocommit = False
        self._read_only = None

    @property
    def read_only(self):
        return self._read_only

    @read_only.setter
    def read_only(self, value):
        self.events.append(("read_only", value))
        self._read_only = value

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.events.append(("commit",))

    def rollback(self):
        self.events.append(("rollback",))

    def close(self):
        self.events.append(("close",))


@pytest.fixture
def settings():
    return DbSettings(host="localhost", dbname="maui", port="5432", user="maui", password=PASSWORD, schema="maui_s")


def _patch_connect(monkeypatch, conn, calls):
    def fake_connect(**kwargs):
        calls.append(kwargs)
        return conn

    monkeypatch.setattr(usage.psycopg, "connect", fake_connect)


def _row(row_id, service):
    return (row_id, 42, "2026-10-06 10:15:30", 100, 20, 0.5, "m", "OpenAI", service, RID, 1200, "datachat")


def test_connection_kwargs_and_protocol(monkeypatch, settings):
    conn, calls = FakeConnection(rows=[_row(7, "/datachat")]), []
    _patch_connect(monkeypatch, conn, calls)

    collection = collect_usage(settings, RID)

    assert calls == [
        {
            "host": "localhost",
            "dbname": "maui",
            "user": "maui",
            "password": PASSWORD,
            "port": "5432",
            "options": "-c default_transaction_read_only=on -c search_path=maui_s",
        }
    ]
    kinds = [event[0] for event in conn.events]
    assert kinds == ["read_only", "execute", "fetchall", "rollback", "close"]
    assert conn.events[0] == ("read_only", True)
    _, query, params, read_only_at_execute = conn.events[1]
    assert (query, params, read_only_at_execute) == (USAGE_QUERY_SQL, (RID,), True)
    assert conn.autocommit is False
    assert collection.record.availability is Availability.PRESENT


def test_rows_mapped_in_db_order(monkeypatch, settings):
    rows = [_row(3, "semantic"), _row(5, "semantic"), _row(9, "/datachat")]
    _patch_connect(monkeypatch, FakeConnection(rows=rows), [])

    collection = collect_usage(settings, RID, primary_log_id=9)

    evidence = collection.evidence
    assert [row["id"] for row in evidence.rows] == [3, 5, 9]
    assert list(evidence.rows[0]) == list(USAGE_COLUMNS)
    assert evidence.rows[2]["service"] == "/datachat"
    assert evidence.primary_log_id == 9
    assert evidence.primary_row_present is True
    assert evidence.query_sql == USAGE_QUERY_SQL
    assert evidence.schema == "maui_s"
    assert evidence.to_dict()["schema_version"] == 1
    assert collection.record.locator["row_count"] == 3


def test_zero_rows_is_present(monkeypatch, settings):
    _patch_connect(monkeypatch, FakeConnection(rows=[]), [])
    collection = collect_usage(settings, RID, primary_log_id=12)
    assert collection.record.availability is Availability.PRESENT
    assert collection.evidence.rows == ()
    assert collection.evidence.primary_row_present is False


def test_db_error_is_unavailable_and_still_closes(monkeypatch, settings):
    error = psycopg.OperationalError(f"connection to localhost failed: password={PASSWORD}")
    conn = FakeConnection(execute_error=error)
    _patch_connect(monkeypatch, conn, [])

    collection = collect_usage(settings, RID)

    assert collection.record.availability is Availability.UNAVAILABLE
    assert collection.record.reason.startswith("db_error: OperationalError")
    assert collection.evidence is None
    assert [e[0] for e in conn.events] == ["read_only", "execute", "close"]
    assert PASSWORD not in repr(collection.record.to_dict())


def test_connect_failure_is_unavailable(monkeypatch, settings):
    def fail(**kwargs):
        raise psycopg.OperationalError(f"bad password {PASSWORD}")

    monkeypatch.setattr(usage.psycopg, "connect", fail)
    collection = collect_usage(settings, RID)
    assert collection.record.availability is Availability.UNAVAILABLE
    assert PASSWORD not in collection.record.reason


def test_non_db_error_propagates_after_close(monkeypatch, settings):
    conn = FakeConnection(execute_error=RuntimeError("bug"))
    _patch_connect(monkeypatch, conn, [])
    with pytest.raises(RuntimeError):
        collect_usage(settings, RID)
    assert conn.events[-1] == ("close",)
    assert ("commit",) not in conn.events


@pytest.mark.parametrize("schema", ["public; DROP TABLE logs", "maui-s", "1abc", "", "a b", "x,public"])
def test_invalid_schema_rejected_before_connect(monkeypatch, schema):
    calls = []
    _patch_connect(monkeypatch, FakeConnection(), calls)
    with pytest.raises(ValueError):
        DbSettings(host="h", dbname="d", port="5432", user="u", password=PASSWORD, schema=schema)
    assert calls == []


def test_settings_from_env_and_password_hygiene():
    settings = DbSettings.from_env({"PGHOST": "h", "PGDB": "d", "PGUSER": "u", "PGPWD": PASSWORD})
    assert (settings.port, settings.schema) == ("5432", "public")
    assert PASSWORD not in repr(settings)
    assert PASSWORD not in repr(settings.public())
    with pytest.raises(ValueError, match="PGPWD"):
        DbSettings.from_env({"PGHOST": "h", "PGDB": "d", "PGUSER": "u"})
    with pytest.raises(ValueError):
        DbSettings.from_env({"PGHOST": "h", "PGDB": "d", "PGUSER": "u", "PGPWD": "p", "MAUI_SCHEMA": "a;b"})


def test_check_read_access(monkeypatch, settings):
    conn = FakeConnection(rows=[(1,)])
    _patch_connect(monkeypatch, conn, [])
    assert check_read_access(settings) is None
    assert [e[0] for e in conn.events] == ["read_only", "execute", "fetchone", "rollback", "close"]
    assert conn.events[1][1] == "SELECT 1"

    _patch_connect(monkeypatch, FakeConnection(execute_error=psycopg.OperationalError("down")), [])
    assert check_read_access(settings) == "db_error: OperationalError"


def test_fetch_user_token_state(monkeypatch, settings):
    conn = FakeConnection(rows=[(42, "fe-eval-v1", 500, "2099-12-31")])
    _patch_connect(monkeypatch, conn, [])

    state = fetch_user_token_state(settings, "fe-eval-v1")

    assert state == UserTokenState(user_id=42, username="fe-eval-v1", tokens=500, date_valid_until="2099-12-31")
    _, query, params, read_only = conn.events[1]
    assert query == USER_STATE_SQL and params == ("fe-eval-v1",) and read_only is True
    assert query.lstrip().upper().startswith("SELECT ")
    assert "api_key" not in query
    assert ("commit",) not in conn.events
    assert conn.events[-1] == ("close",)


def test_fetch_user_token_state_missing_user(monkeypatch, settings):
    _patch_connect(monkeypatch, FakeConnection(rows=[]), [])
    assert fetch_user_token_state(settings, "nobody") is None


def test_no_statement_is_a_write():
    for statement in (usage.USAGE_QUERY_SQL, usage.USER_STATE_SQL, usage.READ_CHECK_SQL):
        assert statement.lstrip().upper().startswith("SELECT")


def test_collector_does_not_import_maui_db_modules():
    code = (
        "import sys, evaluation.datachat.collectors.usage; "
        "bad = [m for m in ('config', 'infrastructure', 'infrastructure.database_pg') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
