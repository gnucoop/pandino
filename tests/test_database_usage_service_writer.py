"""Usage Service Slice B: writer, insert query builder, and admin query/mapping.

Covers:
- build_insert_token_log_query() includes `service` in the explicit column
  list and parameter tuple, and preserves RETURNING id.
- log_token_usage() requires `service` (no default) and passes it through
  unchanged to the INSERT parameters.
- build_get_logs_for_admin_query() selects l.service, l.request_id,
  l.duration_ms, l.source.
- get_logs_for_admin() maps a persisted service/request_id/duration_ms/source
  value through unchanged and maps historical NULL rows to "N/A", following
  the existing model/provider `value or "N/A"` convention. duration_ms uses
  an explicit `is not None` check so a real `0` is preserved rather than
  collapsed to "N/A".

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
        "2026-08-12 00:00:00", 1, 10, 5, 0.01, "gpt-4", "openai", "/datachat", "abc123", "dino"
    )

    query_str = query.as_string(None)
    assert "service" in query_str
    assert "request_id" in query_str
    assert "source" in query_str
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
        "dino",
        None,
        None,
        None,
    )


def test_build_insert_token_log_query_preserves_none_source_as_none():
    _query, params = build_insert_token_log_query(
        "2026-08-12 00:00:00", 1, 10, 5, 0.01, "gpt-4", "openai", "/datachat", "abc123", None
    )

    assert params[9] is None


def test_log_token_usage_requires_service_argument():
    signature = inspect.signature(database_pg.log_token_usage)
    assert "service" in signature.parameters
    assert signature.parameters["service"].default is inspect.Parameter.empty


def test_log_token_usage_requires_request_id_argument():
    signature = inspect.signature(database_pg.log_token_usage)
    assert "request_id" in signature.parameters
    assert signature.parameters["request_id"].default is inspect.Parameter.empty


def test_log_token_usage_requires_source_argument():
    signature = inspect.signature(database_pg.log_token_usage)
    assert "source" in signature.parameters
    assert signature.parameters["source"].default is inspect.Parameter.empty


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
        source="dino",
    )

    assert log_id == 77
    insert_query, insert_params = cursor.executed[1]
    assert "service" in insert_query.as_string(None)
    assert "request_id" in insert_query.as_string(None)
    assert "source" in insert_query.as_string(None)
    assert insert_params[7] == "/datachat"
    assert insert_params[8] == "abc123"
    assert insert_params[9] == "dino"


def test_log_token_usage_persists_none_source_as_none(monkeypatch):
    cursor = _FakeCursor(cost_row=(0.001, 0.002), log_id_row=(78,))
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeConnection(cursor))

    log_id = database_pg.log_token_usage(
        user_id=1,
        token_input=10,
        token_output=5,
        model="gpt-4",
        provider="openai",
        service="/datachat",
        request_id="abc123",
        source=None,
    )

    assert log_id == 78
    _insert_query, insert_params = cursor.executed[1]
    assert insert_params[9] is None


def test_build_get_logs_for_admin_query_selects_service():
    query, _params = build_get_logs_for_admin_query(limit=50)

    assert "l.service" in query.as_string(None)


def test_build_get_logs_for_admin_query_selects_request_id_and_duration_ms():
    query, _params = build_get_logs_for_admin_query(limit=50)

    query_str = query.as_string(None)
    assert "l.request_id" in query_str
    assert "l.duration_ms" in query_str


def test_build_get_logs_for_admin_query_selects_source():
    query, _params = build_get_logs_for_admin_query(limit=50)

    assert "l.source" in query.as_string(None)


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
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", "/datachat", "9bf218009db0127d", 18308, "dino"),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["service"] == "/datachat"


def test_get_logs_for_admin_maps_historical_null_service_to_n_a(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", None, None, None, None),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["service"] == "N/A"
    assert result["logs"][0]["model"] == "gpt-4"
    assert result["logs"][0]["provider"] == "openai"


def test_get_logs_for_admin_maps_non_null_request_id_and_duration_through_unchanged(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", "/agentchat", "9bf218009db0127d", 18308, "dino"),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["request_id"] == "9bf218009db0127d"
    assert result["logs"][0]["duration_ms"] == 18308


def test_get_logs_for_admin_maps_historical_null_request_id_and_duration_to_n_a(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", "/agentchat", None, None, "dino"),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["request_id"] == "N/A"
    assert result["logs"][0]["duration_ms"] == "N/A"


def test_get_logs_for_admin_preserves_real_zero_duration_ms(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", "/agentchat", "9bf218009db0127d", 0, "dino"),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["duration_ms"] == 0


def test_get_logs_for_admin_maps_non_null_source_through_unchanged(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", "/agentchat", "9bf218009db0127d", 18308, "coopi"),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["source"] == "coopi"


def test_get_logs_for_admin_maps_historical_null_source_to_n_a(monkeypatch):
    rows = [
        (1, 10, "alice", "2026-08-12 00:00:00", 5, 3, 0.01, "gpt-4", "openai", "/agentchat", "9bf218009db0127d", 18308, None),
    ]
    cursor = _FakeAdminCursor(rows, total=1)
    monkeypatch.setattr(database_pg, "connect", lambda: _FakeAdminConnection(cursor))

    result = database_pg.get_logs_for_admin()

    assert result["logs"][0]["source"] == "N/A"


# --- Shared resolved-cost persistence foundation (Usage Slice 1) ---


class _TrackingCursor:
    """Like _FakeCursor, but records executed queries by role and never
    needs a costs-table row: fetchone() always answers the INSERT."""

    def __init__(self, log_id_row):
        self._log_id_row = log_id_row
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self._log_id_row

    def close(self):
        pass


class _TrackingConnection:
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


def test_log_usage_with_resolved_cost_skips_cost_lookup_and_inserts_row(monkeypatch):
    cursor = _TrackingCursor(log_id_row=(99,))
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    log_id = database_pg.log_usage_with_resolved_cost(
        user_id=1,
        cost=0.42,
        model="mistral-voxtral",
        provider="mistral",
        service="/transcribe",
        request_id="abc123",
        source="dino",
    )

    assert log_id == 99
    # Exactly one query executed: the INSERT. No costs-table lookup happens.
    assert len(cursor.executed) == 1
    insert_query, insert_params = cursor.executed[0]
    query_str = insert_query.as_string(None)
    assert "costs" not in query_str
    assert "RETURNING id" in query_str
    # token_input/token_output default to 0 (U-2 compatibility convention);
    # cost is the caller-resolved value, unmodified.
    assert insert_params[2] == 0
    assert insert_params[3] == 0
    assert insert_params[4] == 0.42
    assert conn.committed is True
    assert conn.closed is True


def test_log_usage_with_resolved_cost_accepts_explicit_token_counts(monkeypatch):
    cursor = _TrackingCursor(log_id_row=(100,))
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    database_pg.log_usage_with_resolved_cost(
        user_id=1,
        cost=0.42,
        model="mistral-voxtral",
        provider="mistral",
        service="/transcribe",
        request_id="abc123",
        source=None,
        token_input=7,
        token_output=11,
    )

    _insert_query, insert_params = cursor.executed[0]
    assert insert_params[2] == 7
    assert insert_params[3] == 11


def test_log_token_usage_closes_connection_when_cost_row_missing(monkeypatch):
    """Preserves the ValueError-on-missing-cost-row behavior while proving
    the connection is still closed on the raised path (adjacent leak fix)."""
    cursor = _TrackingCursor(log_id_row=None)
    cursor.fetchone = lambda: None  # costs-table lookup returns no row
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(ValueError, match="Cost not found"):
        database_pg.log_token_usage(
            user_id=1,
            token_input=10,
            token_output=5,
            model="gpt-4",
            provider="openai",
            service="/datachat",
            request_id="abc123",
            source="dino",
        )

    assert conn.closed is True
    assert conn.committed is False


def test_log_token_usage_closes_connection_when_insert_returns_no_id(monkeypatch):
    """Preserves the RuntimeError-on-missing-log-id behavior while proving
    the connection is still closed on the raised path (adjacent leak fix)."""
    calls = {"n": 0}

    class _Cursor:
        executed = []

        def execute(self, query, params=None):
            _Cursor.executed.append((query, params))

        def fetchone(self):
            calls["n"] += 1
            if calls["n"] == 1:
                return (0.001, 0.002)  # cost row
            return None  # INSERT ... RETURNING id yields nothing

    cursor = _Cursor()
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError, match="Failed to retrieve log_id"):
        database_pg.log_token_usage(
            user_id=1,
            token_input=10,
            token_output=5,
            model="gpt-4",
            provider="openai",
            service="/datachat",
            request_id="abc123",
            source="dino",
        )

    assert conn.closed is True
    assert conn.committed is False


# --- Embedding Usage Persistence P3: provenance fields + batch writer ---


def test_build_insert_token_log_query_includes_provenance_columns_as_null_by_default():
    query, params = build_insert_token_log_query(
        "2026-09-01 00:00:00", 1, 10, 5, 0.01, "gpt-4", "openai", "/datachat", "abc123", "dino"
    )

    query_str = query.as_string(None)
    assert "embedding_operation_kind" in query_str
    assert "quantity_origin" in query_str
    assert "cost_origin" in query_str
    assert params[10] is None
    assert params[11] is None
    assert params[12] is None


def test_build_insert_token_log_query_places_provenance_values_after_source():
    _query, params = build_insert_token_log_query(
        "2026-09-01 00:00:00",
        1,
        10,
        0,
        0.01,
        "bge-m3",
        "Deepinfra",
        "/completion.json",
        "abc123",
        "dino",
        embedding_operation_kind="query",
        quantity_origin="provider_reported",
        cost_origin="provider_authoritative",
    )

    # Existing columns keep their positions; the new values are additive.
    assert params[:10] == (
        "2026-09-01 00:00:00",
        1,
        10,
        0,
        0.01,
        "bge-m3",
        "Deepinfra",
        "/completion.json",
        "abc123",
        "dino",
    )
    assert params[10] == "query"
    assert params[11] == "provider_reported"
    assert params[12] == "provider_authoritative"


def test_log_usage_with_resolved_cost_writes_null_provenance_when_omitted(monkeypatch):
    cursor = _TrackingCursor(log_id_row=(101,))
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    database_pg.log_usage_with_resolved_cost(
        user_id=1,
        cost=0.42,
        model="mistral-voxtral",
        provider="mistral",
        service="/transcribe",
        request_id="abc123",
        source="dino",
    )

    _insert_query, insert_params = cursor.executed[0]
    assert insert_params[10] is None
    assert insert_params[11] is None
    assert insert_params[12] is None


def test_log_usage_with_resolved_cost_forwards_provenance_values(monkeypatch):
    cursor = _TrackingCursor(log_id_row=(102,))
    conn = _TrackingConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    log_id = database_pg.log_usage_with_resolved_cost(
        user_id=1,
        cost=0.42,
        model="bge-m3",
        provider="Deepinfra",
        service="/completion.json",
        request_id="abc123",
        source="dino",
        embedding_operation_kind="query",
        quantity_origin="provider_reported",
        cost_origin="provider_authoritative",
    )

    assert log_id == 102
    _insert_query, insert_params = cursor.executed[0]
    assert insert_params[10] == "query"
    assert insert_params[11] == "provider_reported"
    assert insert_params[12] == "provider_authoritative"


class _BatchCursor:
    """Returns a distinct log id per INSERT, and can be told to fail on the
    Nth execute so rollback behaviour is observable."""

    def __init__(self, log_ids, fail_on_execute=None):
        self._log_ids = list(log_ids)
        self._fail_on_execute = fail_on_execute
        self.executed = []
        self.closed = False

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if self._fail_on_execute is not None and len(self.executed) == self._fail_on_execute:
            raise RuntimeError("insert exploded")

    def fetchone(self):
        return (self._log_ids.pop(0),)

    def close(self):
        self.closed = True


class _BatchConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1

    def close(self):
        self.closed = True


def _entry(**overrides):
    values = dict(
        user_id=1,
        cost=0.5,
        model="bge-m3",
        provider="Deepinfra",
        service="/completion.json",
        request_id="abc123",
        source="dino",
        token_input=100,
        token_output=0,
    )
    values.update(overrides)
    return database_pg.ResolvedCostUsageEntry(**values)


def test_log_resolved_cost_usage_batch_empty_returns_empty_without_connecting(monkeypatch):
    opened = {"n": 0}

    def _connect():
        opened["n"] += 1
        raise AssertionError("no connection may be opened for an empty batch")

    monkeypatch.setattr(database_pg, "connect", _connect)

    assert database_pg.log_resolved_cost_usage_batch([]) == []
    assert opened["n"] == 0


def test_log_resolved_cost_usage_batch_single_entry_commits_once(monkeypatch):
    cursor = _BatchCursor(log_ids=[201])
    conn = _BatchConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    log_ids = database_pg.log_resolved_cost_usage_batch([_entry()])

    assert log_ids == [201]
    assert len(cursor.executed) == 1
    assert conn.commit_count == 1
    assert conn.rollback_count == 0
    assert conn.closed is True


def test_log_resolved_cost_usage_batch_inserts_in_input_order_in_one_transaction(monkeypatch):
    cursor = _BatchCursor(log_ids=[301, 302, 303])
    conn = _BatchConnection(cursor)
    connections = []

    def _connect():
        connections.append(conn)
        return conn

    monkeypatch.setattr(database_pg, "connect", _connect)

    log_ids = database_pg.log_resolved_cost_usage_batch(
        [
            _entry(model="model-a"),
            _entry(model="model-b"),
            _entry(model="model-c"),
        ]
    )

    assert log_ids == [301, 302, 303]
    assert len(connections) == 1
    assert conn.commit_count == 1
    assert [params[5] for _q, params in cursor.executed] == ["model-a", "model-b", "model-c"]
    assert conn.closed is True


def test_log_resolved_cost_usage_batch_rolls_back_whole_batch_on_failure(monkeypatch):
    cursor = _BatchCursor(log_ids=[401, 402], fail_on_execute=2)
    conn = _BatchConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError, match="insert exploded"):
        database_pg.log_resolved_cost_usage_batch([_entry(), _entry()])

    assert conn.rollback_count == 1
    assert conn.commit_count == 0
    assert conn.closed is True


def test_log_resolved_cost_usage_batch_carries_per_entry_provenance(monkeypatch):
    cursor = _BatchCursor(log_ids=[501, 502])
    conn = _BatchConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    database_pg.log_resolved_cost_usage_batch(
        [
            _entry(
                embedding_operation_kind="query",
                quantity_origin="provider_reported",
                cost_origin="provider_authoritative",
            ),
            _entry(
                embedding_operation_kind="document",
                quantity_origin="maui_derived",
                cost_origin="maui_resolved",
            ),
        ]
    )

    first = cursor.executed[0][1]
    second = cursor.executed[1][1]
    assert first[10:13] == ("query", "provider_reported", "provider_authoritative")
    assert second[10:13] == ("document", "maui_derived", "maui_resolved")


def test_log_resolved_cost_usage_batch_defaults_provenance_to_null(monkeypatch):
    cursor = _BatchCursor(log_ids=[601])
    conn = _BatchConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    database_pg.log_resolved_cost_usage_batch([_entry()])

    _query, params = cursor.executed[0]
    assert params[10:13] == (None, None, None)
