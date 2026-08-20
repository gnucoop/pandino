from infrastructure import database_pg


class FakeCursor:
    def __init__(self):
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass


def _run_init_db_and_capture_statements(monkeypatch):
    cursor = FakeCursor()
    conn = FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    database_pg.init_db()

    return cursor.statements


def _logs_table_statement(statements):
    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS logs" in statement:
            return statement
    raise AssertionError("No 'logs' table creation statement found")


def _users_table_statement(statements):
    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS users" in statement:
            return statement
    raise AssertionError("No 'users' table creation statement found")


def _operational_events_table_statement(statements):
    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS operational_events" in statement:
            return statement
    raise AssertionError("No 'operational_events' table creation statement found")


def test_fresh_logs_schema_includes_nullable_service_column(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    logs_statement = _logs_table_statement(statements)

    assert "service TEXT" in logs_statement
    assert "service TEXT NOT NULL" not in logs_statement


def test_fresh_logs_schema_service_column_is_scoped_to_logs_table(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)

    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS logs" not in statement:
            assert "service" not in statement


def test_fresh_logs_schema_includes_nullable_request_id_column(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    logs_statement = _logs_table_statement(statements)

    assert "request_id TEXT" in logs_statement
    assert "request_id TEXT NOT NULL" not in logs_statement


def test_fresh_logs_schema_request_id_column_is_scoped_to_logs_table(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)

    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS logs" not in statement and (
            "CREATE TABLE IF NOT EXISTS operational_events" not in statement
        ):
            assert "request_id" not in statement


def test_fresh_logs_schema_includes_nullable_duration_ms_column(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    logs_statement = _logs_table_statement(statements)

    assert "duration_ms INTEGER" in logs_statement
    assert "duration_ms INTEGER NOT NULL" not in logs_statement


def test_fresh_logs_schema_duration_ms_column_is_scoped_to_logs_table(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)

    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS logs" not in statement and (
            "CREATE TABLE IF NOT EXISTS operational_events" not in statement
        ):
            assert "duration_ms" not in statement


def test_fresh_users_schema_includes_nullable_client_column(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    users_statement = _users_table_statement(statements)

    assert "client TEXT" in users_statement
    assert "client TEXT NOT NULL" not in users_statement
    assert "client TEXT DEFAULT" not in users_statement


def test_fresh_users_schema_client_column_is_scoped_to_users_table(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)

    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS users" not in statement:
            assert "client" not in statement


def test_fresh_logs_schema_includes_nullable_source_column(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    logs_statement = _logs_table_statement(statements)

    assert "source TEXT" in logs_statement
    assert "source TEXT NOT NULL" not in logs_statement
    assert "source TEXT DEFAULT" not in logs_statement


def test_fresh_logs_schema_source_column_is_scoped_to_logs_and_feedback_tables(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)

    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS logs" not in statement and (
            "CREATE TABLE IF NOT EXISTS feedback" not in statement
        ):
            assert "source" not in statement


def test_fresh_schema_includes_operational_events_table(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)

    # Will raise AssertionError if the table statement is absent.
    _operational_events_table_statement(statements)


def test_fresh_operational_events_schema_has_bigserial_primary_key(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    assert "id BIGSERIAL PRIMARY KEY" in statement


def test_fresh_operational_events_schema_has_not_null_core_fields(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    assert "event_time TIMESTAMPTZ NOT NULL" in statement
    assert "level TEXT NOT NULL" in statement
    assert "logger TEXT NOT NULL" in statement
    assert "event TEXT NOT NULL" in statement


def test_fresh_operational_events_schema_event_time_has_no_default(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    assert "event_time TIMESTAMPTZ NOT NULL," in statement
    assert "DEFAULT NOW()" not in statement


def test_fresh_operational_events_schema_has_nullable_context_fields(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    for column in ("request_id", "app_id", "provider", "model", "error_type", "message"):
        assert f"{column} TEXT" in statement
        assert f"{column} TEXT NOT NULL" not in statement


def test_fresh_operational_events_schema_has_nullable_duration_ms(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    assert "duration_ms INTEGER" in statement
    assert "duration_ms INTEGER NOT NULL" not in statement


def test_fresh_operational_events_schema_has_jsonb_details(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    assert "details JSONB" in statement


def test_fresh_operational_events_schema_has_no_foreign_key(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    assert "REFERENCES" not in statement


def test_fresh_operational_events_schema_has_no_check_constraint(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    statement = _operational_events_table_statement(statements)

    assert "CHECK" not in statement
