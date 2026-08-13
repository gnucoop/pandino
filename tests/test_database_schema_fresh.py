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
        if "CREATE TABLE IF NOT EXISTS logs" not in statement:
            assert "request_id" not in statement


def test_fresh_logs_schema_includes_nullable_duration_ms_column(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)
    logs_statement = _logs_table_statement(statements)

    assert "duration_ms INTEGER" in logs_statement
    assert "duration_ms INTEGER NOT NULL" not in logs_statement


def test_fresh_logs_schema_duration_ms_column_is_scoped_to_logs_table(monkeypatch):
    statements = _run_init_db_and_capture_statements(monkeypatch)

    for statement in statements:
        if "CREATE TABLE IF NOT EXISTS logs" not in statement:
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
