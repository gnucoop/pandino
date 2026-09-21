from infrastructure import database_pg


def test_valid_command_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "init_db", lambda: events.append("init_db"))

    database_pg.run_cli(["database_pg.py", "init_db"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "init_db"]


def test_initialization_occurs_exactly_once(monkeypatch):
    load_dotenv_calls = []
    load_config_calls = []
    init_calls = []
    init_db_calls = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: load_dotenv_calls.append(1))
    monkeypatch.setattr(database_pg, "load_config", lambda: load_config_calls.append(1) or object())
    monkeypatch.setattr(database_pg, "init", lambda config: init_calls.append(config))
    monkeypatch.setattr(database_pg, "init_db", lambda: init_db_calls.append(1))

    database_pg.run_cli(["database_pg.py", "init_db"])

    assert len(load_dotenv_calls) == 1
    assert len(load_config_calls) == 1
    assert len(init_calls) == 1
    assert len(init_db_calls) == 1


def test_no_command_shows_help_without_initializing(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "init_db", lambda: events.append("init_db"))

    database_pg.run_cli(["database_pg.py"])

    assert events == ["print_help"]


def test_unknown_command_shows_help_without_initializing(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))

    database_pg.run_cli(["database_pg.py", "not_a_real_command"])

    assert events == ["print_help"]


def test_invalid_argument_count_shows_help_without_initializing(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_user", lambda username, api_key: events.append("add_user"))

    # add_user requires exactly 4 argv entries (script, add_user, username, api_key)
    database_pg.run_cli(["database_pg.py", "add_user"])

    assert events == ["print_help"]


def test_argument_forwarding_for_command_with_args(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: None)
    monkeypatch.setattr(database_pg, "load_config", lambda: object())
    monkeypatch.setattr(database_pg, "init", lambda config: None)
    monkeypatch.setattr(
        database_pg,
        "add_user",
        lambda username, api_key: events.append((username, api_key)),
    )

    database_pg.run_cli(["database_pg.py", "add_user", "alice", "secret-key"])

    assert events == [("alice", "secret-key")]


def test_add_usage_service_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_usage_service_column", lambda: events.append("add_usage_service_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_service_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_usage_service_column"]


def test_add_usage_service_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_usage_service_column", lambda: events.append("add_usage_service_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_service_column", "foo"])

    assert events == ["print_help"]


def test_add_usage_request_id_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_usage_request_id_column", lambda: events.append("add_usage_request_id_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_request_id_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_usage_request_id_column"]


def test_add_usage_request_id_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_usage_request_id_column", lambda: events.append("add_usage_request_id_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_request_id_column", "foo"])

    assert events == ["print_help"]


def test_add_usage_duration_ms_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_usage_duration_ms_column", lambda: events.append("add_usage_duration_ms_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_duration_ms_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_usage_duration_ms_column"]


def test_add_usage_duration_ms_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_usage_duration_ms_column", lambda: events.append("add_usage_duration_ms_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_duration_ms_column", "foo"])

    assert events == ["print_help"]


def test_add_user_client_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_user_client_column", lambda: events.append("add_user_client_column"))

    database_pg.run_cli(["database_pg.py", "add_user_client_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_user_client_column"]


def test_add_user_client_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_user_client_column", lambda: events.append("add_user_client_column"))

    database_pg.run_cli(["database_pg.py", "add_user_client_column", "foo"])

    assert events == ["print_help"]


def test_add_usage_source_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_usage_source_column", lambda: events.append("add_usage_source_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_source_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_usage_source_column"]


def test_add_usage_source_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_usage_source_column", lambda: events.append("add_usage_source_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_source_column", "foo"])

    assert events == ["print_help"]


def test_add_usage_embedding_operation_kind_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_usage_embedding_operation_kind_column", lambda: events.append("add_usage_embedding_operation_kind_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_embedding_operation_kind_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_usage_embedding_operation_kind_column"]


def test_add_usage_embedding_operation_kind_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_usage_embedding_operation_kind_column", lambda: events.append("add_usage_embedding_operation_kind_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_embedding_operation_kind_column", "foo"])

    assert events == ["print_help"]


def test_add_usage_quantity_origin_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_usage_quantity_origin_column", lambda: events.append("add_usage_quantity_origin_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_quantity_origin_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_usage_quantity_origin_column"]


def test_add_usage_quantity_origin_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_usage_quantity_origin_column", lambda: events.append("add_usage_quantity_origin_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_quantity_origin_column", "foo"])

    assert events == ["print_help"]


def test_add_usage_cost_origin_column_initializes_before_execution(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "add_usage_cost_origin_column", lambda: events.append("add_usage_cost_origin_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_cost_origin_column"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config), "add_usage_cost_origin_column"]


def test_add_usage_cost_origin_column_rejects_unexpected_argument(monkeypatch):
    events = []

    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: events.append("load_config"))
    monkeypatch.setattr(database_pg, "init", lambda config: events.append("init"))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))
    monkeypatch.setattr(database_pg, "add_usage_cost_origin_column", lambda: events.append("add_usage_cost_origin_column"))

    database_pg.run_cli(["database_pg.py", "add_usage_cost_origin_column", "foo"])

    assert events == ["print_help"]


def test_list_users_passes_a_valid_limit_to_the_query_builder(monkeypatch):
    captured = {}

    class _Cursor:
        def execute(self, query, params):
            captured["executed"] = (query, params)

        def fetchall(self):
            return []

    class _Conn:
        def cursor(self):
            return _Cursor()

        def close(self):
            captured["closed"] = True

    def fake_builder(limit, offset=0, search=None):
        captured["limit"] = limit
        captured["offset"] = offset
        return "QUERY", (limit, offset)

    monkeypatch.setattr(database_pg, "connect", lambda: _Conn())
    monkeypatch.setattr(database_pg, "build_list_users_query", fake_builder)

    database_pg.list_users()

    assert isinstance(captured["limit"], int)
    assert captured["limit"] > 0
    assert captured["limit"] == database_pg.CLI_LIST_USERS_LIMIT
    assert captured["executed"] == ("QUERY", (captured["limit"], 0))
    assert captured["closed"] is True


def test_list_users_cli_path_is_operational(monkeypatch):
    events = []

    fake_config = object()
    monkeypatch.setattr(database_pg, "load_dotenv", lambda: events.append("load_dotenv"))
    monkeypatch.setattr(database_pg, "load_config", lambda: (events.append("load_config"), fake_config)[1])
    monkeypatch.setattr(database_pg, "init", lambda config: events.append(("init", config)))
    monkeypatch.setattr(database_pg, "print_help", lambda: events.append("print_help"))

    calls = []

    class _Cursor:
        def execute(self, query, params):
            calls.append(params)

        def fetchall(self):
            return [(1, "alice", b"encrypted", "2030-01-01", 10)]

    class _Conn:
        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    monkeypatch.setattr(database_pg, "connect", lambda: _Conn())
    monkeypatch.setattr(
        database_pg, "build_list_users_query", lambda limit, offset=0, search=None: ("QUERY", (limit, offset))
    )
    monkeypatch.setattr(database_pg, "get_cipher_suite", lambda: _Cipher())

    database_pg.run_cli(["database_pg.py", "list_users"])

    assert events == ["load_dotenv", "load_config", ("init", fake_config)]
    assert calls == [(database_pg.CLI_LIST_USERS_LIMIT, 0)]


class _Cipher:
    def decrypt(self, value):
        return b"super-secret-api-key"


def test_list_users_never_prints_a_decrypted_api_key(monkeypatch, capsys):
    class _Cursor:
        def execute(self, query, params):
            pass

        def fetchall(self):
            return [(1, "alice", b"encrypted", "2030-01-01", 10)]

    class _Conn:
        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    monkeypatch.setattr(database_pg, "connect", lambda: _Conn())
    monkeypatch.setattr(
        database_pg, "build_list_users_query", lambda limit, offset=0, search=None: ("QUERY", (limit, offset))
    )
    monkeypatch.setattr(database_pg, "get_cipher_suite", lambda: _Cipher())

    database_pg.list_users()

    out = capsys.readouterr().out
    assert "alice" in out
    assert "super-secret-api-key" not in out


def test_add_user_help_matches_the_dispatcher(capsys):
    database_pg.print_help()

    out = capsys.readouterr().out
    assert "  add_user <username> <api_key>  Add a new user" in out
    assert "date_valid_until" not in out
