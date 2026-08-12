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
