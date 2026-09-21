"""Regression tests for the ambient request id on POST /datachat.

Intervento 16 removed DataChat's local secrets.token_hex(4) id: the route now
reads the ambient id set by register_request_context_hooks() via
get_request_id(), and engine.chat() no longer takes a request_id argument.

Intervento 17 promoted request_id to a first-class AgentRunRecord field:
log_runresult now reads the ambient id itself instead of receiving it inside
`extra`. _StubEngine.get_last_trace() below is what makes the log_runresult
branch in dataChat() execute at all — in Intervento 16 the stub had no such
method, so that branch was dead code as far as this test module was
concerned.

This is the first test module to exercise routes/datachat.py, so isolation
of the datachat.runtime AND agent_runs loggers is handled entirely here (no
conftest.py), following the pattern in tests/test_logging_config.py.
"""

import io
import json
import logging
import re
from types import SimpleNamespace

import pytest
from flask import Flask

from routes import datachat as datachat_route
from routes import utils as routes_utils
from utils.logging_config import (
    CONTEXT_UNSET,
    _request_id_var,
    register_request_context_hooks,
)
from utils.runtime_logging import setup_datachat_runtime_logger

RUNTIME_LOGGER_NAME = "datachat.runtime"
AGENT_RUNS_LOGGER_NAME = "agent_runs"


@pytest.fixture(autouse=True)
def restore_datachat_runtime_logger():
    """Snapshot and restore datachat.runtime; reset the request id ContextVar."""
    logger = logging.getLogger(RUNTIME_LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate

    logger.handlers = []
    _request_id_var.set(CONTEXT_UNSET)
    try:
        yield
    finally:
        _request_id_var.set(CONTEXT_UNSET)
        for handler in logger.handlers:
            if handler not in saved_handlers:
                handler.close()
        logger.handlers = saved_handlers
        logger.level = saved_level
        logger.propagate = saved_propagate


@pytest.fixture(autouse=True)
def restore_agent_runs_logger():
    """Snapshot and restore agent_runs so this module never touches the real audit log."""
    logger = logging.getLogger(AGENT_RUNS_LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate

    logger.handlers = []
    try:
        yield
    finally:
        for handler in logger.handlers:
            if handler not in saved_handlers:
                handler.close()
        logger.handlers = saved_handlers
        logger.level = saved_level
        logger.propagate = saved_propagate


class _StubEngine:
    """Records the arguments engine.chat() receives; no LLM, no smolagents."""

    def __init__(self):
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"kind": "text", "text": "stub response"}

    def get_last_trace(self):
        """Presence alone satisfies the hasattr check in routes/datachat.py.

        Per the readiness survey, every attribute log_runresult and
        serialize_runresult read off `result` is getattr-with-default, so a
        bare object() survives both without raising and makes the
        log_runresult branch execute.
        """
        return {"run_result": object()}


class _TracelessStubEngine:
    """No get_last_trace(), so the hasattr check in dataChat() fails and
    trace_payload stays None - the no-trace gate."""

    def __init__(self):
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"kind": "text", "text": "stub response"}


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        datachat_token_cost=1,
        models=SimpleNamespace(
            datachat_model="test-model", datachat_provider="test-provider"
        ),
    )
    register_request_context_hooks(app)
    app.register_blueprint(datachat_route.datachat_bp)

    runtime_stream = io.StringIO()
    runtime_handler = logging.StreamHandler(runtime_stream)
    runtime_handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logger = logging.getLogger(RUNTIME_LOGGER_NAME)
    runtime_logger.handlers = [runtime_handler]
    runtime_logger.propagate = False
    runtime_logger.setLevel(logging.INFO)
    app.config["DATACHAT_RUNTIME_LOGGER"] = runtime_logger

    agent_runs_stream = io.StringIO()
    agent_runs_handler = logging.StreamHandler(agent_runs_stream)
    agent_runs_handler.setFormatter(logging.Formatter("%(message)s"))
    agent_runs_logger = logging.getLogger(AGENT_RUNS_LOGGER_NAME)
    agent_runs_logger.handlers = [agent_runs_handler]
    agent_runs_logger.propagate = False
    agent_runs_logger.setLevel(logging.INFO)

    return app, runtime_stream, agent_runs_stream


_DEFAULT_USER = {"id": 123, "username": "user@example.com", "client": "dino"}


def _patch_success_dependencies(
    monkeypatch,
    engine,
    record_calls=None,
    user=_DEFAULT_USER,
    recorded=True,
    read_back_calls=None,
    log_id=999,
):
    """Wire a /datachat app whose only Usage step is the adoption boundary.

    ``record_token_consumption`` and ``get_usage_log_id`` are stubbed at the
    route module, so these tests pin what the *adopter* states and reads,
    never how the boundary persists it. ``user`` may be a dict, ``None``, or
    a callable that raises, to exercise the accounting-side lookup paths.
    """
    monkeypatch.setattr(datachat_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(datachat_route, "get_user_tokens", lambda user_email: 10)
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: engine)
    monkeypatch.setattr(datachat_route, "edit_tokens", lambda *a, **k: None)

    def fake_get_user_by_username(user_email):
        if callable(user):
            return user(user_email)
        return user

    monkeypatch.setattr(
        datachat_route, "get_user_by_username", fake_get_user_by_username
    )

    def fake_record_token_consumption(**kwargs):
        if record_calls is not None:
            record_calls.append(kwargs)
        return recorded

    monkeypatch.setattr(
        datachat_route, "record_token_consumption", fake_record_token_consumption
    )

    def fake_get_usage_log_id():
        if read_back_calls is not None:
            read_back_calls.append(True)
        return log_id

    monkeypatch.setattr(datachat_route, "get_usage_log_id", fake_get_usage_log_id)


def _last_agent_runs_record(agent_runs_stream):
    lines = agent_runs_stream.getvalue().strip().splitlines()
    assert lines, "log_runresult wrote nothing to agent_runs"
    return json.loads(lines[-1])


def _post_chat(client, message="hello"):
    return client.post(
        "/datachat",
        json={"chat": message},
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )


def _extract_line(buffer_text, event):
    for line in buffer_text.splitlines():
        if line.startswith(event):
            return line
    raise AssertionError(f"no {event!r} line found in: {buffer_text!r}")


def test_engine_chat_receives_single_positional_arg_no_request_id_kwarg(monkeypatch):
    """Regression test for the chat() signature change: no request_id kwarg."""
    app, _stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    _patch_success_dependencies(monkeypatch, engine)

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert len(engine.calls) == 1
    args, kwargs = engine.calls[0]
    assert args == ("hello",)
    assert kwargs == {}


def test_runtime_log_request_id_matches_response_header(monkeypatch):
    app, stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    _patch_success_dependencies(monkeypatch, engine)

    response = _post_chat(app.test_client())

    header_id = response.headers["X-Request-ID"]
    start_line = _extract_line(stream.getvalue(), "datachat_request_start")
    assert f"request_id={header_id} " in start_line


def test_two_sequential_requests_get_distinct_runtime_ids(monkeypatch):
    app, stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    _patch_success_dependencies(monkeypatch, engine)
    client = app.test_client()

    _post_chat(client, "first message")
    _post_chat(client, "second message")

    start_lines = [
        line
        for line in stream.getvalue().splitlines()
        if line.startswith("datachat_request_start")
    ]
    assert len(start_lines) == 2

    id_pattern = re.compile(r"request_id=([0-9a-f]+)")
    ids = [id_pattern.search(line).group(1) for line in start_lines]
    assert ids[0] != ids[1]


def test_runtime_request_id_is_ambient_sixteen_hex_not_legacy_eight_hex(monkeypatch):
    app, stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    _patch_success_dependencies(monkeypatch, engine)

    _post_chat(app.test_client())

    start_line = _extract_line(stream.getvalue(), "datachat_request_start")
    match = re.search(r"request_id=([0-9a-f]+)", start_line)
    assert match is not None
    assert re.fullmatch(r"[0-9a-f]{16}", match.group(1))


def test_datachat_runtime_renders_app_id_after_auth_binding(monkeypatch):
    """Source Slice D2: datachat.runtime must render the same request_id/app_id
    that root Operational logging already shows post-auth.

    Uses the real assert_valid_api_key() (only the DB call underneath is
    mocked) so app_id is bound the same way it is in production, and rewires
    the runtime logger through the real setup_datachat_runtime_logger() so
    the production filter/formatter - not this module's bare %(message)s
    override used by the other tests here - is what gets exercised.
    """
    app, _stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    monkeypatch.setattr(
        routes_utils, "validate_api_key", lambda *a, **k: (True, "ok", "dino")
    )
    monkeypatch.setattr(datachat_route, "get_user_tokens", lambda user_email: 10)
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: engine)
    monkeypatch.setattr(datachat_route, "edit_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        datachat_route,
        "get_user_by_username",
        lambda user_email: {"id": 123, "username": user_email, "client": "dino"},
    )
    monkeypatch.setattr(
        datachat_route, "record_token_consumption", lambda **kwargs: True
    )
    monkeypatch.setattr(datachat_route, "get_usage_log_id", lambda: 999)

    runtime_logger = app.config["DATACHAT_RUNTIME_LOGGER"]
    runtime_logger.handlers = []
    setup_datachat_runtime_logger()
    production_stream = io.StringIO()
    runtime_logger.handlers[0].setStream(production_stream)

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    header_id = response.headers["X-Request-ID"]
    output = production_stream.getvalue()
    assert f"request_id={header_id} app_id=dino" in output


def test_agent_runs_record_carries_ambient_request_id(monkeypatch):
    """CRITICAL: asserts on the parsed JSONL record, not on the HTTP status.

    log_runresult swallows AttributeError/TypeError/ValueError internally
    and returns normally, so a broken record still yields a 200 response.
    """
    app, _stream, agent_runs_stream = _make_app()
    engine = _StubEngine()
    _patch_success_dependencies(monkeypatch, engine)

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    header_id = response.headers["X-Request-ID"]
    record = _last_agent_runs_record(agent_runs_stream)
    assert record["request_id"] == header_id


def test_datachat_states_only_consumption_facts_to_the_boundary(monkeypatch):
    """The adopter states exactly the six public consumption facts.

    request_id and source are absent by design: both are derived behind
    record_token_consumption(), so supplying either is impossible rather
    than merely discouraged.
    """
    app, _stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    record_calls = []
    _patch_success_dependencies(monkeypatch, engine, record_calls=record_calls)

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert len(record_calls) == 1
    assert record_calls[0] == {
        "user_id": 123,
        "provider": "test-provider",
        "model": "test-model",
        "service": "/datachat",
        "token_input": 0,
        "token_output": 0,
    }


def test_datachat_records_zero_token_pair_whenever_a_trace_exists(monkeypatch):
    """The absent-count normalization is unchanged: the serialized runtime
    reports None, the adopter states 0/0, and a row is still recorded. No
    token>0 guard was introduced."""
    app, _stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    record_calls = []
    _patch_success_dependencies(monkeypatch, engine, record_calls=record_calls)

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert [(c["token_input"], c["token_output"]) for c in record_calls] == [(0, 0)]


def test_successful_recording_reads_back_log_id_and_exposes_it(monkeypatch):
    """Case A: recording succeeds -> db_log_ok True, the id is read back
    through get_usage_log_id(), and the response contract is preserved."""
    app, stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    read_back_calls = []
    _patch_success_dependencies(
        monkeypatch, engine, read_back_calls=read_back_calls, log_id=999
    )

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert response.get_json()["log_id"] == 999
    assert read_back_calls == [True]

    status_line = _extract_line(stream.getvalue(), "datachat_trace_status")
    assert "db_log_ok=True" in status_line
    assert "log_id=999" in status_line
    end_line = _extract_line(stream.getvalue(), "datachat_request_end")
    assert "log_id=999" in end_line


def test_recording_returning_false_omits_log_id_and_never_reads_back(monkeypatch):
    """Case B: the boundary reports a runtime accounting failure. The
    response stays 200 without log_id, db_log_ok stays False, and no stale
    request-scoped id is read."""
    app, stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    read_back_calls = []
    _patch_success_dependencies(
        monkeypatch, engine, recorded=False, read_back_calls=read_back_calls
    )

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert "log_id" not in response.get_json()
    assert read_back_calls == []

    status_line = _extract_line(stream.getvalue(), "datachat_trace_status")
    assert "db_log_ok=False" in status_line
    assert "log_id=none" in status_line


def test_absent_trace_records_nothing_and_omits_log_id(monkeypatch):
    """Case C: the trace gate is unchanged. No trace means no explicit Usage
    recording at all - not a zero-token row."""
    app, stream, _agent_runs_stream = _make_app()
    engine = _TracelessStubEngine()
    record_calls = []
    read_back_calls = []
    _patch_success_dependencies(
        monkeypatch,
        engine,
        record_calls=record_calls,
        read_back_calls=read_back_calls,
    )

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert record_calls == []
    assert read_back_calls == []
    assert "log_id" not in response.get_json()

    status_line = _extract_line(stream.getvalue(), "datachat_trace_status")
    assert "trace_present=False" in status_line
    assert "db_log_ok=False" in status_line


@pytest.mark.parametrize(
    "user",
    [
        pytest.param(None, id="missing_user_row"),
        pytest.param({"username": "user@example.com"}, id="absent_user_id"),
        pytest.param({"id": "123", "username": "u"}, id="non_int_user_id"),
    ],
)
def test_missing_or_invalid_user_records_nothing(monkeypatch, user):
    """Case D: an unusable accounting-side identity skips recording and
    leaves the DataChat response untouched."""
    app, _stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    record_calls = []
    read_back_calls = []
    _patch_success_dependencies(
        monkeypatch,
        engine,
        user=user,
        record_calls=record_calls,
        read_back_calls=read_back_calls,
    )

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert response.get_json()["response"] is not None
    assert record_calls == []
    assert read_back_calls == []
    assert "log_id" not in response.get_json()


def test_user_lookup_failure_warns_safely_and_changes_nothing(monkeypatch, caplog):
    """Case E: the accounting-side lookup raising is fail-open.

    The exception message carries sensitive-looking text to prove the
    adopter's warning names only the exception type - never the username,
    the message, or a traceback.
    """
    app, _stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    record_calls = []
    read_back_calls = []

    secret = "password=hunter2 user@example.com api_key=test-key"

    def raising_lookup(user_email):
        raise RuntimeError(secret)

    _patch_success_dependencies(
        monkeypatch,
        engine,
        user=raising_lookup,
        record_calls=record_calls,
        read_back_calls=read_back_calls,
    )

    with caplog.at_level(logging.WARNING, logger="routes.datachat"):
        response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert response.get_json()["response"] is not None
    assert "log_id" not in response.get_json()
    assert record_calls == []
    assert read_back_calls == []

    warnings = [
        r
        for r in caplog.records
        if "datachat_usage_user_lookup_failed" in r.getMessage()
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "error_type=RuntimeError" in message
    assert secret not in message
    assert "hunter2" not in message
    assert "user@example.com" not in message
    assert "test-key" not in message


def test_agent_runs_extra_has_channel_and_response_kind_but_not_request_id(monkeypatch):
    app, _stream, agent_runs_stream = _make_app()
    engine = _StubEngine()
    _patch_success_dependencies(monkeypatch, engine)

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    record = _last_agent_runs_record(agent_runs_stream)
    assert record["extra"]["channel"] == "datachat"
    assert "response_kind" in record["extra"]
    assert "request_id" not in record["extra"]
