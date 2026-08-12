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
from utils.logging_config import (
    CONTEXT_UNSET,
    _request_id_var,
    register_request_context_hooks,
)

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


def _patch_success_dependencies(monkeypatch, engine, log_calls=None):
    monkeypatch.setattr(datachat_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(datachat_route, "get_user_tokens", lambda user_email: 10)
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: engine)
    monkeypatch.setattr(datachat_route, "edit_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        datachat_route,
        "get_user_by_username",
        lambda user_email: {"id": 123, "username": user_email},
    )

    def fake_log_token_usage(**kwargs):
        if log_calls is not None:
            log_calls.append(kwargs)
        return 999

    monkeypatch.setattr(datachat_route, "log_token_usage", fake_log_token_usage)


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


def test_log_token_usage_receives_datachat_service_literal(monkeypatch):
    app, _stream, _agent_runs_stream = _make_app()
    engine = _StubEngine()
    log_calls = []
    _patch_success_dependencies(monkeypatch, engine, log_calls)

    response = _post_chat(app.test_client())

    assert response.status_code == 200
    assert len(log_calls) == 1
    assert log_calls[0]["service"] == "/datachat"


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
