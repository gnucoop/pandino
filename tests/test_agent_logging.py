"""Tests for utils.agent_logging.

First test module for this file. Unit-level: no Flask app, no route, no real
smolagents RunResult — every attribute log_runresult reads off `result` is
getattr-with-default, so a bare object() produces a complete record with no
exception.

log_runresult writes to the `agent_runs` logger, which normally carries a
FileHandler on the real audit log (logs/agent_runs.log). Every test here
redirects that logger to an in-memory StreamHandler and restores its
handlers, level and propagate afterwards, closing only the handler this
fixture added, so running this module never appends to the real file. No
conftest.py, following the pattern in tests/test_logging_config.py.
"""

import io
import json
import logging
from types import SimpleNamespace

import pytest

import services.agentchat_service as agentchat_service
from utils.agent_logging import log_runresult
from utils.logging_config import (
    CONTEXT_UNSET,
    _request_id_var,
    reset_request_context,
    set_request_context,
)

AGENT_RUNS_LOGGER_NAME = "agent_runs"


def _discard(handlers):
    for handler in handlers:
        try:
            handler.close()
        except Exception:  # noqa: BLE001 - teardown must never fail
            pass


@pytest.fixture(autouse=True)
def isolated_agent_runs_logger():
    """Redirect agent_runs to an in-memory stream; never touch the real audit log."""
    logger = logging.getLogger(AGENT_RUNS_LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    _request_id_var.set(CONTEXT_UNSET)
    try:
        yield stream
    finally:
        _request_id_var.set(CONTEXT_UNSET)
        _discard([h for h in logger.handlers if h not in saved_handlers])
        logger.handlers = saved_handlers
        logger.level = saved_level
        logger.propagate = saved_propagate


def _last_record(stream):
    lines = stream.getvalue().strip().splitlines()
    assert lines, "log_runresult wrote nothing to agent_runs"
    return json.loads(lines[-1])


def test_bound_request_id_is_recorded(isolated_agent_runs_logger):
    stream = isolated_agent_runs_logger
    tokens = set_request_context(request_id="abc123deadbeef01")
    try:
        log_runresult(
            object(),
            user="user@example.com",
            namespace="datachat",
            language="ENG",
            question="hi",
        )
    finally:
        reset_request_context(tokens)

    record = _last_record(stream)
    assert record["request_id"] == "abc123deadbeef01"


def test_no_bound_context_records_sentinel(isolated_agent_runs_logger):
    """The CLI / background / non-HTTP case: no request in flight."""
    stream = isolated_agent_runs_logger

    log_runresult(
        object(),
        user="user@example.com",
        namespace="datachat",
        language="ENG",
        question="hi",
    )

    record = _last_record(stream)
    assert record["request_id"] == CONTEXT_UNSET


def test_caller_passing_no_extra_still_gets_request_id(isolated_agent_runs_logger):
    """The /agentchat case.

    services/agentchat_service.py:120-126 calls log_runresult with no
    `extra` kwarg at all. This test is what proves /agentchat now receives
    request_id automatically, with zero change to its own call site.
    """
    stream = isolated_agent_runs_logger
    tokens = set_request_context(request_id="feedfacecafebeef")
    try:
        log_runresult(
            object(),
            user="user@example.com",
            namespace="agentchat",
            language="ENG",
            question="hi",
        )
    finally:
        reset_request_context(tokens)

    record = _last_record(stream)
    assert record["request_id"] == "feedfacecafebeef"
    assert record["extra"] is None


def test_request_id_is_top_level_not_nested_in_extra(isolated_agent_runs_logger):
    stream = isolated_agent_runs_logger
    tokens = set_request_context(request_id="0123456789abcdef")
    try:
        log_runresult(
            object(),
            user="user@example.com",
            namespace="datachat",
            language="ENG",
            question="hi",
            extra={"channel": "datachat", "response_kind": "text"},
        )
    finally:
        reset_request_context(tokens)

    record = _last_record(stream)
    assert record["request_id"] == "0123456789abcdef"
    assert "request_id" in record
    assert "request_id" not in record["extra"]


class _StubAgent:
    """Stand-in for smolagents.CodeAgent: no LLM, no tool calls."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return SimpleNamespace(
            steps=[],
            timing=None,
            token_usage=None,
            state=None,
            output={"answer": "real answer", "follow_ups": []},
        )


def _stub_config():
    models = SimpleNamespace(
        completion_model_provider="stub-provider",
        completion_model_agent_chat="stub-model",
        completion_embedding_model_provider="stub-emb-provider",
        completion_embedding_model="stub-emb-model",
    )
    rag = SimpleNamespace(top_k=3, min_sim=0.5)
    return SimpleNamespace(models=models, rag=rag)


def test_audit_log_failure_does_not_discard_agentchat_response(monkeypatch, caplog):
    """Axis 6: run_agentchat must still return its result when log_runresult fails.

    services/agentchat_service.py::run_agentchat calls log_runresult after the
    agent has already produced its answer. A failure in that audit-logging
    call must not turn an already-built AgentChatServiceResult into a
    RuntimeError, discarding a response that was already generated.
    """
    monkeypatch.setattr(agentchat_service, "build_litellm_model", lambda **kwargs: object())
    monkeypatch.setattr(agentchat_service, "RetrieverTool", lambda **kwargs: object())
    monkeypatch.setattr(agentchat_service, "CodeAgent", _StubAgent)
    monkeypatch.setattr(agentchat_service, "load_prompt", lambda *a, **kw: "template")
    monkeypatch.setattr(agentchat_service, "render_prompt", lambda *a, **kw: "rendered")
    monkeypatch.setattr(
        agentchat_service,
        "serialize_runresult",
        lambda result: {
            "answer": "real answer",
            "follow_ups": [],
            "tool_calls": [],
            "vectors": [],
            "metrics": {"token_usage": {}},
        },
    )

    def _raise_on_log(*args, **kwargs):
        raise OSError("audit unavailable")

    monkeypatch.setattr(agentchat_service, "log_runresult", _raise_on_log)

    caplog.set_level(logging.WARNING, logger="services.agentchat_service")

    result = agentchat_service.run_agentchat(
        chat=["hello"],
        namespace="agentchat",
        language="ENG",
        username="user@example.com",
        config=_stub_config(),
    )

    assert result["payload"]["answer"] == "real answer"
    assert result["model"] == "stub-model"
    assert result["provider"] == "stub-provider"
    assert any(
        "event=agentchat_audit_log_failed" in record.getMessage()
        for record in caplog.records
    )
