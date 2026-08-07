"""Regression test for chat_end reporting the delivered answer, not shared instance state.

SmolagentsEngine instances are shared across concurrent same-API-key requests
via activeEngines (infrastructure/agent_manager.py). Before Intervento 39,
chat_end read final_answer_check_passed/final_kind from
self._last_final_answer_check_passed/self._last_final_kind, mutable fields on
that shared instance: a concurrent request using the same API key could
overwrite them before this call's chat_end line was emitted, producing a
false audit record. This module is the first to exercise SmolagentsEngine
directly.
"""

import io
import logging
from types import SimpleNamespace

import pytest

from datachat.smolagents_engine import SmolagentsEngine

RUNTIME_LOGGER_NAME = "datachat.runtime"


@pytest.fixture(autouse=True)
def restore_datachat_runtime_logger():
    """Snapshot and restore datachat.runtime, following the pattern in
    tests/test_datachat_route_request_id.py."""
    logger = logging.getLogger(RUNTIME_LOGGER_NAME)
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


def _make_engine():
    """Construct a minimal SmolagentsEngine without running __post_init__.

    __post_init__ builds the real CodeAgent/LLM stack, which is out of scope for
    this regression test. The fixture provides only the state chat() needs here;
    if chat() later reads additional __post_init__-initialized attributes, update
    this fixture rather than weakening the audit-state regression assertion.
    """
    engine = SmolagentsEngine.__new__(SmolagentsEngine)
    engine.api_key = "test-key"
    engine.user_name = "test-user"
    engine.llm = None
    engine.data = None
    engine._last_run_result = None
    engine._last_run_duration_ms = None
    return engine


class _ForeignWritingAgent:
    """agent.run() stub that simulates another concurrent request overwriting
    the shared engine instance's audit fields before this call's own delivered
    answer is produced."""

    def __init__(self, engine, output):
        self._engine = engine
        self._output = output

    def run(self, *args, **kwargs):
        setattr(self._engine, "_last_final_answer_check_passed", False)
        setattr(self._engine, "_last_final_kind", "other_request_kind")
        return SimpleNamespace(output=self._output)


def _capture_runtime_logger():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    runtime_logger = logging.getLogger(RUNTIME_LOGGER_NAME)
    runtime_logger.handlers = [handler]
    runtime_logger.propagate = False
    runtime_logger.setLevel(logging.INFO)
    return stream


def test_chat_end_reports_own_request_outcome_not_shared_state():
    engine = _make_engine()
    engine._agent = _ForeignWritingAgent(
        engine, '{"kind": "text", "text": "A own answer"}'
    )
    stream = _capture_runtime_logger()

    result = engine.chat("A")

    assert result["kind"] == "text"
    assert result["text"] == "A own answer"

    chat_end_line = next(
        line for line in stream.getvalue().splitlines() if line.startswith("chat_end")
    )
    assert "final_answer_check_passed=True" in chat_end_line
    assert "final_kind=text" in chat_end_line
    assert "final_kind=other_request_kind" not in chat_end_line
