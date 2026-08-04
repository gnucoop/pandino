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

import pytest

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
