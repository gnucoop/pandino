"""Tests for utils.logging_config.

These are the first tests in the suite that mutate global logging state, so
isolation is handled entirely here (no conftest.py) to keep the rest of the
suite unaffected.
"""

import io
import logging
import os
import re
from unittest.mock import patch

import pytest

from utils.logging_config import _HANDLER_MARKER, bootstrap_logging

BASE_ENV = {"DATACHAT_LOG_LEVEL": "INFO"}


def _discard(handlers):
    """Close handlers that are about to be dropped.

    Every bootstrap_logging() call opens a FileHandler on agent_runs;
    dropping it without close() leaks a descriptor per test and can block
    tmp_path cleanup where open files cannot be unlinked.
    """
    for handler in handlers:
        try:
            handler.close()
        except Exception:  # noqa: BLE001 - teardown must never fail
            pass


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Snapshot and restore every logger this module touches."""
    root = logging.getLogger()
    named = {name: logging.getLogger(name) for name in ("agent_runs", "datachat.runtime")}

    saved_root = (list(root.handlers), root.level)
    saved_named = {
        name: (list(lg.handlers), lg.level, lg.propagate) for name, lg in named.items()
    }

    # Start each test from a clean slate so ordering cannot matter. Root
    # handlers are pytest's own capture handlers and are restored verbatim,
    # so they are dropped without closing; the named loggers' handlers are
    # ours to close.
    root.handlers = []
    for lg in named.values():
        _discard(lg.handlers)
        lg.handlers = []

    # bootstrap_logging() calls load_dotenv() itself; a developer .env on disk
    # would otherwise defeat patch.dict(..., clear=True).
    try:
        with patch("utils.logging_config.load_dotenv", lambda *a, **k: None):
            yield
    finally:
        _discard([h for h in root.handlers if getattr(h, _HANDLER_MARKER, False)])
        root.handlers, root.level = saved_root
        for name, (handlers, level, propagate) in saved_named.items():
            _discard([h for h in named[name].handlers if h not in handlers])
            named[name].handlers = handlers
            named[name].level = level
            named[name].propagate = propagate


def _env(**overrides):
    """Environment with an agent-runs path that is always writable."""
    env = dict(BASE_ENV)
    env.update(overrides)
    return env


@pytest.fixture
def agent_runs_env(tmp_path):
    def _factory(**overrides):
        return _env(
            AGENT_RUNS_LOG_PATH=str(tmp_path / "logs" / "agent_runs.log"), **overrides
        )

    return _factory


def _marker_handlers():
    return [h for h in logging.getLogger().handlers if getattr(h, _HANDLER_MARKER, False)]


# --------------------------------------------------------------------------
# Level resolution
# --------------------------------------------------------------------------


def test_default_level_is_warning_when_log_level_unset(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()

    assert logging.getLogger().level == logging.WARNING


@pytest.mark.parametrize("raw", ["info", "INFO", "Info"])
def test_log_level_is_case_insensitive(agent_runs_env, raw):
    with patch.dict(os.environ, agent_runs_env(LOG_LEVEL=raw), clear=True):
        bootstrap_logging()

    assert logging.getLogger().level == logging.INFO


@pytest.mark.parametrize(
    "raw", ["VERBOSE", "", "   ", "42", "NOTSET", "notset", "0"]
)
def test_malformed_log_level_falls_back_to_warning(agent_runs_env, raw):
    with patch.dict(os.environ, agent_runs_env(LOG_LEVEL=raw), clear=True):
        bootstrap_logging()  # must not raise

    assert logging.getLogger().level == logging.WARNING


def test_malformed_log_level_emits_diagnostic_warning(agent_runs_env):
    stream = io.StringIO()
    with patch.dict(os.environ, agent_runs_env(LOG_LEVEL="VERBOSE"), clear=True):
        bootstrap_logging()
        handler = _marker_handlers()[0]
        handler.setStream(stream)
        # Re-run to capture the diagnostic on the redirected stream.
        bootstrap_logging()

    assert "VERBOSE" in stream.getvalue()


# --------------------------------------------------------------------------
# Handler management
# --------------------------------------------------------------------------


def test_bootstrap_installs_exactly_one_marked_handler(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()

    # pytest injects its own capture handlers into root, so only the marked
    # handler is counted.
    assert len(_marker_handlers()) == 1
    assert isinstance(_marker_handlers()[0], logging.StreamHandler)


def test_bootstrap_is_idempotent(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()
        bootstrap_logging()
        bootstrap_logging()

    assert len(_marker_handlers()) == 1


# --------------------------------------------------------------------------
# Formatting / context defaults
# --------------------------------------------------------------------------


def _capture(env, logger_name="some.module", message="hello"):
    stream = io.StringIO()
    with patch.dict(os.environ, env, clear=True):
        bootstrap_logging()
        handler = _marker_handlers()[0]
        handler.setStream(stream)
        logging.getLogger(logger_name).warning(message)
    return stream.getvalue()


def test_propagated_record_renders_context_defaults(agent_runs_env):
    """Regression test for handler-vs-logger filter placement.

    A record emitted by a plain module logger propagates to the root
    handlers without ever consulting the root logger's own filters.
    """
    output = _capture(agent_runs_env())

    assert "request_id=-" in output
    assert "app_id=-" in output
    assert "some.module" in output
    assert "hello" in output


def test_explicit_context_fields_are_not_overwritten(agent_runs_env):
    stream = io.StringIO()
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()
        _marker_handlers()[0].setStream(stream)
        logging.getLogger("some.module").warning(
            "hi", extra={"request_id": "req-1", "app_id": "app-1"}
        )

    assert "request_id=req-1" in stream.getvalue()
    assert "app_id=app-1" in stream.getvalue()


def test_timestamp_is_utc_iso8601_with_offset(agent_runs_env):
    output = _capture(agent_runs_env())
    timestamp = output.split(" ", 1)[0]

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00", timestamp), (
        f"unexpected timestamp: {timestamp!r}"
    )


# --------------------------------------------------------------------------
# Agent-runs audit channel
# --------------------------------------------------------------------------


def test_agent_runs_path_is_honoured_and_parent_created(tmp_path):
    target = tmp_path / "deeply" / "nested" / "agent_runs.log"
    assert not target.parent.exists()

    with patch.dict(
        os.environ, _env(AGENT_RUNS_LOG_PATH=str(target)), clear=True
    ):
        bootstrap_logging()

    assert target.parent.is_dir()
    assert target.exists()

    handlers = logging.getLogger("agent_runs").handlers
    assert any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(target)
        for h in handlers
    )


def test_agent_runs_propagate_remains_false(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()

    assert logging.getLogger("agent_runs").propagate is False


def test_unwritable_agent_runs_path_does_not_abort_startup(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    target = blocker / "logs" / "agent_runs.log"

    stream = io.StringIO()
    with patch.dict(os.environ, _env(AGENT_RUNS_LOG_PATH=str(target)), clear=True):
        logger = bootstrap_logging()  # must not raise
        _marker_handlers()[0].setStream(stream)
        # Re-run to capture the diagnostic on the redirected stream; the
        # handler is reused, so this stays a single-handler bootstrap.
        bootstrap_logging()

    assert logger.name == "datachat.runtime"
    assert not target.exists()

    # Degradation is acceptable only because it is not silent.
    output = stream.getvalue()
    assert str(target) in output
    assert "NotADirectoryError" in output
    assert "WARNING" in output
    assert logging.getLogger("agent_runs").handlers == []


# --------------------------------------------------------------------------
# DataChat runtime channel
# --------------------------------------------------------------------------


def test_returns_datachat_runtime_logger(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        logger = bootstrap_logging()

    assert logger.name == "datachat.runtime"
    assert logger.propagate is False
