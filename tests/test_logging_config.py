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

from utils.logging_config import (
    CONTEXT_UNSET,
    THIRD_PARTY_LOG_LEVELS,
    _app_id_var,
    _HANDLER_MARKER,
    _request_id_var,
    bootstrap_logging,
    get_request_id,
    register_request_context_hooks,
    reset_request_context,
    set_request_context,
)

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
    named = {
        name: logging.getLogger(name)
        for name in ("agent_runs", "datachat.runtime", *THIRD_PARTY_LOG_LEVELS)
    }

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

    # A leaked request_id would otherwise make every later test in the module
    # order-dependent.
    _request_id_var.set(CONTEXT_UNSET)
    _app_id_var.set(CONTEXT_UNSET)

    # bootstrap_logging() calls load_dotenv() itself; a developer .env on disk
    # would otherwise defeat patch.dict(..., clear=True).
    try:
        with patch("utils.logging_config.load_dotenv", lambda *a, **k: None):
            yield
    finally:
        _request_id_var.set(CONTEXT_UNSET)
        _app_id_var.set(CONTEXT_UNSET)
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
    for name, level in THIRD_PARTY_LOG_LEVELS.items():
        assert logging.getLogger(name).level == level


# --------------------------------------------------------------------------
# Third-party logger boundary
# --------------------------------------------------------------------------


def test_third_party_namespaces_are_explicitly_pinned(agent_runs_env):
    """Levels come from Maui's own config, not inheritance from root."""
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()

    for name, level in THIRD_PARTY_LOG_LEVELS.items():
        third_party_logger = logging.getLogger(name)
        assert third_party_logger.level == level
        # NOTSET would mean "inherits from root" rather than "Maui-owned".
        assert third_party_logger.level != logging.NOTSET


def test_root_info_does_not_expose_third_party_info(agent_runs_env):
    stream = io.StringIO()
    with patch.dict(os.environ, agent_runs_env(LOG_LEVEL="INFO"), clear=True):
        bootstrap_logging()
        _marker_handlers()[0].setStream(stream)
        for name in THIRD_PARTY_LOG_LEVELS:
            logging.getLogger(name).info("THIRDPARTY_INFO_MARKER")

    assert "THIRDPARTY_INFO_MARKER" not in stream.getvalue()


def test_third_party_warning_still_reaches_root(agent_runs_env):
    stream = io.StringIO()
    with patch.dict(os.environ, agent_runs_env(LOG_LEVEL="INFO"), clear=True):
        bootstrap_logging()
        _marker_handlers()[0].setStream(stream)
        for name in THIRD_PARTY_LOG_LEVELS:
            logging.getLogger(name).warning("THIRDPARTY_WARNING_MARKER")

    output = stream.getvalue()
    assert output.count("THIRDPARTY_WARNING_MARKER") == len(THIRD_PARTY_LOG_LEVELS)


def test_maui_info_still_reaches_root_when_configured(agent_runs_env):
    stream = io.StringIO()
    with patch.dict(os.environ, agent_runs_env(LOG_LEVEL="INFO"), clear=True):
        bootstrap_logging()
        _marker_handlers()[0].setStream(stream)
        logging.getLogger("some.maui.module").info("MAUI_INFO_MARKER")

    assert "MAUI_INFO_MARKER" in stream.getvalue()


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


# --------------------------------------------------------------------------
# Request context: contextvars bound at the HTTP boundary
# --------------------------------------------------------------------------

_ROUTE_LOGGER = "route.probe"


def _make_app():
    """Throwaway Flask app carrying the hooks and two probe routes."""
    from flask import Flask, abort, request

    app = Flask(__name__)
    register_request_context_hooks(app)

    @app.route("/slowping")
    def slowping():
        """Yields to the hub mid-request so greenlets genuinely interleave."""
        import gevent

        tag = request.args.get("tag", "00")
        gevent.sleep(float(request.args.get("delay", "0.01")))
        logging.getLogger(_ROUTE_LOGGER).warning("SLOWMARKER%s", tag)
        return tag

    @app.route("/ping")
    def ping():
        logging.getLogger(_ROUTE_LOGGER).warning("PING_MARKER")
        return "pong"

    @app.route("/boom")
    def boom():
        logging.getLogger(_ROUTE_LOGGER).warning("BOOM_MARKER")
        abort(403)

    @app.route("/kaboom")
    def kaboom():
        logging.getLogger(_ROUTE_LOGGER).warning("KABOOM_MARKER")
        raise RuntimeError("unhandled")

    return app


def _bootstrapped_stream():
    """Bootstrap logging and redirect the marker handler to a buffer."""
    stream = io.StringIO()
    bootstrap_logging()
    _marker_handlers()[0].setStream(stream)
    return stream


def test_record_inside_request_carries_generated_id(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        stream = _bootstrapped_stream()
        response = _make_app().test_client().get("/ping")

    header_id = response.headers["X-Request-ID"]

    assert response.status_code == 200
    ping_line = next(l for l in stream.getvalue().splitlines() if "PING_MARKER" in l)
    assert f"request_id={header_id}" in ping_line
    # app_id stays unset: its source is a Phase 3 decision.
    assert "app_id=-" in ping_line


def test_request_id_header_is_sixteen_hex_chars(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()
        response = _make_app().test_client().get("/ping")

    assert re.fullmatch(r"[0-9a-f]{16}", response.headers["X-Request-ID"])


def test_two_requests_get_distinct_ids(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()
        client = _make_app().test_client()
        first = client.get("/ping").headers["X-Request-ID"]
        second = client.get("/ping").headers["X-Request-ID"]

    assert first != second


def test_context_is_reset_after_request(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()
        _make_app().test_client().get("/ping")

    assert get_request_id() == CONTEXT_UNSET


def test_context_is_reset_when_view_aborts(agent_runs_env):
    """The abort() path, used by routes/utils.py:17-23 for auth failures.

    Flask handles the HTTPException and finalises a 403 response, so
    after_request does run here and the header is set. The reset is still
    teardown's job.
    """
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        stream = _bootstrapped_stream()
        response = _make_app().test_client().get("/boom")

    assert response.status_code == 403
    boom_line = next(l for l in stream.getvalue().splitlines() if "BOOM_MARKER" in l)
    assert f"request_id={response.headers['X-Request-ID']}" in boom_line
    assert get_request_id() == CONTEXT_UNSET


def test_context_is_reset_when_view_raises_unhandled(agent_runs_env):
    """The path where after_request genuinely does not run.

    With testing=True an unhandled non-HTTP exception propagates out of the
    test client instead of being finalised into a 500, so after_request is
    skipped. teardown_request still runs, which is why the reset lives there.
    """
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        stream = _bootstrapped_stream()
        app = _make_app()
        app.testing = True
        with pytest.raises(RuntimeError, match="unhandled"):
            app.test_client().get("/kaboom")

    line = next(l for l in stream.getvalue().splitlines() if "KABOOM_MARKER" in l)
    assert re.search(r"request_id=[0-9a-f]{16}", line)
    assert get_request_id() == CONTEXT_UNSET


def test_register_request_context_hooks_is_idempotent(agent_runs_env):
    with patch.dict(os.environ, agent_runs_env(), clear=True):
        bootstrap_logging()
        app = _make_app()
        register_request_context_hooks(app)  # second call must be a no-op
        response = app.test_client().get("/ping")

    assert len(app.before_request_funcs[None]) == 1
    assert len(app.after_request_funcs[None]) == 1
    assert re.fullmatch(r"[0-9a-f]{16}", response.headers["X-Request-ID"])


def test_set_request_context_restores_previous_value():
    tokens = set_request_context(request_id="outer")
    assert get_request_id() == "outer"

    inner = set_request_context(request_id="inner")
    assert get_request_id() == "inner"

    reset_request_context(inner)
    assert get_request_id() == "outer"

    reset_request_context(tokens)
    assert get_request_id() == CONTEXT_UNSET


def test_reset_request_context_accepts_none():
    reset_request_context(None)  # must not raise
    assert get_request_id() == CONTEXT_UNSET


def test_context_is_isolated_across_greenlets(agent_runs_env):
    """The regression test for per-greenlet resolution of the ambient value.

    The pytest process is not monkey-patched and does not need to be:
    contextvar isolation across greenlets is provided by greenlet at the C
    level, independently of gevent's monkey-patching.
    """
    gevent = pytest.importorskip("gevent")

    count = 8
    read_back = {}

    with patch.dict(os.environ, agent_runs_env(), clear=True):
        stream = _bootstrapped_stream()

        def worker(n):
            tokens = set_request_context(request_id=f"greenlet{n:02d}")
            try:
                gevent.sleep(0.01)  # force a switch to the hub mid-flight
                logging.getLogger("greenlet.probe").warning("MARKER%02d", n)
                read_back[n] = get_request_id()
            finally:
                reset_request_context(tokens)

        jobs = [gevent.spawn(worker, n) for n in range(count)]
        gevent.joinall(jobs, timeout=30)

    assert all(job.successful() for job in jobs)
    assert read_back == {n: f"greenlet{n:02d}" for n in range(count)}

    lines = [l for l in stream.getvalue().splitlines() if "MARKER" in l]
    assert len(lines) == count

    for line in lines:
        marker = re.search(r"MARKER(\d{2})", line).group(1)
        rid = re.search(r"request_id=(\S+)", line).group(1)
        assert rid == f"greenlet{marker}", f"crossover on line: {line!r}"

    assert get_request_id() == CONTEXT_UNSET


def test_hooks_isolate_context_across_concurrent_requests(agent_runs_env):
    """The hooks themselves, not just set_request_context, under concurrency.

    Each greenlet drives a full request through the Flask test client, so
    before_request/after_request/teardown_request all participate.
    """
    gevent = pytest.importorskip("gevent")

    count = 6
    header_ids = {}

    with patch.dict(os.environ, agent_runs_env(), clear=True):
        stream = _bootstrapped_stream()
        app = _make_app()

        def worker(n):
            # Descending delays, so the requests must complete in reverse
            # spawn order. That ordering is what proves they really were in
            # flight together rather than being served one after another.
            delay = (count - n) * 0.01
            response = app.test_client().get(f"/slowping?tag={n:02d}&delay={delay}")
            header_ids[n] = response.headers["X-Request-ID"]

        jobs = [gevent.spawn(worker, n) for n in range(count)]
        gevent.joinall(jobs, timeout=30)

    assert all(job.successful() for job in jobs), [job.exception for job in jobs]
    assert len(header_ids) == count
    assert len(set(header_ids.values())) == count, "ids were reused across requests"

    lines = [l for l in stream.getvalue().splitlines() if "SLOWMARKER" in l]
    assert len(lines) == count

    emitted = [re.search(r"SLOWMARKER(\d{2})", l).group(1) for l in lines]
    assert emitted == [f"{n:02d}" for n in reversed(range(count))], (
        f"requests did not overlap; emission order was {emitted}"
    )

    for n, request_id in header_ids.items():
        line = next(l for l in lines if f"SLOWMARKER{n:02d}" in l)
        assert f"request_id={request_id}" in line, f"crossover on line: {line!r}"

    assert get_request_id() == CONTEXT_UNSET
