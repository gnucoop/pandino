"""
Bootstrap-wiring tests for utils.operational_persistence.register_operational_persistence
— FOUNDATION INTERVENTION I6.

Scope: attaching the already-built I2-I5 subsystem (snapshot, handler,
bounded delivery queue, gevent consumer) to the real root logger, exactly
once, idempotently, via a single main.py registrar call. No production
event vocabulary, no new persistence semantics, no DB round-trip.

Production inertness is load-bearing throughout: zero production call sites
invoke the I1 emission-contract builder function, so no ordinary Maui log
record carries the persistence marker, and the installed handler ignores
all such records.
"""

import ast
import logging
import os
from datetime import datetime, timezone

import gevent
import pytest

import utils.operational_persistence as op
from utils.logging_config import ContextDefaultsFilter, THIRD_PARTY_LOG_LEVELS
from utils.operational_persistence import (
    OperationalEventSnapshot,
    OperationalPersistenceHandler,
    register_operational_persistence,
)

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeApp:
    """Minimal stand-in: the registrar must not depend on a real Flask app."""


@pytest.fixture(autouse=True)
def _isolated_root_and_delivery():
    """Isolate both root.handlers and the process-local delivery singleton
    across tests, so bootstrap tests never leak a handler, a queue or a
    consumer greenlet into other test modules."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    op._reset_delivery_for_tests()
    yield
    op._reset_delivery_for_tests()

    root.handlers = saved_handlers
    root.level = saved_level


def _persistence_handlers(root):
    return [
        h for h in root.handlers if getattr(h, "_maui_operational_persistence", False)
    ]


def _clear_maui_handlers(root):
    """Drop only Maui-marked handlers, preserving any handler pytest itself
    attached to root (e.g. caplog's), so caplog-based assertions keep
    working inside tests that also want a clean Maui-handler slate."""
    root.handlers = [
        h
        for h in root.handlers
        if not getattr(h, "_maui_bootstrap", False)
        and not getattr(h, "_maui_operational_persistence", False)
    ]


# ---------------------------------------------------------------------------
# 1 & 2. Registrar attaches exactly one OperationalPersistenceHandler
# ---------------------------------------------------------------------------


def test_registrar_attaches_exactly_one_persistence_handler():
    root = logging.getLogger()
    _clear_maui_handlers(root)

    register_operational_persistence(_FakeApp())

    handlers = _persistence_handlers(root)
    assert len(handlers) == 1
    assert isinstance(handlers[0], OperationalPersistenceHandler)


# ---------------------------------------------------------------------------
# 3. Sink ownership — semantic, not bound-method identity
# ---------------------------------------------------------------------------


def test_handler_sink_reaches_the_process_local_delivery_enqueue():
    root = logging.getLogger()
    _clear_maui_handlers(root)

    register_operational_persistence(_FakeApp())

    handler = _persistence_handlers(root)[0]
    delivery = op._get_or_create_delivery()

    # Bound methods of the same instance compare equal even when Python
    # recreates the wrapper object, so this is robust without relying on
    # `is` identity of the bound-method object itself.
    assert handler._sink == delivery.enqueue
    assert handler._sink.__self__ is delivery

    sent = []
    handler._sink = lambda snapshot: sent.append(snapshot)
    snapshot = OperationalEventSnapshot(
        event_time=datetime.now(timezone.utc),
        level="INFO",
        logger="x",
        event="e",
        request_id=None,
        app_id=None,
        provider=None,
        model=None,
        duration_ms=None,
        error_type=None,
        details_json=None,
        message=None,
    )
    handler._sink(snapshot)
    assert sent == [snapshot]


# ---------------------------------------------------------------------------
# 4. Own ContextDefaultsFilter
# ---------------------------------------------------------------------------


def test_persistence_handler_has_exactly_one_context_defaults_filter():
    root = logging.getLogger()
    _clear_maui_handlers(root)

    register_operational_persistence(_FakeApp())

    handler = _persistence_handlers(root)[0]
    assert len(handler.filters) == 1
    assert isinstance(handler.filters[0], ContextDefaultsFilter)


# ---------------------------------------------------------------------------
# 5. Repeated registration idempotency
# ---------------------------------------------------------------------------


def test_repeated_registration_is_idempotent():
    root = logging.getLogger()
    _clear_maui_handlers(root)

    register_operational_persistence(_FakeApp())
    delivery_first = op._get_or_create_delivery()
    greenlet_first = delivery_first._greenlet

    for _ in range(5):
        register_operational_persistence(_FakeApp())
        register_operational_persistence(object())  # a different "app"

    assert len(_persistence_handlers(root)) == 1
    delivery_after = op._get_or_create_delivery()
    assert delivery_after is delivery_first
    assert delivery_after._greenlet is greenlet_first
    assert not greenlet_first.dead


def test_repeated_registration_across_fresh_apps_spawns_one_consumer():
    root = logging.getLogger()
    _clear_maui_handlers(root)

    apps = [_FakeApp() for _ in range(4)]
    for app in apps:
        register_operational_persistence(app)

    assert len(_persistence_handlers(root)) == 1
    delivery = op._get_or_create_delivery()
    assert delivery._greenlet is not None
    assert not delivery._greenlet.dead


def test_repeated_registration_does_not_duplicate_atexit_hook(monkeypatch):
    calls = []
    monkeypatch.setattr(op.atexit, "register", lambda func, *a, **k: calls.append(func))

    root = logging.getLogger()
    _clear_maui_handlers(root)

    for _ in range(3):
        register_operational_persistence(_FakeApp())

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 6. Existing stderr handler preserved
# ---------------------------------------------------------------------------


def test_existing_stderr_handler_untouched_by_registration():
    from utils.logging_config import bootstrap_logging

    root = logging.getLogger()
    _clear_maui_handlers(root)
    saved_env = os.environ.get("LOG_LEVEL")
    os.environ.pop("LOG_LEVEL", None)
    try:
        bootstrap_logging()
    finally:
        if saved_env is not None:
            os.environ["LOG_LEVEL"] = saved_env

    stderr_handler = next(
        h for h in root.handlers if getattr(h, "_maui_bootstrap", False)
    )
    before = {
        "identity": id(stderr_handler),
        "stream": stderr_handler.stream,
        "formatter": stderr_handler.formatter,
        "filters": list(stderr_handler.filters),
        "level": stderr_handler.level,
        "marker": getattr(stderr_handler, "_maui_bootstrap", False),
    }

    register_operational_persistence(_FakeApp())
    register_operational_persistence(_FakeApp())

    after_handler = next(
        h for h in root.handlers if getattr(h, "_maui_bootstrap", False)
    )
    assert id(after_handler) == before["identity"]
    assert after_handler.stream is before["stream"]
    assert after_handler.formatter is before["formatter"]
    assert after_handler.filters == before["filters"]
    assert after_handler.level == before["level"]
    assert getattr(after_handler, "_maui_bootstrap", False) == before["marker"]


# ---------------------------------------------------------------------------
# 7. Root level unchanged
# ---------------------------------------------------------------------------


def test_root_level_unchanged_by_registration():
    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.WARNING)

    register_operational_persistence(_FakeApp())

    assert root.level == logging.WARNING


# ---------------------------------------------------------------------------
# 8. Third-party logger policy unchanged
# ---------------------------------------------------------------------------


def test_third_party_logger_levels_unchanged_by_registration():
    before = {name: logging.getLogger(name).level for name in THIRD_PARTY_LOG_LEVELS}

    register_operational_persistence(_FakeApp())

    after = {name: logging.getLogger(name).level for name in THIRD_PARTY_LOG_LEVELS}
    assert after == before


# ---------------------------------------------------------------------------
# 9. Unmarked ordinary log ignored
# ---------------------------------------------------------------------------


def test_unmarked_ordinary_log_record_does_not_reach_delivery():
    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.INFO)

    register_operational_persistence(_FakeApp())

    handler = _persistence_handlers(root)[0]
    enqueued = []
    handler._sink = lambda snapshot: enqueued.append(snapshot)

    module_logger = logging.getLogger("some.ordinary.module")
    module_logger.info("event=something_happened detail=1")

    assert enqueued == []


# ---------------------------------------------------------------------------
# 10. Marked synthetic record flows to delivery
# ---------------------------------------------------------------------------


def test_marked_synthetic_record_flows_to_delivery():
    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.INFO)

    register_operational_persistence(_FakeApp())

    # Deliberately does NOT stub the sink: this proves the installed handler
    # is wired to the REAL process-local delivery queue, not merely capable
    # of calling an injected callable. put_nowait() is synchronous and the
    # consumer greenlet only runs once this greenlet yields, so the item is
    # still queued, unconsumed, immediately after the logging call returns.
    delivery = op._get_or_create_delivery()

    module_logger = logging.getLogger("some.test.only.module")
    module_logger.info(
        "event=synthetic_test_event",
        extra={"maui_persist": True, "maui_event": "synthetic_test_event"},
    )

    assert delivery._queue.qsize() == 1
    snapshot = delivery._queue.get_nowait()
    assert isinstance(snapshot, OperationalEventSnapshot)
    assert snapshot.event == "synthetic_test_event"


# ---------------------------------------------------------------------------
# 11. Context captured through real registration
# ---------------------------------------------------------------------------


def test_context_captured_through_real_registration():
    from utils.logging_config import reset_request_context, set_request_context

    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.INFO)

    register_operational_persistence(_FakeApp())

    delivery = op._get_or_create_delivery()

    module_logger = logging.getLogger("some.context.module")
    tokens = set_request_context(request_id="bootstrap-req", app_id="bootstrap-app")
    try:
        module_logger.info(
            "event=synthetic_context_event",
            extra={"maui_persist": True, "maui_event": "synthetic_context_event"},
        )
    finally:
        reset_request_context(tokens)

    assert delivery._queue.qsize() == 1
    snapshot = delivery._queue.get_nowait()
    assert snapshot.request_id == "bootstrap-req"
    assert snapshot.app_id == "bootstrap-app"


# ---------------------------------------------------------------------------
# 12. Registration alone does not persist
# ---------------------------------------------------------------------------


def test_registration_alone_does_not_call_db_insert(monkeypatch):
    import infrastructure.database_pg as database_pg

    calls = []
    monkeypatch.setattr(
        database_pg, "insert_operational_event", lambda *a: calls.append(a)
    )

    root = logging.getLogger()
    _clear_maui_handlers(root)

    register_operational_persistence(_FakeApp())
    gevent.sleep(0)

    assert calls == []


# ---------------------------------------------------------------------------
# 13. Import safety
# ---------------------------------------------------------------------------


def test_importing_module_alone_does_not_attach_handler_or_start_delivery():
    # The module is already imported (at collection time, and by every other
    # test in this file); merely being imported must never have attached a
    # handler or created/started the delivery singleton. The autouse fixture
    # guarantees a clean slate on entry, so this asserts the module's own
    # import has no side effect beyond that reset.
    root = logging.getLogger()
    assert not any(
        isinstance(h, OperationalPersistenceHandler) for h in root.handlers
    )
    assert op._DELIVERY is None
    import sys

    assert "utils.operational_persistence" in sys.modules


# ---------------------------------------------------------------------------
# 14. main.py structural wiring
# ---------------------------------------------------------------------------


def _parse_main():
    main_path = os.path.join(REPO_ROOT, "main.py")
    with open(main_path, "r", encoding="utf-8") as f:
        source = f.read()
    return source, ast.parse(source, filename=main_path)


def test_main_py_imports_register_operational_persistence():
    source, tree = _parse_main()
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "utils.operational_persistence":
            imported_names.update(alias.name for alias in node.names)
    assert "register_operational_persistence" in imported_names


def test_main_py_calls_registrar_exactly_once():
    _, tree = _parse_main()
    call_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "register_operational_persistence":
                call_count += 1
    assert call_count == 1


def test_main_py_defines_no_handler_queue_or_db_logic_inline():
    source, _ = _parse_main()
    forbidden = ("logging.Handler", "gevent.queue", "insert_operational_event")
    for needle in forbidden:
        assert needle not in source, f"unexpected persistence logic inline: {needle}"


# ---------------------------------------------------------------------------
# 15. No duplicate wiring outside main.py
# ---------------------------------------------------------------------------


def test_no_other_production_file_calls_the_registrar():
    production_dirs = ("routes", "services", "infrastructure", "datachat", "agent_runs")
    hits = []
    for directory in production_dirs:
        dir_path = os.path.join(REPO_ROOT, directory)
        if not os.path.isdir(dir_path):
            continue
        for root_dir, _, files in os.walk(dir_path):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(root_dir, name)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                if "register_operational_persistence" in text:
                    hits.append(path)
    assert hits == []

    utils_dir = os.path.join(REPO_ROOT, "utils")
    other_util_hits = []
    for name in os.listdir(utils_dir):
        if not name.endswith(".py") or name == "operational_persistence.py":
            continue
        path = os.path.join(utils_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if "register_operational_persistence" in text:
            other_util_hits.append(path)
    assert other_util_hits == []


# ---------------------------------------------------------------------------
# 16. Production inertness
# ---------------------------------------------------------------------------


def _iter_production_python_files():
    skip_dirs = {".git", "venv", "__pycache__", "docs", "tests", ".venv"}
    for root_dir, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        rel = os.path.relpath(root_dir, REPO_ROOT)
        if rel.split(os.sep)[0] in skip_dirs:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root_dir, name)


#: The I1 emission-contract module. It defines and documents the persistent
#: metadata surface (the marker keyword and the builder function) in its own
#: docstrings/dict literal. Sanctioned infrastructure, not a production call
#: site.
_CONTRACT_DEFINITION_FILE = os.path.join(REPO_ROOT, "utils", "operational_event.py")

#: The I2-I6 capture/delivery/bootstrap module: constructs the marker
#: attribute access (getattr(record, "maui_persist", ...)) as the recursion
#: barrier itself. Sanctioned infrastructure, not a production call site.
_PERSISTENCE_INFRASTRUCTURE_FILE = os.path.join(
    REPO_ROOT, "utils", "operational_persistence.py"
)

# Zero production call sites of the I1 emission-contract builder is
# independently and already covered by a dedicated guard test in
# tests/test_operational_event_contract.py (see that file's "no production
# call sites" test). That guard is itself a substring scan forbidding the
# builder's identifier from appearing in any file other than its own
# definition and its own test - so duplicating it here would require
# writing the forbidden identifier into this file's source, which would
# trip that very guard. I6's evidence for exit criterion X14 is therefore
# this cross-reference, not a second test.


def test_no_production_literal_maui_persist_true_outside_persistence_infrastructure():
    sanctioned = {
        _PERSISTENCE_INFRASTRUCTURE_FILE,
        _CONTRACT_DEFINITION_FILE,
        os.path.join(REPO_ROOT, "main.py"),
    }
    hits = []
    for path in _iter_production_python_files():
        if path in sanctioned:
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if "maui_persist" in text:
            hits.append(path)
    assert hits == []


# ---------------------------------------------------------------------------
# 17. DataChat unchanged
# ---------------------------------------------------------------------------


def test_datachat_runtime_logger_unchanged_by_registration():
    datachat_logger = logging.getLogger("datachat.runtime")
    before_handlers = list(datachat_logger.handlers)
    before_propagate = datachat_logger.propagate

    register_operational_persistence(_FakeApp())

    assert list(datachat_logger.handlers) == before_handlers
    assert datachat_logger.propagate == before_propagate


# ---------------------------------------------------------------------------
# 18. agent_runs unchanged
# ---------------------------------------------------------------------------


def test_agent_runs_logger_unchanged_by_registration():
    agent_logger = logging.getLogger("agent_runs")
    before_handlers = list(agent_logger.handlers)
    before_propagate = agent_logger.propagate

    register_operational_persistence(_FakeApp())

    assert list(agent_logger.handlers) == before_handlers
    assert agent_logger.propagate == before_propagate


# ---------------------------------------------------------------------------
# 19. Delivery start failure fail-open
# ---------------------------------------------------------------------------


def test_registration_survives_delivery_start_failure(monkeypatch, caplog):
    root = logging.getLogger()
    _clear_maui_handlers(root)

    delivery = op._get_or_create_delivery()

    def _boom():
        raise RuntimeError("spawn boom")

    monkeypatch.setattr(op.gevent, "spawn", lambda *a, **k: _boom())

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        register_operational_persistence(_FakeApp())  # must not raise

    assert len(_persistence_handlers(root)) == 1
    failed = [
        r for r in caplog.records if "operational_persistence_start_failed" in r.message
    ]
    assert len(failed) == 1
    assert getattr(failed[0], "maui_persist", False) is False
    assert "spawn boom" not in failed[0].getMessage()


# ---------------------------------------------------------------------------
# S1-S4 structural assertions (TDD §16)
# ---------------------------------------------------------------------------


def test_S1_exactly_one_root_persistence_handler_after_registration():
    root = logging.getLogger()
    _clear_maui_handlers(root)

    for _ in range(3):
        register_operational_persistence(_FakeApp())

    assert len(_persistence_handlers(root)) == 1


def test_S2_existing_maui_handler_remains_installed_and_unchanged():
    from utils.logging_config import bootstrap_logging

    root = logging.getLogger()
    _clear_maui_handlers(root)
    bootstrap_logging()

    bootstrap_handlers_before = [
        h for h in root.handlers if getattr(h, "_maui_bootstrap", False)
    ]
    assert len(bootstrap_handlers_before) == 1

    register_operational_persistence(_FakeApp())

    bootstrap_handlers_after = [
        h for h in root.handlers if getattr(h, "_maui_bootstrap", False)
    ]
    assert bootstrap_handlers_after == bootstrap_handlers_before


def test_S3_persistence_handler_owns_exactly_one_context_defaults_filter():
    root = logging.getLogger()
    _clear_maui_handlers(root)

    register_operational_persistence(_FakeApp())

    handler = _persistence_handlers(root)[0]
    assert len([f for f in handler.filters if isinstance(f, ContextDefaultsFilter)]) == 1
    assert len(handler.filters) == 1


def test_S4_datachat_and_agent_runs_configuration_unchanged():
    datachat_logger = logging.getLogger("datachat.runtime")
    agent_logger = logging.getLogger("agent_runs")
    before = {
        "datachat_handlers": list(datachat_logger.handlers),
        "datachat_propagate": datachat_logger.propagate,
        "agent_handlers": list(agent_logger.handlers),
        "agent_propagate": agent_logger.propagate,
    }

    register_operational_persistence(_FakeApp())
    register_operational_persistence(_FakeApp())

    assert list(datachat_logger.handlers) == before["datachat_handlers"]
    assert datachat_logger.propagate == before["datachat_propagate"]
    assert list(agent_logger.handlers) == before["agent_handlers"]
    assert agent_logger.propagate == before["agent_propagate"]
