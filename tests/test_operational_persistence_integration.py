"""
Cross-component integration gates for Operational Log Persistence —
FOUNDATION INTERVENTION I7.

This file proves properties that only hold across the REAL, installed
components together (handler + delivery + consumer + writer + Flask), as
distinguished from the per-component unit tests in
test_operational_event_contract.py, test_operational_persistence_handler.py,
test_operational_persistence_delivery.py and
test_operational_persistence_bootstrap.py. It intentionally does not
duplicate those files' unit-level assertions.

Scope, per docs/logging/operational_persistence_foundation_tdd.md section
17.6 and section 19 (X1, X5, X6, X11 partial):

  E1  the real marker barrier, exercised against unmarked database_pg/
      psycopg-named records;
  E2  the real public insert_operational_event() writer path, forced to
      fail at the connect() boundary, proving containment and no recursive
      second attempt;
  E3  fail-open with respect to a real Flask request/response cycle;
  E4  one integration sanity gate for existing runtime surfaces
      (stderr handler, root level, agent_runs, datachat.runtime) across a
      real registration call.

No production call site is added anywhere by this file. All marked events
here are test-only synthetic events constructed with a raw, literal
`extra={"maui_persist": True, "maui_event": "..."}` mapping, exactly like
test_operational_persistence_bootstrap.py's existing synthetic-event tests -
deliberately not via the emission-contract builder in utils/operational_event.py,
so this file adds no new site against the I1 ratchet asserting zero such call
sites outside that module (see test_operational_event_contract.py).
"""

import logging
import time

import gevent
import pytest

import utils.operational_persistence as op
from infrastructure import database_pg
from utils.logging_config import register_request_context_hooks
from utils.operational_persistence import register_operational_persistence

logger = logging.getLogger(__name__)

_SETTLE_TIMEOUT_SECONDS = 2.0


class _FakeApp:
    """Minimal stand-in: the registrar must not depend on a real Flask app."""


@pytest.fixture(autouse=True)
def _isolated_root_and_delivery():
    """Isolate root.handlers, root.level and the process-local delivery
    singleton across tests in this module, mirroring
    tests/test_operational_persistence_bootstrap.py's fixture."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    op._reset_delivery_for_tests()
    yield
    op._reset_delivery_for_tests()

    root.handlers = saved_handlers
    root.level = saved_level


def _clear_maui_handlers(root):
    root.handlers = [
        h
        for h in root.handlers
        if not getattr(h, "_maui_bootstrap", False)
        and not getattr(h, "_maui_operational_persistence", False)
    ]


def _persistence_handlers(root):
    return [
        h for h in root.handlers if getattr(h, "_maui_operational_persistence", False)
    ]


def _settle(predicate, timeout=_SETTLE_TIMEOUT_SECONDS):
    """Yield to the real gevent consumer greenlet until predicate() is true
    or the timeout elapses. Uses wall-clock + gevent.sleep, matching the
    style already used by tests/test_operational_persistence_delivery.py."""
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        gevent.sleep(0.01)


# ---------------------------------------------------------------------------
# E1 — real marker barrier / recursion shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("logger_name", ["infrastructure.database_pg", "psycopg"])
def test_e1_unmarked_database_and_psycopg_records_stop_at_the_barrier(logger_name):
    """An unmarked record from either recursion-hazard logger name never
    reaches the sink: no snapshot is enqueued, so no DB insert can follow
    and no recursive persistence path can start (ARCH V1, TDD U13/U14, this
    file's E1)."""
    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.INFO)

    register_operational_persistence(_FakeApp())
    delivery = op._get_or_create_delivery()

    target_logger = logging.getLogger(logger_name)
    target_logger.warning(
        "event=simulated_unmarked_diagnostic error_type=OperationalError"
    )
    gevent.sleep(0)

    assert delivery._queue.qsize() == 0


# ---------------------------------------------------------------------------
# E2 — actual failing DB writer path does not recurse (load-bearing)
# ---------------------------------------------------------------------------


def test_e2_real_writer_failure_is_contained_with_no_recursive_attempt(
    monkeypatch, caplog
):
    """Exercises the REAL public infrastructure.database_pg.insert_operational_event
    path: logger -> installed handler -> delivery queue -> real consumer
    greenlet -> insert_operational_event() -> database_pg.connect().

    Only database_pg.connect() is monkeypatched to raise deterministically -
    the lowest practical seam that forces a genuine failure at/under
    connect(), while _insert_operational_event(), insert_operational_event()
    and the consumer's _write() all run unmodified. This mirrors the seam
    tests/test_database_operational_events_writer.py already uses for the
    same function (database_pg.connect), so the writer is never replaced by
    a fake for this test.
    """
    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.INFO)

    register_operational_persistence(_FakeApp())
    delivery = op._get_or_create_delivery()

    def _raising_connect():
        raise RuntimeError("simulated connection failure (E2)")

    monkeypatch.setattr(database_pg, "connect", _raising_connect)

    caplog.set_level(logging.WARNING, logger="utils.operational_persistence")

    module_logger = logging.getLogger("tests.integration.e2_synthetic")
    module_logger.info(
        "event=synthetic_e2_event",
        extra={"maui_persist": True, "maui_event": "synthetic_e2_event"},
    )

    # Exactly one item should have been enqueued by this single emission.
    assert delivery._queue.qsize() == 1

    # Let the real consumer greenlet actually dequeue and attempt the write.
    _settle(lambda: delivery._queue.qsize() == 0)
    assert delivery._queue.qsize() == 0

    # The consumer survives the failure: it is the same greenlet, alive.
    assert delivery._greenlet is not None
    assert not delivery._greenlet.dead

    # Exactly one write-failure diagnostic, and it is unmarked/runtime-only.
    write_failed_records = [
        r
        for r in caplog.records
        if "operational_persistence_write_failed" in r.getMessage()
    ]
    assert len(write_failed_records) == 1
    for record in write_failed_records:
        assert getattr(record, "maui_persist", False) is False

    # No recursive second attempt: the queue stays empty once settled, and
    # no further write-failure diagnostics accumulate from the same event.
    gevent.sleep(0.1)
    assert delivery._queue.qsize() == 0
    write_failed_records_after_settle = [
        r
        for r in caplog.records
        if "operational_persistence_write_failed" in r.getMessage()
    ]
    assert len(write_failed_records_after_settle) == 1

    # The consumer loop is still alive and functional: a second, independent
    # event is still attempted (proves the loop did not silently die).
    module_logger.info(
        "event=synthetic_e2_event_after",
        extra={"maui_persist": True, "maui_event": "synthetic_e2_event_after"},
    )
    _settle(lambda: delivery._queue.qsize() == 0)
    assert delivery._queue.qsize() == 0
    write_failed_records_final = [
        r
        for r in caplog.records
        if "operational_persistence_write_failed" in r.getMessage()
    ]
    assert len(write_failed_records_final) == 2


# ---------------------------------------------------------------------------
# E3 — fail-open relative to Flask/request execution
# ---------------------------------------------------------------------------


def test_e3_persistence_write_failure_does_not_affect_http_response(monkeypatch):
    """A synthetic isolated Flask route emits a marked event through normal
    logging semantics while the real writer path is forced to fail; the
    HTTP response must remain successful and no exception may reach Flask
    (ARCH V8, TDD E2)."""
    from flask import Flask

    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.INFO)

    app = Flask(__name__)
    register_request_context_hooks(app)
    register_operational_persistence(app)

    def _raising_connect():
        raise RuntimeError("simulated connection failure (E3)")

    monkeypatch.setattr(database_pg, "connect", _raising_connect)

    route_logger = logging.getLogger("tests.integration.e3_route")

    @app.route("/synthetic-operational-route")
    def synthetic_route():
        route_logger.info(
            "event=synthetic_e3_route_event",
            extra={"maui_persist": True, "maui_event": "synthetic_e3_route_event"},
        )
        return "ok", 200

    client = app.test_client()
    response = client.get("/synthetic-operational-route")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "ok"

    delivery = op._get_or_create_delivery()
    _settle(lambda: delivery._queue.qsize() == 0)
    assert delivery._queue.qsize() == 0


# ---------------------------------------------------------------------------
# E4 — existing runtime surfaces preserved (one integration sanity gate)
# ---------------------------------------------------------------------------


def test_e4_existing_runtime_surfaces_preserved_across_registration():
    """One combined sanity gate, not a duplicate of the I6 unit suite: root
    keeps its stderr-handler count and level, agent_runs and
    datachat.runtime are untouched, and registration yields exactly one
    persistence handler backed by exactly one live consumer greenlet."""
    root = logging.getLogger()
    _clear_maui_handlers(root)
    root.setLevel(logging.WARNING)

    stderr_handlers_before = [
        h for h in root.handlers if getattr(h, "_maui_bootstrap", False)
    ]
    level_before = root.level

    agent_runs_logger = logging.getLogger("agent_runs")
    datachat_logger = logging.getLogger("datachat.runtime")
    agent_handlers_before = list(agent_runs_logger.handlers)
    agent_propagate_before = agent_runs_logger.propagate
    datachat_handlers_before = list(datachat_logger.handlers)
    datachat_propagate_before = datachat_logger.propagate

    register_operational_persistence(_FakeApp())

    persistence_handlers = _persistence_handlers(root)
    assert len(persistence_handlers) == 1

    stderr_handlers_after = [
        h for h in root.handlers if getattr(h, "_maui_bootstrap", False)
    ]
    assert len(stderr_handlers_after) == len(stderr_handlers_before)
    assert root.level == level_before

    assert agent_runs_logger.handlers == agent_handlers_before
    assert agent_runs_logger.propagate == agent_propagate_before
    assert datachat_logger.handlers == datachat_handlers_before
    assert datachat_logger.propagate == datachat_propagate_before

    delivery = op._get_or_create_delivery()
    assert delivery._greenlet is not None
    assert not delivery._greenlet.dead
