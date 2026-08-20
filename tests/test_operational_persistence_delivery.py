"""
Delivery and lifecycle tests for utils/operational_persistence.py —
FOUNDATION INTERVENTION I5.

Scope: the bounded gevent queue, the single consumer greenlet, DB writer
invocation with contained failure, runtime-only diagnostics, and the
bounded start/stop lifecycle of ``_OperationalDelivery`` plus the
process-local ``_DELIVERY`` singleton. Not in scope: root attachment,
main.py wiring, or any production call-site instrumentation (I6/pilot).
"""

import ast
import logging
import os
from datetime import datetime, timezone

import gevent
import pytest
from gevent.queue import Full

import utils.operational_persistence as op
from utils.operational_persistence import (
    OperationalEventSnapshot,
    OperationalPersistenceHandler,
    _OperationalDelivery,
)


def _make_snapshot(**overrides):
    fields = dict(
        event_time=datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc),
        level="INFO",
        logger="services.flow",
        event="flow_started",
        request_id="req-1",
        app_id="app-1",
        provider=None,
        model=None,
        duration_ms=None,
        error_type=None,
        details_json=None,
        message=None,
    )
    fields.update(overrides)
    return OperationalEventSnapshot(**fields)


@pytest.fixture(autouse=True)
def _reset_delivery_singleton():
    """Test isolation seam: ensure each test starts with no process-local
    delivery singleton and no lingering atexit registration."""
    op._reset_delivery_for_tests()
    yield
    op._reset_delivery_for_tests()


# ---------------------------------------------------------------------------
# 1. Non-blocking enqueue success
# ---------------------------------------------------------------------------


def test_enqueue_succeeds_immediately_without_db():
    delivery = _OperationalDelivery()
    snapshot = _make_snapshot()

    delivery.enqueue(snapshot)

    assert delivery._queue.qsize() == 1
    assert delivery._queue.get_nowait() is snapshot


# ---------------------------------------------------------------------------
# 2. Queue full
# ---------------------------------------------------------------------------


def test_enqueue_on_full_queue_discards_without_blocking_or_raising():
    delivery = _OperationalDelivery()
    delivery._queue = gevent.queue.Queue(maxsize=1)
    delivery.enqueue(_make_snapshot())  # fills the queue

    with pytest.raises(Full):
        delivery._queue.put_nowait(_make_snapshot())  # sanity: queue really is full

    delivery.enqueue(_make_snapshot())  # must not raise, must not block

    assert delivery._queue.qsize() == 1  # second snapshot was discarded


def test_enqueue_on_full_queue_emits_unmarked_diagnostic(caplog):
    delivery = _OperationalDelivery()
    delivery._queue = gevent.queue.Queue(maxsize=1)
    delivery.enqueue(_make_snapshot())

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        delivery.enqueue(_make_snapshot())

    dropped = [r for r in caplog.records if "operational_persistence_event_dropped" in r.message]
    assert len(dropped) == 1
    for record in dropped:
        assert getattr(record, "maui_persist", False) is False


# ---------------------------------------------------------------------------
# 3. Drop damping
# ---------------------------------------------------------------------------


def test_drop_damping_first_emits_then_damps_then_emits_at_101st(caplog):
    delivery = _OperationalDelivery()
    delivery._queue = gevent.queue.Queue(maxsize=1)
    delivery.enqueue(_make_snapshot())  # fills queue, not a drop

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        for _ in range(101):
            delivery.enqueue(_make_snapshot())  # each one is a drop

    dropped = [
        r for r in caplog.records if "operational_persistence_event_dropped" in r.message
    ]
    # first drop (1) emits, then damped until the 101st additional drop.
    assert len(dropped) == 2


def test_drop_episode_resets_on_successful_enqueue(caplog):
    delivery = _OperationalDelivery()
    delivery._queue = gevent.queue.Queue(maxsize=1)
    delivery.enqueue(_make_snapshot())  # fills queue

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        delivery.enqueue(_make_snapshot())  # drop 1: emits
        delivery._queue.get_nowait()  # drain -> queue not full anymore
        delivery.enqueue(_make_snapshot())  # succeeds: episode resets

    dropped_before_reset = [
        r for r in caplog.records if "operational_persistence_event_dropped" in r.message
    ]
    assert len(dropped_before_reset) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        delivery.enqueue(_make_snapshot())  # fills again
        delivery.enqueue(_make_snapshot())  # new episode's first drop: emits

    dropped_after_reset = [
        r for r in caplog.records if "operational_persistence_event_dropped" in r.message
    ]
    assert len(dropped_after_reset) == 1


# ---------------------------------------------------------------------------
# 4. Consumer writes once
# ---------------------------------------------------------------------------


def test_consumer_writes_snapshot_exactly_once_with_scalar_fields(monkeypatch):
    calls = []

    def _fake_insert(*args):
        calls.append(args)

    import infrastructure.database_pg as database_pg

    monkeypatch.setattr(database_pg, "insert_operational_event", _fake_insert)

    delivery = _OperationalDelivery()
    delivery.start()
    try:
        snapshot = _make_snapshot(
            provider="DeepInfra",
            model="m1",
            duration_ms=7,
            error_type="TimeoutError",
            details_json='{"a": 1}',
            message="hello",
        )
        delivery.enqueue(snapshot)

        gevent.sleep(0)
        with gevent.Timeout(2, False):
            while not calls:
                gevent.sleep(0.01)
    finally:
        delivery.stop()

    assert len(calls) == 1
    assert calls[0] == (
        snapshot.event_time,
        snapshot.level,
        snapshot.logger,
        snapshot.event,
        snapshot.request_id,
        snapshot.app_id,
        snapshot.provider,
        snapshot.model,
        snapshot.duration_ms,
        snapshot.error_type,
        snapshot.details_json,
        snapshot.message,
    )
    for arg in calls[0]:
        assert not isinstance(arg, OperationalEventSnapshot)


# ---------------------------------------------------------------------------
# 5. Writer failure does not kill consumer
# ---------------------------------------------------------------------------


def test_writer_failure_does_not_kill_consumer_second_item_still_attempted(
    monkeypatch, caplog
):
    calls = []

    def _fake_insert(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("boom-secret-text")

    import infrastructure.database_pg as database_pg

    monkeypatch.setattr(database_pg, "insert_operational_event", _fake_insert)

    delivery = _OperationalDelivery()
    delivery.start()
    try:
        with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
            delivery.enqueue(_make_snapshot(event="first"))
            with gevent.Timeout(2, False):
                while len(calls) < 1:
                    gevent.sleep(0.01)

            delivery.enqueue(_make_snapshot(event="second"))
            with gevent.Timeout(2, False):
                while len(calls) < 2:
                    gevent.sleep(0.01)
    finally:
        delivery.stop()

    assert len(calls) == 2
    failed = [
        r for r in caplog.records if "operational_persistence_write_failed" in r.message
    ]
    assert len(failed) == 1
    assert getattr(failed[0], "maui_persist", False) is False
    assert "boom-secret-text" not in failed[0].getMessage()


# ---------------------------------------------------------------------------
# 6. Consumer loop is gevent-cooperative
# ---------------------------------------------------------------------------


def test_consumer_loop_is_gevent_cooperative():
    delivery = _OperationalDelivery()
    delivery.start()
    other_ran = []

    def _other():
        other_ran.append(True)

    try:
        gevent.spawn(_other)
        with gevent.Timeout(2, False):
            while not other_ran:
                gevent.sleep(0.01)
    finally:
        delivery.stop()

    assert other_ran == [True]


# ---------------------------------------------------------------------------
# 7. One consumer per delivery
# ---------------------------------------------------------------------------


def test_repeated_start_produces_exactly_one_live_consumer_greenlet():
    delivery = _OperationalDelivery()
    delivery.start()
    first_greenlet = delivery._greenlet

    delivery.start()
    delivery.start()

    try:
        assert delivery._greenlet is first_greenlet
        assert not first_greenlet.dead
    finally:
        delivery.stop()


# ---------------------------------------------------------------------------
# 8. Module-level singleton semantics
# ---------------------------------------------------------------------------


def test_singleton_accessor_does_not_produce_duplicate_delivery_objects():
    first = op._get_or_create_delivery()
    second = op._get_or_create_delivery()

    assert first is second


def test_singleton_start_repeated_does_not_duplicate_queue_or_consumer():
    delivery = op._get_or_create_delivery()
    delivery.start()
    greenlet_before = delivery._greenlet
    queue_before = delivery._queue

    delivery2 = op._get_or_create_delivery()
    delivery2.start()

    try:
        assert delivery2 is delivery
        assert delivery2._greenlet is greenlet_before
        assert delivery2._queue is queue_before
    finally:
        delivery.stop()


# ---------------------------------------------------------------------------
# 9. Stop drains pending item
# ---------------------------------------------------------------------------


def test_stop_drains_pending_item_before_returning(monkeypatch):
    calls = []

    def _fake_insert(*args):
        calls.append(args)

    import infrastructure.database_pg as database_pg

    monkeypatch.setattr(database_pg, "insert_operational_event", _fake_insert)

    delivery = _OperationalDelivery()
    delivery.start()
    delivery.enqueue(_make_snapshot())

    delivery.stop()

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 10. Stop before start
# ---------------------------------------------------------------------------


def test_stop_before_start_is_a_noop():
    delivery = _OperationalDelivery()

    delivery.stop()  # must not raise


# ---------------------------------------------------------------------------
# 11. Stop idempotency
# ---------------------------------------------------------------------------


def test_stop_is_idempotent():
    delivery = _OperationalDelivery()
    delivery.start()

    delivery.stop()
    delivery.stop()
    delivery.stop()  # must not raise, must not re-drain


# ---------------------------------------------------------------------------
# 12. Hanging writer / bounded drain
# ---------------------------------------------------------------------------


def test_stop_returns_within_bound_against_hanging_writer(monkeypatch):
    import infrastructure.database_pg as database_pg

    def _hanging_insert(*args):
        while True:
            gevent.sleep(0.05)

    monkeypatch.setattr(database_pg, "insert_operational_event", _hanging_insert)
    monkeypatch.setenv("MAUI_OPERATIONAL_DRAIN_TIMEOUT_SECONDS", "0.3")

    delivery = _OperationalDelivery()
    delivery.start()
    delivery.enqueue(_make_snapshot())
    gevent.sleep(0)  # let the consumer pick it up and start hanging

    start = gevent.get_hub().loop.now()
    delivery.stop()
    elapsed = gevent.get_hub().loop.now() - start

    assert elapsed < 2.0  # bounded, not unbounded


# ---------------------------------------------------------------------------
# 13. Remaining item loss accepted / drain-timeout diagnostic
# ---------------------------------------------------------------------------


def test_stop_emits_drain_timeout_diagnostic_and_does_not_retry(monkeypatch, caplog):
    import infrastructure.database_pg as database_pg

    def _hanging_insert(*args):
        while True:
            gevent.sleep(0.05)

    monkeypatch.setattr(database_pg, "insert_operational_event", _hanging_insert)
    monkeypatch.setenv("MAUI_OPERATIONAL_DRAIN_TIMEOUT_SECONDS", "0.3")

    delivery = _OperationalDelivery()
    delivery.start()
    delivery.enqueue(_make_snapshot())
    gevent.sleep(0)

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        delivery.stop()

    timeout_diagnostics = [
        r for r in caplog.records if "operational_persistence_drain_timeout" in r.message
    ]
    assert len(timeout_diagnostics) == 1
    assert getattr(timeout_diagnostics[0], "maui_persist", False) is False


# ---------------------------------------------------------------------------
# 14. Runtime-only diagnostics unmarked
# ---------------------------------------------------------------------------


def test_all_diagnostic_families_are_unmarked(monkeypatch, caplog):
    import infrastructure.database_pg as database_pg

    def _failing_insert(*args):
        raise RuntimeError("db down")

    monkeypatch.setattr(database_pg, "insert_operational_event", _failing_insert)

    delivery = _OperationalDelivery()
    delivery._queue = gevent.queue.Queue(maxsize=1)

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        delivery.start()
        delivery.enqueue(_make_snapshot())  # fills queue
        gevent.sleep(0)
        delivery.enqueue(_make_snapshot())  # write_failed once consumed
        delivery.enqueue(_make_snapshot())  # possible drop
        gevent.sleep(0)
        delivery.stop()

    assert len(caplog.records) > 0
    for record in caplog.records:
        assert getattr(record, "maui_persist", False) is False


# ---------------------------------------------------------------------------
# 15. DB write uses scalars
# ---------------------------------------------------------------------------


def test_write_path_never_passes_snapshot_object(monkeypatch):
    received = []

    def _fake_insert(*args):
        received.extend(args)

    import infrastructure.database_pg as database_pg

    monkeypatch.setattr(database_pg, "insert_operational_event", _fake_insert)

    delivery = _OperationalDelivery()
    delivery.start()
    try:
        delivery.enqueue(_make_snapshot())
        with gevent.Timeout(2, False):
            while not received:
                gevent.sleep(0.01)
    finally:
        delivery.stop()

    assert len(received) == 12
    assert all(not isinstance(v, OperationalEventSnapshot) for v in received)


# ---------------------------------------------------------------------------
# 16. No raw exception content
# ---------------------------------------------------------------------------


def test_write_failure_diagnostic_has_error_type_not_raw_message(monkeypatch, caplog):
    import infrastructure.database_pg as database_pg

    distinctive = "super-secret-db-error-detail-xyz"

    def _failing_insert(*args):
        raise ValueError(distinctive)

    monkeypatch.setattr(database_pg, "insert_operational_event", _failing_insert)

    delivery = _OperationalDelivery()
    delivery.start()
    try:
        with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
            delivery.enqueue(_make_snapshot())
            with gevent.Timeout(2, False):
                while not any(
                    "operational_persistence_write_failed" in r.message
                    for r in caplog.records
                ):
                    gevent.sleep(0.01)
    finally:
        delivery.stop()

    failed = [
        r for r in caplog.records if "operational_persistence_write_failed" in r.message
    ]
    assert len(failed) >= 1
    for record in failed:
        message = record.getMessage()
        assert "ValueError" in message
        assert distinctive not in message


# ---------------------------------------------------------------------------
# 17. Consumer unexpected failure
# ---------------------------------------------------------------------------


def test_consumer_unexpected_failure_emits_diagnostic_and_does_not_propagate(
    monkeypatch, caplog
):
    delivery = _OperationalDelivery()

    class _BoomQueue:
        def get(self, timeout=None):
            raise RuntimeError("unexpected internal failure")

        def empty(self):
            return True

    delivery._queue = _BoomQueue()

    with caplog.at_level(logging.ERROR, logger="utils.operational_persistence"):
        greenlet = gevent.spawn(delivery._consume_loop)
        greenlet.join(timeout=2)

    assert greenlet.dead
    assert greenlet.exception is None  # failure did not propagate out of the greenlet

    failed = [
        r for r in caplog.records if "operational_persistence_consumer_failed" in r.message
    ]
    assert len(failed) == 1
    assert getattr(failed[0], "maui_persist", False) is False


# ---------------------------------------------------------------------------
# 18. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults_when_no_env_vars(monkeypatch):
    monkeypatch.delenv("MAUI_OPERATIONAL_QUEUE_MAXSIZE", raising=False)
    monkeypatch.delenv("MAUI_OPERATIONAL_DRAIN_TIMEOUT_SECONDS", raising=False)

    delivery = _OperationalDelivery()

    assert delivery._queue.maxsize == 1000
    assert delivery._drain_timeout == 2.0


# ---------------------------------------------------------------------------
# 19. Config valid overrides
# ---------------------------------------------------------------------------


def test_config_valid_overrides_are_honored(monkeypatch):
    monkeypatch.setenv("MAUI_OPERATIONAL_QUEUE_MAXSIZE", "50")
    monkeypatch.setenv("MAUI_OPERATIONAL_DRAIN_TIMEOUT_SECONDS", "5.5")

    delivery = _OperationalDelivery()

    assert delivery._queue.maxsize == 50
    assert delivery._drain_timeout == 5.5


# ---------------------------------------------------------------------------
# 20. Config invalid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["0", "-5", "not-a-number", ""])
def test_config_invalid_queue_maxsize_falls_back_safely(monkeypatch, bad_value, caplog):
    monkeypatch.setenv("MAUI_OPERATIONAL_QUEUE_MAXSIZE", bad_value)

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        delivery = _OperationalDelivery()

    assert delivery._queue.maxsize == 1000
    for record in caplog.records:
        assert getattr(record, "maui_persist", False) is False


@pytest.mark.parametrize("bad_value", ["0", "-1.5", "not-a-number", ""])
def test_config_invalid_drain_timeout_falls_back_safely(monkeypatch, bad_value, caplog):
    monkeypatch.setenv("MAUI_OPERATIONAL_DRAIN_TIMEOUT_SECONDS", bad_value)

    with caplog.at_level(logging.WARNING, logger="utils.operational_persistence"):
        delivery = _OperationalDelivery()

    assert delivery._drain_timeout == 2.0
    for record in caplog.records:
        assert getattr(record, "maui_persist", False) is False


# ---------------------------------------------------------------------------
# 21. Atexit registration idempotency
# ---------------------------------------------------------------------------


def test_atexit_registered_once_across_repeated_starts(monkeypatch):
    calls = []

    def _fake_register(func, *args, **kwargs):
        calls.append(func)

    monkeypatch.setattr(op.atexit, "register", _fake_register)

    delivery = op._get_or_create_delivery()
    delivery.start()
    delivery.start()

    delivery2 = _OperationalDelivery()
    op._DELIVERY = delivery2
    delivery2.start()

    try:
        assert len(calls) == 1
    finally:
        delivery.stop()
        delivery2.stop()


# ---------------------------------------------------------------------------
# 22. No root attachment
# ---------------------------------------------------------------------------


def test_no_root_operational_persistence_handler_after_start_stop():
    delivery = op._get_or_create_delivery()
    delivery.start()
    delivery.stop()

    root = logging.getLogger()
    assert not any(
        isinstance(h, OperationalPersistenceHandler) for h in root.handlers
    )


# ---------------------------------------------------------------------------
# 23. No main.py / registrar
# ---------------------------------------------------------------------------


def test_no_registrar_function_defined_in_module():
    source_text = __import__("pathlib").Path(op.__file__).read_text()
    assert "def register_operational_persistence" not in source_text


def test_main_py_unchanged_no_operational_persistence_reference():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(repo_root, "main.py")
    with open(main_path, "r", encoding="utf-8") as f:
        source = f.read()

    assert "operational_persistence" not in source
    assert "register_operational_persistence" not in source


# ---------------------------------------------------------------------------
# 24. No later features
# ---------------------------------------------------------------------------


def test_module_has_no_retry_batch_pool_broker_or_metrics_constructs():
    source_text = __import__("pathlib").Path(op.__file__).read_text().lower()

    forbidden = (
        "retry",
        "batch",
        "connectionpool",
        "broker",
        "spool",
        "prometheus",
        "/metrics",
    )
    for needle in forbidden:
        assert needle not in source_text, f"unexpected later-feature construct: {needle}"
