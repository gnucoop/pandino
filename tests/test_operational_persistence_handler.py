"""
Snapshot and capture-handler tests for utils/operational_persistence.py —
FOUNDATION INTERVENTIONS I3 (snapshot/normalization) and I4 (capture
handler).

I3 scope: OperationalEventSnapshot and snapshot_from_record.
I4 scope: OperationalPersistenceHandler — marker-first selection, its own
ContextDefaultsFilter, and handing valid snapshots to an injected sink. No
queue, no gevent consumer, no lifecycle, no DB and no root production
attachment — those belong to later interventions.
"""

import ast
import json
import logging
from dataclasses import FrozenInstanceError
from datetime import timezone

import pytest

import utils.operational_persistence as op
from utils.logging_config import (
    ContextDefaultsFilter,
    reset_request_context,
    set_request_context,
)
from utils.operational_persistence import (
    OperationalEventSnapshot,
    OperationalPersistenceHandler,
    snapshot_from_record,
)

logger = logging.getLogger(__name__)


def _make_record(
    *,
    name="some.module",
    level=logging.INFO,
    msg="event=text_only_name irrelevant_to_snapshot",
    args=(),
    exc_info=None,
    extra=None,
):
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


# ---------------------------------------------------------------------------
# 1. Complete snapshot mapping
# ---------------------------------------------------------------------------


def test_complete_snapshot_mapping():
    record = _make_record(
        name="services.flow",
        level=logging.WARNING,
        extra={
            "maui_event": "provider_call_failed",
            "request_id": "abc123",
            "app_id": "app-1",
            "maui_provider": "DeepInfra",
            "maui_model": "some-model",
            "maui_duration_ms": 42,
            "maui_error_type": "TimeoutError",
            "maui_details": {"b": 2, "a": 1},
            "maui_message": "Provider did not respond in time",
        },
    )

    snapshot = snapshot_from_record(record)

    assert snapshot is not None
    assert snapshot.level == "WARNING"
    assert snapshot.logger == "services.flow"
    assert snapshot.event == "provider_call_failed"
    assert snapshot.request_id == "abc123"
    assert snapshot.app_id == "app-1"
    assert snapshot.provider == "DeepInfra"
    assert snapshot.model == "some-model"
    assert snapshot.duration_ms == 42
    assert snapshot.error_type == "TimeoutError"
    assert snapshot.details_json == json.dumps({"a": 1, "b": 2}, sort_keys=True)
    assert snapshot.message == "Provider did not respond in time"

    assert snapshot.event_time.tzinfo is timezone.utc
    from datetime import datetime

    expected = datetime.fromtimestamp(record.created, timezone.utc)
    assert snapshot.event_time == expected


# ---------------------------------------------------------------------------
# 2. Missing optional fields
# ---------------------------------------------------------------------------


def test_missing_optional_fields_become_none():
    record = _make_record(extra={"maui_event": "flow_started"})

    snapshot = snapshot_from_record(record)

    assert snapshot is not None
    assert snapshot.provider is None
    assert snapshot.model is None
    assert snapshot.duration_ms is None
    assert snapshot.error_type is None
    assert snapshot.details_json is None
    assert snapshot.message is None
    assert snapshot.request_id is None
    assert snapshot.app_id is None


# ---------------------------------------------------------------------------
# 3. Sentinel translation
# ---------------------------------------------------------------------------


def test_sentinel_request_and_app_id_become_none():
    record = _make_record(
        extra={"maui_event": "flow_started", "request_id": "-", "app_id": "-"}
    )

    snapshot = snapshot_from_record(record)

    assert snapshot.request_id is None
    assert snapshot.app_id is None


def test_real_request_and_app_id_survive_unchanged():
    record = _make_record(
        extra={
            "maui_event": "flow_started",
            "request_id": "real-request-id",
            "app_id": "real-app-id",
        }
    )

    snapshot = snapshot_from_record(record)

    assert snapshot.request_id == "real-request-id"
    assert snapshot.app_id == "real-app-id"


# ---------------------------------------------------------------------------
# 4. Structured event authority
# ---------------------------------------------------------------------------


def test_event_comes_only_from_maui_event_not_from_msg():
    record = _make_record(
        msg="event=text_says_this_name",
        extra={"maui_event": "structured_says_this_name"},
    )

    snapshot = snapshot_from_record(record)

    assert snapshot.event == "structured_says_this_name"


# ---------------------------------------------------------------------------
# 5. Missing event
# ---------------------------------------------------------------------------


def test_missing_maui_event_yields_none():
    record = _make_record()

    assert snapshot_from_record(record) is None


# ---------------------------------------------------------------------------
# 6. Invalid event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_event", [None, 123, "", []])
def test_invalid_maui_event_yields_none_without_raising(bad_event):
    record = _make_record(extra={"maui_event": bad_event})

    assert snapshot_from_record(record) is None


# ---------------------------------------------------------------------------
# 7. Details serialization
# ---------------------------------------------------------------------------


def test_details_none_yields_none():
    record = _make_record(extra={"maui_event": "e", "maui_details": None})
    assert snapshot_from_record(record).details_json is None


def test_details_empty_dict_yields_none():
    record = _make_record(extra={"maui_event": "e", "maui_details": {}})
    assert snapshot_from_record(record).details_json is None


def test_details_nonempty_dict_yields_deterministic_sorted_json():
    record = _make_record(
        extra={"maui_event": "e", "maui_details": {"b": 2, "a": 1}}
    )
    snapshot = snapshot_from_record(record)
    assert snapshot.details_json == '{"a": 1, "b": 2}'


# ---------------------------------------------------------------------------
# 8. Detachment
# ---------------------------------------------------------------------------


def test_snapshot_details_detached_from_caller_mutation():
    details = {"a": 1, "b": "x"}
    record = _make_record(extra={"maui_event": "e", "maui_details": details})

    snapshot = snapshot_from_record(record)
    before = snapshot.details_json

    details["a"] = 999
    details["new_key"] = "mutated"

    assert snapshot.details_json == before
    assert "999" not in snapshot.details_json
    assert "mutated" not in snapshot.details_json


# ---------------------------------------------------------------------------
# 9. Serialization failure
# ---------------------------------------------------------------------------


class _Unserializable:
    pass


def test_unserializable_details_drops_details_but_keeps_event():
    record = _make_record(
        extra={"maui_event": "e", "maui_details": {"bad": _Unserializable()}}
    )

    snapshot = snapshot_from_record(record)

    assert snapshot is not None
    assert snapshot.event == "e"
    assert snapshot.details_json is None


# ---------------------------------------------------------------------------
# 10. record.getMessage() never called
# ---------------------------------------------------------------------------


def test_get_message_never_invoked():
    record = _make_record(extra={"maui_event": "e"})

    def _boom():
        raise AssertionError("record.getMessage() must never be called")

    record.getMessage = _boom

    snapshot = snapshot_from_record(record)

    assert snapshot is not None


def test_snapshot_survives_record_whose_message_formatting_would_raise():
    record = _make_record(
        msg="event=e value=%s",
        args=(object(),),
        extra={"maui_event": "e"},
    )

    snapshot = snapshot_from_record(record)

    assert snapshot is not None
    assert snapshot.event == "e"


# ---------------------------------------------------------------------------
# 11. exc_info ignored
# ---------------------------------------------------------------------------


def test_exc_info_is_ignored():
    try:
        raise ValueError("boom-secret-traceback-text")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = _make_record(
        extra={"maui_event": "e", "maui_message": "safe message"},
        exc_info=exc_info,
    )

    snapshot = snapshot_from_record(record)

    assert snapshot is not None
    for field_name in (
        "level",
        "logger",
        "event",
        "request_id",
        "app_id",
        "provider",
        "model",
        "error_type",
        "details_json",
        "message",
    ):
        value = getattr(snapshot, field_name)
        if value is not None:
            assert "boom-secret-traceback-text" not in str(value)
            assert "ValueError" not in str(value)
    assert not hasattr(snapshot, "exc_info")
    assert not hasattr(snapshot, "traceback")


# ---------------------------------------------------------------------------
# 12. Frozen snapshot
# ---------------------------------------------------------------------------


def test_snapshot_is_frozen():
    record = _make_record(extra={"maui_event": "e"})
    snapshot = snapshot_from_record(record)

    with pytest.raises(FrozenInstanceError):
        snapshot.event = "other"


# ---------------------------------------------------------------------------
# 13. slots
# ---------------------------------------------------------------------------


def test_snapshot_has_no_dict_and_rejects_new_attributes():
    record = _make_record(extra={"maui_event": "e"})
    snapshot = snapshot_from_record(record)

    assert not hasattr(snapshot, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        snapshot.new_attribute = "value"


# ---------------------------------------------------------------------------
# 14. No marker responsibility
# ---------------------------------------------------------------------------


def test_snapshot_from_record_ignores_persistence_marker():
    record = _make_record(extra={"maui_event": "e"})

    assert not hasattr(record, "maui_persist")
    snapshot = snapshot_from_record(record)

    assert snapshot is not None
    assert snapshot.event == "e"


# ---------------------------------------------------------------------------
# 15. No subsystem logging / I/O
# ---------------------------------------------------------------------------


def test_module_emits_no_log_records(caplog):
    with caplog.at_level(logging.DEBUG):
        record = _make_record(
            extra={"maui_event": "e", "maui_details": {"bad": _Unserializable()}}
        )
        snapshot_from_record(record)
        snapshot_from_record(_make_record())

    assert caplog.records == []


def test_module_imports_no_infrastructure_gevent_or_flask():
    source = ast.parse(
        __import__("pathlib").Path(op.__file__).read_text(), filename=op.__file__
    )
    forbidden_prefixes = ("infrastructure", "gevent", "flask", "database_pg")
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith(forbidden_prefixes), (
                f"unexpected import: {name}"
            )


def test_module_has_no_forbidden_later_intervention_constructs():
    """I4 introduces OperationalPersistenceHandler; it must not introduce
    anything belonging to I5 (queue/gevent/lifecycle) or I6 (registrar/
    root wiring)."""
    source_text = __import__("pathlib").Path(op.__file__).read_text()
    tree = ast.parse(source_text, filename=op.__file__)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith(("gevent", "flask", "infrastructure")), (
                f"unexpected import: {name}"
            )
            assert "database_pg" not in name

    forbidden_substrings = (
        "register_operational_persistence",
        "gevent.queue",
        "gevent.spawn",
        "atexit",
        "def start(",
        "def stop(",
    )
    for needle in forbidden_substrings:
        assert needle not in source_text, f"unexpected construct: {needle}"


# ===========================================================================
# FOUNDATION INTERVENTION I4 — OperationalPersistenceHandler
# ===========================================================================


class _ListSink:
    def __init__(self):
        self.received = []

    def __call__(self, snapshot):
        self.received.append(snapshot)


class _RaisingSink:
    def __init__(self, exc=None):
        self.calls = 0
        self._exc = exc or RuntimeError("sink boom")

    def __call__(self, snapshot):
        self.calls += 1
        raise self._exc


def _marked_record(**overrides):
    extra = {"maui_persist": True, "maui_event": "flow_started"}
    extra.update(overrides.pop("extra", {}))
    return _make_record(extra=extra, **overrides)


# ---------------------------------------------------------------------------
# 1. Unmarked record ignored
# ---------------------------------------------------------------------------


def test_unmarked_record_ignored_no_sink_call():
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _make_record()  # no maui_persist at all

    assert not hasattr(record, "maui_persist")
    handler.emit(record)

    assert sink.received == []


def test_unmarked_record_does_not_invoke_snapshot_from_record(monkeypatch):
    calls = []
    monkeypatch.setattr(
        op, "snapshot_from_record", lambda r: calls.append(r) or None
    )
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _make_record()

    handler.emit(record)

    assert calls == []
    assert sink.received == []


# ---------------------------------------------------------------------------
# 2. Explicit maui_persist=False ignored
# ---------------------------------------------------------------------------


def test_explicit_maui_persist_false_ignored():
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _make_record(extra={"maui_persist": False, "maui_event": "flow_started"})

    handler.emit(record)

    assert sink.received == []


# ---------------------------------------------------------------------------
# 3. Marked valid record captured
# ---------------------------------------------------------------------------


def test_marked_valid_record_captured_exactly_once():
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _marked_record(
        name="services.flow",
        level=logging.WARNING,
        extra={
            "maui_provider": "DeepInfra",
            "maui_duration_ms": 7,
            "request_id": "req-1",
            "app_id": "app-1",
        },
    )

    handler.emit(record)

    assert len(sink.received) == 1
    snapshot = sink.received[0]
    assert isinstance(snapshot, OperationalEventSnapshot)
    assert snapshot.event == "flow_started"
    assert snapshot.level == "WARNING"
    assert snapshot.logger == "services.flow"
    assert snapshot.provider == "DeepInfra"
    assert snapshot.duration_ms == 7
    assert snapshot.request_id == "req-1"
    assert snapshot.app_id == "app-1"


# ---------------------------------------------------------------------------
# 4. Invalid snapshot skipped
# ---------------------------------------------------------------------------


def test_marked_record_with_missing_event_skips_sink():
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _make_record(extra={"maui_persist": True})  # no maui_event

    handler.emit(record)

    assert sink.received == []


def test_marked_record_with_invalid_event_skips_sink_without_raising():
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _make_record(extra={"maui_persist": True, "maui_event": ""})

    handler.emit(record)

    assert sink.received == []


# ---------------------------------------------------------------------------
# 5. Sink failure contained
# ---------------------------------------------------------------------------


def test_sink_failure_does_not_propagate():
    sink = _RaisingSink()
    handler = OperationalPersistenceHandler(sink)
    record = _marked_record()

    handler.emit(record)  # must not raise

    assert sink.calls == 1


def test_sink_failure_is_not_retried():
    sink = _RaisingSink()
    handler = OperationalPersistenceHandler(sink)
    record = _marked_record()

    handler.emit(record)

    assert sink.calls == 1  # exactly once, no internal retry


# ---------------------------------------------------------------------------
# 6. Snapshot failure contained
# ---------------------------------------------------------------------------


def test_snapshot_from_record_failure_is_contained(monkeypatch):
    def _boom(record):
        raise RuntimeError("normalization boom")

    monkeypatch.setattr(op, "snapshot_from_record", _boom)
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _marked_record()

    handler.emit(record)  # must not raise

    assert sink.received == []


# ---------------------------------------------------------------------------
# 7. Own ContextDefaultsFilter
# ---------------------------------------------------------------------------


def test_handler_has_exactly_one_context_defaults_filter():
    handler = OperationalPersistenceHandler(_ListSink())

    assert len(handler.filters) == 1
    assert isinstance(handler.filters[0], ContextDefaultsFilter)


# ---------------------------------------------------------------------------
# 8. Context enrichment works
# ---------------------------------------------------------------------------


def test_context_enrichment_populates_request_and_app_id():
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    test_logger = logging.getLogger("test_operational_persistence.context")
    test_logger.propagate = False
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.INFO)

    tokens = set_request_context(request_id="ctx-request", app_id="ctx-app")
    try:
        test_logger.info(
            "event=flow_started",
            extra={"maui_persist": True, "maui_event": "flow_started"},
        )
    finally:
        reset_request_context(tokens)
        test_logger.handlers = []

    assert len(sink.received) == 1
    snapshot = sink.received[0]
    assert snapshot.request_id == "ctx-request"
    assert snapshot.app_id == "ctx-app"


# ---------------------------------------------------------------------------
# 9. Handler order independence
# ---------------------------------------------------------------------------


def test_handler_order_does_not_affect_captured_context():
    def _capture_with_order(persistence_first):
        sink = _ListSink()
        persistence_handler = OperationalPersistenceHandler(sink)
        stderr_handler = logging.StreamHandler()
        stderr_handler.addFilter(ContextDefaultsFilter())

        test_logger = logging.getLogger(
            f"test_operational_persistence.order.{persistence_first}"
        )
        test_logger.propagate = False
        test_logger.setLevel(logging.INFO)
        if persistence_first:
            test_logger.handlers = [persistence_handler, stderr_handler]
        else:
            test_logger.handlers = [stderr_handler, persistence_handler]

        tokens = set_request_context(request_id="order-request", app_id="order-app")
        try:
            test_logger.info(
                "event=flow_started",
                extra={"maui_persist": True, "maui_event": "flow_started"},
            )
        finally:
            reset_request_context(tokens)
            test_logger.handlers = []

        return sink.received[0]

    snapshot_first = _capture_with_order(persistence_first=True)
    snapshot_last = _capture_with_order(persistence_first=False)

    assert snapshot_first.request_id == snapshot_last.request_id == "order-request"
    assert snapshot_first.app_id == snapshot_last.app_id == "order-app"


# ---------------------------------------------------------------------------
# 10. Handler level
# ---------------------------------------------------------------------------


def test_handler_level_is_notset():
    handler = OperationalPersistenceHandler(_ListSink())

    assert handler.level == logging.NOTSET


# ---------------------------------------------------------------------------
# 11. Marker attribute
# ---------------------------------------------------------------------------


def test_handler_carries_operational_persistence_marker():
    handler = OperationalPersistenceHandler(_ListSink())

    assert handler._maui_operational_persistence is True


# ---------------------------------------------------------------------------
# 12. No message parsing
# ---------------------------------------------------------------------------


def test_selection_depends_only_on_marker_and_maui_event_not_msg():
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _marked_record(msg="totally unrelated text with no event= prefix")

    handler.emit(record)

    assert len(sink.received) == 1
    assert sink.received[0].event == "flow_started"


# ---------------------------------------------------------------------------
# 13. database_pg / third-party unmarked barrier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "logger_name", ["infrastructure.database_pg", "psycopg", "psycopg.connection"]
)
def test_unmarked_records_from_db_and_third_party_loggers_never_reach_sink(
    logger_name,
):
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _make_record(name=logger_name, msg="some warning during connect")

    handler.emit(record)

    assert sink.received == []


# ---------------------------------------------------------------------------
# 14. record with exc_info
# ---------------------------------------------------------------------------


def test_marked_record_with_exc_info_captured_without_traceback_data():
    try:
        raise ValueError("boom-secret-traceback-text")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    record = _marked_record(
        exc_info=exc_info, extra={"maui_message": "safe message"}
    )

    handler.emit(record)

    assert len(sink.received) == 1
    snapshot = sink.received[0]
    assert snapshot.message == "safe message"
    assert not hasattr(snapshot, "exc_info")


# ---------------------------------------------------------------------------
# 15. No root production mutation
# ---------------------------------------------------------------------------


def test_importing_module_adds_no_root_handler():
    root = logging.getLogger()
    assert not any(
        isinstance(h, OperationalPersistenceHandler) for h in root.handlers
    )


def test_constructing_handler_class_does_not_touch_root():
    root_handlers_before = list(logging.getLogger().handlers)

    OperationalPersistenceHandler(_ListSink())

    assert logging.getLogger().handlers == root_handlers_before


# ---------------------------------------------------------------------------
# 16. No forbidden imports / later components (handler-focused)
# ---------------------------------------------------------------------------


def test_handler_module_declares_no_registrar_or_lifecycle():
    source_text = __import__("pathlib").Path(op.__file__).read_text()

    assert "register_operational_persistence" not in source_text
    assert "class OperationalDelivery" not in source_text
    assert "import gevent" not in source_text
