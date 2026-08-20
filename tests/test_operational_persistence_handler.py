"""
Snapshot / normalization tests for utils/operational_persistence.py —
FOUNDATION INTERVENTION I3.

Scope is deliberately narrow: only OperationalEventSnapshot and
snapshot_from_record. No Handler, no marker gate, no queue, no DB — those
belong to later interventions. Despite the filename (inherited from the
TDD's naming for the eventual handler test file), this file covers ONLY
snapshot behavior for I3.
"""

import ast
import json
import logging
from dataclasses import FrozenInstanceError
from datetime import timezone

import pytest

import utils.operational_persistence as op
from utils.operational_persistence import OperationalEventSnapshot, snapshot_from_record

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


def test_module_has_no_handler_subclass():
    source = op.__file__
    tree = ast.parse(__import__("pathlib").Path(source).read_text(), filename=source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = getattr(base, "attr", None) or getattr(base, "id", None)
                assert base_name != "Handler", (
                    "I3 must not implement a logging.Handler subclass"
                )
