"""
Snapshot model, capture handler and normalization boundary for persistent
Operational events.

This module owns the LogRecord -> OperationalEventSnapshot boundary (a
frozen, detached, self-contained representation of a marked LogRecord,
built eagerly on the producing greenlet) and the dedicated capture handler
that selects marked records and hands their snapshots to an injected sink.

It does not implement a delivery queue, a gevent consumer, a lifecycle or
any database access. Those belong to later interventions.
``snapshot_from_record`` works only with the LogRecord it is given: it does
not read request-context ContextVars, and does not call
``record.getMessage()`` or inspect ``record.msg``/``record.exc_info``.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from utils.logging_config import ContextDefaultsFilter

__all__ = [
    "OperationalEventSnapshot",
    "OperationalPersistenceHandler",
    "snapshot_from_record",
]

#: Runtime sentinel used by ContextDefaultsFilter for unset request/app
#: context. The persistent representation of "no known context" is None,
#: not this string.
_CONTEXT_UNSET = "-"


@dataclass(frozen=True, slots=True)
class OperationalEventSnapshot:
    event_time: datetime
    level: str
    logger: str
    event: str
    request_id: Optional[str]
    app_id: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    duration_ms: Optional[int]
    error_type: Optional[str]
    details_json: Optional[str]
    message: Optional[str]


def _normalize_context_value(value: Optional[str]) -> Optional[str]:
    if value is None or value == _CONTEXT_UNSET:
        return None
    return value


def _serialize_details(details: Optional[dict]) -> Optional[str]:
    if not details:
        return None
    try:
        return json.dumps(details, sort_keys=True)
    except (TypeError, ValueError):
        return None


def snapshot_from_record(record) -> Optional[OperationalEventSnapshot]:
    """Normalize a marked LogRecord into a detached OperationalEventSnapshot.

    Returns None, without raising, when ``maui_event`` is missing, not a
    str, or empty. Does not inspect the persistence marker: eligibility is
    the caller's responsibility, not this normalization primitive's.
    """
    event = getattr(record, "maui_event", None)
    if not isinstance(event, str) or not event:
        return None

    return OperationalEventSnapshot(
        event_time=datetime.fromtimestamp(record.created, timezone.utc),
        level=record.levelname,
        logger=record.name,
        event=event,
        request_id=_normalize_context_value(getattr(record, "request_id", None)),
        app_id=_normalize_context_value(getattr(record, "app_id", None)),
        provider=getattr(record, "maui_provider", None),
        model=getattr(record, "maui_model", None),
        duration_ms=getattr(record, "maui_duration_ms", None),
        error_type=getattr(record, "maui_error_type", None),
        details_json=_serialize_details(getattr(record, "maui_details", None)),
        message=getattr(record, "maui_message", None),
    )


class OperationalPersistenceHandler(logging.Handler):
    """Capture marked LogRecords and hand their snapshot to a sink.

    Marker-first barrier: ``emit()`` checks ``maui_persist`` before doing
    anything else, so every unmarked record - including every third-party
    and database-layer record - returns immediately with zero allocation
    and zero I/O. This is the recursion barrier described in the design; it
    must never be reordered behind normalization or sink work.

    Owns its own :class:`ContextDefaultsFilter` instance so request_id/app_id
    enrichment never depends on another handler (e.g. the stderr handler)
    having run first.

    The sink is a constructor-injected callable ``snapshot -> None`` - the
    seam a later intervention replaces with the real delivery boundary. This
    handler knows nothing about queues, gevent or the database.
    """

    def __init__(self, sink):
        super().__init__(level=logging.NOTSET)
        self._sink = sink
        self.addFilter(ContextDefaultsFilter())
        self._maui_operational_persistence = True

    def emit(self, record):
        if not getattr(record, "maui_persist", False):
            return

        try:
            snapshot = snapshot_from_record(record)
        except Exception:  # noqa: BLE001 - capture failures must never escape
            return

        if snapshot is None:
            return

        try:
            self._sink(snapshot)
        except Exception:  # noqa: BLE001 - sink failures must never escape
            return
