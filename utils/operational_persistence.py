"""
Snapshot model, capture handler, delivery boundary and lifecycle for
persistent Operational events.

This module owns the LogRecord -> OperationalEventSnapshot boundary (a
frozen, detached, self-contained representation of a marked LogRecord,
built eagerly on the producing greenlet), the dedicated capture handler
that selects marked records and hands their snapshots to an injected sink,
and the process-local delivery boundary behind that sink: a bounded gevent
queue, one consumer greenlet, and a bounded best-effort start/stop
lifecycle.

``snapshot_from_record`` works only with the LogRecord it is given: it does
not read request-context ContextVars, and does not call
``record.getMessage()`` or inspect ``record.msg``/``record.exc_info``.

``register_operational_persistence(app)`` is the single production
attachment point: it starts the process-local delivery singleton and
attaches one ``OperationalPersistenceHandler`` to root, alongside the
existing stderr handler. ``_DELIVERY`` is a process-local singleton by
design: with N gunicorn workers there are N independent delivery
queues/consumers, all writing to the same PostgreSQL store. Only records
carrying the persistence marker travel this path; every other record -
third-party, database-layer, and any unmarked application record - returns
at the handler's marker check and is never persisted.
"""

import atexit
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import gevent
from gevent.event import Event
from gevent.queue import Empty, Full, Queue

from utils.logging_config import ContextDefaultsFilter

__all__ = [
    "OperationalEventSnapshot",
    "OperationalPersistenceHandler",
    "register_operational_persistence",
    "snapshot_from_record",
]

logger = logging.getLogger(__name__)

#: Default bounded-queue capacity, used when MAUI_OPERATIONAL_QUEUE_MAXSIZE
#: is absent or invalid. Not added to AppConfig: this subsystem is
#: configured independently, matching LOG_LEVEL/AGENT_RUNS_LOG_PATH.
_DEFAULT_QUEUE_MAXSIZE = 1000

#: Default bounded-drain timeout in seconds, used when
#: MAUI_OPERATIONAL_DRAIN_TIMEOUT_SECONDS is absent or invalid.
_DEFAULT_DRAIN_TIMEOUT_SECONDS = 2.0

#: How long the consumer's queue.get() blocks before re-checking the stop
#: signal. Bounds only how long a stopped consumer lingers on an empty
#: queue; never bounds request latency.
_POLL_SECONDS = 0.2

#: Emit a drop diagnostic on the first drop of an episode, then at most
#: once per this many further drops while pressure continues.
_DROP_DAMPING_INTERVAL = 100

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
    and zero I/O. This ordering is a re-entry barrier: the handler sits on
    the root logger, while this module's own delivery path and the database
    layer beneath it emit ordinary unmarked records of their own. Checking
    the marker before any other work keeps those records from re-entering
    the persistence path, so it must never be reordered behind normalization
    or sink work.

    Owns its own :class:`ContextDefaultsFilter` instance so request_id/app_id
    enrichment never depends on another handler (e.g. the stderr handler)
    having run first.

    The sink is a constructor-injected callable ``snapshot -> None``: this
    handler hands the detached snapshot to it and nothing more. It knows
    nothing about queues, gevent or the database, so it stays independent of
    the concrete delivery mechanism behind that sink.
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


def _parse_positive_int(env_name: str, default: int) -> "tuple[int, bool]":
    """Parse a positive int from the environment. Returns (value, fell_back).

    Absence of the variable is not a fallback: it is the ordinary default
    case and is not diagnosed. A present-but-invalid value IS a fallback.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return default, False
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("non-positive")
        return value, False
    except (TypeError, ValueError):
        return default, True


def _parse_positive_float(env_name: str, default: float) -> "tuple[float, bool]":
    """Parse a positive float from the environment. Returns (value, fell_back)."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default, False
    try:
        value = float(raw)
        if not value > 0:
            raise ValueError("non-positive")
        return value, False
    except (TypeError, ValueError):
        return default, True


class _OperationalDelivery:
    """Process-local delivery boundary and lifecycle behind the
    :class:`OperationalPersistenceHandler` sink.

    Owns a bounded ``gevent.queue.Queue``, exactly one consumer greenlet, a
    ``gevent.event.Event`` stop signal, and a plain-integer drop-diagnostic
    counter. Not a general framework and not a reusable task system: this is
    the smallest lifecycle owner for exactly this queue/consumer pair.

    Enqueue (the producer side, used as that handler's sink) NEVER blocks and
    NEVER raises: ``put_nowait`` either succeeds or the snapshot is discarded
    on ``Full``, with a damped runtime-only diagnostic. The consumer drains the
    queue via one gevent-cooperative loop, calling the
    ``insert_operational_event`` database writer with scalar fields
    unpacked from the snapshot; a write failure is contained and the loop
    continues. Shutdown is a single bounded ``join()`` on the
    consumer greenlet; remaining queued events may be lost after the bound.
    """

    def __init__(self):
        maxsize, maxsize_fell_back = _parse_positive_int(
            "MAUI_OPERATIONAL_QUEUE_MAXSIZE", _DEFAULT_QUEUE_MAXSIZE
        )
        drain_timeout, drain_fell_back = _parse_positive_float(
            "MAUI_OPERATIONAL_DRAIN_TIMEOUT_SECONDS", _DEFAULT_DRAIN_TIMEOUT_SECONDS
        )
        self._queue = Queue(maxsize=maxsize)
        self._drain_timeout = drain_timeout
        self._stop_event = Event()
        self._greenlet = None
        self._started = False
        self._drop_episode_active = False
        self._drop_count = 0

        if maxsize_fell_back:
            logger.warning(
                "event=operational_persistence_start_failed "
                "reason=invalid_queue_maxsize_env fallback=%s",
                _DEFAULT_QUEUE_MAXSIZE,
            )
        if drain_fell_back:
            logger.warning(
                "event=operational_persistence_start_failed "
                "reason=invalid_drain_timeout_env fallback=%s",
                _DEFAULT_DRAIN_TIMEOUT_SECONDS,
            )

    # -- producer side --------------------------------------------------

    def enqueue(self, snapshot) -> None:
        """Non-blocking handoff. Never raises, never retries, never writes
        synchronously. Full -> discard + damped diagnostic."""
        try:
            self._queue.put_nowait(snapshot)
        except Full:
            self._record_drop()
            return
        except Exception as exc:  # noqa: BLE001 - enqueue must stay fail-open
            self._safe_report_drop(exc)
            return

        self._drop_episode_active = False
        self._drop_count = 0

    def _record_drop(self) -> None:
        if not self._drop_episode_active:
            self._drop_episode_active = True
            self._drop_count = 1
            self._safe_report_drop(None)
        else:
            self._drop_count += 1
            if (self._drop_count - 1) % _DROP_DAMPING_INTERVAL == 0:
                self._safe_report_drop(None)

    def _safe_report_drop(self, exc: Optional[Exception]) -> None:
        try:
            if exc is None:
                logger.warning(
                    "event=operational_persistence_event_dropped "
                    "dropped_count=%s",
                    self._drop_count,
                )
            else:
                logger.warning(
                    "event=operational_persistence_event_dropped "
                    "error_type=%s",
                    type(exc).__name__,
                )
        except Exception:  # noqa: BLE001 - diagnostics must never raise
            pass

    # -- consumer side ----------------------------------------------------

    def _write(self, snapshot) -> None:
        try:
            from infrastructure import database_pg

            database_pg.insert_operational_event(
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
        except Exception as exc:  # noqa: BLE001 - DB failure must not kill the consumer
            logger.warning(
                "event=operational_persistence_write_failed error_type=%s",
                type(exc).__name__,
            )

    def _consume_loop(self) -> None:
        try:
            while True:
                try:
                    snapshot = self._queue.get(timeout=_POLL_SECONDS)
                except Empty:
                    if self._stop_event.is_set() and self._queue.empty():
                        return
                    continue
                self._write(snapshot)
        except Exception as exc:  # noqa: BLE001 - protect the greenlet itself
            logger.error(
                "event=operational_persistence_consumer_failed error_type=%s",
                type(exc).__name__,
            )

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Idempotent: spawns exactly one consumer greenlet. Safe to call
        repeatedly; never spawns a duplicate."""
        if self._started:
            return
        self._stop_event.clear()
        try:
            self._greenlet = gevent.spawn(self._consume_loop)
            self._started = True
        except Exception as exc:  # noqa: BLE001 - startup must never abort the caller
            logger.warning(
                "event=operational_persistence_start_failed error_type=%s",
                type(exc).__name__,
            )
            return
        _register_atexit_hook()

    def stop(self) -> None:
        """Idempotent bounded best-effort drain. Safe if never started.
        Waits exactly once via greenlet.join(timeout=...) and returns
        unconditionally; remaining queued events may be lost."""
        if not self._started:
            return
        self._started = False
        self._stop_event.set()

        greenlet = self._greenlet
        if greenlet is None:
            return

        greenlet.join(timeout=self._drain_timeout)

        if not greenlet.dead or not self._queue.empty():
            logger.warning(
                "event=operational_persistence_drain_timeout pending=%s",
                self._queue.qsize(),
            )


#: Process-local delivery singleton. By design: one queue/consumer pair per
#: worker process, never shared across processes.
_DELIVERY: Optional[_OperationalDelivery] = None

#: Guards against registering the atexit shutdown hook more than once per
#: process, across repeated start()/delivery-creation calls.
_ATEXIT_REGISTERED = False


def _atexit_shutdown() -> None:
    if _DELIVERY is not None:
        _DELIVERY.stop()


def _register_atexit_hook() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(_atexit_shutdown)
    _ATEXIT_REGISTERED = True


def _get_or_create_delivery() -> _OperationalDelivery:
    """Process-local singleton accessor. Does not start the consumer;
    callers that need delivery running call ``.start()`` themselves."""
    global _DELIVERY
    if _DELIVERY is None:
        _DELIVERY = _OperationalDelivery()
    return _DELIVERY


def _reset_delivery_for_tests() -> None:
    """Test-only seam: stop and drop the process-local delivery singleton
    and clear atexit-registration state. Not a production management API."""
    global _DELIVERY, _ATEXIT_REGISTERED
    if _DELIVERY is not None:
        _DELIVERY.stop()
    _DELIVERY = None
    _ATEXIT_REGISTERED = False


def register_operational_persistence(app) -> None:
    """Attach the Operational Persistence subsystem to the real root logger.

    Idempotent: a no-op if a root handler already carries
    ``_maui_operational_persistence`` - across any number of calls, on the
    same or a freshly created Flask app. Starts the process-local delivery
    singleton (itself idempotent at the queue/consumer level) and
    attaches exactly one :class:`OperationalPersistenceHandler`, bound to
    ``delivery.enqueue``, as a sibling of the existing stderr handler. Does
    not touch the stderr handler, root's level, or any other handler.

    ``app`` is accepted only for signature symmetry with the other
    ``register_*_hooks(app)`` registrars in ``main.py``; the subsystem is
    process-local and does not depend on the Flask app or request context.
    """
    root = logging.getLogger()
    if any(
        getattr(handler, "_maui_operational_persistence", False)
        for handler in root.handlers
    ):
        return

    delivery = _get_or_create_delivery()
    delivery.start()

    handler = OperationalPersistenceHandler(delivery.enqueue)
    root.addHandler(handler)
