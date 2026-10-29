"""
run.py
------

AnalysisRun: one analysis of one brief, as the outside world sees it.

A run happens in the background, so a client follows it by polling: the run
keeps a status and an append-only list of progress events, each numbered, and
GET /analysis-status?since=N hands back the events after N. The events say what
the analyst is doing (which sub-task, which tools) and never carry data or SQL.

The orchestrator is the only writer; route handlers only read. Both happen on
different threads (greenlets under gunicorn gevent), hence the lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

QUEUED = "queued"
PLANNING = "planning"
RUNNING = "running"
CHARTING = "charting"
WRITING_REPORT = "writing_report"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

FINISHED_STATUSES = (COMPLETED, FAILED, CANCELLED)

# Oldest events are dropped past this, so a long run cannot grow without bound.
# A client polling every few seconds never notices.
_MAX_EVENTS = 1000


@dataclass
class ProgressLog:
    """Numbered progress events, readable from a cursor."""

    _events: list[dict[str, Any]] = field(default_factory=list)
    _seq: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def emit(
        self,
        event_type: str,
        message: str,
        *,
        task_id: Optional[str] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._seq += 1
            event: dict[str, Any] = {
                "seq": self._seq,
                "ts": time.time(),
                "type": event_type,
                "message": message,
            }
            if task_id is not None:
                event["task_id"] = task_id
            if data:
                event["data"] = data
            self._events.append(event)
            del self._events[:-_MAX_EVENTS]

    def since(self, seq: int) -> tuple[list[dict[str, Any]], int]:
        """The events after `seq`, and the cursor to pass next time."""
        with self._lock:
            return [dict(e) for e in self._events if e["seq"] > seq], self._seq


@dataclass
class AnalysisRun:
    run_id: str
    brief_id: int
    api_key: str
    user_email: str

    status: str = QUEUED
    current_task: Optional[str] = None
    tasks_done: int = 0
    tasks_total: int = 0
    error: Optional[str] = None
    report_ready: bool = False

    progress: ProgressLog = field(default_factory=ProgressLog)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    def cancel(self) -> None:
        """Ask the run to stop; the orchestrator stops at the next agent step."""
        self._cancel.set()

    def close(self) -> None:
        """Called by agent_manager when the run is removed from the registry."""
        self.cancel()

    def status_payload(self, since: int = 0) -> dict[str, Any]:
        events, cursor = self.progress.since(since)
        return {
            "run_id": self.run_id,
            "brief_id": self.brief_id,
            "status": self.status,
            "current_task": self.current_task,
            "tasks_done": self.tasks_done,
            "tasks_total": self.tasks_total,
            "error": self.error,
            "report_ready": self.report_ready,
            "events": events,
            "next_since": cursor,
        }
