"""Runtime stderr collector (``datachat.runtime`` records in ``maui_stderr.log``).

Record shape: ``<UTC ISO> <LEVEL> <logger> request_id=<id> app_id=<client> <message>``.
The ``request_id=`` token is authoritative for extraction; byte offsets only
bound the read. Only complete (newline-terminated) lines are considered, so
a record still being written is never sliced in half.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evaluation.datachat.model import Availability, EvidenceSourceRecord

_REQUEST_ID = re.compile(r"[0-9a-f]{16}")
_REQUEST_ID_TOKEN = re.compile(rb"(?:^|\s)request_id=(\S+)")
_BOOL = {"True": True, "False": False}


def _check_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ValueError(f"invalid request id: {request_id!r}")


def _check_offset(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def current_offset(path: Path) -> int:
    """Current size of ``path`` in bytes."""
    return Path(path).stat().st_size


def read_range(path: Path, start: int, end: int | None = None) -> bytes:
    """Bytes ``[start, end)`` of ``path`` (to EOF when ``end`` is None)."""
    _check_offset(start, "start")
    if end is not None:
        _check_offset(end, "end")
        if end < start:
            raise ValueError(f"end {end} < start {start}")
    with open(path, "rb") as handle:
        handle.seek(start)
        return handle.read() if end is None else handle.read(end - start)


def _complete_lines(data: bytes) -> list[bytes]:
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    return lines


def _carries(line: bytes, request_id: bytes) -> bool:
    return request_id in _REQUEST_ID_TOKEN.findall(line)


def filter_request_lines(data: bytes, request_id: str) -> bytes:
    """Complete lines carrying ``request_id=<request_id>``, byte-exact and in order."""
    _check_request_id(request_id)
    target = request_id.encode("ascii")
    return b"".join(line for line in _complete_lines(data) if _carries(line, target))


def _has_marker(lines: bytes, marker: str) -> bool:
    token = marker.encode("ascii")
    return any(token in line.split() for line in lines.splitlines())


@dataclass(frozen=True)
class MarkerWait:
    found: bool
    offset_to: int
    lines: bytes
    polls: int


def wait_for_marker(
    path: Path,
    start_offset: int,
    request_id: str,
    marker: str,
    deadline_s: float,
    *,
    poll_interval_s: float = 0.1,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> MarkerWait:
    """Re-read from ``start_offset`` until ``marker`` appears under ``request_id``.

    Gives up once ``deadline_s`` seconds have elapsed on ``clock``. Read
    errors propagate as ``OSError``.
    """
    _check_request_id(request_id)
    _check_offset(start_offset, "start_offset")
    give_up_at = clock() + deadline_s
    polls = 0
    while True:
        polls += 1
        data = read_range(path, start_offset)
        complete = b"".join(_complete_lines(data))
        lines = filter_request_lines(complete, request_id)
        offset_to = start_offset + len(complete)
        if _has_marker(lines, marker):
            return MarkerWait(True, offset_to, lines, polls)
        if clock() >= give_up_at:
            return MarkerWait(False, offset_to, lines, polls)
        sleep(poll_interval_s)


def collect_runtime(
    source: str,
    path: Path,
    start_offset: int,
    request_id: str,
    marker: str,
    deadline_s: float,
    *,
    poll_interval_s: float = 0.1,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[EvidenceSourceRecord, bytes]:
    """Runtime slice for one request: ``PRESENT`` once ``marker`` is seen, else ``UNAVAILABLE``.

    The returned bytes are the matching lines (possibly partial when unavailable).
    """
    locator = {"request_id": request_id, "marker": marker, "offset_from": start_offset}
    try:
        wait = wait_for_marker(
            path, start_offset, request_id, marker, deadline_s,
            poll_interval_s=poll_interval_s, clock=clock, sleep=sleep,
        )
    except OSError as exc:
        locator.update(offset_to=None, line_count=0)
        reason = f"read_failed: {type(exc).__name__}: {exc.strerror or exc}"
        return EvidenceSourceRecord(source, Availability.UNAVAILABLE, reason, locator=locator), b""

    locator.update(
        offset_to=wait.offset_to,
        line_count=len(wait.lines.splitlines()),
        marker_found=wait.found,
        polls=wait.polls,
    )
    if wait.found:
        return EvidenceSourceRecord(source, Availability.PRESENT, locator=locator), wait.lines
    reason = f"marker_missing: {marker} not seen within {deadline_s}s"
    return EvidenceSourceRecord(source, Availability.UNAVAILABLE, reason, locator=locator), wait.lines


@dataclass(frozen=True)
class TraceStatus:
    """``datachat_trace_status`` flags; ``None`` when the line or field is absent."""

    trace_present: bool | None = None
    structured_log_ok: bool | None = None


def parse_trace_status(lines: bytes) -> TraceStatus:
    """Flags from the last ``datachat_trace_status`` line of a datachat runtime slice."""
    status_line = None
    for line in lines.splitlines():
        if b"datachat_trace_status" in line.split():
            status_line = line
    if status_line is None:
        return TraceStatus()
    fields = {}
    for token in status_line.decode("utf-8", errors="replace").split():
        key, sep, value = token.partition("=")
        if sep:
            fields[key] = value
    return TraceStatus(
        trace_present=_BOOL.get(fields.get("trace_present", "")),
        structured_log_ok=_BOOL.get(fields.get("structured_log_ok", "")),
    )
