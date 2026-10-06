"""agent_runs collector: the run's dedicated append-only JSONL file.

Only bytes appended after the offset recorded before ``/datachat`` are read.
Matching records (``namespace == "datachat"`` and the datachat request ID)
are returned byte-exact for ``agent_runs/record.jsonl``.

Zero matches can be legitimate: CURRENT writes a record only when the agent
run returned a trace. ``trace_hint`` (``trace_present`` from the runtime
``datachat_trace_status`` line) decides between ``EXPECTED_ABSENT`` and
``UNAVAILABLE``; without a hint, absence is ``EXPECTED_ABSENT`` with reason
``no_trace_signal``.
"""

import json
from pathlib import Path

from evaluation.datachat.model import Availability, EvidenceSourceRecord

SOURCE = "agent_runs"


def collect_agent_runs(
    path: Path, offset: int, request_id: str, trace_hint: bool | None = None
) -> tuple[EvidenceSourceRecord, bytes]:
    """``(record, payload)``; payload holds every matching raw line (all of them when ambiguous)."""
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError(f"offset must be int, got {type(offset).__name__}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")

    locator: dict = {"request_id": request_id, "offset_from": offset}
    try:
        with open(path, "rb") as handle:
            size = handle.seek(0, 2)
            if offset > size:
                locator["offset_to"] = size
                reason = f"offset_beyond_eof: {offset} > {size}"
                return EvidenceSourceRecord(SOURCE, Availability.UNAVAILABLE, reason, locator=locator), b""
            handle.seek(offset)
            data = handle.read()
    except OSError as exc:
        reason = f"read_failed: {type(exc).__name__}: {exc.strerror or exc}"
        return EvidenceSourceRecord(SOURCE, Availability.UNAVAILABLE, reason, locator=locator), b""

    target = request_id.encode("utf-8")
    matches: list[bytes] = []
    malformed_candidates: list[bytes] = []
    line_count = malformed = 0
    for line in data.splitlines(keepends=True):
        if not line.strip():
            continue
        line_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            if target in line:
                malformed_candidates.append(line)
            continue
        if (
            isinstance(record, dict)
            and record.get("namespace") == "datachat"
            and record.get("request_id") == request_id
        ):
            matches.append(line)

    locator.update(
        offset_to=offset + len(data),
        line_count=line_count,
        malformed_lines=malformed,
        malformed_candidates=len(malformed_candidates),
        matches=len(matches),
        trace_hint=trace_hint,
    )
    payload = b"".join(matches + malformed_candidates)

    def record(availability: Availability, reason: str | None = None) -> EvidenceSourceRecord:
        return EvidenceSourceRecord(SOURCE, availability, reason, locator=locator)

    if malformed_candidates:
        return record(Availability.UNAVAILABLE, "malformed_candidate"), payload
    if len(matches) > 1:
        return record(Availability.UNAVAILABLE, "ambiguous"), payload
    if len(matches) == 1:
        return record(Availability.PRESENT), payload
    if trace_hint is True:
        return record(Availability.UNAVAILABLE, "missing_despite_trace"), payload
    if trace_hint is False:
        return record(Availability.EXPECTED_ABSENT, "trace_absent"), payload
    return record(Availability.EXPECTED_ABSENT, "no_trace_signal"), payload
