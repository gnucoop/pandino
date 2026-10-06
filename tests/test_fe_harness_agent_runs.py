"""Tests for evaluation.datachat.collectors.agent_runs, the agent_runs JSONL collector.

Fixture JSONL files under tmp_path only. Pins the PRESENT / EXPECTED_ABSENT /
UNAVAILABLE semantics, byte-exact payloads and the offset boundary.
"""

import json

import pytest

from evaluation.datachat.collectors.agent_runs import collect_agent_runs
from evaluation.datachat.model import Availability

RID = "0123456789abcdef"


def _jsonl(**fields):
    record = {"namespace": "datachat", "request_id": RID, "state": "success", "note": "è"}
    record.update(fields)
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _write(tmp_path, *lines):
    path = tmp_path / "agent_runs.log"
    path.write_bytes(b"".join(lines))
    return path


def test_single_match_is_present_and_byte_exact(tmp_path):
    match = _jsonl()
    path = _write(tmp_path, _jsonl(namespace="agentchat"), match, _jsonl(request_id="ffffffffffffffff"))
    record, payload = collect_agent_runs(path, 0, RID)
    assert record.availability is Availability.PRESENT
    assert payload == match
    assert record.locator["matches"] == 1
    assert record.locator["line_count"] == 3


@pytest.mark.parametrize(
    "hint, availability, reason",
    [
        (False, Availability.EXPECTED_ABSENT, "trace_absent"),
        (True, Availability.UNAVAILABLE, "missing_despite_trace"),
        (None, Availability.EXPECTED_ABSENT, "no_trace_signal"),
    ],
)
def test_zero_matches_follow_trace_hint(tmp_path, hint, availability, reason):
    path = _write(tmp_path, _jsonl(namespace="agentchat"), _jsonl(request_id="ffffffffffffffff"))
    record, payload = collect_agent_runs(path, 0, RID, trace_hint=hint)
    assert (record.availability, record.reason, payload) == (availability, reason, b"")


def test_two_matches_are_ambiguous_and_both_kept(tmp_path):
    first, second = _jsonl(state="a"), _jsonl(state="b")
    record, payload = collect_agent_runs(_write(tmp_path, first, second), 0, RID, trace_hint=True)
    assert record.availability is Availability.UNAVAILABLE
    assert record.reason == "ambiguous"
    assert payload == first + second


def test_malformed_unrelated_line_is_tolerated(tmp_path):
    match = _jsonl()
    record, payload = collect_agent_runs(_write(tmp_path, b"{not json\n", match), 0, RID)
    assert record.availability is Availability.PRESENT
    assert payload == match
    assert record.locator["malformed_lines"] == 1
    assert record.locator["malformed_candidates"] == 0


def test_malformed_candidate_line_is_unavailable(tmp_path):
    broken = b'{"namespace": "datachat", "request_id": "' + RID.encode() + b'", \n'
    record, payload = collect_agent_runs(_write(tmp_path, broken), 0, RID, trace_hint=False)
    assert record.availability is Availability.UNAVAILABLE
    assert record.reason == "malformed_candidate"
    assert payload == broken


def test_missing_file_is_unavailable(tmp_path):
    record, payload = collect_agent_runs(tmp_path / "missing.log", 0, RID)
    assert record.availability is Availability.UNAVAILABLE
    assert record.reason.startswith("read_failed: FileNotFoundError")
    assert payload == b""


def test_unreadable_path_is_unavailable(tmp_path):
    record, _ = collect_agent_runs(tmp_path, 0, RID)  # a directory
    assert record.availability is Availability.UNAVAILABLE
    assert record.reason.startswith("read_failed")


def test_only_appended_bytes_are_considered(tmp_path):
    old = _jsonl(state="previous scenario")
    path = _write(tmp_path, old)
    offset = path.stat().st_size
    record, payload = collect_agent_runs(path, offset, RID, trace_hint=False)
    assert record.availability is Availability.EXPECTED_ABSENT
    assert payload == b""

    new = _jsonl(state="this scenario")
    with open(path, "ab") as handle:
        handle.write(new)
    record, payload = collect_agent_runs(path, offset, RID)
    assert record.availability is Availability.PRESENT
    assert payload == new
    assert record.locator["offset_from"] == offset
    assert record.locator["offset_to"] == path.stat().st_size


def test_offset_beyond_eof_is_unavailable(tmp_path):
    record, _ = collect_agent_runs(_write(tmp_path, b"x\n"), 100, RID)
    assert record.availability is Availability.UNAVAILABLE
    assert record.reason.startswith("offset_beyond_eof")


@pytest.mark.parametrize("offset, exc", [(-1, ValueError), ("0", TypeError)])
def test_invalid_offset_rejected(tmp_path, offset, exc):
    with pytest.raises(exc):
        collect_agent_runs(tmp_path / "x", offset, RID)
