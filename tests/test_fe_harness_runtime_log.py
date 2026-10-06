"""Tests for evaluation.datachat.collectors.runtime_log, the runtime stderr collector.

Fixture stderr files under tmp_path only; marker polling uses an injected
fake clock and sleep, so nothing really waits. No Maui, DB or network.
"""

import pytest

from evaluation.datachat.collectors.runtime_log import (
    TraceStatus,
    collect_runtime,
    current_offset,
    filter_request_lines,
    parse_trace_status,
    read_range,
    wait_for_marker,
)
from evaluation.datachat.model import Availability

RID = "0123456789abcdef"
OTHER = "fedcba9876543210"


def _line(request_id, message):
    return f"2026-10-06T08:15:30.000Z INFO datachat.runtime request_id={request_id} app_id=fe {message}\n".encode()


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_offsets_and_read_range(tmp_path):
    path = tmp_path / "stderr.log"
    path.write_bytes(b"abcdef")
    assert current_offset(path) == 6
    assert read_range(path, 2) == b"cdef"
    assert read_range(path, 1, 3) == b"bc"


@pytest.mark.parametrize("start, end, exc", [(-1, None, ValueError), (3, 2, ValueError), (0, -1, ValueError), ("0", None, TypeError)])
def test_read_range_rejects_invalid_offsets(tmp_path, start, end, exc):
    path = tmp_path / "stderr.log"
    path.write_bytes(b"abc")
    with pytest.raises(exc):
        read_range(path, start, end)


def test_filter_extracts_exact_id_in_order():
    data = b"".join([
        _line(RID, "engine_init_start"),
        _line(OTHER, "engine_init_start"),
        _line("-", "unrelated"),
        _line(RID + "00", "longer id"),
        _line(RID, "engine_init_result status=ok sql=off"),
        b"note mentions " + RID.encode() + b" without token\n",
    ])
    assert filter_request_lines(data, RID) == _line(RID, "engine_init_start") + _line(
        RID, "engine_init_result status=ok sql=off"
    )


def test_filter_matches_request_id_inside_message():
    line = _line("-", f"datachat_trace_status request_id={RID} trace_present=True")
    assert filter_request_lines(line, RID) == line


def test_filter_rejects_non_hex_request_id():
    with pytest.raises(ValueError):
        filter_request_lines(b"", "-")


def test_filter_ignores_incomplete_last_line():
    data = _line(RID, "a") + _line(RID, "b").rstrip(b"\n")
    assert filter_request_lines(data, RID) == _line(RID, "a")


def test_wait_finds_marker_immediately(tmp_path):
    path = tmp_path / "stderr.log"
    path.write_bytes(_line("-", "boot") + _line(RID, "cleanup_result status=ok"))
    clock = FakeClock()
    result = wait_for_marker(path, 0, RID, "cleanup_result", 5, clock=clock, sleep=clock.sleep)
    assert result.found and result.polls == 1
    assert result.offset_to == path.stat().st_size
    assert clock.sleeps == []


def test_wait_ignores_marker_of_other_request(tmp_path):
    path = tmp_path / "stderr.log"
    path.write_bytes(_line(OTHER, "cleanup_result status=ok"))
    clock = FakeClock()
    result = wait_for_marker(path, 0, RID, "cleanup_result", 1, poll_interval_s=0.25, clock=clock, sleep=clock.sleep)
    assert not result.found
    assert result.lines == b""
    assert clock.now >= 1


def test_wait_sees_file_growing_between_polls(tmp_path):
    path = tmp_path / "stderr.log"
    old = _line(RID, "datachat_request_end")  # before start offset: must be ignored
    path.write_bytes(old)
    start = current_offset(path)
    clock = FakeClock()

    def sleep(seconds):
        clock.sleep(seconds)
        if len(clock.sleeps) == 2:
            with open(path, "ab") as handle:
                handle.write(_line(RID, "chat_start") + _line(RID, "datachat_request_end"))

    result = wait_for_marker(path, start, RID, "datachat_request_end", 10, clock=clock, sleep=sleep)
    assert result.found
    assert result.polls == 3
    assert result.lines == _line(RID, "chat_start") + _line(RID, "datachat_request_end")


def test_marker_must_be_a_whole_token(tmp_path):
    path = tmp_path / "stderr.log"
    path.write_bytes(_line(RID, "engine_init_result_extra"))
    clock = FakeClock()
    assert not wait_for_marker(path, 0, RID, "engine_init_result", 0, clock=clock, sleep=clock.sleep).found


def test_collect_present(tmp_path):
    path = tmp_path / "stderr.log"
    body = _line(RID, "engine_init_result status=ok sql=off") + _line(RID, "engine_init_tools sql_tools=0")
    path.write_bytes(_line(OTHER, "x") + body)
    clock = FakeClock()
    record, payload = collect_runtime("runtime_start", path, 0, RID, "engine_init_result", 5, clock=clock, sleep=clock.sleep)
    assert record.availability is Availability.PRESENT
    assert record.reason is None
    assert payload == body
    assert record.locator["request_id"] == RID
    assert record.locator["offset_from"] == 0
    assert record.locator["offset_to"] == path.stat().st_size
    assert record.locator["line_count"] == 2


def test_collect_marker_missing_is_unavailable(tmp_path):
    path = tmp_path / "stderr.log"
    path.write_bytes(_line(RID, "chat_start"))
    clock = FakeClock()
    record, payload = collect_runtime("runtime_datachat", path, 0, RID, "datachat_request_end", 2, clock=clock, sleep=clock.sleep)
    assert record.availability is Availability.UNAVAILABLE
    assert record.reason.startswith("marker_missing")
    assert payload == _line(RID, "chat_start")


def test_collect_missing_file_is_unavailable(tmp_path):
    clock = FakeClock()
    record, payload = collect_runtime("runtime_teardown", tmp_path / "missing.log", 0, RID, "cleanup_result", 1, clock=clock, sleep=clock.sleep)
    assert record.availability is Availability.UNAVAILABLE
    assert record.reason.startswith("read_failed: FileNotFoundError")
    assert payload == b""


def test_collect_rejects_invalid_offset(tmp_path):
    with pytest.raises(ValueError):
        collect_runtime("runtime_start", tmp_path / "x", -5, RID, "m", 1)


@pytest.mark.parametrize(
    "fields, expected",
    [
        ("trace_present=True structured_log_ok=True", TraceStatus(True, True)),
        ("trace_present=False structured_log_ok=False", TraceStatus(False, False)),
        ("trace_present=True structured_log_ok=False", TraceStatus(True, False)),
        ("trace_present=True", TraceStatus(True, None)),
        ("trace_present=maybe structured_log_ok=", TraceStatus(None, None)),
    ],
)
def test_parse_trace_status(fields, expected):
    lines = _line(RID, "chat_start") + _line(
        "-", f"datachat_trace_status request_id={RID} user=fe engine=x {fields} db_log_ok=True log_id=7"
    )
    assert parse_trace_status(lines) == expected


def test_parse_trace_status_absent_is_unknown():
    assert parse_trace_status(_line(RID, "datachat_request_end")) == TraceStatus(None, None)
    assert parse_trace_status(b"") == TraceStatus()
