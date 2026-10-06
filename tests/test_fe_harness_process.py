"""Tests for evaluation.datachat.maui_process, the harness-owned Maui process lifecycle.

Children are trivial `sys.executable -c` scripts, never Gunicorn or Maui.
Readiness polling uses a fake health function, clock and sleep. SIGKILL
escalation is checked with a fake Popen so it never depends on timing.
"""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest

from evaluation.datachat import maui_process
from evaluation.datachat.maui_process import EXITED, READY, TIMEOUT, MauiProcess, port_free


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _python(code):
    return [sys.executable, "-c", code]


def _wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --- Port --------------------------------------------------------------------


def test_port_free_and_occupied():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert port_free("127.0.0.1", port) is False
    assert port_free("127.0.0.1", port) is True


@pytest.mark.parametrize("port, exc", [(0, ValueError), (65536, ValueError), ("5000", TypeError), (True, TypeError)])
def test_port_free_rejects_invalid_port(port, exc):
    with pytest.raises(exc):
        port_free("127.0.0.1", port)


# --- Start / log / env / stop -------------------------------------------------


def test_start_captures_output_and_env_then_stops(tmp_path, monkeypatch):
    monkeypatch.delenv("FE_CHILD_VALUE", raising=False)
    before = dict(os.environ)
    log = tmp_path / "suite" / "maui_stderr.log"
    code = (
        "import os, sys, time\n"
        "sys.stderr.write('child-ready ' + os.environ['FE_CHILD_VALUE'] + '\\n'); sys.stderr.flush()\n"
        "print('stdout-line', flush=True)\n"
        "time.sleep(60)\n"
    )
    env = {**os.environ, "FE_CHILD_VALUE": "from-explicit-env"}
    process = MauiProcess(_python(code), tmp_path, env, log)

    pid = process.start()
    try:
        assert pid == process.pid
        assert process.alive()
        assert _wait_for(lambda: b"stdout-line" in log.read_bytes())
        content = log.read_bytes()
        assert b"child-ready from-explicit-env" in content
    finally:
        code_after = process.stop(grace_s=5)

    assert not process.alive()
    assert code_after == -signal.SIGTERM
    assert dict(os.environ) == before


def test_start_twice_rejected(tmp_path):
    process = MauiProcess(_python("import time; time.sleep(60)"), tmp_path, dict(os.environ), tmp_path / "log")
    process.start()
    try:
        with pytest.raises(RuntimeError):
            process.start()
    finally:
        process.stop(grace_s=5)


def test_start_failure_propagates_oserror(tmp_path):
    process = MauiProcess([str(tmp_path / "no-such-binary")], tmp_path, {}, tmp_path / "log")
    with pytest.raises(OSError):
        process.start()
    assert process.stop() is None


def test_invalid_argv_rejected(tmp_path):
    with pytest.raises(TypeError):
        MauiProcess("gunicorn main:app", tmp_path, {}, tmp_path / "log")
    with pytest.raises(TypeError):
        MauiProcess([], tmp_path, {}, tmp_path / "log")


def test_stop_is_idempotent(tmp_path):
    process = MauiProcess(_python("pass"), tmp_path, dict(os.environ), tmp_path / "log")
    assert process.stop() is None  # never started
    process.start()
    assert _wait_for(lambda: not process.alive())
    assert process.stop() == 0
    assert process.stop() == 0


# --- Readiness -----------------------------------------------------------------


def test_wait_ready_detects_early_exit(tmp_path):
    process = MauiProcess(_python("import sys; sys.exit(3)"), tmp_path, dict(os.environ), tmp_path / "log")
    process.start()
    assert _wait_for(lambda: not process.alive())
    clock = FakeClock()
    calls = []

    result = process.wait_ready(lambda: calls.append(1) or False, 30, clock=clock, sleep=clock.sleep)

    assert (result.ready, result.outcome, result.exit_code) == (False, EXITED, 3)
    assert calls == []
    assert clock.now == 0
    process.stop()


def test_wait_ready_succeeds_after_polls(tmp_path):
    process = MauiProcess(_python("import time; time.sleep(60)"), tmp_path, dict(os.environ), tmp_path / "log")
    process.start()
    try:
        clock = FakeClock()
        answers = iter([False, False, True])
        result = process.wait_ready(lambda: next(answers), 30, poll_interval_s=1, clock=clock, sleep=clock.sleep)
        assert (result.ready, result.outcome, result.polls) == (True, READY, 3)
        assert clock.now == 2
    finally:
        process.stop(grace_s=5)


def test_wait_ready_times_out(tmp_path):
    process = MauiProcess(_python("import time; time.sleep(60)"), tmp_path, dict(os.environ), tmp_path / "log")
    process.start()
    try:
        clock = FakeClock()
        result = process.wait_ready(lambda: False, 5, poll_interval_s=1, clock=clock, sleep=clock.sleep)
        assert (result.ready, result.outcome) == (False, TIMEOUT)
        assert result.polls == 6
        assert process.alive()
    finally:
        process.stop(grace_s=5)


def test_wait_ready_requires_started_process(tmp_path):
    with pytest.raises(RuntimeError):
        MauiProcess(_python("pass"), tmp_path, {}, tmp_path / "log").wait_ready(lambda: True, 1)


# --- Stop escalation (fake Popen: deterministic) ------------------------------


class FakePopen:
    def __init__(self, *args, **kwargs):
        self.pid = 4242
        self.returncode = None
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = -signal.SIGKILL
        return self.returncode


def test_stop_escalates_to_sigkill(tmp_path, monkeypatch):
    signals = []
    monkeypatch.setattr(maui_process.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(maui_process.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    process = MauiProcess(["gunicorn"], tmp_path, {}, tmp_path / "log")
    process.start()

    code = process.stop(grace_s=7)

    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert process._popen.waits == [7, None]
    assert code == -signal.SIGKILL
    assert process._log is None


def test_stop_tolerates_group_already_gone(tmp_path, monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(maui_process.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(maui_process.os, "killpg", gone)
    process = MauiProcess(["gunicorn"], tmp_path, {}, tmp_path / "log")
    process.start()
    assert process.stop(grace_s=1) == -signal.SIGKILL
