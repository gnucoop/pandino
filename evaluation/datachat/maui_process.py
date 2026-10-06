"""Lifecycle of the Maui (Gunicorn) process launched and owned by the harness.

The harness never attaches to an existing server: ``port_free`` lets
preflight refuse an occupied port, and ``MauiProcess`` manages exactly the
child it started. The child runs in its own session (process group), with
stdout and stderr appended to the suite log and an explicit, already
resolved environment (this module never reads ``.env`` or touches
``os.environ``). Shutdown is SIGTERM to the group, a grace period, then
SIGKILL. No ``INFRA_*`` classification happens here.
"""

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

READY = "ready"
EXITED = "exited"
TIMEOUT = "timeout"


def _check_port(port: int) -> None:
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError(f"port must be int, got {type(port).__name__}")
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid port: {port}")


def port_free(host: str, port: int) -> bool:
    """True when ``host:port`` can be bound now. Never touches whoever holds it."""
    _check_port(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


@dataclass(frozen=True)
class ReadyResult:
    ready: bool
    outcome: str  # READY | EXITED | TIMEOUT
    polls: int
    exit_code: int | None = None


class MauiProcess:
    def __init__(self, argv: Sequence[str], cwd: Path, env: Mapping[str, str], log_path: Path):
        if isinstance(argv, (str, bytes)) or not argv:
            raise TypeError("argv must be a non-empty sequence of strings")
        self.argv = [str(arg) for arg in argv]
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.log_path = Path(log_path)
        self._popen: subprocess.Popen | None = None
        self._log = None

    @property
    def pid(self) -> int | None:
        return self._popen.pid if self._popen is not None else None

    @property
    def returncode(self) -> int | None:
        return self._popen.poll() if self._popen is not None else None

    def start(self) -> int:
        if self._popen is not None:
            raise RuntimeError("process already started")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        try:
            self._popen = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            self._log.close()
            self._log = None
            raise
        return self._popen.pid

    def alive(self) -> bool:
        return self._popen is not None and self._popen.poll() is None

    def wait_ready(
        self,
        health_check: Callable[[], bool],
        deadline_s: float,
        *,
        poll_interval_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> ReadyResult:
        """Poll ``health_check`` until it returns True, the child exits, or the deadline passes."""
        if self._popen is None:
            raise RuntimeError("process not started")
        give_up_at = clock() + deadline_s
        polls = 0
        while True:
            if not self.alive():
                return ReadyResult(False, EXITED, polls, self._popen.returncode)
            polls += 1
            if health_check():
                return ReadyResult(True, READY, polls)
            if not self.alive():
                return ReadyResult(False, EXITED, polls, self._popen.returncode)
            if clock() >= give_up_at:
                return ReadyResult(False, TIMEOUT, polls)
            sleep(poll_interval_s)

    def stop(self, grace_s: float = 10.0) -> int | None:
        """SIGTERM the process group, wait ``grace_s``, then SIGKILL. Idempotent."""
        popen = self._popen
        try:
            if popen is not None and popen.poll() is None:
                self._signal_group(signal.SIGTERM)
                try:
                    popen.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    popen.wait()
            return popen.returncode if popen is not None else None
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(self._popen.pid, sig)
        except ProcessLookupError:
            pass
