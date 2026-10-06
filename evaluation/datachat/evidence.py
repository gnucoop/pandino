"""Evidence persistence primitives of the DataChat Functional Evaluation harness.

- ``RunLayout``: path-only view of ``runs/<run_id>/``; constructing it touches
  nothing on disk, and every path it returns stays below the run root.
- Raw evidence (HTTP bodies, runtime slices, agent_runs records, Usage rows,
  artifacts) is write-once: ``write_raw`` creates the file exclusively and
  never overwrites it.
- Execution metadata (manifests, ``http/*.json``) is mutable while the run
  progresses: ``write_json_atomic`` replaces it via temp file, fsync and
  ``os.replace``.
- Redaction happens before writing, so a stored hash always describes the
  stored (redacted) bytes.

Path-contract violations raise ``ValueError``; filesystem failures propagate
as ``OSError``. Classifying them as ``INFRA_*`` is the runner's job.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.datachat.model import FileRef

REDACTED = "<redacted>"

HTTP_KEYS = frozenset({"pre_cleanup", "start", "datachat", "teardown", "export"})
RUNTIME_NAMES = frozenset({"start", "datachat", "teardown"})
ARTIFACT_FILES = frozenset({"image.png", "image.json", "export.csv"})

_SCENARIO_ID = re.compile(r"FE-\d{2}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_HASH_CHUNK = 1024 * 1024


def safe_child(root: Path, relative_path: str | Path) -> Path:
    """``root / relative_path``, rejecting absolute paths and escapes from ``root``."""
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"absolute path not allowed: {relative_path}")
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes run root: {relative_path}")
    return Path(root) / relative


def _check_scenario_id(scenario_id: str) -> str:
    if not isinstance(scenario_id, str) or not _SCENARIO_ID.fullmatch(scenario_id):
        raise ValueError(f"invalid scenario id: {scenario_id!r}")
    return scenario_id


def _check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
    return run_id


class RunLayout:
    """Locations inside one run directory. Never creates anything."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, *parts: str) -> Path:
        return safe_child(self.root, Path(*parts))

    # Suite level

    @property
    def run_manifest(self) -> Path:
        return self._path("run_manifest.json")

    @property
    def environment(self) -> Path:
        return self._path("environment.json")

    @property
    def maui_stderr(self) -> Path:
        return self._path("maui_stderr.log")

    @property
    def agent_runs_log(self) -> Path:
        return self._path("agent_runs.log")

    @property
    def plots_dir(self) -> Path:
        return self._path("plots")

    @property
    def failures_dir(self) -> Path:
        return self._path("failures")

    # Scenario level

    def scenario_dir(self, scenario_id: str) -> Path:
        return self._path("scenarios", _check_scenario_id(scenario_id))

    def execution(self, scenario_id: str) -> Path:
        return self.scenario_dir(scenario_id) / "execution.json"

    def http_paths(self, scenario_id: str, key: str, attempt: int = 1) -> tuple[Path, Path, Path]:
        """``(metadata, request body, response body)`` for one HTTP attempt.

        Attempt 1 is ``http/<key>.*``; attempt n > 1 is ``http/<key>.<n>.*``.
        """
        if key not in HTTP_KEYS:
            raise ValueError(f"invalid http key: {key!r}")
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            raise TypeError(f"attempt must be int, got {type(attempt).__name__}")
        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt}")
        stem = key if attempt == 1 else f"{key}.{attempt}"
        http_dir = self.scenario_dir(scenario_id) / "http"
        return (
            http_dir / f"{stem}.json",
            http_dir / f"{stem}.request.bin",
            http_dir / f"{stem}.response.bin",
        )

    def runtime_log(self, scenario_id: str, name: str) -> Path:
        if name not in RUNTIME_NAMES:
            raise ValueError(f"invalid runtime log name: {name!r}")
        return self.scenario_dir(scenario_id) / "runtime" / f"{name}.log"

    def agent_runs_record(self, scenario_id: str) -> Path:
        return self.scenario_dir(scenario_id) / "agent_runs" / "record.jsonl"

    def usage_rows(self, scenario_id: str) -> Path:
        return self.scenario_dir(scenario_id) / "usage" / "rows.json"

    def artifact(self, scenario_id: str, filename: str) -> Path:
        if filename not in ARTIFACT_FILES:
            raise ValueError(f"invalid artifact file: {filename!r}")
        return self.scenario_dir(scenario_id) / "artifacts" / filename


def create_run_dir(base_dir: Path, run_id: str) -> tuple[str, RunLayout]:
    """Create ``base_dir/<run_id>``, or ``<run_id>-2``, ``-3``, ... if taken.

    Never reuses an existing directory. Scenario directories are not created.
    """
    _check_run_id(run_id)
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    candidate, suffix = run_id, 1
    while True:
        try:
            (base / candidate).mkdir()
        except FileExistsError:
            suffix += 1
            candidate = f"{run_id}-{suffix}"
            continue
        return candidate, RunLayout(base / candidate)


# --- Hashing -----------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path, relative_to: Path) -> FileRef:
    """``FileRef`` of an existing file, its path stored POSIX-style relative to ``relative_to``."""
    path = Path(path)
    return FileRef(
        path=path.relative_to(relative_to).as_posix(),
        sha256=sha256_file(path),
        size=path.stat().st_size,
    )


# --- Writes ------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_raw(path: Path, data: bytes, relative_to: Path) -> FileRef:
    """Write raw evidence once. Raises ``FileExistsError`` if ``path`` exists.

    The returned ``FileRef.path`` is relative to ``relative_to`` (normally the
    scenario directory, per the model's ``FileRef`` contract).
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"raw evidence must be bytes, got {type(data).__name__}")
    path = Path(path)
    relative = path.relative_to(relative_to).as_posix()  # ValueError if outside
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return FileRef(path=relative, sha256=sha256_bytes(bytes(data)), size=len(data))


def write_json_atomic(path: Path, obj: Any) -> None:
    """Replace ``path`` atomically with ``obj`` as UTF-8 JSON (temp, fsync, ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --- Redaction ---------------------------------------------------------------


def redact_headers(
    headers: Mapping[str, Any], secret_names: Iterable[str] = ("x-api-key",)
) -> dict[str, str]:
    """Copy of ``headers`` with secret header values (case-insensitive name) redacted."""
    secret = {name.lower() for name in secret_names}
    return {
        str(name): REDACTED if str(name).lower() in secret else str(value)
        for name, value in headers.items()
    }


def redact_bytes(data: bytes, secrets: Iterable[str | bytes]) -> tuple[bytes, bool]:
    """Literal replacement of each non-empty secret; returns ``(bytes, replaced)``."""
    replaced = False
    marker = REDACTED.encode("ascii")
    for secret in secrets:
        needle = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if needle and needle in data:
            data = data.replace(needle, marker)
            replaced = True
    return data, replaced
