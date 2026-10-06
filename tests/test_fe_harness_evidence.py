"""Tests for evaluation.datachat.evidence, the harness evidence persistence primitives.

Offline unit tests confined to pytest's tmp_path: no Maui, DB, network or
providers. They pin the run layout and path safety, run-directory collision
suffixes, SHA-256 helpers, the write-once raw evidence vs atomically
replaceable metadata boundary, and header/body redaction.
"""

import hashlib
import json

import pytest

from evaluation.datachat.evidence import (
    REDACTED,
    RunLayout,
    create_run_dir,
    file_ref,
    read_json,
    redact_bytes,
    redact_headers,
    safe_child,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_raw,
)
from evaluation.datachat.model import FileRef

API_KEY = "sk-eval-0123456789"


# --- Layout ------------------------------------------------------------------


def test_constructing_layout_creates_nothing(tmp_path):
    root = tmp_path / "run"
    layout = RunLayout(root)
    layout.scenario_dir("FE-01")
    layout.http_paths("FE-01", "datachat")
    assert not root.exists()


def test_suite_level_paths(tmp_path):
    layout = RunLayout(tmp_path)
    assert layout.run_manifest == tmp_path / "run_manifest.json"
    assert layout.environment == tmp_path / "environment.json"
    assert layout.maui_stderr == tmp_path / "maui_stderr.log"
    assert layout.agent_runs_log == tmp_path / "agent_runs.log"
    assert layout.plots_dir == tmp_path / "plots"
    assert layout.failures_dir == tmp_path / "failures"


def test_scenario_paths(tmp_path):
    layout = RunLayout(tmp_path)
    scenario = tmp_path / "scenarios" / "FE-11"
    assert layout.scenario_dir("FE-11") == scenario
    assert layout.execution("FE-11") == scenario / "execution.json"
    assert layout.runtime_log("FE-11", "start") == scenario / "runtime" / "start.log"
    assert layout.runtime_log("FE-11", "teardown") == scenario / "runtime" / "teardown.log"
    assert layout.agent_runs_record("FE-11") == scenario / "agent_runs" / "record.jsonl"
    assert layout.usage_rows("FE-11") == scenario / "usage" / "rows.json"
    assert layout.artifact("FE-11", "image.png") == scenario / "artifacts" / "image.png"
    assert layout.artifact("FE-11", "export.csv") == scenario / "artifacts" / "export.csv"


def test_http_attempt_naming(tmp_path):
    layout = RunLayout(tmp_path)
    http = tmp_path / "scenarios" / "FE-01" / "http"
    assert layout.http_paths("FE-01", "teardown") == (
        http / "teardown.json",
        http / "teardown.request.bin",
        http / "teardown.response.bin",
    )
    assert layout.http_paths("FE-01", "teardown", attempt=2) == (
        http / "teardown.2.json",
        http / "teardown.2.request.bin",
        http / "teardown.2.response.bin",
    )


@pytest.mark.parametrize("attempt", [0, -1])
def test_http_attempt_below_one_rejected(tmp_path, attempt):
    with pytest.raises(ValueError):
        RunLayout(tmp_path).http_paths("FE-01", "teardown", attempt=attempt)


@pytest.mark.parametrize("scenario_id", ["FE-1", "FE-001", "fe-01", "../FE-01", "FE-01/..", ""])
def test_invalid_scenario_id_rejected(tmp_path, scenario_id):
    with pytest.raises(ValueError):
        RunLayout(tmp_path).scenario_dir(scenario_id)


@pytest.mark.parametrize("key", ["health", "../datachat", "DATACHAT", ""])
def test_invalid_http_key_rejected(tmp_path, key):
    with pytest.raises(ValueError):
        RunLayout(tmp_path).http_paths("FE-01", key)


def test_invalid_runtime_and_artifact_names_rejected(tmp_path):
    layout = RunLayout(tmp_path)
    with pytest.raises(ValueError):
        layout.runtime_log("FE-01", "export")
    with pytest.raises(ValueError):
        layout.artifact("FE-01", "../../run_manifest.json")


@pytest.mark.parametrize("relative", ["../outside", "a/../../outside", "/etc/passwd"])
def test_safe_child_rejects_traversal(tmp_path, relative):
    with pytest.raises(ValueError):
        safe_child(tmp_path, relative)


def test_safe_child_accepts_nested_path(tmp_path):
    assert safe_child(tmp_path, "scenarios/FE-01") == tmp_path / "scenarios" / "FE-01"


# --- Run creation ------------------------------------------------------------


def test_create_run_dir_and_collisions(tmp_path):
    run_id = "20261006T081530Z_aa1ff33"
    first_id, first = create_run_dir(tmp_path, run_id)
    second_id, second = create_run_dir(tmp_path, run_id)
    third_id, third = create_run_dir(tmp_path, run_id)

    assert first_id == run_id
    assert second_id == f"{run_id}-2"
    assert third_id == f"{run_id}-3"
    assert first.root == tmp_path / run_id and first.root.is_dir()
    assert second.root.is_dir() and third.root.is_dir()
    assert list(first.root.iterdir()) == []


def test_create_run_dir_preserves_existing_content(tmp_path):
    existing = tmp_path / "r1"
    existing.mkdir()
    (existing / "run_manifest.json").write_text("{}")
    run_id, _ = create_run_dir(tmp_path, "r1")
    assert run_id == "r1-2"
    assert (existing / "run_manifest.json").read_text() == "{}"


@pytest.mark.parametrize("run_id", ["../escape", "a/b", ""])
def test_create_run_dir_rejects_unsafe_run_id(tmp_path, run_id):
    with pytest.raises(ValueError):
        create_run_dir(tmp_path, run_id)


# --- Hashing -----------------------------------------------------------------


def test_sha256_helpers_match_hashlib(tmp_path):
    data = b"a,b\n1,2\n" * 1000
    expected = hashlib.sha256(data).hexdigest()
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    assert sha256_bytes(data) == expected
    assert sha256_file(path) == expected
    assert len(expected) == 64 and expected == expected.lower()

    ref = file_ref(path, relative_to=tmp_path)
    assert ref == FileRef(path="f.bin", sha256=expected, size=len(data))


# --- Raw evidence ------------------------------------------------------------


def test_write_raw_preserves_bytes_and_creates_parents(tmp_path):
    layout = RunLayout(tmp_path)
    scenario = layout.scenario_dir("FE-11")
    _, _, response = layout.http_paths("FE-11", "datachat")
    data = bytes(range(256)) * 3

    ref = write_raw(response, data, relative_to=scenario)

    assert response.read_bytes() == data
    assert ref == FileRef(
        path="http/datachat.response.bin", sha256=hashlib.sha256(data).hexdigest(), size=len(data)
    )
    assert not (scenario / "artifacts").exists()


def test_write_raw_never_overwrites(tmp_path):
    path = tmp_path / "runtime" / "start.log"
    write_raw(path, b"original", relative_to=tmp_path)
    with pytest.raises(FileExistsError):
        write_raw(path, b"x", relative_to=tmp_path)
    assert path.read_bytes() == b"original"


def test_write_raw_rejects_path_outside_relative_root(tmp_path):
    with pytest.raises(ValueError):
        write_raw(tmp_path / "elsewhere.bin", b"x", relative_to=tmp_path / "scenarios")
    assert not (tmp_path / "elsewhere.bin").exists()


def test_write_raw_rejects_text(tmp_path):
    with pytest.raises(TypeError):
        write_raw(tmp_path / "a.log", "text", relative_to=tmp_path)


# --- Atomic JSON -------------------------------------------------------------


def test_write_json_atomic_writes_readable_unicode_json(tmp_path):
    path = tmp_path / "scenarios" / "FE-01" / "execution.json"
    obj = {"question": "Qual è il fatturato medio?", "schema_version": 1}
    write_json_atomic(path, obj)

    text = path.read_text(encoding="utf-8")
    assert "è" in text
    assert json.loads(text) == obj
    assert read_json(path) == obj
    assert [p.name for p in path.parent.iterdir()] == ["execution.json"]


def test_write_json_atomic_replaces_existing_metadata(tmp_path):
    path = tmp_path / "run_manifest.json"
    write_json_atomic(path, {"status": "RUNNING"})
    write_json_atomic(path, {"status": "COMPLETED"})
    assert read_json(path) == {"status": "COMPLETED"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run_manifest.json"]


def test_write_json_atomic_failure_keeps_previous_content(tmp_path):
    path = tmp_path / "execution.json"
    write_json_atomic(path, {"status": "RUNNING"})
    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": object()})
    assert read_json(path) == {"status": "RUNNING"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["execution.json"]


def test_raw_is_write_once_but_metadata_is_replaceable(tmp_path):
    raw = tmp_path / "datachat.response.bin"
    meta = tmp_path / "datachat.json"
    write_raw(raw, b"v1", relative_to=tmp_path)
    write_json_atomic(meta, {"v": 1})

    with pytest.raises(FileExistsError):
        write_raw(raw, b"v2", relative_to=tmp_path)
    write_json_atomic(meta, {"v": 2})

    assert raw.read_bytes() == b"v1"
    assert read_json(meta) == {"v": 2}


# --- Redaction ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["X-API-KEY", "x-api-key", "X-Api-Key"])
def test_redact_headers_case_insensitive(name):
    source = {name: API_KEY, "Content-Type": "application/json", "Content-Length": 12}
    snapshot = dict(source)
    redacted = redact_headers(source)
    assert redacted == {name: REDACTED, "Content-Type": "application/json", "Content-Length": "12"}
    assert source == snapshot


def test_redact_headers_custom_secret_names():
    redacted = redact_headers({"Authorization": "Bearer t", "Accept": "*/*"}, secret_names={"AUTHORIZATION"})
    assert redacted == {"Authorization": REDACTED, "Accept": "*/*"}


def test_redact_bytes_replaces_echoed_key():
    body = f'{{"message": "Agent was not active for this key {API_KEY}"}}'.encode()
    redacted, replaced = redact_bytes(body, [API_KEY])
    assert replaced is True
    assert API_KEY.encode() not in redacted
    assert redacted == b'{"message": "Agent was not active for this key <redacted>"}'


def test_redact_bytes_multiple_and_empty_secrets():
    redacted, replaced = redact_bytes(b"a=SECRET1 b=SECRET2 c=SECRET1", ["SECRET1", b"SECRET2", "", b""])
    assert replaced is True
    assert redacted == b"a=<redacted> b=<redacted> c=<redacted>"


def test_redact_bytes_reports_no_replacement():
    assert redact_bytes(b"clean body", [API_KEY, ""]) == (b"clean body", False)


def test_redaction_precedes_hashing(tmp_path):
    body = f"key {API_KEY}".encode()
    stored, replaced = redact_bytes(body, [API_KEY])
    ref = write_raw(tmp_path / "pre_cleanup.response.bin", stored, relative_to=tmp_path)
    assert replaced
    assert ref.sha256 == hashlib.sha256(b"key <redacted>").hexdigest()
    assert API_KEY.encode() not in (tmp_path / "pre_cleanup.response.bin").read_bytes()
