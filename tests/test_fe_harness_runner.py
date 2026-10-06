"""Tests for evaluation.datachat.runner, the FE harness scenario/suite orchestration.

Fully offline. A FakeMaui answers the real DataChatClient through
httpx.MockTransport and appends the matching runtime lines to the suite
stderr file and agent_runs JSONL, so the real runtime and agent_runs
collectors run against real temp files. The process, Git, artifact/drift
checks and DB reads are fakes; marker polling uses a fake clock. Faults are
injected per scenario number to exercise every INFRA_* and completion rule.
"""

import base64
import json
import os
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg
import pytest

from evaluation.datachat import runner as runner_module
from evaluation.datachat.client import DataChatClient
from evaluation.datachat.collectors.usage import UsageCollection, UserTokenState
from evaluation.datachat.contract import resolve_child_env
from evaluation.datachat.evidence import read_json
from evaluation.datachat.maui_process import ReadyResult
from evaluation.datachat.model import (
    Availability,
    EvidenceSourceRecord,
    InfraCode,
    Phase,
    ScenarioExecution,
    SuiteStatus,
    UsageEvidence,
)
from evaluation.datachat.runner import GitState, RunnerDeps, RunnerOptions, run_suite
from evaluation.datachat.scenarios import SCENARIOS, DriftResult

API_KEY = "sk-fe-eval-secret-0123"
EMAIL = "fe-eval@example.invalid"
PHASE_ORDER = [p.value for p in Phase]

BASE_ENV = {
    "DATACHAT_PROVIDER": "OpenAI",
    "DATACHAT_MODEL": "gpt-test",
    "DATACHAT_TOKEN_COST": "1",
    "PGHOST": "localhost",
    "PGDB": "maui",
    "PGUSER": "maui",
    "PGPWD": "pg-secret",
    "ENCRYPTION_KEY": "enc-secret",
    "OPENAI_API_KEY": "provider-secret",
}


def _png():
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00\x00")) + chunk(b"IEND", b"")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeMaui:
    """Scripted Maui. ``faults`` is a set of (fault, scenario_number)."""

    def __init__(self, faults=()):
        self.faults = set(faults)
        self.stderr: Path | None = None
        self.agent_runs: Path | None = None
        self.engine = False
        self.started = 0
        self.counter = 0
        self.calls = []

    def has(self, fault):
        return (fault, self.started) in self.faults

    def rid(self):
        self.counter += 1
        return f"{self.counter:016x}"

    def log(self, rid, message):
        with open(self.stderr, "ab") as handle:
            handle.write(f"2026-10-06T08:00:00Z INFO datachat.runtime request_id={rid} app_id=fe {message}\n".encode())

    def respond(self, status, payload, rid, raw=None):
        content = raw if raw is not None else json.dumps(payload).encode()
        return httpx.Response(status, content=content, headers={"X-Request-ID": rid, "content-type": "application/json"})

    def __call__(self, request):
        request.read()
        path = request.url.path
        self.calls.append((path, self.started))
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/enddatachat":
            return self.end()
        if path == "/startdatachat":
            return self.start()
        if path == "/datachat":
            return self.datachat(json.loads(request.content)["chat"])
        if path.startswith("/datachat/export/"):
            if self.has("export_fail"):
                raise httpx.ConnectError("export refused")
            return httpx.Response(200, content=b"region,count\nNord,10\n", headers={"X-Request-ID": self.rid()})
        raise AssertionError(path)

    def end(self):
        teardown = self.engine
        scenario = self.started if teardown else self.started + 1
        if teardown and ("teardown_fail", scenario) in self.faults:
            return httpx.Response(500, text="boom")
        rid = self.rid()
        if ("prev_engine", scenario) in self.faults and not teardown:
            self.faults.discard(("prev_engine", scenario))
            self.log(rid, "cleanup_result status=ok")
            return self.respond(200, {"Agent deleted succesfully": "active"}, rid)
        if self.engine:
            self.engine = False
            self.log(rid, "cleanup_result status=ok")
            return self.respond(200, {"Agent deleted succesfully": "active"}, rid)
        return self.respond(200, {"Agent was not active for this key": API_KEY}, rid)

    def start(self):
        self.started += 1
        rid = self.rid()
        self.engine = True
        if not self.has("no_fresh"):
            self.log(rid, "engine_init_tools engine=smolagents user=fe tool_count=9 sql_tools=0")
            self.log(rid, "engine_init_result engine=smolagents user=fe status=ok sql=off")
        return self.respond(200, {"Agent active": "active"}, rid)

    def datachat(self, question):
        if self.has("timeout"):
            raise httpx.ReadTimeout("slow")
        if self.has("connect"):
            raise httpx.ConnectError("refused")
        rid = self.rid()
        trace = not self.has("no_trace")
        self.log(rid, "datachat_request_start")
        if not self.has("no_end_marker"):
            self.log(rid, "datachat_request_end status=ok http_status=200")
        self.log("-", f"datachat_trace_status request_id={rid} user=fe engine=x trace_present={trace} structured_log_ok={trace} db_log_ok=True log_id=7")
        if trace and not self.has("agent_runs_missing"):
            record = {"namespace": "datachat", "request_id": rid, "question": question}
            with open(self.agent_runs, "ab") as handle:
                handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
        response = {"type": "str", "value": "answer"}
        if self.has("kind_error"):
            response = {"kind": "error", "code": "RUN_FAILED"}
        if self.has("image_ok"):
            response = {"type": "image", "value": str(base64.b64encode(_png()))}
        if self.has("image_bad"):
            response = {"type": "image", "value": "b'not-base64!'"}
        if self.has("export") or self.has("export_fail"):
            response = {"type": "table", "value": [], "download_url": "/datachat/export/tok"}
        return self.respond(200, {"response": response, "explanation": None, "log_id": 7}, rid)


class FakeProcess:
    def __init__(self, maui, argv, cwd, env, log_path, ready=True, write_log=True):
        self.maui, self.argv, self.env, self.log_path = maui, argv, env, Path(log_path)
        self.ready, self.write_log = ready, write_log
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_bytes(b"[INFO] Starting gunicorn\n" if self.write_log else b"")
        self.maui.stderr = self.log_path
        self.maui.agent_runs = Path(self.env["AGENT_RUNS_LOG_PATH"])
        self.maui.agent_runs.touch()  # CURRENT Maui opens the audit FileHandler eagerly
        return 1234

    def wait_ready(self, health_check, deadline_s, **kwargs):
        if not self.ready:
            return ReadyResult(False, "exited", 1, 3)
        return ReadyResult(health_check(), "ready", 1)

    def stop(self, grace_s=10):
        self.stopped = True
        return 0


class Harness:
    def __init__(self, tmp_path, faults=(), **overrides):
        self.tmp_path = tmp_path
        self.maui = FakeMaui(faults)
        self.processes = []
        self.usage_calls = []
        self.repo = tmp_path / "repo"
        (self.repo / "docs" / "datachat_functional_evaluation").mkdir(parents=True)
        (self.repo / "docs" / "datachat_functional_evaluation" / "evaluation_dataset.csv").write_bytes(b"region\nNord\n")
        self.process_kwargs = {k: overrides.pop(k) for k in ("ready", "write_log") if k in overrides}
        clock = FakeClock()

        def make_process(argv, cwd, env, log_path):
            process = FakeProcess(self.maui, argv, cwd, env, log_path, **self.process_kwargs)
            self.processes.append(process)
            return process

        def collect_usage(settings, request_id, primary_log_id):
            self.usage_calls.append((request_id, primary_log_id))
            if ("usage_down", self.maui.started) in self.maui.faults:
                return UsageCollection(EvidenceSourceRecord("usage", Availability.UNAVAILABLE, "db_error: OperationalError"), None)
            rows = () if ("usage_zero", self.maui.started) in self.maui.faults else ({"id": 7, "request_id": request_id},)
            evidence = UsageEvidence(request_id, "SELECT ...", settings.schema, "t", rows, primary_log_id, bool(rows))
            return UsageCollection(EvidenceSourceRecord("usage", Availability.PRESENT, locator={"row_count": len(rows)}), evidence)

        defaults = dict(
            git_state=lambda root: GitState("aa1ff33" + "0" * 33, "feat/x", False, ()),
            verify_artifacts=lambda root: (),
            check_drift=lambda root, scenarios: DriftResult(True, "design.md", "x" * 64),
            resolve_env=resolve_child_env,
            port_free=lambda host, port: True,
            make_process=make_process,
            make_client=lambda base, identity: DataChatClient(base, identity, transport=httpx.MockTransport(self.maui)),
            check_read_access=lambda settings: None,
            fetch_user=lambda settings, username: UserTokenState(42, username, 1000, "2099-12-31"),
            collect_usage=collect_usage,
            package_versions=lambda: {"httpx": "0.28.1"},
            clock=clock,
            sleep=clock.sleep,
            now=lambda: datetime(2026, 10, 6, 8, 15, 30, tzinfo=timezone.utc),
        )
        defaults.update(overrides)
        self.deps = RunnerDeps(**defaults)

    def options(self, **kw):
        values = dict(
            repo_root=self.repo, runs_dir=self.tmp_path / "runs", eval_label="fe-eval-v1",
            username="fe-eval-v1", email=EMAIL, port=5099, base_env=dict(BASE_ENV),
            marker_deadline_s=1.0, poll_interval_s=0.5,
        )
        values.update(kw)
        return RunnerOptions(**values)

    def run(self, scenarios=SCENARIOS, **kw):
        self.result = run_suite(self.options(**kw), API_KEY, self.deps, scenarios)
        return self.result

    def manifest(self):
        return read_json(self.result.run_dir / "run_manifest.json")

    def execution(self, sid):
        return ScenarioExecution.from_dict(read_json(self.result.run_dir / "scenarios" / sid / "execution.json"))


def _phase_status(execution):
    return {o.phase.value: o.status.value for o in execution.phases}


def _assert_not_run(harness, ids):
    for sid in ids:
        execution = harness.execution(sid)
        assert execution.status.value == "NOT_RUN"
        assert set(_phase_status(execution).values()) == {"SKIPPED"}


# --- Single scenario -------------------------------------------------------------


def test_happy_single_scenario(tmp_path):
    h = Harness(tmp_path)
    result = h.run(SCENARIOS[:1])

    assert result.status is SuiteStatus.COMPLETED and result.failure is None
    ex = h.execution("FE-01")
    assert ex.status.value == "EXECUTED"
    assert [o.phase.value for o in ex.phases] == PHASE_ORDER
    assert set(_phase_status(ex).values()) == {"OK"}
    flags = ex.flags
    assert flags.fresh_engine_proven and flags.reached_datachat and flags.teardown_verified
    assert flags.execution_complete and flags.evidence_complete and flags.scoreable
    assert not flags.previous_engine_present
    assert [c[0] for c in h.maui.calls].count("/datachat") == 1

    sdir = result.run_dir / "scenarios" / "FE-01"
    for rel in ["http/pre_cleanup.json", "http/start.json", "http/datachat.json", "http/datachat.request.bin",
                "http/datachat.response.bin", "http/teardown.json", "runtime/start.log", "runtime/datachat.log",
                "runtime/teardown.log", "agent_runs/record.jsonl", "usage/rows.json"]:
        assert (sdir / rel).is_file(), rel
    assert not (sdir / "artifacts").exists()
    assert ex.evidence["agent_runs"].availability is Availability.PRESENT
    assert ex.evidence["image"].availability is Availability.NOT_APPLICABLE
    assert ex.evidence["export"].availability is Availability.NOT_APPLICABLE
    assert h.usage_calls == [(ex.evidence["http_datachat"].locator["request_id"], 7)]
    assert h.processes[0].stopped


def test_datachat_kind_error_is_a_result_not_a_failure(tmp_path):
    h = Harness(tmp_path, {("kind_error", 1)})
    result = h.run(SCENARIOS[:2])
    assert result.status is SuiteStatus.COMPLETED
    assert h.execution("FE-01").status.value == "EXECUTED"
    assert h.execution("FE-01").flags.scoreable
    assert h.execution("FE-02").status.value == "EXECUTED"


def test_previous_engine_present_is_an_anomaly_not_a_failure(tmp_path):
    h = Harness(tmp_path, {("prev_engine", 2)})
    result = h.run(SCENARIOS[:2])
    assert result.status is SuiteStatus.COMPLETED
    ex = h.execution("FE-02")
    assert ex.flags.previous_engine_present is True
    assert ex.status.value == "EXECUTED" and ex.failure is None
    assert any("previous engine" in n for n in ex.notes)


def test_pre_cleanup_not_active_body_is_redacted_on_disk(tmp_path):
    h = Harness(tmp_path)
    h.run(SCENARIOS[:1])
    for path in h.result.run_dir.rglob("*"):
        if path.is_file():
            assert API_KEY.encode() not in path.read_bytes(), path
    meta = read_json(h.result.run_dir / "scenarios" / "FE-01" / "http" / "pre_cleanup.json")
    assert meta["response"]["redacted"] is True


# --- Blocking failures -------------------------------------------------------------


def test_freshness_failure_aborts_suite(tmp_path):
    h = Harness(tmp_path, {("no_fresh", 1)})
    result = h.run()
    assert result.status is SuiteStatus.ABORTED_INFRA
    assert result.failure.code is InfraCode.INFRA_FRESH_ENGINE_UNPROVEN
    ex = h.execution("FE-01")
    assert ex.status.value == "NOT_SCOREABLE_INFRA"
    assert ex.failure.code is InfraCode.INFRA_FRESH_ENGINE_UNPROVEN
    assert not ex.flags.fresh_engine_proven and not ex.flags.scoreable
    status = _phase_status(ex)
    assert status["START"] == "FAILED" and status["DATACHAT"] == "SKIPPED" and status["COLLECT"] == "SKIPPED"
    assert status["TEARDOWN"] == "OK"  # best-effort cleanup of the engine that may exist
    assert status["CLOSE"] == "OK"
    assert [c for c in h.maui.calls if c[0] == "/datachat"] == []
    _assert_not_run(h, [s.scenario_id for s in SCENARIOS[1:]])
    assert len(h.manifest()["scenarios"]) == 14


@pytest.mark.parametrize("fault, code", [("timeout", InfraCode.INFRA_HTTP_TIMEOUT), ("connect", InfraCode.INFRA_MAUI_UNAVAILABLE)])
def test_datachat_transport_failure(tmp_path, fault, code):
    h = Harness(tmp_path, {(fault, 1)})
    result = h.run(SCENARIOS[:3])
    assert result.failure.code is code
    ex = h.execution("FE-01")
    assert ex.status.value == "NOT_SCOREABLE_INFRA"
    assert ex.flags.reached_datachat is True
    assert ex.evidence["http_datachat"].availability is Availability.UNAVAILABLE
    assert (h.result.run_dir / "scenarios" / "FE-01" / "http" / "datachat.request.bin").is_file()
    assert not (h.result.run_dir / "scenarios" / "FE-01" / "http" / "datachat.response.bin").exists()
    assert _phase_status(ex)["TEARDOWN"] == "OK"  # cleanup attempted
    assert [c[0] for c in h.maui.calls].count("/datachat") == 1
    _assert_not_run(h, ["FE-02", "FE-03"])


@pytest.mark.parametrize(
    "fault, source",
    [("no_end_marker", "runtime_datachat"), ("agent_runs_missing", "agent_runs"), ("usage_down", "usage")],
)
def test_required_evidence_unavailable(tmp_path, fault, source):
    h = Harness(tmp_path, {(fault, 1)})
    result = h.run(SCENARIOS[:2])
    assert result.failure.code is InfraCode.INFRA_EVIDENCE_UNAVAILABLE
    ex = h.execution("FE-01")
    assert ex.status.value == "NOT_SCOREABLE_INFRA"
    assert ex.evidence[source].availability is Availability.UNAVAILABLE
    assert _phase_status(ex)["COLLECT"] == "FAILED"
    assert not ex.flags.scoreable and not ex.flags.evidence_complete
    _assert_not_run(h, ["FE-02"])


def test_agent_runs_expected_absent_stays_scoreable(tmp_path):
    h = Harness(tmp_path, {("no_trace", 1)})
    result = h.run(SCENARIOS[:1])
    assert result.status is SuiteStatus.COMPLETED
    ex = h.execution("FE-01")
    assert ex.evidence["agent_runs"].availability is Availability.EXPECTED_ABSENT
    assert ex.evidence["agent_runs"].reason == "trace_absent"
    assert ex.flags.scoreable and ex.flags.evidence_complete


def test_usage_zero_rows_stays_scoreable(tmp_path):
    h = Harness(tmp_path, {("usage_zero", 1)})
    h.run(SCENARIOS[:1])
    ex = h.execution("FE-01")
    assert ex.flags.scoreable
    rows = read_json(h.result.run_dir / "scenarios" / "FE-01" / "usage" / "rows.json")
    assert rows["rows"] == [] and rows["schema_version"] == 1


def test_valid_image_persisted(tmp_path):
    h = Harness(tmp_path, {("image_ok", 1)})
    h.run(SCENARIOS[:1])
    sdir = h.result.run_dir / "scenarios" / "FE-01" / "artifacts"
    assert (sdir / "image.png").read_bytes() == _png()
    assert read_json(sdir / "image.json")["decode_status"] == "OK"


def test_malformed_image_is_not_infra(tmp_path):
    h = Harness(tmp_path, {("image_bad", 1)})
    result = h.run(SCENARIOS[:1])
    assert result.status is SuiteStatus.COMPLETED
    ex = h.execution("FE-01")
    assert ex.status.value == "EXECUTED" and ex.flags.scoreable
    assert ex.evidence["image"].availability is Availability.PRESENT
    sdir = h.result.run_dir / "scenarios" / "FE-01" / "artifacts"
    assert not (sdir / "image.png").exists()
    assert read_json(sdir / "image.json")["decode_status"] == "BASE64_INVALID"


def test_export_fetched_before_teardown(tmp_path):
    h = Harness(tmp_path, {("export", 1)})
    h.run(SCENARIOS[:1])
    paths = [c[0] for c in h.maui.calls]
    assert paths.index("/datachat/export/tok") < len(paths) - 1 - paths[::-1].index("/enddatachat")
    ex = h.execution("FE-01")
    assert ex.evidence["export"].availability is Availability.PRESENT
    sdir = h.result.run_dir / "scenarios" / "FE-01"
    assert (sdir / "artifacts" / "export.csv").read_bytes() == b"region,count\nNord,10\n"
    assert (sdir / "http" / "export.json").is_file()


def test_export_failure_keeps_scenario_scoreable(tmp_path):
    h = Harness(tmp_path, {("export_fail", 1)})
    result = h.run(SCENARIOS[:2])
    assert result.status is SuiteStatus.COMPLETED
    ex = h.execution("FE-01")
    assert ex.evidence["export"].availability is Availability.UNAVAILABLE
    assert ex.status.value == "EXECUTED"
    assert ex.flags.scoreable is True
    assert ex.flags.evidence_complete is False


def test_teardown_only_failure(tmp_path):
    h = Harness(tmp_path, {("teardown_fail", 1)})
    result = h.run(SCENARIOS[:3])

    ex = h.execution("FE-01")
    assert ex.status.value == "EXECUTED"
    assert ex.failure is None
    assert ex.flags.scoreable is True
    assert ex.flags.teardown_verified is False
    status = _phase_status(ex)
    assert status["TEARDOWN"] == "FAILED" and status["CLOSE"] == "OK"
    assert ex.phases[PHASE_ORDER.index("TEARDOWN")].failure.code is InfraCode.INFRA_TEARDOWN_UNVERIFIED
    assert result.status is SuiteStatus.ABORTED_INFRA
    assert result.failure.code is InfraCode.INFRA_TEARDOWN_UNVERIFIED
    manifest = h.manifest()
    assert manifest["failure"]["code"] == "INFRA_TEARDOWN_UNVERIFIED"
    assert [e["status"] for e in manifest["scenarios"]] == ["EXECUTED", "NOT_RUN", "NOT_RUN"]
    _assert_not_run(h, ["FE-02", "FE-03"])


# --- Suite ------------------------------------------------------------------------


def test_full_happy_suite(tmp_path):
    h = Harness(tmp_path)
    result = h.run()
    assert result.status is SuiteStatus.COMPLETED and result.failure is None
    manifest = h.manifest()
    assert manifest["status"] == "COMPLETED" and manifest["ended_at"]
    assert [e["scenario_id"] for e in manifest["scenarios"]] == [s.scenario_id for s in SCENARIOS]
    assert {e["status"] for e in manifest["scenarios"]} == {"EXECUTED"}
    assert manifest["token_balance_at_preflight"] == 1000
    assert manifest["eval_identity"]["user_id"] == 42
    assert manifest["comparable"] is False  # not official
    for scenario in SCENARIOS:
        ex = h.execution(scenario.scenario_id)
        assert ex.flags.teardown_verified and ex.flags.scoreable
        assert ex.question == scenario.question
    starts = [c for c in h.maui.calls if c[0] == "/startdatachat"]
    assert len(starts) == 14
    rids = {h.execution(s.scenario_id).evidence["http_datachat"].locator["request_id"] for s in SCENARIOS}
    assert len(rids) == 14


def test_abort_midway(tmp_path):
    h = Harness(tmp_path, {("no_end_marker", 6)})
    result = h.run()
    assert result.status is SuiteStatus.ABORTED_INFRA
    statuses = [e["status"] for e in h.manifest()["scenarios"]]
    assert statuses == ["EXECUTED"] * 5 + ["NOT_SCOREABLE_INFRA"] + ["NOT_RUN"] * 8
    assert h.processes[0].stopped


def test_raw_evidence_written_before_manifest_references_it(tmp_path, monkeypatch):
    real = runner_module.write_json_atomic
    checked = []

    def checking(path, obj):
        path = Path(path)
        if path.name == "execution.json":
            sdir = path.parent
            for record in obj["evidence"].values():
                for ref in record["files"]:
                    assert (sdir / ref["path"]).is_file(), ref
            for refs in obj["requests"].values():
                for ref in refs:
                    assert (sdir / ref).is_file(), ref
            checked.append(path)
        real(path, obj)

    monkeypatch.setattr(runner_module, "write_json_atomic", checking)
    h = Harness(tmp_path, {("image_ok", 1), ("export", 2)})
    h.run(SCENARIOS[:2])
    assert len(checked) > 10


def test_unexpected_exception_keeps_run_inspectable(tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError(f"collector bug {API_KEY}")

    h = Harness(tmp_path, collect_usage=boom)
    with pytest.raises(RuntimeError):
        h.run(SCENARIOS[:2])
    run_dir = next((tmp_path / "runs").iterdir())
    manifest = read_json(run_dir / "run_manifest.json")
    assert manifest["status"] == "RUNNING"  # interrupted, never rewritten to ABORTED_INFRA
    tracebacks = list((run_dir / "failures").iterdir())
    assert len(tracebacks) == 1
    text = tracebacks[0].read_text()
    assert "collector bug" in text and API_KEY not in text
    assert read_json(run_dir / "scenarios" / "FE-01" / "execution.json")["status"] == "RUNNING"
    assert h.processes[0].stopped


# --- Preflight ------------------------------------------------------------------


def _assert_no_run(h, code=InfraCode.INFRA_PREFLIGHT):
    assert h.result.status is SuiteStatus.ABORTED_INFRA
    assert h.result.failure.code is code
    assert h.result.run_id is None and h.result.run_dir is None
    assert not (h.tmp_path / "runs").exists()
    assert h.processes == []


def test_preflight_artifact_mismatch(tmp_path):
    from evaluation.datachat.contract import ArtifactCheck

    bad = ArtifactCheck("ground_truth.json", "docs/x", True, "a" * 64, True, "b" * 64, False)
    h = Harness(tmp_path, verify_artifacts=lambda root: (bad,))
    h.run()
    _assert_no_run(h)
    assert "ground_truth.json" in h.result.failure.detail


def test_preflight_scenario_drift(tmp_path):
    h = Harness(tmp_path, check_drift=lambda root, scenarios: DriftResult(False, "d", None, ("FE-05: question differs",)))
    h.run()
    _assert_no_run(h)


def test_preflight_dirty_tree_official_vs_not(tmp_path):
    dirty = lambda root: GitState("aa1ff33" + "0" * 33, "feat/x", True, ("main.py",))
    h = Harness(tmp_path, git_state=dirty)
    h.run(official=True)
    _assert_no_run(h)

    h2 = Harness(tmp_path / "second", git_state=dirty)
    result = h2.run(SCENARIOS[:1])
    assert result.status is SuiteStatus.COMPLETED
    assert result.run_id.endswith("_aa1ff33_dirty")
    manifest = h2.manifest()
    assert manifest["comparable"] is False and manifest["maui"]["dirty_files"] == ["main.py"]


def test_preflight_port_busy(tmp_path):
    h = Harness(tmp_path, port_free=lambda host, port: False)
    h.run()
    _assert_no_run(h)


def test_preflight_missing_model(tmp_path):
    h = Harness(tmp_path)
    h.run(base_env={**BASE_ENV, "DATACHAT_MODEL": ""})
    _assert_no_run(h)
    assert "DATACHAT_MODEL" in h.result.failure.detail


def _assert_aborted_after_launch(h, code):
    assert h.result.status is SuiteStatus.ABORTED_INFRA
    assert h.result.failure.code is code
    manifest = h.manifest()
    assert manifest["status"] == "ABORTED_INFRA" and manifest["failure"]["code"] == code.value
    assert [e["status"] for e in manifest["scenarios"]] == ["NOT_RUN"] * 14
    assert h.processes[0].stopped
    assert [c for c in h.maui.calls if c[0] == "/startdatachat"] == []


def test_preflight_readiness_failure(tmp_path):
    h = Harness(tmp_path, ready=False)
    h.run()
    _assert_aborted_after_launch(h, InfraCode.INFRA_MAUI_UNAVAILABLE)


def test_preflight_stderr_empty(tmp_path):
    h = Harness(tmp_path, write_log=False)
    h.run()
    _assert_aborted_after_launch(h, InfraCode.INFRA_PREFLIGHT)


def test_preflight_db_read_failure(tmp_path):
    h = Harness(tmp_path, check_read_access=lambda settings: "db_error: OperationalError")
    h.run()
    _assert_aborted_after_launch(h, InfraCode.INFRA_PREFLIGHT)


def test_preflight_user_missing_and_lookup_error(tmp_path):
    h = Harness(tmp_path, fetch_user=lambda settings, username: None)
    h.run()
    _assert_aborted_after_launch(h, InfraCode.INFRA_PREFLIGHT)

    def broken(settings, username):
        raise psycopg.OperationalError("down")

    h2 = Harness(tmp_path / "second", fetch_user=broken)
    h2.run()
    _assert_aborted_after_launch(h2, InfraCode.INFRA_PREFLIGHT)


def test_user_looked_up_by_email(tmp_path):
    seen = []
    h = Harness(tmp_path, fetch_user=lambda settings, username: seen.append(username) or UserTokenState(1, username, 100, "x"))
    h.run(SCENARIOS[:1])
    assert seen == [EMAIL]


def test_preflight_insufficient_tokens(tmp_path):
    # 2 × 14 × 1 + margin 2 = 30
    h = Harness(tmp_path, fetch_user=lambda settings, username: UserTokenState(42, username, 29, "x"))
    h.run()
    _assert_aborted_after_launch(h, InfraCode.INFRA_TOKEN_BUDGET)
    assert h.manifest()["token_balance_at_preflight"] == 29
    assert h.manifest()["runtime"]["token_budget"]["required"] == 30

    h2 = Harness(tmp_path / "second", fetch_user=lambda settings, username: UserTokenState(42, username, 30, "x"))
    assert h2.run(SCENARIOS[:1]).status is SuiteStatus.COMPLETED


def test_successful_preflight_manifest(tmp_path):
    h = Harness(tmp_path)
    h.run(SCENARIOS[:1])
    manifest = h.manifest()
    assert manifest["schema_version"] == 1
    assert manifest["maui"]["git_sha"].startswith("aa1ff33")
    assert manifest["runtime"]["gunicorn_argv"][-1] == "127.0.0.1:5099"
    assert manifest["runtime"]["preflight_sweep"]["outcome"] == "not_active"
    assert manifest["datachat_config"]["secrets_present"]["OPENAI_API_KEY"] is True
    env_json = (h.result.run_dir / "environment.json").read_text()
    for secret in ("pg-secret", "enc-secret", "provider-secret", API_KEY):
        assert secret not in json.dumps(manifest) and secret not in env_json
    assert h.processes[0].env["DATACHAT_SQL_ENABLED"] == "false"
    assert h.processes[0].env["AGENT_RUNS_LOG_PATH"] == str(h.result.run_dir / "agent_runs.log")
    assert "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] != "provider-secret"


def test_official_subset_rejected_before_anything_runs(tmp_path):
    h = Harness(tmp_path)
    with pytest.raises(ValueError, match="full frozen scenario catalogue"):
        h.run(SCENARIOS[:1], official=True)
    assert h.processes == []
    assert not (tmp_path / "runs").exists()
