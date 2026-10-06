"""Scenario and suite orchestration of the DataChat Functional Evaluation harness.

Composes the contract, process, client, collector and evidence modules:

- suite preflight, in the frozen order: repository identity, artifact hashes
  and catalogue drift, port, effective environment (all before anything is
  written), then run directory and ``RUNNING`` manifest, Maui launch,
  ``/health``, stderr capture, DB read, evaluation user, token budget and a
  defensive ``/enddatachat`` sweep;
- per scenario: ``PRE_CLEANUP`` → ``START`` (freshness proof) → ``DATACHAT``
  (exactly one request) → ``COLLECT`` → ``TEARDOWN`` (verified) → ``CLOSE``;
- the first blocking ``INFRA_*`` aborts the suite; remaining scenarios are
  written as ``NOT_RUN``; Maui is always stopped.

Raw evidence is written (write-once) before any manifest references it, and
manifests are replaced atomically after every phase. A teardown-only failure
leaves an otherwise valid scenario ``EXECUTED`` but aborts the suite.

An unexpected harness exception is not an ``INFRA_*`` condition: its
traceback is saved under ``failures/``, manifests keep whatever state was
already persisted (``RUNNING`` means interrupted), Maui is stopped and the
exception propagates.

No grading happens here: DataChat answers, ``kind:error`` bodies and HTTP
error statuses from a well-formed request are evidence, not failures.
"""

import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psycopg

from evaluation.datachat import contract
from evaluation.datachat.client import DataChatClient, ExchangeResult, Identity
from evaluation.datachat.collectors import agent_runs as agent_runs_collector
from evaluation.datachat.collectors import runtime_log
from evaluation.datachat.collectors import usage as usage_collector
from evaluation.datachat.collectors.artifacts import (
    WRAPPER_MISSING,
    ImageDecode,
    decode_image_value,
    export_artifact,
    image_artifact,
)
from evaluation.datachat.collectors.usage import DbSettings
from evaluation.datachat.evidence import (
    RunLayout,
    create_run_dir,
    file_ref,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_raw,
)
from evaluation.datachat.maui_process import MauiProcess, port_free
from evaluation.datachat.model import (
    Availability,
    EvaluationRun,
    EvidenceSourceRecord,
    Failure,
    FileRef,
    InfraCode,
    Phase,
    PhaseStatus,
    ScenarioExecution,
    ScenarioStatus,
    SuiteStatus,
    TransportError,
    utc_now,
)
from evaluation.datachat.scenarios import SCENARIOS, Scenario, check_drift

SOURCES = (
    "http_datachat",
    "runtime_start",
    "runtime_datachat",
    "runtime_teardown",
    "agent_runs",
    "usage",
    "image",
    "export",
)
REQUIRED_SOURCES = ("http_datachat", "runtime_start", "runtime_datachat", "agent_runs", "usage")

MARKER_START = "engine_init_result"
MARKER_DATACHAT = "datachat_request_end"
MARKER_TEARDOWN = "cleanup_result"

END_DELETED = "Agent deleted succesfully"  # sic, CURRENT route spelling
END_NOT_ACTIVE = "Agent was not active for this key"
START_ACTIVE = "Agent active"
NOT_ENOUGH_TOKENS = "Not enough tokens"

_REQUEST_ID = re.compile(r"[0-9a-f]{16}")
_PHASES = list(Phase)


# --- Options, dependencies, results -------------------------------------------


@dataclass(frozen=True)
class RunnerOptions:
    """Operator-supplied run settings. Holds no secret; the API key is passed separately.

    ``email`` is sent as ``X-USER-EMAIL`` and is also the ``users.username``
    value CURRENT routes look up; ``username`` is sent as ``X-USER-NAME``.
    ``token_margin`` defaults to 2 × ``DATACHAT_TOKEN_COST`` (harness-local
    default; the authority only requires "+ margin"). Deadlines are local
    defaults too: the authority requires them to be bounded, not specific.
    ``base_env`` (default ``os.environ``) may hold secrets and is never persisted.
    """

    repo_root: Path
    runs_dir: Path
    eval_label: str
    username: str
    email: str
    port: int = contract.DEFAULT_PORT
    official: bool = False
    token_margin: int | None = None
    health_deadline_s: float = 60.0
    marker_deadline_s: float = 15.0
    poll_interval_s: float = 0.2
    stop_grace_s: float = 10.0
    base_env: Mapping[str, str] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class GitState:
    sha: str
    branch: str
    dirty: bool
    dirty_files: tuple[str, ...]


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout


def read_git_state(repo_root: Path) -> GitState:
    """Raises ``OSError`` / ``subprocess.CalledProcessError`` when Git cannot answer."""
    status = _git(repo_root, "status", "--porcelain")
    dirty_files = tuple(line[3:] for line in status.splitlines() if line.strip())
    return GitState(
        sha=_git(repo_root, "rev-parse", "HEAD").strip(),
        branch=_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        dirty=bool(dirty_files),
        dirty_files=dirty_files,
    )


@dataclass
class RunnerDeps:
    """Collaborators, replaceable in tests. Defaults are the real harness modules."""

    git_state: Callable[[Path], GitState] = read_git_state
    verify_artifacts: Callable[[Path], tuple] = contract.verify_artifacts
    check_drift: Callable[..., Any] = check_drift
    resolve_env: Callable[..., contract.ChildEnv] = contract.resolve_child_env
    port_free: Callable[[str, int], bool] = port_free
    make_process: Callable[[list[str], Path, Mapping[str, str], Path], MauiProcess] = MauiProcess
    make_client: Callable[[str, Identity], DataChatClient] = DataChatClient
    check_read_access: Callable[[DbSettings], str | None] = usage_collector.check_read_access
    fetch_user: Callable[[DbSettings, str], Any] = usage_collector.fetch_user_token_state
    collect_runtime: Callable[..., tuple] = runtime_log.collect_runtime
    collect_agent_runs: Callable[..., tuple] = agent_runs_collector.collect_agent_runs
    collect_usage: Callable[..., usage_collector.UsageCollection] = usage_collector.collect_usage
    package_versions: Callable[[], dict] = contract.package_versions
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


@dataclass(frozen=True)
class RunResult:
    run_id: str | None
    status: SuiteStatus
    failure: Failure | None
    run_dir: Path | None


class _Blocking(Exception):
    """Internal unwinding of a scenario on its first blocking INFRA_* condition."""

    def __init__(self, failure: Failure):
        super().__init__(failure.summary)
        self.failure = failure


# --- Small pure helpers ----------------------------------------------------------


def _json_body(data: bytes | None) -> Any:
    if not data:
        return None
    try:
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _valid_request_id(value: str | None) -> bool:
    return isinstance(value, str) and bool(_REQUEST_ID.fullmatch(value))


def transport_code(error: TransportError) -> InfraCode:
    """Client timeout → ``INFRA_HTTP_TIMEOUT``; any other no-response failure → Maui unavailable."""
    return InfraCode.INFRA_HTTP_TIMEOUT if "Timeout" in error.type else InfraCode.INFRA_MAUI_UNAVAILABLE


def end_outcome(result: ExchangeResult) -> str | None:
    """``"deleted"`` / ``"not_active"`` for an accepted ``/enddatachat`` reply, else None."""
    response = result.exchange.response
    if response is None or response.status != 200:
        return None
    body = _json_body(result.response_bytes)
    if not isinstance(body, dict):
        return None
    if END_DELETED in body:
        return "deleted"
    if END_NOT_ACTIVE in body:
        return "not_active"
    return None


def _not_enough_tokens(body: Any) -> bool:
    return isinstance(body, dict) and body.get("error") == NOT_ENOUGH_TOKENS


def freshness_proven(start_lines: bytes) -> bool:
    """Both freshness records: ``engine_init_result status=ok sql=off`` and ``engine_init_tools sql_tools=0``."""
    result_ok = tools_ok = False
    for line in start_lines.splitlines():
        tokens = set(line.split())
        if b"engine_init_result" in tokens and {b"status=ok", b"sql=off"} <= tokens:
            result_ok = True
        if b"engine_init_tools" in tokens and b"sql_tools=0" in tokens:
            tools_ok = True
    return result_ok and tools_ok


def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def make_run_id(now: datetime, git_sha: str, dirty: bool) -> str:
    return f"{now:%Y%m%dT%H%M%SZ}_{git_sha[:7]}" + ("_dirty" if dirty else "")


# --- Scenario ------------------------------------------------------------------


class _ScenarioRun:
    def __init__(self, suite: "_Suite", scenario: Scenario):
        self.suite = suite
        self.deps = suite.deps
        self.layout = suite.layout
        self.sid = scenario.scenario_id
        self.question = scenario.question
        self.dir = self.layout.scenario_dir(self.sid)
        self.ex = ScenarioExecution.new(suite.run_id, self.sid, scenario.ordinal, scenario.question)
        self.engine_may_exist = False
        self.datachat_rid: str | None = None
        self.datachat_body: Any = None
        self.datachat_response = False
        self.stderr_offset = 0
        self.agent_runs_offset = 0

    # Persistence

    def save(self) -> None:
        write_json_atomic(self.layout.execution(self.sid), self.ex.to_dict())

    def persist_http(self, phase: Phase, result: ExchangeResult) -> list[FileRef]:
        """Raw bodies first (write-once), then the exchange metadata."""
        exchange = result.exchange
        meta, request_path, response_path = self.layout.http_paths(self.sid, exchange.key, exchange.attempt)
        refs = []
        if result.request_bytes is not None:
            refs.append(write_raw(request_path, result.request_bytes, relative_to=self.dir))
        if result.response_bytes is not None:
            refs.append(write_raw(response_path, result.response_bytes, relative_to=self.dir))
        write_json_atomic(meta, exchange.to_dict())
        self.ex.requests.setdefault(exchange.key, []).append(meta.relative_to(self.dir).as_posix())
        outcome = self._outcome(phase)
        if exchange.key not in outcome.request_keys:
            outcome.request_keys.append(exchange.key)
        return refs

    def persist_runtime(self, name: str, record: EvidenceSourceRecord, payload: bytes) -> None:
        if payload:
            ref = write_raw(self.layout.runtime_log(self.sid, name), payload, relative_to=self.dir)
            record = replace(record, files=(ref,))
        self.ex.evidence[record.source] = record

    # Phase discipline

    def _outcome(self, phase: Phase):
        return self.ex.phases[_PHASES.index(phase)]

    def begin(self, phase: Phase) -> None:
        index = _PHASES.index(phase)
        if any(o.status is PhaseStatus.PENDING for o in self.ex.phases[:index]):
            raise RuntimeError(f"{phase.value} started before earlier phases settled")
        outcome = self.ex.phases[index]
        if outcome.status is not PhaseStatus.PENDING:
            raise RuntimeError(f"{phase.value} already {outcome.status.value}")
        outcome.started_at = utc_now()

    def end(self, phase: Phase, status: PhaseStatus, failure: Failure | None = None, note: str | None = None) -> None:
        outcome = self._outcome(phase)
        outcome.status = status
        outcome.ended_at = utc_now()
        outcome.failure = failure
        if note:
            outcome.notes.append(note)
        self.save()

    def block(self, phase: Phase, code: InfraCode, summary: str, detail: str | None = None):
        failure = Failure(code=code, summary=summary, phase=phase, detail=detail)
        self.end(phase, PhaseStatus.FAILED, failure)
        raise _Blocking(failure)

    def block_transport(self, phase: Phase, result: ExchangeResult, what: str):
        error = result.exchange.transport_error
        self.block(phase, transport_code(error), f"{what}: no HTTP response", f"{error.type}: {error.message}")

    def stderr_now(self, phase: Phase) -> int:
        try:
            return runtime_log.current_offset(self.layout.maui_stderr)
        except OSError as exc:
            self.block(phase, InfraCode.INFRA_EVIDENCE_UNAVAILABLE, "maui stderr log unreadable", _detail(exc))

    def runtime(self, source: str, offset: int, request_id: str, marker: str):
        options = self.suite.options
        return self.deps.collect_runtime(
            source, self.layout.maui_stderr, offset, request_id, marker, options.marker_deadline_s,
            poll_interval_s=options.poll_interval_s, clock=self.deps.clock, sleep=self.deps.sleep,
        )

    # Lifecycle

    def run(self) -> Failure | None:
        """Execute and close the scenario. Returns the failure that must abort the suite, if any."""
        self.save()
        failure = None
        try:
            self.pre_cleanup()
            self.start()
            self.datachat()
            self.collect()
        except _Blocking as blocking:
            failure = blocking.failure
            self.ex.failure = failure
            for outcome in self.ex.phases[: _PHASES.index(Phase.TEARDOWN)]:
                if outcome.status is PhaseStatus.PENDING:
                    outcome.status = PhaseStatus.SKIPPED
                    outcome.notes.append("skipped after blocking failure")
            self.save()

        if failure is None or self.engine_may_exist:
            teardown_failure = self.teardown(best_effort=failure is not None)
        else:
            teardown_failure = None
            self._outcome(Phase.TEARDOWN).status = PhaseStatus.SKIPPED
            self._outcome(Phase.TEARDOWN).notes.append("no engine could exist")
        self.close(failure)
        return failure or teardown_failure

    def pre_cleanup(self) -> None:
        phase = Phase.PRE_CLEANUP
        self.begin(phase)
        attempts = self.suite.client.end("pre_cleanup")
        for result in attempts:
            self.persist_http(phase, result)
        last = attempts[-1]
        if last.exchange.response is None:
            self.block_transport(phase, last, "pre-cleanup /enddatachat")
        outcome = end_outcome(last)
        if outcome is None:
            self.block(
                phase, InfraCode.INFRA_FRESH_ENGINE_UNPROVEN,
                f"pre-cleanup did not confirm a clean precondition (HTTP {last.exchange.response.status})",
            )
        note = None
        if outcome == "deleted":
            self.ex.flags.previous_engine_present = True
            note = "anomaly: a previous engine was present and deleted"
            self.ex.notes.append(note)
        self.end(phase, PhaseStatus.OK, note=note)

    def start(self) -> None:
        phase = Phase.START
        self.begin(phase)
        offset = self.stderr_now(phase)
        self.engine_may_exist = True
        result = self.suite.client.start(self.suite.dataset_path)
        self.persist_http(phase, result)
        response = result.exchange.response
        if response is None:
            self.block_transport(phase, result, "/startdatachat")
        body = _json_body(result.response_bytes)
        if _not_enough_tokens(body):
            self.block(phase, InfraCode.INFRA_TOKEN_BUDGET, "/startdatachat: not enough tokens")
        if response.status != 200 or not (isinstance(body, dict) and START_ACTIVE in body):
            self.block(phase, InfraCode.INFRA_FRESH_ENGINE_UNPROVEN, f"/startdatachat did not activate an agent (HTTP {response.status})")
        request_id = result.exchange.request_id
        if not _valid_request_id(request_id):
            self.block(phase, InfraCode.INFRA_FRESH_ENGINE_UNPROVEN, "/startdatachat returned no usable X-Request-ID")

        record, payload = self.runtime("runtime_start", offset, request_id, MARKER_START)
        self.persist_runtime("start", record, payload)
        if record.availability is not Availability.PRESENT:
            self.block(phase, InfraCode.INFRA_FRESH_ENGINE_UNPROVEN, f"freshness marker missing: {record.reason}")
        if not freshness_proven(payload):
            self.block(phase, InfraCode.INFRA_FRESH_ENGINE_UNPROVEN, "freshness proof incomplete (status=ok sql=off / sql_tools=0)")
        self.ex.flags.fresh_engine_proven = True
        self.end(phase, PhaseStatus.OK)

    def datachat(self) -> None:
        phase = Phase.DATACHAT
        self.begin(phase)
        self.stderr_offset = self.stderr_now(phase)
        try:
            self.agent_runs_offset = runtime_log.current_offset(self.layout.agent_runs_log)
        except FileNotFoundError:
            self.agent_runs_offset = 0
        except OSError as exc:
            self.block(phase, InfraCode.INFRA_EVIDENCE_UNAVAILABLE, "agent_runs log unreadable", _detail(exc))

        self.ex.flags.reached_datachat = True
        result = self.suite.client.datachat(self.question)
        refs = self.persist_http(phase, result)
        exchange = result.exchange
        locator = {"request_id": exchange.request_id, "status": exchange.response.status if exchange.response else None}
        if exchange.response is None:
            self.ex.evidence["http_datachat"] = EvidenceSourceRecord(
                "http_datachat", Availability.UNAVAILABLE, "no HTTP response", tuple(refs), locator
            )
            self.block_transport(phase, result, "/datachat")
        self.ex.evidence["http_datachat"] = EvidenceSourceRecord(
            "http_datachat", Availability.PRESENT, None, tuple(refs), locator
        )
        self.datachat_response = True
        self.datachat_body = _json_body(result.response_bytes)
        if _not_enough_tokens(self.datachat_body):
            self.block(phase, InfraCode.INFRA_TOKEN_BUDGET, "/datachat: not enough tokens")
        if not _valid_request_id(exchange.request_id):
            self.block(phase, InfraCode.INFRA_EVIDENCE_UNAVAILABLE, "/datachat returned no usable X-Request-ID")
        self.datachat_rid = exchange.request_id
        self.end(phase, PhaseStatus.OK)

    def collect(self) -> None:
        phase = Phase.COLLECT
        self.begin(phase)
        first: str | None = None
        rid = self.datachat_rid

        record, payload = self.runtime("runtime_datachat", self.stderr_offset, rid, MARKER_DATACHAT)
        self.persist_runtime("datachat", record, payload)
        if record.availability is Availability.UNAVAILABLE:
            first = first or f"runtime_datachat: {record.reason}"
        trace_hint = runtime_log.parse_trace_status(payload).trace_present

        record, payload = self.deps.collect_agent_runs(self.layout.agent_runs_log, self.agent_runs_offset, rid, trace_hint)
        if payload:
            ref = write_raw(self.layout.agent_runs_record(self.sid), payload, relative_to=self.dir)
            record = replace(record, files=(ref,))
        self.ex.evidence["agent_runs"] = record
        if record.availability is Availability.UNAVAILABLE:
            first = first or f"agent_runs: {record.reason}"

        log_id = self.datachat_body.get("log_id") if isinstance(self.datachat_body, dict) else None
        if isinstance(log_id, bool) or not isinstance(log_id, int):
            log_id = None
        collection = self.deps.collect_usage(self.suite.db_settings, rid, log_id)
        record = collection.record
        if collection.evidence is not None:
            path = self.layout.usage_rows(self.sid)
            write_json_atomic(path, collection.evidence.to_dict())
            record = replace(record, files=(file_ref(path, self.dir),))
        self.ex.evidence["usage"] = record
        if record.availability is Availability.UNAVAILABLE:
            first = first or f"usage: {record.reason}"

        self.collect_image()
        self.collect_export()

        if first is not None:
            self.block(phase, InfraCode.INFRA_EVIDENCE_UNAVAILABLE, f"required evidence unavailable: {first}")
        self.end(phase, PhaseStatus.OK)

    def _response_object(self) -> dict:
        body = self.datachat_body
        response = body.get("response") if isinstance(body, dict) else None
        return response if isinstance(response, dict) else {}

    def collect_image(self) -> None:
        response = self._response_object()
        if response.get("type") != "image":
            self.ex.evidence["image"] = EvidenceSourceRecord("image", Availability.NOT_APPLICABLE, "response is not an image")
            return
        value = response.get("value")
        decoded = decode_image_value(value) if isinstance(value, str) else ImageDecode(WRAPPER_MISSING, False, False, None, None)
        artifact = image_artifact(self.datachat_rid, decoded)
        files = []
        if decoded.data is not None:
            files.append(write_raw(self.layout.artifact(self.sid, "image.png"), decoded.data, relative_to=self.dir))
        meta_path = self.layout.artifact(self.sid, "image.json")
        write_json_atomic(meta_path, artifact.to_dict())
        files.append(file_ref(meta_path, self.dir))
        self.ex.evidence["image"] = EvidenceSourceRecord(
            "image", Availability.PRESENT, None, tuple(files),
            {"request_id": self.datachat_rid, "decode_status": artifact.decode_status},
        )

    def collect_export(self) -> None:
        url = self._response_object().get("download_url")
        if not isinstance(url, str):
            self.ex.evidence["export"] = EvidenceSourceRecord("export", Availability.NOT_APPLICABLE, "no download_url")
            return
        try:
            result = self.suite.client.export(url)
        except ValueError as exc:
            self.ex.evidence["export"] = EvidenceSourceRecord(
                "export", Availability.UNAVAILABLE, f"download_url rejected: {exc}", locator={"download_url": url}
            )
            return
        self.persist_http(Phase.COLLECT, result)
        exchange = result.exchange
        locator = {"download_url": url, "request_id": exchange.request_id, "source_request_id": self.datachat_rid}
        if exchange.response is None:
            error = exchange.transport_error
            self.ex.evidence["export"] = EvidenceSourceRecord(
                "export", Availability.UNAVAILABLE, f"no HTTP response: {error.type}", locator=locator
            )
            return
        artifact, payload = export_artifact(result.response_bytes, exchange.response.status, self.datachat_rid)
        files = ()
        if payload is not None:
            files = (write_raw(self.layout.artifact(self.sid, "export.csv"), payload, relative_to=self.dir),)
        reason = None if artifact.status is Availability.PRESENT else f"export fetch returned HTTP {exchange.response.status}"
        locator.update(http_status=artifact.http_status, sha256=artifact.sha256, size=artifact.size)
        self.ex.evidence["export"] = EvidenceSourceRecord("export", artifact.status, reason, files, locator)

    def teardown(self, best_effort: bool) -> Failure | None:
        phase = Phase.TEARDOWN
        self.begin(phase)
        note = "best-effort cleanup after blocking failure" if best_effort else None

        def failed(summary: str, detail: str | None = None) -> Failure:
            failure = Failure(InfraCode.INFRA_TEARDOWN_UNVERIFIED, summary, phase, detail)
            self.end(phase, PhaseStatus.FAILED, failure, note)
            return failure

        try:
            offset = runtime_log.current_offset(self.layout.maui_stderr)
        except OSError as exc:
            return failed("maui stderr log unreadable", _detail(exc))
        attempts = self.suite.client.end("teardown")
        for result in attempts:
            self.persist_http(phase, result)
        last = attempts[-1]
        if last.exchange.response is None:
            error = last.exchange.transport_error
            return failed("teardown /enddatachat: no HTTP response", f"{error.type}: {error.message}")
        outcome = end_outcome(last)
        if outcome != "deleted":
            return failed(f"teardown did not confirm deletion (HTTP {last.exchange.response.status}, {outcome})")
        request_id = last.exchange.request_id
        if not _valid_request_id(request_id):
            return failed("teardown returned no usable X-Request-ID")
        record, payload = self.runtime("runtime_teardown", offset, request_id, MARKER_TEARDOWN)
        self.persist_runtime("teardown", record, payload)
        if record.availability is not Availability.PRESENT:
            return failed(f"cleanup_result not observed: {record.reason}")
        self.ex.flags.teardown_verified = True
        self.end(phase, PhaseStatus.OK, note=note)
        return None

    def close(self, failure: Failure | None) -> None:
        self.begin(Phase.CLOSE)
        for source in SOURCES:
            if source not in self.ex.evidence:
                self.ex.evidence[source] = EvidenceSourceRecord(source, Availability.UNAVAILABLE, "not_collected")
        self.ex.status = ScenarioStatus.NOT_SCOREABLE_INFRA if failure else ScenarioStatus.EXECUTED
        close = self._outcome(Phase.CLOSE)
        close.status = PhaseStatus.OK
        close.ended_at = utc_now()
        flags = self.ex.flags
        flags.execution_complete = all(o.status is not PhaseStatus.PENDING for o in self.ex.phases)
        flags.evidence_complete = all(r.availability is not Availability.UNAVAILABLE for r in self.ex.evidence.values())
        flags.scoreable = (
            self.ex.status is ScenarioStatus.EXECUTED
            and flags.fresh_engine_proven
            and self.datachat_response
            and all(self.ex.evidence[s].availability is not Availability.UNAVAILABLE for s in REQUIRED_SOURCES)
        )
        self.ex.ended_at = utc_now()
        self.save()


# --- Suite ---------------------------------------------------------------------


class _Suite:
    def __init__(self, options, deps, identity, layout, run_id, run, env, db_settings, token_cost, scenarios):
        self.options = options
        self.deps = deps
        self.identity = identity
        self.layout = layout
        self.run_id = run_id
        self.run = run
        self.env = env
        self.db_settings = db_settings
        self.token_cost = token_cost
        self.scenarios = scenarios
        self.dataset_path = contract.evaluation_dir(options.repo_root) / contract.DATASET_NAME
        self.process = None
        self.client: DataChatClient | None = None

    def save(self) -> None:
        write_json_atomic(self.layout.run_manifest, self.run.to_dict())

    def index(self, execution: ScenarioExecution) -> None:
        path = self.layout.execution(execution.scenario_id)
        self.run.scenarios.append({
            "scenario_id": execution.scenario_id,
            "status": execution.status.value,
            "dir": path.parent.relative_to(self.layout.root).as_posix(),
            "execution_sha256": sha256_file(path),
        })

    def fail(self, code: InfraCode, summary: str, detail: str | None = None) -> Failure:
        return Failure(code=code, summary=summary, detail=detail)

    def launch_and_check(self) -> Failure | None:
        options, deps = self.options, self.deps
        argv = self.run.runtime["gunicorn_argv"]
        self.process = deps.make_process(argv, options.repo_root, self.env, self.layout.maui_stderr)
        try:
            self.process.start()
        except OSError as exc:
            return self.fail(InfraCode.INFRA_MAUI_UNAVAILABLE, "maui could not be launched", _detail(exc))
        self.client = deps.make_client(f"http://{contract.BIND_HOST}:{options.port}", self.identity)

        ready = self.process.wait_ready(
            lambda: self.client.health().status == 200, options.health_deadline_s,
            poll_interval_s=options.poll_interval_s, clock=deps.clock, sleep=deps.sleep,
        )
        if not ready.ready:
            return self.fail(
                InfraCode.INFRA_MAUI_UNAVAILABLE, f"maui not ready: {ready.outcome}",
                f"polls={ready.polls} exit_code={ready.exit_code}",
            )
        try:
            stderr_size = runtime_log.current_offset(self.layout.maui_stderr)
        except OSError as exc:
            return self.fail(InfraCode.INFRA_PREFLIGHT, "maui stderr capture unreadable", _detail(exc))
        if stderr_size == 0:
            return self.fail(InfraCode.INFRA_PREFLIGHT, "maui stderr capture produced no output")

        reason = deps.check_read_access(self.db_settings)
        if reason is not None:
            return self.fail(InfraCode.INFRA_PREFLIGHT, "database read check failed", reason)
        try:
            user = deps.fetch_user(self.db_settings, options.email)
        except psycopg.Error as exc:
            return self.fail(InfraCode.INFRA_PREFLIGHT, "evaluation user lookup failed", f"db_error: {type(exc).__name__}")
        if user is None:
            return self.fail(InfraCode.INFRA_PREFLIGHT, "evaluation user not found")
        self.run.eval_identity["user_id"] = user.user_id
        if isinstance(user.tokens, bool) or not isinstance(user.tokens, int):
            return self.fail(InfraCode.INFRA_TOKEN_BUDGET, "token balance unreadable")
        self.run.token_balance_at_preflight = user.tokens
        margin = options.token_margin if options.token_margin is not None else 2 * self.token_cost
        required = 2 * len(self.scenarios) * self.token_cost + margin
        self.run.runtime["token_budget"] = {"required": required, "margin": margin, "token_cost": self.token_cost}
        self.save()
        if user.tokens < required:
            return self.fail(InfraCode.INFRA_TOKEN_BUDGET, f"token balance {user.tokens} < required {required}")

        attempts = self.client.end("pre_cleanup")
        last = attempts[-1]
        sweep = {
            "attempts": len(attempts),
            "request_id": last.exchange.request_id,
            "status": last.exchange.response.status if last.exchange.response else None,
            "outcome": end_outcome(last),
        }
        self.run.runtime["preflight_sweep"] = sweep
        self.save()
        if last.exchange.response is None:
            error = last.exchange.transport_error
            return self.fail(transport_code(error), "preflight /enddatachat sweep: no HTTP response", error.type)
        if sweep["outcome"] is None:
            return self.fail(InfraCode.INFRA_PREFLIGHT, f"preflight /enddatachat sweep rejected (HTTP {sweep['status']})")
        return None

    def run_scenarios(self) -> Failure | None:
        for position, scenario in enumerate(self.scenarios):
            scenario_run = _ScenarioRun(self, scenario)
            failure = scenario_run.run()
            self.index(scenario_run.ex)
            self.save()
            if failure is not None:
                self.abort(failure, self.scenarios[position + 1:])
                return failure
        return None

    def abort(self, failure: Failure, remaining: Sequence[Scenario]) -> None:
        """Persist the first blocking failure first, then the NOT_RUN packages."""
        self.run.failure = failure
        self.run.status = SuiteStatus.ABORTED_INFRA
        self.save()
        for scenario in remaining:
            execution = ScenarioExecution.new(self.run_id, scenario.scenario_id, scenario.ordinal, scenario.question)
            execution.started_at = None  # never started
            execution.mark_not_run()
            write_json_atomic(self.layout.execution(scenario.scenario_id), execution.to_dict())
            self.index(execution)
        self.run.ended_at = utc_now()
        self.save()

    def complete(self) -> None:
        statuses = {entry["status"] for entry in self.run.scenarios}
        if (
            len(self.run.scenarios) != len(self.scenarios)
            or statuses != {ScenarioStatus.EXECUTED.value}
            or self.run.failure is not None
        ):
            raise RuntimeError("suite finished without failure but completion invariants do not hold")
        self.run.status = SuiteStatus.COMPLETED
        self.run.ended_at = utc_now()
        self.save()

    def record_crash(self, exc: BaseException) -> None:
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        text = text.replace(self.identity.api_key, "<redacted>")
        name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_traceback.txt"
        write_raw(self.layout.failures_dir / name, text.encode("utf-8"), relative_to=self.layout.root)


# --- Entry point -----------------------------------------------------------------


@dataclass(frozen=True)
class _LocalPreflight:
    git: GitState
    checks: tuple
    run_id: str
    db_settings: DbSettings
    token_cost: int


def _local_preflight(options: RunnerOptions, deps: RunnerDeps, scenarios) -> _LocalPreflight | Failure:
    """Steps that write nothing: repository, artifacts, drift, port, environment."""

    def fail(summary: str, detail: str | None = None) -> Failure:
        return Failure(InfraCode.INFRA_PREFLIGHT, summary, None, detail)

    try:
        git = deps.git_state(options.repo_root)
    except (OSError, subprocess.CalledProcessError) as exc:
        return fail("repository identity unavailable", _detail(exc))
    if options.official and git.dirty:
        return fail("official run requires a clean working tree", ", ".join(git.dirty_files)[:300])

    checks = deps.verify_artifacts(options.repo_root)
    if not contract.required_artifacts_match(checks):
        bad = [c.name for c in checks if c.required and not c.matches]
        return fail("evaluation artifact hash mismatch", ", ".join(bad))
    drift = deps.check_drift(options.repo_root, scenarios=SCENARIOS)
    if not drift.ok:
        return fail("scenario catalogue drift", "; ".join(drift.errors)[:300])

    if not deps.port_free(contract.BIND_HOST, options.port):
        return fail(f"port {options.port} already in use")

    run_id = make_run_id(deps.now(), git.sha, git.dirty)
    provisional = deps.resolve_env(options.repo_root, Path(options.runs_dir) / run_id, options.base_env)
    problems = contract.validate_env(provisional.env)
    if problems:
        return fail("effective environment incomplete", ", ".join(problems))
    try:
        token_cost = int(provisional.env.get("DATACHAT_TOKEN_COST") or "1")
        db_settings = DbSettings.from_env(provisional.env)
    except ValueError as exc:
        return fail("effective environment invalid", _detail(exc))
    return _LocalPreflight(git, tuple(checks), run_id, db_settings, token_cost)


def run_suite(
    options: RunnerOptions,
    api_key: str,
    deps: RunnerDeps | None = None,
    scenarios: Sequence[Scenario] = SCENARIOS,
) -> RunResult:
    """Run the frozen suite (or a prefix of it, for tests) and return the final run state."""
    deps = deps or RunnerDeps()
    identity = Identity(api_key=api_key, user_email=options.email, user_name=options.username)
    scenarios = tuple(scenarios)

    local = _local_preflight(options, deps, scenarios)
    if isinstance(local, Failure):
        return RunResult(None, SuiteStatus.ABORTED_INFRA, local, None)

    run_id, layout = create_run_dir(options.runs_dir, local.run_id)
    child = deps.resolve_env(options.repo_root, layout.root, options.base_env)
    report = contract.environment_report(child.env, child.dotenv_present)
    checks = local.checks
    run = EvaluationRun(
        run_id=run_id,
        started_at=utc_now(),
        official=options.official,
        comparable=options.official and not local.git.dirty and all(c.matches for c in checks),
        maui={
            "git_sha": local.git.sha,
            "branch": local.git.branch,
            "dirty": local.git.dirty,
            "dirty_files": list(local.git.dirty_files),
        },
        evaluation_contract={
            "artifacts": [c.to_dict() for c in checks],
            "all_match": all(c.matches for c in checks),
            "required_match": contract.required_artifacts_match(checks),
            "scenario_catalogue_match": True,
        },
        runtime={
            "python": sys.version.split()[0],
            "packages": deps.package_versions(),
            "gunicorn_argv": contract.gunicorn_argv(options.port),
            "port": options.port,
            "cwd": str(options.repo_root),
        },
        datachat_config=report,
        eval_identity={
            "label": options.eval_label,
            "username": options.username,
            "email": options.email,
            "user_id": None,
            "api_key_sha256": sha256_bytes(api_key.encode("utf-8")),
        },
        scenario_order=[s.scenario_id for s in scenarios],
    )
    suite = _Suite(options, deps, identity, layout, run_id, run, child.env, local.db_settings, local.token_cost, scenarios)
    suite.save()
    write_json_atomic(layout.environment, report)

    try:
        failure = suite.launch_and_check()
        if failure is not None:
            suite.abort(failure, scenarios)
        else:
            failure = suite.run_scenarios()
            if failure is None:
                suite.complete()
    except Exception as exc:  # outer boundary only: keep the run inspectable, then propagate
        suite.record_crash(exc)
        raise
    finally:
        if suite.process is not None:
            suite.process.stop(options.stop_grace_s)
        if suite.client is not None:
            suite.client.close()
    return RunResult(run_id, run.status, run.failure, layout.root)
