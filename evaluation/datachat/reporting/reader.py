"""Read-only loader of graded run packages for reporting.

``load_run_for_reporting(run_dir)`` turns a closed, graded run directory into
a typed source model:

    run package + grading/results/ → integrity verification → ReportingRunInput

Inputs: ``grading/results/run_grade.json``, ``grading/results/scenarios/*.json``,
``run_manifest.json``, ``scenarios/<id>/execution.json``, the ``/datachat``
exchange and response body, and the delivered image/export artifacts. Never
read: the repository scenario catalogue, the
evaluation contract under ``docs/`` (recorded in ``RunGrade.contract``,
verified by the grader), ``agent_runs``, runtime logs or Usage as answer.

Only grading schema 2 (technical outcomes) is accepted.

Any inconsistency or unsupported schema raises ``ReportingInputError``: an
input error, never a scenario outcome. The reader never writes, repairs,
reconciles or regrades, and recomputes no judgment: grading values are
deserialized verbatim and only compared for identity across files.
"""

import json
from pathlib import Path
from typing import Any, Callable

from evaluation.datachat.evidence import RunLayout, read_json, safe_child, sha256_file
from evaluation.datachat.grading.grader import RESULTS_DIR, RUN_GRADE_NAME
from evaluation.datachat.grading.model import (
    ActivitySource,
    CheckResult,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ObservedActivity,
    RunGrade,
    ScenarioGrade,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.grading.reader import ParseStatus, normalize_response
from evaluation.datachat.model import (
    ArtifactRecord,
    Availability,
    EvaluationRun,
    FileRef,
    HttpExchange,
    ScenarioExecution,
    SuiteStatus,
)
from evaluation.datachat.reporting.model import (
    AgentResult,
    DatachatBody,
    DeliveredArtifact,
    ReportingRunInput,
    ReportingScenarioInput,
)

# Schema versions this reader understands. Anything else is refused, never best-effort parsed.
SUPPORTED_GRADING_SCHEMAS = frozenset({2})
SUPPORTED_EXECUTION_SCHEMAS = frozenset({1})


class ReportingInputError(Exception):
    """The run package cannot be reported: missing, inconsistent or corrupted inputs."""


class UnsupportedSchemaError(ReportingInputError):
    """A persisted record carries a schema version this reader does not understand."""


# --- Low-level reads ---------------------------------------------------------------


def _read(path: Path, what: str) -> Any:
    if not path.is_file():
        raise ReportingInputError(f"{what} missing")
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        raise ReportingInputError(f"{what} unreadable: {type(exc).__name__}") from exc


def _safe(root: Path, relative: str, what: str) -> Path:
    try:
        return safe_child(root, relative)
    except (ValueError, TypeError) as exc:
        raise ReportingInputError(f"{what}: unsafe path {relative!r}") from exc


def _verify(path: Path, sha256: str | None, size: int | None, what: str) -> None:
    if not path.is_file():
        raise ReportingInputError(f"{what}: referenced file missing")
    if size is not None and path.stat().st_size != size:
        raise ReportingInputError(f"{what}: size mismatch")
    if sha256_file(path) != sha256:
        raise ReportingInputError(f"{what}: sha256 mismatch")


def _check_schema(data: Any, supported: frozenset[int], what: str) -> None:
    if not isinstance(data, dict):
        raise ReportingInputError(f"{what} malformed: not an object")
    version = data.get("schema_version")
    if version not in supported:
        raise UnsupportedSchemaError(f"{what}: unsupported schema_version {version!r}")


def _parse(data: dict, build: Callable[[dict], Any], what: str) -> Any:
    """Build a persisted record; refuse unless it serializes back to exactly ``data``."""
    try:
        record = build(data)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ReportingInputError(f"{what} malformed: {type(exc).__name__}: {exc}") from exc
    if record.to_dict() != data:
        raise ReportingInputError(f"{what} malformed: fields differ from the grading model")
    return record


# --- Grading results ---------------------------------------------------------------


def _opt(enum, value):
    return None if value is None else enum(value)


def _failures(values: list) -> tuple[TechnicalFailure, ...]:
    return tuple(TechnicalFailure(f) for f in values)


def _check_result(d: dict) -> CheckResult:
    return CheckResult(
        check_id=d["check_id"], evidence_class=EvidenceClass(d["evidence_class"]),
        status=CheckStatus(d["status"]), expected=d["expected"], observed=d["observed"],
        evidence_refs=tuple(d["evidence_refs"]), failure=_opt(TechnicalFailure, d["failure"]), note=d["note"],
    )


def _activity(d: dict) -> ObservedActivity:
    return ObservedActivity(name=d["name"], source=ActivitySource(d["source"]), count=d["count"],
                            outcome=d["outcome"])


def _scenario_grade(d: dict) -> ScenarioGrade:
    return ScenarioGrade(
        scenario_id=d["scenario_id"], gate=Gate(d["gate"]), outcome=_opt(TechnicalOutcome, d["outcome"]),
        failures=_failures(d["failures"]),
        checks=tuple(_check_result(c) for c in d["checks"]),
        execution_assessment=_opt(ExecutionAssessment, d["execution_assessment"]),
        execution_assessment_reason=_opt(ExecutionAssessmentReason, d["execution_assessment_reason"]),
        execution_sha256=d["execution_sha256"], teardown_verified=d["teardown_verified"],
        activity=tuple(_activity(a) for a in d["activity"]),
        schema_version=d["schema_version"],
    )


def _run_grade(d: dict) -> RunGrade:
    return RunGrade(
        run_id=d["run_id"], official=d["official"], comparable=d["comparable"],
        grader=d["grader"], contract=d["contract"],
        scenarios=tuple(
            ScenarioIndexEntry(e["scenario_id"], Gate(e["gate"]), _opt(TechnicalOutcome, e["outcome"]),
                               _failures(e["failures"]))
            for e in d["scenarios"]),
        counts_by_outcome=d["counts_by_outcome"], counts_by_failure=d["counts_by_failure"],
        schema_version=d["schema_version"],
    )


# --- Agent result ------------------------------------------------------------------


def _datachat_body(body: bytes, status: int, content_type: str | None) -> DatachatBody:
    """Structural parse shared with grading (same ``ParseStatus``); no runtime evidence is consulted."""
    n = normalize_response(body, status, content_type)
    parsed = n.raw if n.parse_status is not ParseStatus.NON_JSON else None
    response = parsed.get("response") if isinstance(parsed, dict) else None
    return DatachatBody(
        parse_status=n.parse_status,
        parsed=parsed,
        response_type=n.type,
        value=response.get("value") if n.parse_status is ParseStatus.OK else None,
        table_meta=dict(n.table_meta),
        charts=n.charts,
        note=n.note,
        more_rows_available=n.more_rows_available,
        download_url=n.download_url,
        explanation=parsed.get("explanation") if isinstance(parsed, dict) else None,
        log_id=n.log_id,
        error=n.error,
    )


def _ref(execution: ScenarioExecution, source: str, filename: str) -> FileRef | None:
    record = execution.evidence.get(source)
    if record is None:
        return None
    return next((ref for ref in record.files if ref.path.rsplit("/", 1)[-1] == filename), None)


def _verified(scenario_dir: Path, ref: FileRef | None, what: str) -> FileRef | None:
    if ref is not None:
        _verify(_safe(scenario_dir, ref.path, what), ref.sha256, ref.size, f"{what} {ref.path}")
    return ref


def _image(execution: ScenarioExecution, scenario_dir: Path, sid: str) -> DeliveredArtifact | None:
    record = execution.evidence.get("image")
    if record is None:
        return None
    png = _verified(scenario_dir, _ref(execution, "image", "image.png"), f"{sid} image")
    meta_ref = _verified(scenario_dir, _ref(execution, "image", "image.json"), f"{sid} image metadata")
    metadata = None
    if meta_ref is not None:
        data = _read(scenario_dir / meta_ref.path, f"{sid} image metadata")
        metadata = _parse(data, lambda d: ArtifactRecord(**{**d, "status": Availability(d["status"])}),
                          f"{sid} image metadata")
        if metadata.kind != "image" or (metadata.path, metadata.sha256) != (
                (png.path, png.sha256) if png else (None, None)):
            raise ReportingInputError(f"{sid} image metadata does not match the persisted image file")
    return DeliveredArtifact("image", record.availability, record.reason, png, metadata, meta_ref,
                             dict(record.locator))


def _export(execution: ScenarioExecution, scenario_dir: Path, sid: str) -> DeliveredArtifact | None:
    record = execution.evidence.get("export")
    if record is None:
        return None
    csv = _verified(scenario_dir, _ref(execution, "export", "export.csv"), f"{sid} export")
    return DeliveredArtifact("export", record.availability, record.reason, csv, locator=dict(record.locator))


def _agent_result(execution: ScenarioExecution, scenario_dir: Path, sid: str) -> AgentResult | None:
    record = execution.evidence.get("http_datachat")
    attempts = execution.requests.get("datachat", [])
    if record is None and not attempts:
        return None
    exchange = body = content = body_file = None
    if attempts:
        meta = _safe(scenario_dir, attempts[-1], f"{sid} datachat metadata")
        data = _read(meta, f"{sid} datachat metadata")
        _check_schema(data, SUPPORTED_EXECUTION_SCHEMAS, f"{sid} datachat metadata")
        try:
            exchange = HttpExchange.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ReportingInputError(f"{sid} datachat metadata malformed") from exc
        if exchange.key != "datachat":
            raise ReportingInputError(f"{sid} datachat metadata records key {exchange.key!r}")
        response = exchange.response
        if response is not None:
            body_file = response.body_file
            path = _safe(scenario_dir, body_file, f"{sid} datachat response")
            _verify(path, response.body_sha256, response.size, f"{sid} datachat response {body_file}")
            recorded = next((r for r in (record.files if record else ()) if r.path == body_file), None)
            _verified(scenario_dir, recorded, f"{sid} http_datachat")
            body = path.read_bytes()
            content = _datachat_body(body, response.status, response.content_type)
    return AgentResult(
        availability=record.availability if record else None,
        reason=record.reason if record else None,
        exchange=exchange,
        attempts=len(attempts),
        body=body,
        body_file=body_file,
        content=content,
        image=_image(execution, scenario_dir, sid),
        export=_export(execution, scenario_dir, sid),
    )


# --- Scenario and run --------------------------------------------------------------


def _load_scenario(
    root: Path, run: RunGrade, manifest: EvaluationRun, entry: ScenarioIndexEntry
) -> ReportingScenarioInput:
    sid = entry.scenario_id
    layout = RunLayout(root)
    try:
        scenario_dir = layout.scenario_dir(sid)
    except ValueError as exc:
        raise ReportingInputError(f"invalid scenario id in run_grade.json: {sid!r}") from exc
    run_rel = f"scenarios/{sid}"

    grade_path = _safe(root, (RESULTS_DIR / "scenarios" / f"{sid}.json").as_posix(), f"{sid} grade")
    data = _read(grade_path, f"{sid} scenario grade")
    _check_schema(data, SUPPORTED_GRADING_SCHEMAS, f"{sid} scenario grade")
    grade = _parse(data, _scenario_grade, f"{sid} scenario grade")
    if (grade.scenario_id, grade.gate, grade.outcome, grade.failures) != (
            sid, entry.gate, entry.outcome, entry.failures):
        raise ReportingInputError(f"{sid}: scenario grade does not match run_grade.json (id/gate/outcome/failures)")

    manifest_entries = [e for e in manifest.scenarios if e.get("scenario_id") == sid]
    if len(manifest_entries) > 1:
        raise ReportingInputError(f"{sid}: duplicate run manifest entries")
    manifest_entry = manifest_entries[0] if manifest_entries else None
    if manifest_entry is not None and manifest_entry.get("dir") != run_rel:
        raise ReportingInputError(f"{sid}: manifest dir {manifest_entry.get('dir')!r} is not {run_rel!r}")
    recorded = {
        "ScenarioGrade.execution_sha256": grade.execution_sha256,
        "RunGrade.contract.scenario_execution_sha256": run.contract["scenario_execution_sha256"].get(sid),
        "run_manifest.json execution_sha256": manifest_entry.get("execution_sha256") if manifest_entry else None,
    }
    execution_path = layout.execution(sid)
    if manifest_entry is None:
        # Suite aborted before this scenario's package was written: nothing to read.
        if any(v is not None for v in recorded.values()) or execution_path.exists():
            raise ReportingInputError(f"{sid}: execution digest recorded for a scenario absent from the manifest")
        return ReportingScenarioInput(sid, entry, grade, None, None, run_rel)
    if not execution_path.is_file():
        raise ReportingInputError(f"{sid}: execution.json missing")
    recorded["execution.json on disk"] = sha256_file(execution_path)
    if len(set(recorded.values())) != 1 or None in recorded.values():
        raise ReportingInputError(f"{sid}: execution digest mismatch: {recorded}")

    data = _read(execution_path, f"{sid} execution.json")
    _check_schema(data, SUPPORTED_EXECUTION_SCHEMAS, f"{sid} execution.json")
    try:
        execution = ScenarioExecution.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportingInputError(f"{sid}: execution.json malformed") from exc
    if (execution.run_id, execution.scenario_id) != (run.run_id, sid):
        raise ReportingInputError(f"{sid}: execution.json identity does not match the run")
    return ReportingScenarioInput(
        scenario_id=sid, index=entry, grade=grade, execution=execution,
        execution_sha256=recorded["execution.json on disk"], dir=run_rel,
        agent_result=_agent_result(execution, scenario_dir, sid),
    )


def load_run_for_reporting(run_dir: Path) -> ReportingRunInput:
    """Load and verify one graded run directory. Read-only and offline.

    Raises ``ReportingInputError`` (``UnsupportedSchemaError`` for unknown
    schema versions) when the inputs cannot be trusted to belong together.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise ReportingInputError(f"run directory not found: {root}")

    run_grade_path = _safe(root, (RESULTS_DIR / RUN_GRADE_NAME).as_posix(), "run grade")
    if not run_grade_path.is_file():
        raise ReportingInputError("run is not graded: grading/results/run_grade.json missing")
    data = _read(run_grade_path, "run_grade.json")
    _check_schema(data, SUPPORTED_GRADING_SCHEMAS, "run_grade.json")
    run = _parse(data, _run_grade, "run_grade.json")
    run_grade_sha256 = sha256_file(run_grade_path)

    data = _read(RunLayout(root).run_manifest, "run_manifest.json")
    _check_schema(data, SUPPORTED_EXECUTION_SCHEMAS, "run_manifest.json")
    try:
        manifest = EvaluationRun.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportingInputError("run_manifest.json malformed") from exc
    if manifest.status is SuiteStatus.RUNNING:
        raise ReportingInputError("run is still RUNNING (interrupted or in progress)")
    if not (root.name == manifest.run_id == run.run_id):
        raise ReportingInputError("run_id differs between directory, run_manifest.json and run_grade.json")
    if (run.official, run.comparable) != (manifest.official, manifest.comparable):
        raise ReportingInputError("official/comparable differ between run_manifest.json and run_grade.json")

    ids = [e.scenario_id for e in run.scenarios]
    if ids != list(manifest.scenario_order):
        raise ReportingInputError("run_grade.json scenarios differ from the manifest scenario_order")
    digests = run.contract.get("scenario_execution_sha256")
    if not isinstance(digests, dict) or sorted(digests) != sorted(ids):
        raise ReportingInputError("run_grade.json contract.scenario_execution_sha256 does not index the scenarios")

    scenarios = tuple(_load_scenario(root, run, manifest, entry) for entry in run.scenarios)
    return ReportingRunInput(root, run, run_grade_sha256, manifest, scenarios)
