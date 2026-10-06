"""Execution domain model of the DataChat Functional Evaluation harness.

Dataclasses and string enums describing what one evaluation run executed and
which evidence it captured. No I/O and no grading happen here.

Two kinds of record:

- frozen value records (``Failure``, ``FileRef``, ``HttpExchange``,
  ``EvidenceSourceRecord``, ``ArtifactRecord``, ``UsageEvidence``, ...):
  facts captured once and never changed;
- mutable execution records (``PhaseOutcome``, ``ScenarioFlags``,
  ``ScenarioExecution``, ``EvaluationRun``): manifests the runner updates
  while a run progresses and freezes on disk at closure.

Every record serializes with ``to_dict()`` to plain JSON-compatible values
(enums become their string value). Top-level persisted records carry
``schema_version``. ``from_dict()`` exists only for the records read back from
disk: ``HttpExchange``, ``ScenarioExecution`` and ``EvaluationRun``.
"""

import hashlib
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1


def utc_now() -> str:
    """Current time as a timezone-aware UTC ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class _Serializable:
    def to_dict(self) -> dict:
        return _to_jsonable(self)


# --- Enums -------------------------------------------------------------------


class SuiteStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ABORTED_INFRA = "ABORTED_INFRA"


class ScenarioStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    EXECUTED = "EXECUTED"
    NOT_SCOREABLE_INFRA = "NOT_SCOREABLE_INFRA"


class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class Availability(str, Enum):
    """Execution availability of one evidence source (not a grading class)."""

    PRESENT = "PRESENT"
    EXPECTED_ABSENT = "EXPECTED_ABSENT"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Phase(str, Enum):
    """Per-scenario phases, declared in execution order."""

    PRE_CLEANUP = "PRE_CLEANUP"
    START = "START"
    DATACHAT = "DATACHAT"
    COLLECT = "COLLECT"
    TEARDOWN = "TEARDOWN"
    CLOSE = "CLOSE"


class InfraCode(str, Enum):
    """Infrastructure failure codes; never an analytical scenario verdict."""

    INFRA_PREFLIGHT = "INFRA_PREFLIGHT"
    INFRA_MAUI_UNAVAILABLE = "INFRA_MAUI_UNAVAILABLE"
    INFRA_EVIDENCE_UNAVAILABLE = "INFRA_EVIDENCE_UNAVAILABLE"
    INFRA_FRESH_ENGINE_UNPROVEN = "INFRA_FRESH_ENGINE_UNPROVEN"
    INFRA_TEARDOWN_UNVERIFIED = "INFRA_TEARDOWN_UNVERIFIED"
    INFRA_TOKEN_BUDGET = "INFRA_TOKEN_BUDGET"
    INFRA_HTTP_TIMEOUT = "INFRA_HTTP_TIMEOUT"


# --- Frozen value records ----------------------------------------------------


@dataclass(frozen=True)
class Failure(_Serializable):
    """An INFRA_* condition. ``detail`` holds exception type and truncated message."""

    code: InfraCode
    summary: str
    phase: Phase | None = None
    detail: str | None = None
    traceback_file: str | None = None

    @classmethod
    def _from_dict(cls, data: dict | None) -> "Failure | None":
        if data is None:
            return None
        phase = data.get("phase")
        return cls(
            code=InfraCode(data["code"]),
            summary=data["summary"],
            phase=Phase(phase) if phase is not None else None,
            detail=data.get("detail"),
            traceback_file=data.get("traceback_file"),
        )


@dataclass(frozen=True)
class FileRef(_Serializable):
    """A persisted evidence file; ``path`` is relative to the scenario directory."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class TransportError(_Serializable):
    """No HTTP response was received."""

    type: str
    message: str


@dataclass(frozen=True)
class HttpRequestRecord(_Serializable):
    """Request side of an exchange. The API key never appears in ``headers_redacted``.

    ``body_file``/``body_sha256`` are null when no body is stored; the
    multipart start request lists its non-file fields in ``form`` and
    references the dataset hash there instead of duplicating the CSV.
    """

    headers_redacted: dict[str, str]
    content_type: str | None = None
    body_file: str | None = None
    body_sha256: str | None = None
    form: dict[str, Any] | None = None


@dataclass(frozen=True)
class HttpResponseRecord(_Serializable):
    """Response side; the hash is of the stored (possibly redacted) bytes."""

    status: int
    headers: dict[str, str]
    content_type: str | None
    body_file: str
    body_sha256: str
    size: int
    redacted: bool = False


@dataclass(frozen=True)
class HttpExchange(_Serializable):
    """One HTTP attempt. Without a response, ``request_id`` and ``response`` are null."""

    key: str
    attempt: int
    method: str
    path: str
    sent_at: str
    completed_at: str
    duration_ms: int
    request: HttpRequestRecord
    response: HttpResponseRecord | None = None
    request_id: str | None = None
    transport_error: TransportError | None = None
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict) -> "HttpExchange":
        response = data.get("response")
        transport_error = data.get("transport_error")
        return cls(
            schema_version=data["schema_version"],
            key=data["key"],
            attempt=data["attempt"],
            method=data["method"],
            path=data["path"],
            sent_at=data["sent_at"],
            completed_at=data["completed_at"],
            duration_ms=data["duration_ms"],
            request=HttpRequestRecord(**data["request"]),
            response=HttpResponseRecord(**response) if response is not None else None,
            request_id=data.get("request_id"),
            transport_error=(
                TransportError(**transport_error) if transport_error is not None else None
            ),
        )


@dataclass(frozen=True)
class EvidenceSourceRecord(_Serializable):
    """Availability of one evidence source; ``reason`` is set unless PRESENT.

    ``locator`` is source-specific (request ID, offsets, line counts,
    matches, query text).
    """

    source: str
    availability: Availability
    reason: str | None = None
    files: tuple[FileRef, ...] = ()
    locator: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _from_dict(cls, data: dict) -> "EvidenceSourceRecord":
        return cls(
            source=data["source"],
            availability=Availability(data["availability"]),
            reason=data.get("reason"),
            files=tuple(FileRef(**ref) for ref in data.get("files", [])),
            locator=dict(data.get("locator", {})),
        )


@dataclass(frozen=True)
class ArtifactRecord(_Serializable):
    """A plot image or CSV export.

    Image-only fields: ``wrapper_stripped``, ``decode_status`` (``OK`` |
    ``WRAPPER_MISSING`` | ``BASE64_INVALID`` | ``NOT_PNG``),
    ``png_signature_ok``, ``detected_format``. Export-only: ``http_status``.
    Fields of the other kind stay null.
    """

    kind: str  # "image" | "export"
    source_request_id: str | None
    status: Availability
    path: str | None = None
    sha256: str | None = None
    size: int | None = None
    wrapper_stripped: bool | None = None
    decode_status: str | None = None
    png_signature_ok: bool | None = None
    detected_format: str | None = None
    http_status: int | None = None


@dataclass(frozen=True)
class UsageEvidence(_Serializable):
    """Verbatim Usage (``logs``) rows for one request ID, in ``id`` order."""

    request_id: str
    query_sql: str
    schema: str
    executed_at: str
    rows: tuple[dict[str, Any], ...] = ()
    primary_log_id: int | None = None
    primary_row_present: bool = False
    schema_version: int = SCHEMA_VERSION


# --- Mutable execution records -----------------------------------------------


@dataclass
class PhaseOutcome(_Serializable):
    phase: Phase
    status: PhaseStatus = PhaseStatus.PENDING
    started_at: str | None = None
    ended_at: str | None = None
    request_keys: list[str] = field(default_factory=list)
    failure: Failure | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def _from_dict(cls, data: dict) -> "PhaseOutcome":
        return cls(
            phase=Phase(data["phase"]),
            status=PhaseStatus(data["status"]),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            request_keys=list(data.get("request_keys", [])),
            failure=Failure._from_dict(data.get("failure")),
            notes=list(data.get("notes", [])),
        )


@dataclass
class ScenarioFlags(_Serializable):
    """Completeness flags; all start False until the runner proves otherwise."""

    reached_datachat: bool = False
    previous_engine_present: bool = False
    fresh_engine_proven: bool = False
    execution_complete: bool = False
    evidence_complete: bool = False
    scoreable: bool = False
    teardown_verified: bool = False


@dataclass
class ScenarioExecution(_Serializable):
    """Manifest of one scenario (``scenarios/<id>/execution.json``).

    ``requests`` maps an exchange key (``pre_cleanup``, ``start``,
    ``datachat``, ``teardown``, ``export``) to the scenario-relative paths of
    its ``HttpExchange`` metadata files, one per attempt. ``evidence`` maps a
    source name to its ``EvidenceSourceRecord``.
    """

    run_id: str
    scenario_id: str
    ordinal: int
    question: str
    question_sha256: str
    status: ScenarioStatus = ScenarioStatus.RUNNING
    failure: Failure | None = None
    started_at: str | None = None
    ended_at: str | None = None
    requests: dict[str, list[str]] = field(default_factory=dict)
    phases: list[PhaseOutcome] = field(default_factory=list)
    evidence: dict[str, EvidenceSourceRecord] = field(default_factory=dict)
    flags: ScenarioFlags = field(default_factory=ScenarioFlags)
    notes: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def new(cls, run_id: str, scenario_id: str, ordinal: int, question: str) -> "ScenarioExecution":
        """A RUNNING scenario with the six phases PENDING, in execution order."""
        return cls(
            run_id=run_id,
            scenario_id=scenario_id,
            ordinal=ordinal,
            question=question,
            question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
            started_at=utc_now(),
            phases=[PhaseOutcome(phase=phase) for phase in Phase],
        )

    def mark_not_run(self) -> None:
        """The suite aborted before this scenario started."""
        self.status = ScenarioStatus.NOT_RUN
        for outcome in self.phases:
            outcome.status = PhaseStatus.SKIPPED

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioExecution":
        return cls(
            schema_version=data["schema_version"],
            run_id=data["run_id"],
            scenario_id=data["scenario_id"],
            ordinal=data["ordinal"],
            question=data["question"],
            question_sha256=data["question_sha256"],
            status=ScenarioStatus(data["status"]),
            failure=Failure._from_dict(data.get("failure")),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            requests={key: list(refs) for key, refs in data.get("requests", {}).items()},
            phases=[PhaseOutcome._from_dict(item) for item in data.get("phases", [])],
            evidence={
                name: EvidenceSourceRecord._from_dict(item)
                for name, item in data.get("evidence", {}).items()
            },
            flags=ScenarioFlags(**data.get("flags", {})),
            notes=list(data.get("notes", [])),
        )


@dataclass
class EvaluationRun(_Serializable):
    """Run manifest (``run_manifest.json``).

    Nested metadata stays as plain JSON dicts:

    - ``maui``: ``git_sha``, ``branch``, ``dirty``, ``dirty_files``;
    - ``evaluation_contract``: artifact hashes and ``all_match``;
    - ``runtime``: Python/package versions, Gunicorn argv, port, cwd;
    - ``datachat_config``: effective non-secret values and secret-presence flags;
    - ``eval_identity``: label, ``user_id``, optional ``api_key_sha256``;
    - ``scenarios``: index entries ``{scenario_id, status, dir, execution_sha256}``.
    """

    run_id: str
    started_at: str | None = None
    ended_at: str | None = None
    status: SuiteStatus = SuiteStatus.RUNNING
    official: bool = False
    comparable: bool = False
    failure: Failure | None = None
    maui: dict[str, Any] = field(default_factory=dict)
    evaluation_contract: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    datachat_config: dict[str, Any] = field(default_factory=dict)
    eval_identity: dict[str, Any] = field(default_factory=dict)
    token_balance_at_preflight: float | None = None
    scenario_order: list[str] = field(default_factory=list)
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationRun":
        return cls(
            schema_version=data["schema_version"],
            run_id=data["run_id"],
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            status=SuiteStatus(data["status"]),
            official=data.get("official", False),
            comparable=data.get("comparable", False),
            failure=Failure._from_dict(data.get("failure")),
            maui=dict(data.get("maui", {})),
            evaluation_contract=dict(data.get("evaluation_contract", {})),
            runtime=dict(data.get("runtime", {})),
            datachat_config=dict(data.get("datachat_config", {})),
            eval_identity=dict(data.get("eval_identity", {})),
            token_balance_at_preflight=data.get("token_balance_at_preflight"),
            scenario_order=list(data.get("scenario_order", [])),
            scenarios=[dict(entry) for entry in data.get("scenarios", [])],
        )
