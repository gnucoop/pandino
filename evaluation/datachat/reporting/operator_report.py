"""Operator-oriented projection of a graded run (Reporting v1, design §9–§15).

``build_operator_report(run_input)`` turns a validated ``ReportingRunInput``
into a typed, renderer-independent model for a human Functional Evaluation
operator:

    ReportingRunInput → OperatorRunReport

Pure: no filesystem or network I/O, no mutation of the input, no clock or
environment value. It reorganizes persisted values only; it never grades,
renders or persists.

The visibility classes of design §9 are encoded structurally, not as UI
metadata:

- top-level fields of ``OperatorRunReport`` / ``ScenarioReport`` are MUST SHOW.
  Conditional elements (§9.1) are empty / None in their normal state and
  populated only from an exceptional persisted state;
- ``detail`` holds USEFUL DETAIL (all checks, timing, ...);
- ``provenance`` holds PROVENANCE / DEBUG (hashes, refs, request IDs, grader).

Grading input is schema 2: one technical outcome per gradeable scenario and
every established failure category, unranked.

Allowed derivations (§12) only: selection by persisted status, ID or category
equality, counts of persisted values, splitting ``counts_by_outcome`` into
outcome and gate groups, structural classification of the Agent result.
Forbidden (§13): new outcomes, gates, assessments, check statuses, failure
order or primary failure, severity, scores or rates.
"""

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
    outcome_label,
)
from evaluation.datachat.grading.reader import ParseStatus
from evaluation.datachat.model import EvidenceSourceRecord, Failure, ScenarioStatus, SuiteStatus
from evaluation.datachat.reporting.model import (
    AgentResult,
    DeliveredArtifact,
    ReportingRunInput,
    ReportingScenarioInput,
)

# Manifest config keys rendered as provider/model provenance (design §6.2); nothing else is copied.
_DATACHAT_IDENTITY_KEYS = ("DATACHAT_PROVIDER", "DATACHAT_MODEL", "DATACHAT_ENGINE")

_ATTENTION_CHECK_STATUSES = (CheckStatus.NOT_MET, CheckStatus.INDETERMINATE)


class OperatorReportError(Exception):
    """A validated reporting input violates an invariant the operator model relies on."""


# --- Agent result -----------------------------------------------------------------------


class ResultShape(str, Enum):
    """Structural shape of the persisted Agent result (design §7.2). Says nothing about correctness."""

    TEXT = "TEXT"                # response.type == "str"
    TABLE = "TABLE"              # response.type == "dataframe"
    IMAGE = "IMAGE"              # response.type == "image"
    STRUCTURED = "STRUCTURED"    # response.type == "dict"
    ERROR_BODY = "ERROR_BODY"    # {"error": ...}
    RAW = "RAW"                  # non-JSON, malformed JSON or unknown envelope: body verbatim
    NO_RESPONSE = "NO_RESPONSE"  # request recorded, no response persisted (transport / not captured)
    ABSENT = "ABSENT"            # no /datachat record at all (e.g. NOT_RUN)


_SHAPE_BY_TYPE = {"str": ResultShape.TEXT, "dataframe": ResultShape.TABLE,
                  "image": ResultShape.IMAGE, "dict": ResultShape.STRUCTURED}


@dataclass(frozen=True)
class OperatorAgentResult:
    """The persisted Agent result plus its structural shape.

    ``source`` is the Slice 1 ``AgentResult`` itself (exact body bytes, parsed
    envelope, text, notes, explanation, charts, artifacts): nothing is copied,
    rewritten or summarized. ``source`` is None only for ``ABSENT``.
    """

    shape: ResultShape
    source: AgentResult | None

    @property
    def charts(self) -> tuple[dict[str, Any], ...]:
        return self.source.content.charts if self.source and self.source.content else ()

    @property
    def image(self) -> DeliveredArtifact | None:
        return self.source.image if self.source else None

    @property
    def export(self) -> DeliveredArtifact | None:
        """The export artifact, kept apart from any inline table preview."""
        return self.source.export if self.source else None


def _agent_result(result: AgentResult | None) -> OperatorAgentResult:
    if result is None:
        return OperatorAgentResult(ResultShape.ABSENT, None)
    content = result.content
    if content is None:
        shape = ResultShape.NO_RESPONSE
    elif content.parse_status is ParseStatus.ERROR_BODY:
        shape = ResultShape.ERROR_BODY
    elif content.parse_status is ParseStatus.OK and content.response_type in _SHAPE_BY_TYPE:
        shape = _SHAPE_BY_TYPE[content.response_type]
    else:
        shape = ResultShape.RAW
    return OperatorAgentResult(shape, result)


# --- Attention ---------------------------------------------------------------------------


class AttentionReason(str, Enum):
    """Persisted exceptional states of design §11.

    Declaration order is a fixed listing order, not a severity ranking.
    """

    FAIL_OUTCOME = "FAIL_OUTCOME"
    INDETERMINATE_OUTCOME = "INDETERMINATE_OUTCOME"
    NOT_GRADEABLE = "NOT_GRADEABLE"
    EXECUTION_NOT_OK = "EXECUTION_NOT_OK"
    TEARDOWN_NOT_VERIFIED = "TEARDOWN_NOT_VERIFIED"


@dataclass(frozen=True)
class Attention:
    """Navigation aid: which persisted exceptional states hold. Not a judgment, not a rank."""

    reasons: tuple[AttentionReason, ...] = ()

    @property
    def required(self) -> bool:
        return bool(self.reasons)


def _attention(s: ReportingScenarioInput) -> Attention:
    g = s.grade
    held = {
        AttentionReason.FAIL_OUTCOME: g.outcome is TechnicalOutcome.FAIL,
        AttentionReason.INDETERMINATE_OUTCOME: g.outcome is TechnicalOutcome.INDETERMINATE,
        AttentionReason.NOT_GRADEABLE: g.gate is not Gate.GRADEABLE,
        AttentionReason.EXECUTION_NOT_OK: (g.execution_assessment is not None
                                           and g.execution_assessment is not ExecutionAssessment.OK),
        AttentionReason.TEARDOWN_NOT_VERIFIED: g.teardown_verified is False,
    }
    return Attention(tuple(reason for reason in AttentionReason if held[reason]))


# --- Scenario ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionIssue:
    """A persisted execution assessment other than OK, foregrounded with its reason."""

    assessment: ExecutionAssessment
    reason: ExecutionAssessmentReason


@dataclass(frozen=True)
class FailureView:
    """One persisted technical failure category with the checks carrying that category.

    ``checks_carrying_category`` are NOT_MET checks whose persisted ``failure``
    equals ``category``. Equality only: this is not a recorded cause. For
    EXECUTION (and a RESPONSE CONTRACT established by the execution
    assessment) the explanation is the execution assessment/reason.
    """

    category: TechnicalFailure
    checks_carrying_category: tuple[CheckResult, ...] = ()


@dataclass(frozen=True)
class ScenarioDetail:
    """USEFUL DETAIL: everything reachable on demand."""

    checks: tuple[CheckResult, ...]
    normal_checks: tuple[CheckResult, ...]
    execution_assessment_reason: ExecutionAssessmentReason | None
    execution_status: ScenarioStatus | None
    execution_failure: Failure | None
    teardown_verified: bool | None


@dataclass(frozen=True)
class ScenarioProvenance:
    """PROVENANCE / DEBUG: identities and navigation targets in the run package."""

    dir: str
    execution_sha256: str | None
    question_sha256: str | None
    evidence: tuple[EvidenceSourceRecord, ...]  # execution.json evidence records with FileRefs
    evidence_refs: tuple[str, ...]  # run-relative, from checks, first-seen order
    request_ids: tuple[str, ...]
    teardown_verified: bool | None


@dataclass(frozen=True)
class ScenarioReport:
    """One scenario for the operator. Top-level fields are MUST SHOW.

    Question, Agent result and Grading stay separate. ``outcome`` None means
    no technical outcome exists (non-gradeable gate). Conditional fields are
    empty/None in the normal state.
    """

    scenario_id: str
    attention: Attention
    # Question
    question: str | None
    question_sent: bool
    # Agent result
    agent_result: OperatorAgentResult
    # Grading
    gate: Gate
    outcome: TechnicalOutcome | None
    execution_assessment: ExecutionAssessment | None
    execution_issue: ExecutionIssue | None
    failures: tuple[FailureView, ...]  # persisted order, unranked
    attention_checks: tuple[CheckResult, ...]  # persisted NOT_MET / INDETERMINATE
    check_status_counts: tuple[tuple[CheckStatus, int], ...]  # compact summary, CheckStatus order
    gate_execution_failure: Failure | None  # execution.json failure, non-gradeable gates only
    detail: ScenarioDetail
    provenance: ScenarioProvenance

    @property
    def has_outcome(self) -> bool:
        return self.outcome is not None

    @property
    def label(self) -> str | None:
        """``TECHNICAL PASS``, ``INDETERMINATE`` or ``FAIL — A, B`` from persisted values."""
        return outcome_label(self.outcome, tuple(f.category for f in self.failures))


def _failures(s: ReportingScenarioInput) -> tuple[FailureView, ...]:
    g = s.grade
    return tuple(
        FailureView(category, tuple(c for c in g.checks if c.failure is category))
        for category in g.failures)


def _unique(values) -> tuple:
    return tuple(dict.fromkeys(v for v in values if v is not None))


def _scenario(s: ReportingScenarioInput) -> ScenarioReport:
    g = s.grade
    execution = s.execution
    gradeable = g.gate is Gate.GRADEABLE
    statuses = Counter(c.status for c in g.checks)
    issue = None
    if g.execution_assessment is not None and g.execution_assessment is not ExecutionAssessment.OK:
        issue = ExecutionIssue(g.execution_assessment, g.execution_assessment_reason)
    result = s.agent_result
    request_ids = _unique([
        result.exchange.request_id if result and result.exchange else None,
        result.image.metadata.source_request_id if result and result.image and result.image.metadata else None,
    ])
    return ScenarioReport(
        scenario_id=s.scenario_id,
        attention=_attention(s),
        question=s.question,
        question_sent=bool(result and result.attempts),
        agent_result=_agent_result(result),
        gate=g.gate,
        outcome=g.outcome,
        execution_assessment=g.execution_assessment,
        execution_issue=issue,
        failures=_failures(s),
        attention_checks=tuple(c for c in g.checks if c.status in _ATTENTION_CHECK_STATUSES),
        check_status_counts=tuple((st, statuses[st]) for st in CheckStatus if statuses[st]),
        gate_execution_failure=execution.failure if execution and not gradeable else None,
        detail=ScenarioDetail(
            checks=g.checks,
            normal_checks=tuple(c for c in g.checks if c.status not in _ATTENTION_CHECK_STATUSES),
            execution_assessment_reason=g.execution_assessment_reason,
            execution_status=execution.status if execution else None,
            execution_failure=execution.failure if execution else None,
            teardown_verified=g.teardown_verified,
        ),
        provenance=ScenarioProvenance(
            dir=s.dir,
            execution_sha256=s.execution_sha256,
            question_sha256=s.question_sha256,
            evidence=tuple(execution.evidence.values()) if execution else (),
            evidence_refs=_unique(ref for c in g.checks for ref in c.evidence_refs),
            request_ids=request_ids,
            teardown_verified=g.teardown_verified,
        ),
    )


# --- Run ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunDetail:
    """USEFUL DETAIL from ``run_manifest.json`` (design §6.2), never interpreted."""

    status: SuiteStatus
    failure: Failure | None
    started_at: str | None
    ended_at: str | None
    maui: dict[str, Any]
    datachat: dict[str, Any]  # provider / model / engine only
    scenario_order: tuple[str, ...]


@dataclass(frozen=True)
class RunProvenance:
    """PROVENANCE / DEBUG: what produced the grade and which file this report reflects."""

    run_grade_sha256: str
    grader: dict[str, Any]
    contract: dict[str, Any]
    grading_schema_version: int
    manifest_schema_version: int


@dataclass(frozen=True)
class OperatorRunReport:
    """One graded run for the operator. Top-level fields are MUST SHOW.

    There is no run outcome, score or rate. ``outcome_counts`` and
    ``gate_counts`` are the persisted ``counts_by_outcome`` split by key kind;
    ``failure_counts`` is the persisted ``counts_by_failure``; all in
    persisted order, zeros included.
    """

    run_id: str
    official: bool
    comparable: bool
    inventory: tuple[ScenarioIndexEntry, ...]  # frozen run order
    outcome_counts: tuple[tuple[TechnicalOutcome, int], ...]
    failure_counts: tuple[tuple[TechnicalFailure, int], ...]
    gate_counts: tuple[tuple[Gate, int], ...]
    scenarios: tuple[ScenarioReport, ...]  # frozen run order
    detail: RunDetail
    provenance: RunProvenance

    @property
    def attention_scenarios(self) -> tuple[str, ...]:
        return tuple(s.scenario_id for s in self.scenarios if s.attention.required)


def _split_counts(counts: dict[str, int]) -> tuple[tuple, tuple]:
    outcomes, gates = [], []
    for key, value in counts.items():
        if key in TechnicalOutcome._value2member_map_:
            outcomes.append((TechnicalOutcome(key), value))
        elif key in Gate._value2member_map_ and key != Gate.GRADEABLE.value:
            gates.append((Gate(key), value))
        else:
            raise OperatorReportError(f"counts_by_outcome has an unknown key {key!r}")
    return tuple(outcomes), tuple(gates)


def _failure_counts(counts: dict[str, int]) -> tuple[tuple[TechnicalFailure, int], ...]:
    try:
        return tuple((TechnicalFailure(key), value) for key, value in counts.items())
    except ValueError as exc:
        raise OperatorReportError(f"counts_by_failure has an unknown key: {exc}") from exc


def build_operator_report(run_input: ReportingRunInput) -> OperatorRunReport:
    """Project a validated ``ReportingRunInput`` for the operator. Pure; reads and writes nothing."""
    rg, manifest = run_input.run_grade, run_input.manifest
    if [e.scenario_id for e in rg.scenarios] != [s.scenario_id for s in run_input.scenarios]:
        raise OperatorReportError("scenario inputs do not follow RunGrade.scenarios")
    outcome_counts, gate_counts = _split_counts(rg.counts_by_outcome)
    config = manifest.datachat_config.get("values", {})
    return OperatorRunReport(
        run_id=rg.run_id,
        official=rg.official,
        comparable=rg.comparable,
        inventory=rg.scenarios,
        outcome_counts=outcome_counts,
        failure_counts=_failure_counts(rg.counts_by_failure),
        gate_counts=gate_counts,
        scenarios=tuple(_scenario(s) for s in run_input.scenarios),
        detail=RunDetail(
            status=manifest.status,
            failure=manifest.failure,
            started_at=manifest.started_at,
            ended_at=manifest.ended_at,
            maui=dict(manifest.maui),
            datachat={k: config.get(k) for k in _DATACHAT_IDENTITY_KEYS},
            scenario_order=tuple(manifest.scenario_order),
        ),
        provenance=RunProvenance(
            run_grade_sha256=run_input.run_grade_sha256,
            grader=dict(rg.grader),
            contract=dict(rg.contract),
            grading_schema_version=rg.schema_version,
            manifest_schema_version=manifest.schema_version,
        ),
    )
