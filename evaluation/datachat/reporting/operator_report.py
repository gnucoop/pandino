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
- ``detail`` holds USEFUL DETAIL (all checks, resolved reviews, timing, ...);
- ``provenance`` holds PROVENANCE / DEBUG (hashes, refs, request IDs, grader).

Allowed derivations (§12) only: selection by persisted status, ID or category
equality, counts of persisted values, splitting ``counts_by_verdict`` into
verdict and gate groups, structural classification of the Agent result.
Forbidden (§13): new verdicts, gates, assessments, check or review outcomes,
failure order, caveats, severity, scores, rates, or a stated basis for a
``NEEDS REVIEW`` verdict the grader did not persist.
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
    FailureCategory,
    Gate,
    ManualReviewItem,
    ReviewStatus,
    ScenarioIndexEntry,
    Verdict,
)
from evaluation.datachat.grading.reader import ParseStatus
from evaluation.datachat.model import EvidenceSourceRecord, Failure, ScenarioStatus, SuiteStatus
from evaluation.datachat.reporting.model import (
    AgentResult,
    DeliveredArtifact,
    ReportingRunInput,
    ReportingScenarioInput,
)

# Where pending review answers are recorded (grading §review workflow); shown, never written.
REVIEW_FILE_TEMPLATE = "grading/reviews/{scenario_id}.json"

# Manifest config keys rendered as provider/model provenance (design §6.2); nothing else is copied.
_DATACHAT_IDENTITY_KEYS = ("DATACHAT_PROVIDER", "DATACHAT_MODEL", "DATACHAT_ENGINE")

_FAIL_VERDICTS = frozenset(c.verdict for c in FailureCategory)
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

    FAIL_VERDICT = "FAIL_VERDICT"
    NEEDS_REVIEW_VERDICT = "NEEDS_REVIEW_VERDICT"
    PASS_WITH_CAVEAT = "PASS_WITH_CAVEAT"
    NOT_GRADEABLE = "NOT_GRADEABLE"
    PENDING_REVIEWS = "PENDING_REVIEWS"
    EXECUTION_NOT_OK = "EXECUTION_NOT_OK"
    TEARDOWN_NOT_VERIFIED = "TEARDOWN_NOT_VERIFIED"


@dataclass(frozen=True)
class Attention:
    """Navigation aid: which persisted exceptional states hold. Not a judgment, not a rank."""

    reasons: tuple[AttentionReason, ...] = ()

    @property
    def required(self) -> bool:
        return bool(self.reasons)


def _attention(entry: ScenarioIndexEntry, s: ReportingScenarioInput) -> Attention:
    g = s.grade
    held = {
        AttentionReason.FAIL_VERDICT: g.verdict in _FAIL_VERDICTS,
        AttentionReason.NEEDS_REVIEW_VERDICT: g.verdict is Verdict.NEEDS_REVIEW,
        AttentionReason.PASS_WITH_CAVEAT: g.verdict is Verdict.PASS_WITH_CAVEAT,
        AttentionReason.NOT_GRADEABLE: g.gate is not Gate.GRADEABLE,
        AttentionReason.PENDING_REVIEWS: entry.pending_reviews > 0,
        AttentionReason.EXECUTION_NOT_OK: (g.execution_assessment is not None
                                           and g.execution_assessment is not ExecutionAssessment.OK),
        AttentionReason.TEARDOWN_NOT_VERIFIED: g.teardown_verified is False,
    }
    return Attention(tuple(reason for reason in AttentionReason if held[reason]))


# --- Scenario ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionIssue:
    """A persisted execution assessment other than OK, foregrounded with its reason.

    ``resolvable_by_review`` is False in v1: review files cannot change the
    execution assessment (grading design; reporting design §8.3).
    """

    assessment: ExecutionAssessment
    reason: ExecutionAssessmentReason
    resolvable_by_review: bool = False


@dataclass(frozen=True)
class FailureView:
    """One persisted failure category with the items carrying that category.

    ``items_carrying_category`` are checks whose persisted ``failure_category``
    and FAILED reviews whose ``failure_category_if_failed`` equal ``category``.
    Equality only: this is not a recorded cause. For EXECUTION the explanation
    is the execution assessment/reason, so no item is associated by design.
    """

    category: FailureCategory
    is_root: bool
    checks_carrying_category: tuple[CheckResult, ...] = ()
    reviews_carrying_category: tuple[ManualReviewItem, ...] = ()


@dataclass(frozen=True)
class CaveatItem:
    """A persisted caveat ID resolved to exactly one check or review item."""

    check_id: str
    check: CheckResult | None = None
    review: ManualReviewItem | None = None


@dataclass(frozen=True)
class PendingReview:
    """A PENDING review item and where its answer is recorded (never answered here)."""

    item: ManualReviewItem
    answer_file: str  # run-relative


@dataclass(frozen=True)
class ScenarioDetail:
    """USEFUL DETAIL: everything reachable on demand."""

    checks: tuple[CheckResult, ...]
    normal_checks: tuple[CheckResult, ...]
    resolved_reviews: tuple[ManualReviewItem, ...]
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
    evidence_refs: tuple[str, ...]  # run-relative, from checks and reviews, first-seen order
    request_ids: tuple[str, ...]
    review_evidence_sha256: tuple[tuple[str, str], ...]  # (check_id, evidence_sha256)
    teardown_verified: bool | None


@dataclass(frozen=True)
class ScenarioReport:
    """One scenario for the operator. Top-level fields are MUST SHOW.

    Question, Agent result and Grading stay separate. ``verdict`` None means
    no analytical verdict exists (non-gradeable gate). Conditional fields are
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
    verdict: Verdict | None
    execution_assessment: ExecutionAssessment | None
    execution_issue: ExecutionIssue | None
    root_failure: FailureCategory | None
    failures: tuple[FailureView, ...]  # root first, then other_failures, in persisted order
    caveats: tuple[CaveatItem, ...]
    pending_reviews: tuple[PendingReview, ...]
    attention_checks: tuple[CheckResult, ...]  # persisted NOT_MET / INDETERMINATE
    check_status_counts: tuple[tuple[CheckStatus, int], ...]  # compact summary, CheckStatus order
    gate_execution_failure: Failure | None  # execution.json failure, non-gradeable gates only
    detail: ScenarioDetail
    provenance: ScenarioProvenance

    @property
    def has_verdict(self) -> bool:
        return self.verdict is not None

    @property
    def other_failures(self) -> tuple[FailureCategory, ...]:
        return tuple(f.category for f in self.failures if not f.is_root)


def _failures(s: ReportingScenarioInput) -> tuple[FailureView, ...]:
    g = s.grade
    if g.root_failure is None:
        if g.other_failures:
            raise OperatorReportError(f"{s.scenario_id}: other_failures without a root_failure")
        return ()
    views = []
    for category in (g.root_failure, *g.other_failures):
        views.append(FailureView(
            category=category,
            is_root=category is g.root_failure,
            checks_carrying_category=tuple(c for c in g.checks if c.failure_category is category),
            reviews_carrying_category=tuple(
                r for r in g.manual_reviews
                if r.status is ReviewStatus.FAILED and r.failure_category_if_failed is category),
        ))
    return tuple(views)


def _caveats(s: ReportingScenarioInput) -> tuple[CaveatItem, ...]:
    g = s.grade
    items = []
    for check_id in g.caveats:
        checks = [c for c in g.checks if c.check_id == check_id]
        reviews = [r for r in g.manual_reviews if r.check_id == check_id]
        if len(checks) + len(reviews) != 1:
            raise OperatorReportError(
                f"{s.scenario_id}: caveat {check_id!r} resolves to {len(checks) + len(reviews)} items")
        items.append(CaveatItem(check_id, checks[0] if checks else None, reviews[0] if reviews else None))
    return tuple(items)


def _unique(values) -> tuple:
    return tuple(dict.fromkeys(v for v in values if v is not None))


def _scenario(entry: ScenarioIndexEntry, s: ReportingScenarioInput) -> ScenarioReport:
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
        attention=_attention(entry, s),
        question=s.question,
        question_sent=bool(result and result.attempts),
        agent_result=_agent_result(result),
        gate=g.gate,
        verdict=g.verdict,
        execution_assessment=g.execution_assessment,
        execution_issue=issue,
        root_failure=g.root_failure,
        failures=_failures(s),
        caveats=_caveats(s),
        pending_reviews=tuple(
            PendingReview(r, REVIEW_FILE_TEMPLATE.format(scenario_id=s.scenario_id))
            for r in g.manual_reviews if r.status is ReviewStatus.PENDING),
        attention_checks=tuple(c for c in g.checks if c.status in _ATTENTION_CHECK_STATUSES),
        check_status_counts=tuple((st, statuses[st]) for st in CheckStatus if statuses[st]),
        gate_execution_failure=execution.failure if execution and not gradeable else None,
        detail=ScenarioDetail(
            checks=g.checks,
            normal_checks=tuple(c for c in g.checks if c.status not in _ATTENTION_CHECK_STATUSES),
            resolved_reviews=tuple(r for r in g.manual_reviews if r.status is not ReviewStatus.PENDING),
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
            evidence_refs=_unique(ref for item in (*g.checks, *g.manual_reviews) for ref in item.evidence_refs),
            request_ids=request_ids,
            review_evidence_sha256=tuple((r.check_id, r.evidence_sha256) for r in g.manual_reviews),
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

    There is no run verdict, score or rate. ``verdict_counts`` and
    ``gate_counts`` are the persisted ``counts_by_verdict`` split by key kind,
    in persisted order, zeros included.
    """

    run_id: str
    official: bool
    comparable: bool
    grading_complete: bool
    inventory: tuple[ScenarioIndexEntry, ...]  # frozen run order
    verdict_counts: tuple[tuple[Verdict, int], ...]
    gate_counts: tuple[tuple[Gate, int], ...]
    scenarios: tuple[ScenarioReport, ...]  # frozen run order
    detail: RunDetail
    provenance: RunProvenance

    @property
    def attention_scenarios(self) -> tuple[str, ...]:
        return tuple(s.scenario_id for s in self.scenarios if s.attention.required)

    @property
    def incomplete_scenarios(self) -> tuple[str, ...]:
        """Scenarios keeping grading incomplete (persisted NEEDS REVIEW); empty when complete."""
        return tuple(s.scenario_id for s in self.scenarios if s.verdict is Verdict.NEEDS_REVIEW)


def _split_counts(counts: dict[str, int]) -> tuple[tuple, tuple]:
    verdicts, gates = [], []
    for key, value in counts.items():
        if key in Verdict._value2member_map_:
            verdicts.append((Verdict(key), value))
        elif key in Gate._value2member_map_ and key != Gate.GRADEABLE.value:
            gates.append((Gate(key), value))
        else:
            raise OperatorReportError(f"counts_by_verdict has an unknown key {key!r}")
    return tuple(verdicts), tuple(gates)


def build_operator_report(run_input: ReportingRunInput) -> OperatorRunReport:
    """Project a validated ``ReportingRunInput`` for the operator. Pure; reads and writes nothing."""
    rg, manifest = run_input.run_grade, run_input.manifest
    if [e.scenario_id for e in rg.scenarios] != [s.scenario_id for s in run_input.scenarios]:
        raise OperatorReportError("scenario inputs do not follow RunGrade.scenarios")
    verdict_counts, gate_counts = _split_counts(rg.counts_by_verdict)
    config = manifest.datachat_config.get("values", {})
    return OperatorRunReport(
        run_id=rg.run_id,
        official=rg.official,
        comparable=rg.comparable,
        grading_complete=rg.grading_complete,
        inventory=rg.scenarios,
        verdict_counts=verdict_counts,
        gate_counts=gate_counts,
        scenarios=tuple(_scenario(e, s) for e, s in zip(rg.scenarios, run_input.scenarios)),
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
