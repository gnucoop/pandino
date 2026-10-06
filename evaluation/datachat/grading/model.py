"""Grading domain model of the DataChat Functional Evaluation.

Frozen vocabulary (dimensions, verdicts, requirement levels, evidence
classes) and the derived result records. No I/O and no grading logic happen
here: nothing in this module decides a verdict.

Records serialize with ``to_dict()`` to plain JSON-compatible values, using
the same convention as the execution model (enums become their string
value). Top-level persisted records carry ``schema_version``.

There is deliberately no numeric score and no run-level verdict.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from evaluation.datachat.model import _Serializable

GRADING_SCHEMA_VERSION = 1


# --- Frozen vocabulary -------------------------------------------------------


class Dimension(str, Enum):
    UNDERSTANDING = "Understanding"
    TOOL_SELECTION = "Tool selection"
    ANALYTICAL = "Analytical correctness"
    CHAINING = "Chaining"
    HONESTY = "Honesty / coverage"
    INTERPRETATION = "Final interpretation"
    BACKEND_CONTRACT = "Backend contract"


class FailureCategory(str, Enum):
    """FAIL verdict categories. EXECUTION is a category, not a dimension."""

    EXECUTION = "EXECUTION"
    UNDERSTANDING = "UNDERSTANDING"
    TOOL_SELECTION = "TOOL SELECTION"
    CHAINING = "CHAINING"
    ANALYTICAL = "ANALYTICAL"
    HONESTY = "HONESTY"
    INTERPRETATION = "INTERPRETATION"
    BACKEND_CONTRACT = "BACKEND CONTRACT"

    @property
    def verdict(self) -> "Verdict":
        return Verdict(f"FAIL — {self.value}")


class Verdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_CAVEAT = "PASS WITH CAVEAT"
    FAIL_UNDERSTANDING = "FAIL — UNDERSTANDING"
    FAIL_TOOL_SELECTION = "FAIL — TOOL SELECTION"
    FAIL_ANALYTICAL = "FAIL — ANALYTICAL"
    FAIL_CHAINING = "FAIL — CHAINING"
    FAIL_HONESTY = "FAIL — HONESTY"
    FAIL_INTERPRETATION = "FAIL — INTERPRETATION"
    FAIL_BACKEND_CONTRACT = "FAIL — BACKEND CONTRACT"
    FAIL_EXECUTION = "FAIL — EXECUTION"
    NEEDS_REVIEW = "NEEDS REVIEW"


class RequirementLevel(str, Enum):
    MUST = "MUST"
    SHOULD = "SHOULD"
    MAY = "MAY"


class EvidenceClass(str, Enum):
    DIRECT = "DIRECT"
    INFERRED = "INFERRED"
    MANUAL_REVIEW_ONLY = "MANUAL_REVIEW_ONLY"
    NOT_SCOREABLE_IN_V1 = "NOT_SCOREABLE_IN_V1"


class CheckStatus(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    INDETERMINATE = "INDETERMINATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ExecutionAssessment(str, Enum):
    """Generic usability of the DataChat outcome, derived outside the scenario catalogue.

    - OK: the outcome is usable; grade checks and reviews normally;
    - FAILED: the outcome itself is unusable → FAIL — EXECUTION (root);
    - NEEDS_REVIEW: borderline execution evidence → NEEDS REVIEW unless a FAIL is established.
    """

    OK = "OK"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ExecutionAssessmentReason(str, Enum):
    """Stable, scenario-agnostic reason for an ``ExecutionAssessment`` (one per derivation branch)."""

    USABLE_OUTCOME = "USABLE_OUTCOME"
    ERROR_OUTCOME = "ERROR_OUTCOME"
    HTTP_5XX = "HTTP_5XX"
    EMPTY_OUTCOME = "EMPTY_OUTCOME"
    MISSING_RESPONSE = "MISSING_RESPONSE"
    CONFLICTING_KINDS = "CONFLICTING_KINDS"
    HTTP_NON_2XX = "HTTP_NON_2XX"
    UNPARSEABLE_RESPONSE = "UNPARSEABLE_RESPONSE"
    RUNTIME_STATUS_CONFLICT = "RUNTIME_STATUS_CONFLICT"
    ADAPTER_FALLBACK = "ADAPTER_FALLBACK"


_R = ExecutionAssessmentReason
EXECUTION_REASONS: dict[ExecutionAssessment, frozenset[ExecutionAssessmentReason]] = {
    ExecutionAssessment.OK: frozenset({_R.USABLE_OUTCOME}),
    ExecutionAssessment.FAILED: frozenset({_R.ERROR_OUTCOME, _R.HTTP_5XX, _R.EMPTY_OUTCOME}),
    ExecutionAssessment.NEEDS_REVIEW: frozenset({
        _R.MISSING_RESPONSE, _R.CONFLICTING_KINDS, _R.HTTP_NON_2XX, _R.UNPARSEABLE_RESPONSE,
        _R.RUNTIME_STATUS_CONFLICT, _R.ADAPTER_FALLBACK,
    }),
}


def check_execution_provenance(
    assessment: ExecutionAssessment | None, reason: ExecutionAssessmentReason | None
) -> None:
    """Both enums and a valid pair, or both None. Raises ``ValueError``."""
    if assessment is None and reason is None:
        return
    if not isinstance(assessment, ExecutionAssessment) or not isinstance(reason, ExecutionAssessmentReason):
        raise ValueError(f"execution assessment and reason must both be set: {assessment!r}, {reason!r}")
    if reason not in EXECUTION_REASONS[assessment]:
        raise ValueError(f"reason {reason.value} is not valid for execution assessment {assessment.value}")


class Gate(str, Enum):
    """Whether a scenario may be analytically graded. Not a verdict."""

    GRADEABLE = "GRADEABLE"
    NOT_RUN = "NOT_RUN"
    NOT_SCOREABLE_INFRA = "NOT_SCOREABLE_INFRA"
    NOT_SCOREABLE = "NOT_SCOREABLE"


# --- Small-value guard -------------------------------------------------------

_SCALARS = (str, int, float, bool, type(None))


def _is_small_value(value: Any) -> bool:
    """Scalars, or flat lists/tuples/dicts of scalars. Never raw evidence bytes."""
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, _SCALARS) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and isinstance(v, _SCALARS) for k, v in value.items())
    return False


def _check_refs(refs: tuple[str, ...]) -> None:
    for ref in refs:
        if not isinstance(ref, str) or not ref or ref.startswith("/") or ".." in ref.split("/"):
            raise ValueError(f"evidence ref must be a run-relative path: {ref!r}")


# --- Records -----------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult(_Serializable):
    """One automatic check. ``expected``/``observed`` are small JSON values."""

    check_id: str
    dimension: Dimension
    requirement_level: RequirementLevel
    evidence_class: EvidenceClass
    status: CheckStatus
    expected: Any = None
    observed: Any = None
    evidence_refs: tuple[str, ...] = ()
    failure_category: FailureCategory | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        for name in ("expected", "observed"):
            if not _is_small_value(getattr(self, name)):
                raise ValueError(f"{name} must be a scalar or a flat list/dict of scalars")
        _check_refs(self.evidence_refs)


@dataclass(frozen=True)
class ManualReviewItem(_Serializable):
    """A targeted human question about specific evidence.

    An answer is valid only for ``evidence_sha256``, the hash of the evidence
    the question was asked against.
    """

    check_id: str
    dimension: Dimension
    requirement_level: RequirementLevel
    question: str
    evidence_refs: tuple[str, ...]
    failure_category_if_failed: FailureCategory | None
    evidence_sha256: str
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("a review item needs a specific question")
        if not self.evidence_refs:
            raise ValueError("a review item needs specific evidence")
        _check_refs(self.evidence_refs)


@dataclass(frozen=True)
class ScenarioGrade(_Serializable):
    """Grade of one scenario. ``verdict`` is null unless the gate is GRADEABLE.

    A GRADEABLE grade always records the execution assessment and its stable
    reason; non-gradeable gates record neither (infrastructure is explained by
    the gate, never by the execution vocabulary). The assessment is derived
    from execution evidence only: human review files cannot change it in v1.
    """

    scenario_id: str
    gate: Gate
    verdict: Verdict | None = None
    root_failure: FailureCategory | None = None
    other_failures: tuple[FailureCategory, ...] = ()
    caveats: tuple[str, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    manual_reviews: tuple[ManualReviewItem, ...] = ()
    execution_assessment: ExecutionAssessment | None = None
    execution_assessment_reason: ExecutionAssessmentReason | None = None
    execution_sha256: str | None = None
    teardown_verified: bool | None = None
    schema_version: int = GRADING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        check_execution_provenance(self.execution_assessment, self.execution_assessment_reason)
        if self.gate is not Gate.GRADEABLE:
            if self.verdict is not None:
                raise ValueError(f"gate {self.gate.value} admits no verdict")
            if self.execution_assessment is not None:
                raise ValueError(f"gate {self.gate.value} admits no execution assessment")
        elif self.execution_assessment is None:
            raise ValueError("a GRADEABLE grade needs an execution assessment and reason")


@dataclass(frozen=True)
class ScenarioIndexEntry(_Serializable):
    scenario_id: str
    gate: Gate
    verdict: Verdict | None
    pending_reviews: int


@dataclass(frozen=True)
class RunGrade(_Serializable):
    """Run-level grading index. No run verdict and no aggregate score.

    - ``grader``: implementation identity, git SHA and dirty state at grading time;
    - ``contract``: evaluation contract hashes the grade was computed against.
    """

    run_id: str
    official: bool
    comparable: bool
    grader: dict[str, Any] = field(default_factory=dict)
    contract: dict[str, Any] = field(default_factory=dict)
    scenarios: tuple[ScenarioIndexEntry, ...] = ()
    counts_by_verdict: dict[str, int] = field(default_factory=dict)
    grading_complete: bool = False
    schema_version: int = GRADING_SCHEMA_VERSION
