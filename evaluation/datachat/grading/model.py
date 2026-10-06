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
    """Grade of one scenario. ``verdict`` is null unless the gate is GRADEABLE."""

    scenario_id: str
    gate: Gate
    verdict: Verdict | None = None
    root_failure: FailureCategory | None = None
    other_failures: tuple[FailureCategory, ...] = ()
    caveats: tuple[str, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    manual_reviews: tuple[ManualReviewItem, ...] = ()
    execution_sha256: str | None = None
    teardown_verified: bool | None = None
    schema_version: int = GRADING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.gate is not Gate.GRADEABLE and self.verdict is not None:
            raise ValueError(f"gate {self.gate.value} admits no verdict")


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
