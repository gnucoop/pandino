"""Grading domain model of the DataChat Functional Evaluation.

Frozen technical vocabulary (outcomes, failure categories, evidence classes,
check statuses, execution assessment) and the derived result records. No I/O
and no grading logic happen here: nothing in this module decides an outcome.

Records serialize with ``to_dict()`` to plain JSON-compatible values, using
the same convention as the execution model (enums become their string
value). Top-level persisted records carry ``schema_version``.

There is deliberately no numeric score and no run-level outcome. Grading
schema 2 is technical only: no semantic dimension, verdict, caveat or
manual-review record exists.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from evaluation.datachat.model import _Serializable

GRADING_SCHEMA_VERSION = 2


# --- Frozen vocabulary -------------------------------------------------------


class TechnicalOutcome(str, Enum):
    """Technical outcome of one GRADEABLE scenario (grading design §3).

    ``FAIL`` always comes with one or more ``TechnicalFailure``; they are
    recorded together, unranked. A TECHNICAL PASS says nothing about semantic
    correctness.
    """

    TECHNICAL_PASS = "TECHNICAL PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


class TechnicalFailure(str, Enum):
    """Category of a reliably established technical failure. Declaration order is a listing order, not a rank."""

    EXECUTION = "EXECUTION"
    RESPONSE_CONTRACT = "RESPONSE CONTRACT"
    TOOL_EXECUTION = "TOOL EXECUTION"
    ARTIFACT = "ARTIFACT"


class EvidenceClass(str, Enum):
    """DIRECT: observed fact. INFERRED: best-effort reading; its negative never fails a scenario."""

    DIRECT = "DIRECT"
    INFERRED = "INFERRED"


class CheckStatus(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    INDETERMINATE = "INDETERMINATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ExecutionAssessment(str, Enum):
    """Generic usability of the DataChat outcome, derived outside the scenario catalogue.

    - OK: the outcome is usable; scenario checks decide the outcome;
    - FAILED: the outcome itself is unusable → FAIL — EXECUTION;
    - RESPONSE_CONTRACT_VIOLATION: non-2xx (non-5xx) status or unparseable body → FAIL — RESPONSE CONTRACT;
    - INDETERMINATE: contradictory or missing execution markers → INDETERMINATE unless a FAIL is established.
    """

    OK = "OK"
    FAILED = "FAILED"
    RESPONSE_CONTRACT_VIOLATION = "RESPONSE_CONTRACT_VIOLATION"
    INDETERMINATE = "INDETERMINATE"


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
    ExecutionAssessment.RESPONSE_CONTRACT_VIOLATION: frozenset({_R.HTTP_NON_2XX, _R.UNPARSEABLE_RESPONSE}),
    ExecutionAssessment.INDETERMINATE: frozenset({
        _R.MISSING_RESPONSE, _R.CONFLICTING_KINDS, _R.RUNTIME_STATUS_CONFLICT, _R.ADAPTER_FALLBACK,
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
    """Whether a scenario may be technically graded. Not an outcome."""

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
    """One mandatory technical check. ``expected``/``observed`` are small JSON values.

    ``failure`` is the category a reliable NOT_MET establishes; it is set only on NOT_MET.
    """

    check_id: str
    evidence_class: EvidenceClass
    status: CheckStatus
    expected: Any = None
    observed: Any = None
    evidence_refs: tuple[str, ...] = ()
    failure: TechnicalFailure | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        for name in ("expected", "observed"):
            if not _is_small_value(getattr(self, name)):
                raise ValueError(f"{name} must be a scalar or a flat list/dict of scalars")
        _check_refs(self.evidence_refs)
        if (self.failure is not None) != (self.status is CheckStatus.NOT_MET):
            raise ValueError(f"{self.check_id}: a failure category is set exactly on NOT_MET")


class ActivitySource(str, Enum):
    """Where an observed activity comes from (grading design §5.1). Declaration order is the listing order.

    - RUNTIME: a runtime ``tool_call_*`` event — observed invocation with its logged outcome;
    - GENERATED_CODE: a call written in generated agent code — inferred, not proof of execution;
    - USAGE: Usage rows beyond the primary one — provider-backed work, tool not identified.
    """

    RUNTIME = "RUNTIME"
    GENERATED_CODE = "GENERATED_CODE"
    USAGE = "USAGE"


@dataclass(frozen=True)
class ObservedActivity(_Serializable):
    """Aggregated, source-labelled activity of one scenario. Descriptive only: never graded.

    ``outcome`` is the logged ``tool_call_<outcome>`` suffix and exists only for RUNTIME.
    """

    name: str
    source: ActivitySource
    count: int
    outcome: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"activity name must be a non-empty string: {self.name!r}")
        if not isinstance(self.source, ActivitySource):
            raise ValueError(f"activity source must be an ActivitySource: {self.source!r}")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError(f"{self.name}: activity count must be a positive integer")
        if (self.outcome is not None) != (self.source is ActivitySource.RUNTIME):
            raise ValueError(f"{self.name}: an outcome is recorded exactly for RUNTIME activity")


def _check_outcome(outcome: TechnicalOutcome | None, failures: tuple[TechnicalFailure, ...]) -> None:
    if len(set(failures)) != len(failures):
        raise ValueError(f"duplicate technical failures: {failures}")
    if (outcome is TechnicalOutcome.FAIL) != bool(failures):
        raise ValueError(f"outcome {outcome} inconsistent with failures {failures}")


@dataclass(frozen=True)
class ScenarioGrade(_Serializable):
    """Grade of one scenario. ``outcome`` is null unless the gate is GRADEABLE.

    A GRADEABLE grade always records the technical outcome, the execution
    assessment and its stable reason; non-gradeable gates record none of them
    (infrastructure is explained by the gate, never by the execution
    vocabulary). ``failures`` lists every established failure, unranked.
    ``activity`` is the observed-activity summary (descriptive, never an
    input to the outcome); non-gradeable gates carry none.
    """

    scenario_id: str
    gate: Gate
    outcome: TechnicalOutcome | None = None
    failures: tuple[TechnicalFailure, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    execution_assessment: ExecutionAssessment | None = None
    execution_assessment_reason: ExecutionAssessmentReason | None = None
    execution_sha256: str | None = None
    teardown_verified: bool | None = None
    activity: tuple[ObservedActivity, ...] = ()
    schema_version: int = GRADING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        check_execution_provenance(self.execution_assessment, self.execution_assessment_reason)
        _check_outcome(self.outcome, self.failures)
        keys = [(a.source, a.name, a.outcome) for a in self.activity]
        if len(set(keys)) != len(keys):
            raise ValueError("activity records must be aggregated: duplicate (source, name, outcome)")
        if self.gate is not Gate.GRADEABLE:
            if self.outcome is not None or self.checks or self.activity:
                raise ValueError(f"gate {self.gate.value} admits no outcome, no checks and no activity")
            if self.execution_assessment is not None:
                raise ValueError(f"gate {self.gate.value} admits no execution assessment")
        elif self.outcome is None or self.execution_assessment is None:
            raise ValueError("a GRADEABLE grade needs an outcome, an execution assessment and reason")

    @property
    def label(self) -> str | None:
        return outcome_label(self.outcome, self.failures)


def outcome_label(outcome: TechnicalOutcome | None, failures: tuple[TechnicalFailure, ...]) -> str | None:
    """``TECHNICAL PASS``, ``INDETERMINATE`` or ``FAIL — A, B`` (failures in recorded order)."""
    if outcome is TechnicalOutcome.FAIL:
        return f"FAIL — {', '.join(f.value for f in failures)}"
    return outcome.value if outcome is not None else None


@dataclass(frozen=True)
class ScenarioIndexEntry(_Serializable):
    scenario_id: str
    gate: Gate
    outcome: TechnicalOutcome | None
    failures: tuple[TechnicalFailure, ...] = ()

    def __post_init__(self) -> None:
        _check_outcome(self.outcome, self.failures)

    @property
    def label(self) -> str | None:
        return outcome_label(self.outcome, self.failures)


@dataclass(frozen=True)
class RunGrade(_Serializable):
    """Run-level grading index. No run outcome and no aggregate score.

    - ``grader``: implementation identity, git SHA and dirty state at grading time;
    - ``contract``: evaluation contract hashes the grade was computed against;
    - ``counts_by_outcome``: every technical outcome and non-gradeable gate, zeros included;
    - ``counts_by_failure``: scenarios establishing each failure category (a FAIL may count in several).
    """

    run_id: str
    official: bool
    comparable: bool
    grader: dict[str, Any] = field(default_factory=dict)
    contract: dict[str, Any] = field(default_factory=dict)
    scenarios: tuple[ScenarioIndexEntry, ...] = ()
    counts_by_outcome: dict[str, int] = field(default_factory=dict)
    counts_by_failure: dict[str, int] = field(default_factory=dict)
    schema_version: int = GRADING_SCHEMA_VERSION
