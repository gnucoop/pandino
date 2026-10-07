"""Scenario technical outcome construction (grading design §3–§5).

    execution assessment + ScenarioCheckSet → ScenarioGrade

Pure. Nothing here inspects raw evidence or knows scenario policy: it
operates generically on ``CheckResult`` as the catalogue emitted them.

Outcome rules (one outcome per GRADEABLE scenario):

1. failures — every reliably established failure is recorded, unranked:
   execution assessment FAILED (EXECUTION), RESPONSE_CONTRACT_VIOLATION
   (RESPONSE CONTRACT) and each DIRECT check NOT_MET (its own category)
   → FAIL;
2. ambiguity — execution assessment INDETERMINATE, any check INDETERMINATE
   or any INFERRED check NOT_MET → INDETERMINATE. An INFERRED negative never
   fails by itself;
3. otherwise TECHNICAL PASS (every check MET or NOT_APPLICABLE).

The execution assessment and its stable reason are inputs: they are derived
from normalized evidence elsewhere, never here, and are recorded verbatim.
"""

from typing import Sequence

from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ScenarioGrade,
    TechnicalFailure,
    TechnicalOutcome,
    check_execution_provenance,
)

_EXECUTION_FAILURE = {
    ExecutionAssessment.FAILED: TechnicalFailure.EXECUTION,
    ExecutionAssessment.RESPONSE_CONTRACT_VIOLATION: TechnicalFailure.RESPONSE_CONTRACT,
}


class OutcomeBuildError(Exception):
    """Catalogue output violates an invariant the outcome builder relies on."""


def _check_invariants(checks: Sequence[CheckResult]) -> None:
    ids = [x.check_id for x in checks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise OutcomeBuildError(f"check ids appear more than once: {duplicates}")


def build_scenario_grade(
    *,
    execution_assessment: ExecutionAssessment | None,
    execution_assessment_reason: ExecutionAssessmentReason | None,
    scenario_id: str,
    gate: Gate,
    execution_sha256: str | None,
    teardown_verified: bool | None,
    checks: Sequence[CheckResult] = (),
) -> ScenarioGrade:
    """One ``ScenarioGrade``. Non-gradeable gates get no outcome and must carry no checks.

    ``execution_assessment`` and ``execution_assessment_reason`` are mandatory and never
    defaulted: a valid pair for GRADEABLE, both None for any other gate.
    """
    try:
        check_execution_provenance(execution_assessment, execution_assessment_reason)
    except ValueError as exc:
        raise OutcomeBuildError(f"{scenario_id}: {exc}") from exc
    if gate is not Gate.GRADEABLE:
        if execution_assessment is not None:
            raise OutcomeBuildError(f"{scenario_id}: gate {gate.value} admits no execution assessment")
        if checks:
            raise OutcomeBuildError(f"{scenario_id}: gate {gate.value} cannot carry checks")
        return ScenarioGrade(scenario_id=scenario_id, gate=gate, execution_sha256=execution_sha256,
                             teardown_verified=teardown_verified)

    if execution_assessment is None:
        raise OutcomeBuildError(f"{scenario_id}: a GRADEABLE scenario needs an execution assessment")
    checks = tuple(checks)
    _check_invariants(checks)

    established = {c.failure for c in checks
                   if c.status is CheckStatus.NOT_MET and c.evidence_class is EvidenceClass.DIRECT}
    if execution_assessment in _EXECUTION_FAILURE:
        established.add(_EXECUTION_FAILURE[execution_assessment])
    failures = tuple(f for f in TechnicalFailure if f in established)
    if failures:
        outcome = TechnicalOutcome.FAIL
    elif (execution_assessment is ExecutionAssessment.INDETERMINATE
          or any(c.status is CheckStatus.INDETERMINATE
                 or (c.status is CheckStatus.NOT_MET and c.evidence_class is EvidenceClass.INFERRED)
                 for c in checks)):
        outcome = TechnicalOutcome.INDETERMINATE
    else:
        outcome = TechnicalOutcome.TECHNICAL_PASS
    return ScenarioGrade(
        scenario_id=scenario_id,
        gate=gate,
        outcome=outcome,
        failures=failures,
        checks=checks,
        execution_assessment=execution_assessment,
        execution_assessment_reason=execution_assessment_reason,
        execution_sha256=execution_sha256,
        teardown_verified=teardown_verified,
    )
