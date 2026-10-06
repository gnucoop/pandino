"""Manual-review merge and scenario verdict construction.

    ScenarioCheckSet + durable human answers → ScenarioGrade

Pure apart from ``load_review_answers``, which reads one explicit review file
(``<run>/grading/reviews/FE-NN.json``) and never writes. Nothing here
inspects raw evidence or knows scenario policy: it operates generically on
``CheckResult``/``ManualReviewItem`` as the catalogue emitted them.

Review file shape (human-authored)::

    {"scenario_id": "FE-05",
     "reviews": {"fe05.no_causal_claim": {"status": "SATISFIED", "reviewer": "...",
                                          "note": "...", "evidence_sha256": "..."}}}

An answer applies only when its ``evidence_sha256`` equals the item's; a
stale answer is ignored (the item stays PENDING) and reported. Unknown or
duplicate check IDs are rejected: a human decision never floats unattached.

Verdict order (one verdict per GRADEABLE scenario):

1. failures — execution assessment FAILED (EXECUTION), MUST checks NOT_MET
   with non-INFERRED evidence, and MUST reviews FAILED → root failure by
   reasoning-chain order; pending reviews do not override it;
2. unresolved — execution assessment NEEDS_REVIEW, MUST INDETERMINATE, any
   INFERRED MUST NOT_MET, or any PENDING review (MUST or SHOULD) → NEEDS REVIEW.
   An INFERRED negative never fails by itself and is never "confirmed" by
   another check: a DIRECT failure establishes only its own category;
3. caveats — SHOULD checks NOT_MET (any evidence class) or SHOULD reviews
   FAILED → PASS WITH CAVEAT;
4. otherwise PASS.

NOT_APPLICABLE (check or review), SHOULD INDETERMINATE and MAY never affect
the verdict. The execution assessment is an input: it is derived from
normalized evidence elsewhere, never here.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    FailureCategory,
    Gate,
    ManualReviewItem,
    RequirementLevel,
    ReviewStatus,
    ScenarioGrade,
    Verdict,
)

MUST, SHOULD = RequirementLevel.MUST, RequirementLevel.SHOULD

# Reasoning-chain order used to select the root failure.
FAILURE_ORDER: tuple[FailureCategory, ...] = (
    FailureCategory.EXECUTION,
    FailureCategory.UNDERSTANDING,
    FailureCategory.TOOL_SELECTION,
    FailureCategory.CHAINING,
    FailureCategory.ANALYTICAL,
    FailureCategory.HONESTY,
    FailureCategory.INTERPRETATION,
    FailureCategory.BACKEND_CONTRACT,
)

_ANSWER_STATUSES = (ReviewStatus.SATISFIED, ReviewStatus.FAILED, ReviewStatus.NOT_APPLICABLE)
_ANSWER_KEYS = {"status", "reviewer", "note", "evidence_sha256"}


class ReviewInputError(Exception):
    """A human review file or answer set is malformed, ambiguous or unattached."""


class VerdictBuildError(Exception):
    """Catalogue output violates an invariant the verdict builder relies on."""


# --- Review answers -----------------------------------------------------------------


@dataclass(frozen=True)
class ReviewAnswer:
    """One durable human decision, valid only for ``evidence_sha256``."""

    check_id: str
    status: ReviewStatus
    reviewer: str
    evidence_sha256: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id:
            raise ReviewInputError("review answer needs a check_id")
        if self.status not in _ANSWER_STATUSES:
            raise ReviewInputError(f"{self.check_id}: status must be one of "
                                   f"{[s.value for s in _ANSWER_STATUSES]}, got {self.status!r}")
        if not isinstance(self.reviewer, str) or not self.reviewer.strip():
            raise ReviewInputError(f"{self.check_id}: reviewer required")
        if not isinstance(self.evidence_sha256, str) or len(self.evidence_sha256) != 64:
            raise ReviewInputError(f"{self.check_id}: evidence_sha256 must be a SHA-256 hex digest")
        if self.note is not None and not isinstance(self.note, str):
            raise ReviewInputError(f"{self.check_id}: note must be text")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ReviewInputError(f"duplicate keys in review file: {duplicates}")
    return dict(pairs)


def load_review_answers(path: Path, scenario_id: str) -> tuple[ReviewAnswer, ...]:
    """Strictly parse one review file. A missing file means no answers (all items stay PENDING)."""
    path = Path(path)
    if not path.exists():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, ReviewInputError):
            raise
        raise ReviewInputError(f"review file unreadable: {path.name}: {type(exc).__name__}") from exc
    if not isinstance(data, dict) or set(data) != {"scenario_id", "reviews"}:
        raise ReviewInputError(f"{path.name}: expected exactly the keys 'scenario_id' and 'reviews'")
    if data["scenario_id"] != scenario_id:
        raise ReviewInputError(f"{path.name}: scenario_id {data['scenario_id']!r} is not {scenario_id!r}")
    reviews = data["reviews"]
    if not isinstance(reviews, dict):
        raise ReviewInputError(f"{path.name}: 'reviews' must be an object keyed by check_id")
    answers = []
    for check_id, entry in reviews.items():
        if not isinstance(entry, dict) or not set(entry) <= _ANSWER_KEYS or not {
                "status", "reviewer", "evidence_sha256"} <= set(entry):
            raise ReviewInputError(f"{path.name}: {check_id}: expected keys {sorted(_ANSWER_KEYS)}")
        try:
            status = ReviewStatus(entry["status"])
        except ValueError as exc:
            raise ReviewInputError(f"{path.name}: {check_id}: unknown status {entry['status']!r}") from exc
        answers.append(ReviewAnswer(check_id, status, entry["reviewer"], entry["evidence_sha256"],
                                    entry.get("note")))
    return tuple(answers)


@dataclass(frozen=True)
class ReviewMerge:
    """Catalogue review items with current human answers applied; ``stale`` lists ignored answers."""

    items: tuple[ManualReviewItem, ...]
    stale: tuple[str, ...] = ()


def merge_reviews(items: Sequence[ManualReviewItem], answers: Iterable[ReviewAnswer] = ()) -> ReviewMerge:
    """Apply answers whose evidence hash matches. Inputs are never mutated."""
    by_id: dict[str, ReviewAnswer] = {}
    for answer in answers:
        if answer.check_id in by_id:
            raise ReviewInputError(f"duplicate review answer: {answer.check_id}")
        by_id[answer.check_id] = answer
    known = {item.check_id for item in items}
    unknown = sorted(set(by_id) - known)
    if unknown:
        raise ReviewInputError(f"review answers for unknown check ids: {unknown}")
    merged, stale = [], []
    for item in items:
        answer = by_id.get(item.check_id)
        if answer is None:
            merged.append(item)
        elif answer.evidence_sha256 != item.evidence_sha256:
            stale.append(item.check_id)
            merged.append(replace(item, status=ReviewStatus.PENDING, reviewer=None, note=None))
        else:
            merged.append(replace(item, status=answer.status, reviewer=answer.reviewer, note=answer.note))
    return ReviewMerge(tuple(merged), tuple(stale))


# --- Verdict --------------------------------------------------------------------------


def _check_invariants(checks: Sequence[CheckResult], reviews: Sequence[ManualReviewItem]) -> None:
    ids = [x.check_id for x in (*checks, *reviews)]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise VerdictBuildError(f"check ids appear more than once: {duplicates}")
    for check in checks:
        if check.requirement_level is MUST and check.status is CheckStatus.NOT_MET and check.failure_category is None:
            raise VerdictBuildError(f"{check.check_id}: MUST NOT_MET without a failure category")
    for item in reviews:
        if item.requirement_level is MUST and item.failure_category_if_failed is None:
            raise VerdictBuildError(f"{item.check_id}: MUST review without failure_category_if_failed")


def _failures(checks: Sequence[CheckResult], reviews: Sequence[ManualReviewItem]) -> tuple[set, list[str]]:
    """(established failure categories, unresolved INFERRED MUST negatives)."""
    must_not_met = [c for c in checks if c.requirement_level is MUST and c.status is CheckStatus.NOT_MET]
    failures = {c.failure_category for c in must_not_met if c.evidence_class is not EvidenceClass.INFERRED}
    inferred = [c.check_id for c in must_not_met if c.evidence_class is EvidenceClass.INFERRED]
    failures |= {r.failure_category_if_failed for r in reviews
                 if r.requirement_level is MUST and r.status is ReviewStatus.FAILED}
    return failures, inferred


def build_scenario_grade(
    *,
    execution_assessment: ExecutionAssessment,
    scenario_id: str,
    gate: Gate,
    execution_sha256: str | None,
    teardown_verified: bool | None,
    checks: Sequence[CheckResult] = (),
    manual_reviews: Sequence[ManualReviewItem] = (),
    review_answers: Iterable[ReviewAnswer] = (),
) -> ScenarioGrade:
    """One ``ScenarioGrade``. Non-gradeable gates get no verdict and must carry no checks.

    ``execution_assessment`` is mandatory: OK asserts that execution evidence was examined
    and found usable, so it is never assumed.
    """
    if not isinstance(execution_assessment, ExecutionAssessment):
        raise VerdictBuildError(f"{scenario_id}: invalid execution assessment {execution_assessment!r}")
    if gate is not Gate.GRADEABLE:
        if checks or manual_reviews or tuple(review_answers):
            raise VerdictBuildError(f"{scenario_id}: gate {gate.value} cannot carry checks or reviews")
        return ScenarioGrade(scenario_id=scenario_id, gate=gate, execution_sha256=execution_sha256,
                             teardown_verified=teardown_verified)

    checks = tuple(checks)
    reviews = merge_reviews(manual_reviews, review_answers).items
    _check_invariants(checks, reviews)

    caveats = tuple(sorted(
        [c.check_id for c in checks if c.requirement_level is SHOULD and c.status is CheckStatus.NOT_MET]
        + [r.check_id for r in reviews if r.requirement_level is SHOULD and r.status is ReviewStatus.FAILED]))
    failures, unresolved_inferred = _failures(checks, reviews)
    if execution_assessment is ExecutionAssessment.FAILED:
        failures.add(FailureCategory.EXECUTION)
    root, others = None, ()
    if failures:
        ordered = [f for f in FAILURE_ORDER if f in failures]
        root, others = ordered[0], tuple(ordered[1:])
        verdict = root.verdict
    elif (execution_assessment is ExecutionAssessment.NEEDS_REVIEW
          or unresolved_inferred
          or any(c.requirement_level is MUST and c.status is CheckStatus.INDETERMINATE for c in checks)
          or any(r.status is ReviewStatus.PENDING for r in reviews)):
        verdict = Verdict.NEEDS_REVIEW
    elif caveats:
        verdict = Verdict.PASS_WITH_CAVEAT
    else:
        verdict = Verdict.PASS
    return ScenarioGrade(
        scenario_id=scenario_id,
        gate=gate,
        verdict=verdict,
        root_failure=root,
        other_failures=others,
        caveats=caveats,
        checks=checks,
        manual_reviews=reviews,
        execution_sha256=execution_sha256,
        teardown_verified=teardown_verified,
    )
