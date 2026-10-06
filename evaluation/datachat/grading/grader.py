"""Run-level grading orchestration and derived result persistence.

    run directory → reader → per scenario: execution assessment → catalogue
    → durable review answers → verdict → RunGrade → grading/results/

``grade_run`` is the entry point. The whole ``RunGrade`` is built in memory
first; derived files are written only once every scenario graded, scenario
files first and ``run_grade.json`` last, each atomically. Only
``<run>/grading/results/`` is ever written: raw evidence and the human
answers under ``<run>/grading/reviews/`` are read-only inputs.

Results are regenerable and deterministic: identical evidence, review answers
and grader version give byte-identical files (no timestamps, no randomness).

Execution assessment (``assess_execution``) is generic and scenario-agnostic.
It reads only the normalized ``/datachat`` response and runtime markers:

- FAILED — direct evidence that the DataChat outcome is unusable: every
  recorded final/response kind is ``error``; a 5xx ``/datachat`` response (the
  harness already excluded transport and infrastructure failures from
  GRADEABLE); or a parsed answer carrying no text, rows, value or chart;
- NEEDS_REVIEW — contradictory or ambiguous markers: error kind contradicted
  by another marker, adapter fallback, a non-5xx non-2xx status, an
  unparseable/unknown body, no recorded response, or a request end that is
  not ``ok`` despite a 2xx answer;
- OK — otherwise: there is a usable result to grade (not: the answer is right).

Each branch yields one stable ``ExecutionAssessmentReason``; both values are
persisted in every GRADEABLE ``ScenarioGrade``. Execution NEEDS_REVIEW is not
resolvable through ``grading/reviews/`` in v1: review answers apply only to
catalogue review items.

Non-gradeable gates (NOT_RUN, NOT_SCOREABLE_INFRA, NOT_SCOREABLE) get no
assessment, no checks and a null verdict: infrastructure never becomes
``FAIL — EXECUTION``.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evaluation.datachat import contract
from evaluation.datachat.evidence import safe_child, sha256_file, write_json_atomic
from evaluation.datachat.grading import catalogue
from evaluation.datachat.grading.model import (
    GRADING_SCHEMA_VERSION,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ReviewStatus,
    RunGrade,
    ScenarioGrade,
    ScenarioIndexEntry,
    Verdict,
)
from evaluation.datachat.grading.reader import (
    GradingInputError,
    ParseStatus,
    RunForGrading,
    ScenarioEvidence,
    evidence_digest,
    load_run_for_grading,
)
from evaluation.datachat.grading.verdict import (
    ReviewInputError,
    VerdictBuildError,
    build_scenario_grade,
    load_review_answers,
)

# Bump when grading logic changes in a way that can change results.
GRADER_VERSION = 1
GRADER_IMPLEMENTATION = "evaluation.datachat.grading"

RESULTS_DIR = Path("grading") / "results"
REVIEWS_DIR = Path("grading") / "reviews"
RUN_GRADE_NAME = "run_grade.json"

_NON_GRADEABLE = (Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE)


class RunGradingError(Exception):
    """The run cannot be graded or its results cannot be persisted. Never a scenario verdict."""


# --- Execution assessment ---------------------------------------------------------


@dataclass(frozen=True)
class ExecutionFinding:
    """Assessment, its stable reason, and a human-oriented detail (not persisted)."""

    assessment: ExecutionAssessment
    reason: ExecutionAssessmentReason
    detail: str = ""


def _usable_answer(ev: ScenarioEvidence) -> bool:
    r = ev.response
    if r.charts:
        return True
    if r.type == "str":
        return bool(r.text and r.text.strip())
    if r.type == "dataframe":
        return bool(r.table_rows)
    value = r.raw.get("response", {}).get("value") if isinstance(r.raw, dict) else None
    return bool(value)  # image (base64 string) or dict


def assess_execution(ev: ScenarioEvidence) -> ExecutionFinding:
    """Generic usability of one GRADEABLE scenario's DataChat outcome."""
    if ev.gate is not Gate.GRADEABLE:
        raise RunGradingError(f"{ev.scenario_id}: no execution assessment for gate {ev.gate.value}")
    ok, failed, review = ExecutionAssessment.OK, ExecutionAssessment.FAILED, ExecutionAssessment.NEEDS_REVIEW
    R = ExecutionAssessmentReason
    r, rt = ev.response, ev.runtime
    if r is None:
        return ExecutionFinding(review, R.MISSING_RESPONSE, "no /datachat response recorded")

    kinds = {
        "final_kind": r.final_kind,
        "runtime response_kind": rt.response_kind if rt else None,
        "agent_runs response_kind": ev.agent_run.response_kind if ev.agent_run else None,
    }
    recorded = {k: v for k, v in kinds.items() if v is not None}
    errors = sorted(k for k, v in recorded.items() if v == "error")
    if errors and len(errors) == len(recorded):
        return ExecutionFinding(failed, R.ERROR_OUTCOME, f"runtime recorded an error final answer ({', '.join(errors)})")
    if errors:
        return ExecutionFinding(review, R.CONFLICTING_KINDS, f"contradictory answer kinds: {recorded}")

    status = r.http_status
    if status is not None and status >= 500:
        return ExecutionFinding(failed, R.HTTP_5XX, f"/datachat returned HTTP {status}")
    if status is None or not 200 <= status < 300:
        return ExecutionFinding(review, R.HTTP_NON_2XX, f"/datachat returned HTTP {status}")
    if r.parse_status is not ParseStatus.OK:
        return ExecutionFinding(review, R.UNPARSEABLE_RESPONSE, f"/datachat body not a CURRENT answer shape ({r.parse_status.value})")
    if rt is not None and rt.request_end_status not in (None, "ok"):
        return ExecutionFinding(review, R.RUNTIME_STATUS_CONFLICT, f"request end status {rt.request_end_status!r} despite HTTP {status}")
    if r.adapter_fallback_used:
        return ExecutionFinding(review, R.ADAPTER_FALLBACK, "adapter fallback used")
    if not _usable_answer(ev):
        return ExecutionFinding(failed, R.EMPTY_OUTCOME, f"empty {r.type} answer with no chart")
    return ExecutionFinding(ok, R.USABLE_OUTCOME, "usable answer")


# --- Grading ------------------------------------------------------------------------


def review_path(run_dir: Path, scenario_id: str) -> Path:
    return safe_child(Path(run_dir), REVIEWS_DIR / f"{scenario_id}.json")


def _grade_scenario(run: RunForGrading, ev: ScenarioEvidence, ground_truth: dict[str, Any]) -> ScenarioGrade:
    if ev.gate in _NON_GRADEABLE:
        return build_scenario_grade(
            execution_assessment=None, execution_assessment_reason=None, scenario_id=ev.scenario_id,
            gate=ev.gate, execution_sha256=ev.execution_sha256, teardown_verified=ev.teardown_verified)
    finding = assess_execution(ev)
    check_set = catalogue.grade_scenario_checks(
        ev, ground_truth, digest=lambda refs: evidence_digest(run.root, refs))
    answers = load_review_answers(review_path(run.root, ev.scenario_id), ev.scenario_id)
    return build_scenario_grade(
        execution_assessment=finding.assessment,
        execution_assessment_reason=finding.reason,
        scenario_id=ev.scenario_id,
        gate=ev.gate,
        execution_sha256=ev.execution_sha256,
        teardown_verified=ev.teardown_verified,
        checks=check_set.checks,
        manual_reviews=check_set.manual_reviews,
        review_answers=answers,
    )


def _counts(grades: tuple[ScenarioGrade, ...]) -> dict[str, int]:
    """Every verdict and non-gradeable gate, in a fixed order, zeros included."""
    counts = {v.value: 0 for v in Verdict} | {g.value: 0 for g in _NON_GRADEABLE}
    for g in grades:
        counts[g.verdict.value if g.gate is Gate.GRADEABLE else g.gate.value] += 1
    return counts


def read_grader_identity(repo_root: Path) -> dict[str, Any]:
    """Grader provenance: implementation, version, schema and Pandino git state."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunGradingError(f"cannot read pandino git state: {type(exc).__name__}") from exc
    return {
        "implementation": GRADER_IMPLEMENTATION,
        "version": GRADER_VERSION,
        "grading_schema_version": GRADING_SCHEMA_VERSION,
        "git_sha": sha,
        "dirty": dirty,
    }


def build_run_grade(
    run: RunForGrading, ground_truth: dict[str, Any], grader: dict[str, Any]
) -> tuple[RunGrade, tuple[ScenarioGrade, ...]]:
    """Grade every scenario in the run's frozen order. Pure apart from reading review files."""
    grades = []
    for ev in run.scenarios:
        try:
            grades.append(_grade_scenario(run, ev, ground_truth))
        except (ReviewInputError, VerdictBuildError, catalogue.CatalogueError, GradingInputError) as exc:
            raise RunGradingError(f"{ev.scenario_id}: {exc}") from exc
    grades = tuple(grades)
    p = run.provenance
    run_grade = RunGrade(
        run_id=p.run_id,
        official=p.official,
        comparable=p.comparable,
        grader=dict(grader),
        contract={
            "artifacts": dict(sorted(p.contract_hashes.items())),
            "dataset_sha256": p.dataset_sha256,
            "ground_truth_sha256": p.ground_truth_sha256,
            "scenario_execution_sha256": {g.scenario_id: g.execution_sha256 for g in grades},
        },
        scenarios=tuple(
            ScenarioIndexEntry(g.scenario_id, g.gate, g.verdict,
                               sum(r.status is ReviewStatus.PENDING for r in g.manual_reviews))
            for g in grades),
        counts_by_verdict=_counts(grades),
        grading_complete=all(g.verdict is not Verdict.NEEDS_REVIEW for g in grades if g.gate is Gate.GRADEABLE),
    )
    return run_grade, grades


# --- Persistence --------------------------------------------------------------------


def persist_results(run_dir: Path, run_grade: RunGrade, grades: tuple[ScenarioGrade, ...]) -> Path:
    """Atomically (re)write ``grading/results/``; ``run_grade.json`` last. Returns the results dir."""
    results = safe_child(Path(run_dir), RESULTS_DIR)
    try:
        for g in grades:
            write_json_atomic(safe_child(results, Path("scenarios") / f"{g.scenario_id}.json"), g.to_dict())
        write_json_atomic(results / RUN_GRADE_NAME, run_grade.to_dict())
    except (OSError, ValueError, TypeError) as exc:
        raise RunGradingError(f"cannot persist grading results: {type(exc).__name__}: {exc}") from exc
    return results


def grade_run(
    run_dir: Path,
    repo_root: Path = contract.REPO_ROOT,
    *,
    grader_identity: Callable[[Path], dict[str, Any]] = read_grader_identity,
) -> RunGrade:
    """Load, grade and persist one finished run. Raises ``RunGradingError``; never partial on grading errors."""
    repo_root = Path(repo_root)
    try:
        run = load_run_for_grading(run_dir, repo_root)
        ground_truth = catalogue.load_ground_truth(repo_root)
        gt_path = contract.evaluation_dir(repo_root) / contract.GROUND_TRUTH_JSON_NAME
        if sha256_file(gt_path) != run.provenance.ground_truth_sha256:
            raise RunGradingError("ground_truth.json changed after contract verification")
    except (GradingInputError, catalogue.GroundTruthError) as exc:
        raise RunGradingError(str(exc)) from exc
    run_grade, grades = build_run_grade(run, ground_truth, grader_identity(repo_root))
    persist_results(run.root, run_grade, grades)
    return run_grade
