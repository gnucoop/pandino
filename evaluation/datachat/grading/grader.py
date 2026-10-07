"""Run-level grading orchestration and derived result persistence.

    run directory → reader → per scenario: execution assessment → catalogue
    → technical outcome → RunGrade → grading/results/

``grade_run`` is the entry point. The whole ``RunGrade`` is built in memory
first; derived files are written only once every scenario graded, scenario
files first and ``run_grade.json`` last, each atomically. Only
``<run>/grading/results/`` is ever written: raw evidence is a read-only input.
There is no human review input: grading reads no review files.

Results are regenerable and deterministic: identical evidence and grader
version give byte-identical files (no timestamps, no randomness).

Execution assessment (``assess_execution``) is generic and scenario-agnostic.
It reads only the normalized ``/datachat`` response and runtime markers:

- FAILED — direct evidence that the DataChat outcome is unusable: every
  recorded final/response kind is ``error``; a 5xx ``/datachat`` response (the
  harness already excluded transport and infrastructure failures from
  GRADEABLE); or a parsed answer carrying no text, rows, value or chart;
- RESPONSE_CONTRACT_VIOLATION — the ``/datachat`` response breaks the
  CURRENT response contract: a non-5xx non-2xx status, or an
  unparseable/unknown body (→ FAIL — RESPONSE CONTRACT);
- INDETERMINATE — contradictory or missing markers: no recorded response,
  error kind contradicted by another marker, a request end that is not
  ``ok`` despite a 2xx answer, or adapter fallback;
- OK — otherwise: there is a usable result to grade (not: the answer is right).

Each branch yields one stable ``ExecutionAssessmentReason``; both values are
persisted in every GRADEABLE ``ScenarioGrade``.

Non-gradeable gates (NOT_RUN, NOT_SCOREABLE_INFRA, NOT_SCOREABLE) get no
assessment, no checks and a null outcome: infrastructure never becomes
``FAIL — EXECUTION``.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evaluation.datachat import contract
from evaluation.datachat.evidence import safe_child, write_json_atomic
from evaluation.datachat.grading import catalogue
from evaluation.datachat.grading.activity import observed_activity
from evaluation.datachat.grading.model import (
    GRADING_SCHEMA_VERSION,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    RunGrade,
    ScenarioGrade,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.grading.reader import (
    GradingInputError,
    ParseStatus,
    RunForGrading,
    ScenarioEvidence,
    load_run_for_grading,
)
from evaluation.datachat.grading.outcome import OutcomeBuildError, build_scenario_grade

# Bump when grading logic changes in a way that can change results.
GRADER_VERSION = 2
GRADER_IMPLEMENTATION = "evaluation.datachat.grading"

RESULTS_DIR = Path("grading") / "results"
RUN_GRADE_NAME = "run_grade.json"

_NON_GRADEABLE = (Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE)


class RunGradingError(Exception):
    """The run cannot be graded or its results cannot be persisted. Never a scenario outcome."""


class ResultPersistenceError(RunGradingError):
    """Grading succeeded in memory but ``grading/results/`` could not be written."""


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
    ok, failed = ExecutionAssessment.OK, ExecutionAssessment.FAILED
    contract_violation, unclear = ExecutionAssessment.RESPONSE_CONTRACT_VIOLATION, ExecutionAssessment.INDETERMINATE
    R = ExecutionAssessmentReason
    r, rt = ev.response, ev.runtime
    if r is None:
        return ExecutionFinding(unclear, R.MISSING_RESPONSE, "no /datachat response recorded")

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
        return ExecutionFinding(unclear, R.CONFLICTING_KINDS, f"contradictory answer kinds: {recorded}")

    status = r.http_status
    if status is not None and status >= 500:
        return ExecutionFinding(failed, R.HTTP_5XX, f"/datachat returned HTTP {status}")
    if status is None or not 200 <= status < 300:
        return ExecutionFinding(contract_violation, R.HTTP_NON_2XX, f"/datachat returned HTTP {status}")
    if r.parse_status is not ParseStatus.OK:
        return ExecutionFinding(contract_violation, R.UNPARSEABLE_RESPONSE, f"/datachat body not a CURRENT answer shape ({r.parse_status.value})")
    if rt is not None and rt.request_end_status not in (None, "ok"):
        return ExecutionFinding(unclear, R.RUNTIME_STATUS_CONFLICT, f"request end status {rt.request_end_status!r} despite HTTP {status}")
    if r.adapter_fallback_used:
        return ExecutionFinding(unclear, R.ADAPTER_FALLBACK, "adapter fallback used")
    if not _usable_answer(ev):
        return ExecutionFinding(failed, R.EMPTY_OUTCOME, f"empty {r.type} answer with no chart")
    return ExecutionFinding(ok, R.USABLE_OUTCOME, "usable answer")


# --- Grading ------------------------------------------------------------------------


def _grade_scenario(ev: ScenarioEvidence) -> ScenarioGrade:
    if ev.gate in _NON_GRADEABLE:
        return build_scenario_grade(
            execution_assessment=None, execution_assessment_reason=None, scenario_id=ev.scenario_id,
            gate=ev.gate, execution_sha256=ev.execution_sha256, teardown_verified=ev.teardown_verified)
    finding = assess_execution(ev)
    check_set = catalogue.grade_scenario_checks(ev)
    return build_scenario_grade(
        execution_assessment=finding.assessment,
        execution_assessment_reason=finding.reason,
        scenario_id=ev.scenario_id,
        gate=ev.gate,
        execution_sha256=ev.execution_sha256,
        teardown_verified=ev.teardown_verified,
        checks=check_set.checks,
        activity=observed_activity(ev),
    )


def _counts(grades: tuple[ScenarioGrade, ...]) -> tuple[dict[str, int], dict[str, int]]:
    """(every outcome and non-gradeable gate, every failure category), fixed order, zeros included."""
    outcomes = {o.value: 0 for o in TechnicalOutcome} | {g.value: 0 for g in _NON_GRADEABLE}
    failures = {f.value: 0 for f in TechnicalFailure}
    for g in grades:
        outcomes[g.outcome.value if g.gate is Gate.GRADEABLE else g.gate.value] += 1
        for f in g.failures:
            failures[f.value] += 1
    return outcomes, failures


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
    run: RunForGrading, grader: dict[str, Any]
) -> tuple[RunGrade, tuple[ScenarioGrade, ...]]:
    """Grade every scenario in the run's frozen order. Pure."""
    grades = []
    for ev in run.scenarios:
        try:
            grades.append(_grade_scenario(ev))
        except (OutcomeBuildError, catalogue.CatalogueError, GradingInputError) as exc:
            raise RunGradingError(f"{ev.scenario_id}: {exc}") from exc
    grades = tuple(grades)
    counts_by_outcome, counts_by_failure = _counts(grades)
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
        scenarios=tuple(ScenarioIndexEntry(g.scenario_id, g.gate, g.outcome, g.failures) for g in grades),
        counts_by_outcome=counts_by_outcome,
        counts_by_failure=counts_by_failure,
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
        raise ResultPersistenceError(f"cannot persist grading results: {type(exc).__name__}: {exc}") from exc
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
    except GradingInputError as exc:
        raise RunGradingError(str(exc)) from exc
    run_grade, grades = build_run_grade(run, grader_identity(repo_root))
    persist_results(run.root, run_grade, grades)
    return run_grade
