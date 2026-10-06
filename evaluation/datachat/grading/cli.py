"""Operator command for grading an existing DataChat Functional Evaluation run.

``python -m evaluation.datachat.grading RUN_ID | --run-dir PATH``

A thin layer over ``grader.grade_run``: it resolves one explicit run, grades
it offline (reading immutable evidence and durable reviews from
``grading/reviews/``), regenerates ``grading/results/`` and prints a short
summary. No Maui, no HTTP, no DB, no provider, no API key. It is a separate
entry point because the execution CLI treats every positional as a scenario ID.

Exit codes: 0 grading complete (scenario FAIL verdicts included — they are
results, not command failures), 1 grading written but incomplete (NEEDS REVIEW
remains), 2 usage/input error (missing run, contract mismatch, malformed
review, corrupted package), 3 unexpected error, 130 interrupted.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Callable, Sequence, TextIO

from evaluation.datachat import contract
from evaluation.datachat.evidence import _check_run_id, safe_child
from evaluation.datachat.grading import grader
from evaluation.datachat.grading.model import Gate, RunGrade, Verdict
from evaluation.datachat.grading.reader import ContractMismatchError

EXIT_COMPLETE = 0
EXIT_INCOMPLETE = 1
EXIT_USAGE = 2
EXIT_UNEXPECTED = 3
EXIT_INTERRUPTED = 130

RUNS_DIR_RELATIVE = Path("evaluation") / "datachat" / "runs"


class _UsageError(Exception):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.datachat.grading",
        description="Grade an existing DataChat Functional Evaluation run offline. "
                    "Does not execute DataChat. Reads human reviews from <run>/grading/reviews/ "
                    "and (re)writes results under <run>/grading/results/.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("run_id", nargs="?", metavar="RUN_ID", help=f"run directory name under {RUNS_DIR_RELATIVE}/")
    target.add_argument("--run-dir", type=Path, metavar="PATH", help="explicit run directory")
    return parser


def resolve_run_dir(run_id: str | None, run_dir: Path | None, repo_root: Path) -> Path:
    """The one run the operator named. No latest-run or partial-ID guessing. Raises ``_UsageError``."""
    if run_dir is not None:
        path = Path(run_dir)
        if not path.is_dir():
            raise _UsageError(f"run directory not found: {path}")
        if not (path / "run_manifest.json").is_file():
            raise _UsageError(f"not a run directory (no run_manifest.json): {path}")
        return path
    try:
        _check_run_id(run_id)
        path = safe_child(repo_root / RUNS_DIR_RELATIVE, run_id)
    except ValueError:
        raise _UsageError(f"invalid run ID: {run_id!r}") from None
    if not path.is_dir():
        raise _UsageError(f"run not found: {RUNS_DIR_RELATIVE / run_id}")
    return path


# --- Rendering -------------------------------------------------------------------


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _display(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _execution_reason(results_dir: Path, scenario_id: str) -> str | None:
    """Stable execution reason from the persisted scenario grade, when it matters for display."""
    try:
        data = json.loads((results_dir / "scenarios" / f"{scenario_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("execution_assessment") in ("FAILED", "NEEDS_REVIEW"):
        return data.get("execution_assessment_reason")
    return None


def render(run_grade: RunGrade, run_dir: Path, repo_root: Path, out: TextIO) -> None:
    results = run_dir / grader.RESULTS_DIR
    print("DataChat FE grading completed", file=out)
    print("", file=out)
    print(f"Run:         {run_grade.run_id}", file=out)
    print(f"Official:    {_yes(run_grade.official)}", file=out)
    print(f"Comparable:  {_yes(run_grade.comparable)}", file=out)
    print("", file=out)
    print("Scenarios:", file=out)
    execution_review = 0
    for entry in run_grade.scenarios:
        if entry.gate is not Gate.GRADEABLE:
            print(f"  {entry.scenario_id}  {entry.gate.value}", file=out)
            continue
        row = f"  {entry.scenario_id}  {entry.verdict.value}"
        reason = _execution_reason(results, entry.scenario_id)
        if reason:
            row += f"  [execution: {reason}]"
            execution_review += entry.verdict is Verdict.NEEDS_REVIEW
        if entry.pending_reviews:
            row += f"  ({entry.pending_reviews} pending review{'s' if entry.pending_reviews != 1 else ''})"
        print(row, file=out)
    pending = sum(e.pending_reviews for e in run_grade.scenarios)
    print("", file=out)
    print(f"Grading complete: {_yes(run_grade.grading_complete)}", file=out)
    if not run_grade.grading_complete:
        if pending:
            print(f"Pending manual reviews: {pending}", file=out)
            print(f"Review files: {_display(run_dir / grader.REVIEWS_DIR, repo_root)}/<FE-NN>.json "
                  "(edit, then re-run this command)", file=out)
        if execution_review:
            print(f"Execution needs review: {execution_review} scenario(s) "
                  "(not resolvable through review files in v1)", file=out)
    print(f"Results: {_display(results, repo_root)}", file=out)


# --- Entry point -----------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    grade_fn: Callable[..., RunGrade] = grader.grade_run,
    repo_root: Path | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                args = _parser().parse_args(argv)
            except SystemExit as exc:  # --help (0) or argparse usage error (2)
                return exc.code if isinstance(exc.code, int) else EXIT_USAGE

        try:
            root = contract.resolve_repo_root(repo_root)
            run_dir = resolve_run_dir(args.run_id, args.run_dir, root)
        except (_UsageError, ValueError) as exc:
            print(f"error: {exc}", file=err)
            return EXIT_USAGE

        try:
            run_grade = grade_fn(run_dir, root)
        except grader.ResultPersistenceError as exc:
            print(f"Cannot write grading results: {type(exc.__cause__ or exc).__name__}", file=err)
            print(f"Results directory: {_display(run_dir / grader.RESULTS_DIR, root)}", file=err)
            return EXIT_UNEXPECTED
        except grader.RunGradingError as exc:
            if isinstance(exc.__cause__, ContractMismatchError):
                print("Cannot grade run: evaluation contract does not match the CURRENT FE contract.", file=err)
                print(f"  ({exc})", file=err)
                print("Re-run the Functional Evaluation under the current contract.", file=err)
            else:
                print(f"Cannot grade run: {exc}", file=err)
            return EXIT_USAGE
        except Exception as exc:  # message may carry anything; only the type is shown
            print(f"Unexpected grading error: {type(exc).__name__}", file=err)
            print(f"Run directory: {_display(run_dir, root)} (raw evidence and reviews are never modified)", file=err)
            return EXIT_UNEXPECTED
        render(run_grade, run_dir, root, out)
        return EXIT_COMPLETE if run_grade.grading_complete else EXIT_INCOMPLETE
    except KeyboardInterrupt:
        print("Interrupted.", file=err)
        return EXIT_INTERRUPTED
