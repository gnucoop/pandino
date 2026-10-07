"""Tests for evaluation.datachat.grading.cli, the offline grading operator command.

Parser, rendering and exit codes use an injected fake ``grade_run``; one
integration test grades a synthetic CURRENT-contract package end to end.
No Maui, no DB, no provider, no API key.
"""

import ast
import functools
import io
import json
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.grading import cli
from evaluation.datachat.grading import grader as gr
from evaluation.datachat.grading.model import (
    Gate,
    RunGrade,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.grading.reader import ContractMismatchError, GradingInputError
from tests.test_fe_grading_grader import FE01, FE05, IDENTITY, repo_root  # noqa: F401  (fixture)
from tests.test_fe_grading_reader import RUN_ID, build_run

T, F = TechnicalOutcome, TechnicalFailure


@pytest.fixture
def repo(tmp_path):
    """A minimal repository root with one empty synthetic run directory."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "main.py").write_text("")
    run = root / cli.RUNS_DIR_RELATIVE / RUN_ID
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text("{}")
    return root


def entry(sid, outcome=None, *failures, gate=Gate.GRADEABLE):
    return ScenarioIndexEntry(sid, gate, outcome, failures)


def run_grade(*entries):
    return RunGrade(run_id=RUN_ID, official=False, comparable=False, scenarios=entries)


def invoke(argv, repo_root, grade_fn=None, result=None):
    out, err = io.StringIO(), io.StringIO()
    calls = []

    def fake(run_dir, root):
        calls.append((run_dir, root))
        return result or run_grade(entry("FE-01", T.TECHNICAL_PASS))

    code = cli.main(argv, grade_fn=grade_fn or fake, repo_root=repo_root, out=out, err=err)
    return code, out.getvalue(), err.getvalue(), calls


def write_scenario_result(repo_root, sid, assessment, reason):
    path = repo_root / cli.RUNS_DIR_RELATIVE / RUN_ID / "grading" / "results" / "scenarios" / f"{sid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"execution_assessment": assessment, "execution_assessment_reason": reason}))


# --- Parser and resolution ---------------------------------------------------------


def test_help(repo):
    code, out, _, calls = invoke(["--help"], repo)
    out = " ".join(out.split())
    assert code == 0 and "offline" in out and "grading/results/" in out and "review" not in out
    assert "Does not execute DataChat" in out and not calls


@pytest.mark.parametrize("argv", [[], ["--bogus"], [RUN_ID, "--run-dir", "x"], [RUN_ID, "other"]])
def test_usage_errors(repo, argv):
    code, _, _, calls = invoke(argv, repo)
    assert code == 2 and not calls


def test_run_id_resolves_under_runs_dir(repo):
    code, _, _, calls = invoke([RUN_ID], repo)
    assert code == 0 and calls == [(repo / cli.RUNS_DIR_RELATIVE / RUN_ID, repo)]


def test_explicit_run_dir(repo):
    path = repo / cli.RUNS_DIR_RELATIVE / RUN_ID
    code, _, _, calls = invoke(["--run-dir", str(path)], repo)
    assert code == 0 and calls[0][0] == path


@pytest.mark.parametrize("argv,message", [
    (["20990101T000000Z_missing"], "run not found"),
    (["../escape"], "invalid run ID"),
    ([RUN_ID[:10]], "run not found"),  # partial IDs are never guessed
    (["--run-dir", "/nonexistent/run"], "run directory not found"),
])
def test_bad_run_targets(repo, argv, message):
    code, _, err, calls = invoke(argv, repo)
    assert code == 2 and message in err and not calls


def test_run_dir_without_manifest(repo, tmp_path):
    code, _, err, _ = invoke(["--run-dir", str(tmp_path)], repo)
    assert code == 2 and "no run_manifest.json" in err


def test_no_latest_run_auto_selection(repo):
    code, _, _, calls = invoke([], repo)
    assert code == 2 and not calls


def test_bad_repo_root(tmp_path):
    code, _, err, _ = invoke([RUN_ID], tmp_path)
    assert code == 2 and "not a Maui repository root" in err


# --- Summary and exit codes ---------------------------------------------------------


def test_graded_run_exit_0(repo):
    code, out, _, _ = invoke([RUN_ID], repo, result=run_grade(entry("FE-01", T.TECHNICAL_PASS)))
    assert code == cli.EXIT_GRADED == 0
    assert f"Run:         {RUN_ID}" in out and "Official:    no" in out and "Comparable:  no" in out
    assert "FE-01  TECHNICAL PASS\n" in out
    assert "Grading complete" not in out and "review" not in out.lower()
    assert f"Results: {cli.RUNS_DIR_RELATIVE.as_posix()}/{RUN_ID}/grading/results" in out


def test_fail_and_indeterminate_are_results_not_command_failures(repo):
    result = run_grade(entry("FE-04", T.FAIL, F.TOOL_EXECUTION, F.ARTIFACT), entry("FE-05", T.INDETERMINATE))
    code, out, _, _ = invoke([RUN_ID], repo, result=result)
    assert code == 0
    assert "FE-04  FAIL — TOOL EXECUTION, ARTIFACT\n" in out and "FE-05  INDETERMINATE\n" in out
    assert not hasattr(cli, "EXIT_INCOMPLETE")


def test_non_gradeable_shown_as_gates(repo):
    result = run_grade(entry("FE-01", T.TECHNICAL_PASS), entry("FE-02", gate=Gate.NOT_RUN),
                       entry("FE-03", gate=Gate.NOT_SCOREABLE_INFRA), entry("FE-04", gate=Gate.NOT_SCOREABLE))
    code, out, _, _ = invoke([RUN_ID], repo, result=result)
    assert code == 0
    assert "FE-02  NOT_RUN\n" in out and "FE-03  NOT_SCOREABLE_INFRA\n" in out and "FE-04  NOT_SCOREABLE\n" in out
    assert "FAIL" not in out


def test_execution_reasons_shown(repo):
    write_scenario_result(repo, "FE-01", "FAILED", "ERROR_OUTCOME")
    write_scenario_result(repo, "FE-02", "INDETERMINATE", "ADAPTER_FALLBACK")
    write_scenario_result(repo, "FE-03", "OK", "USABLE_OUTCOME")
    write_scenario_result(repo, "FE-04", "RESPONSE_CONTRACT_VIOLATION", "HTTP_NON_2XX")
    result = run_grade(entry("FE-01", T.FAIL, F.EXECUTION), entry("FE-02", T.INDETERMINATE),
                       entry("FE-03", T.TECHNICAL_PASS), entry("FE-04", T.FAIL, F.RESPONSE_CONTRACT))
    code, out, _, _ = invoke([RUN_ID], repo, result=result)
    assert code == 0
    assert "FE-01  FAIL — EXECUTION  [execution: ERROR_OUTCOME]" in out
    assert "FE-02  INDETERMINATE  [execution: ADAPTER_FALLBACK]" in out
    assert "FE-03  TECHNICAL PASS\n" in out and "USABLE_OUTCOME" not in out
    assert "FE-04  FAIL — RESPONSE CONTRACT  [execution: HTTP_NON_2XX]" in out


# --- Errors ----------------------------------------------------------------------------


def _raising(exc):
    def fn(run_dir, root):
        raise exc
    return fn


def _wrapped(cause):
    try:
        raise gr.RunGradingError(str(cause)) from cause
    except gr.RunGradingError as exc:
        return exc


@pytest.mark.parametrize("exc,expected", [
    (_wrapped(ContractMismatchError("run manifest does not record the current contract hash: x")),
     "evaluation contract does not match the CURRENT FE contract"),
    (_wrapped(GradingInputError("FE-01 http body: sha256 mismatch")), "sha256 mismatch"),
])
def test_expected_errors_exit_2(repo, exc, expected):
    code, out, err, _ = invoke([RUN_ID], repo, grade_fn=_raising(exc))
    assert code == 2 and expected in err and "Traceback" not in err and out == ""


def test_contract_mismatch_says_rerun(repo):
    exc = _wrapped(ContractMismatchError("run was executed against a non-matching evaluation contract"))
    _, _, err, _ = invoke([RUN_ID], repo, grade_fn=_raising(exc))
    assert "Re-run the Functional Evaluation under the current contract." in err


def test_unexpected_error_exit_3_type_only(repo):
    code, _, err, _ = invoke([RUN_ID], repo, grade_fn=_raising(RuntimeError("secret-token-123")))
    assert code == 3 and "RuntimeError" in err and "secret-token-123" not in err and "Traceback" not in err


def test_persistence_error_exit_3(repo):
    try:
        raise gr.ResultPersistenceError("cannot persist: /secret") from OSError("disk full secret")
    except gr.ResultPersistenceError as exc:
        error = exc
    code, _, err, _ = invoke([RUN_ID], repo, grade_fn=_raising(error))
    assert code == 3 and "OSError" in err and "secret" not in err


def test_interrupt_exit_130(repo):
    code, _, err, _ = invoke([RUN_ID], repo, grade_fn=_raising(KeyboardInterrupt()))
    assert code == 130 and "Interrupted" in err


# --- Offline boundary ------------------------------------------------------------------


def test_cli_imports_no_execution_dependencies():
    forbidden = {"psycopg", "requests", "flask", "subprocess", "getpass",
                 "evaluation.datachat.runner", "evaluation.datachat.client", "evaluation.datachat.maui_process",
                 "evaluation.datachat.cli"}
    tree = ast.parse(Path(cli.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if any(m == f or m.startswith(f + ".") for f in forbidden)}
    assert "MAUI_FE_" not in Path(cli.__file__).read_text()


# --- Real integration -------------------------------------------------------------------


def test_real_grading_end_to_end(tmp_path, repo_root):  # noqa: F811  (fixture from the grader tests)
    (repo_root / ".git").mkdir()
    (repo_root / "main.py").write_text("")
    run_dir = build_run(tmp_path, repo_root, {"FE-01": FE01, "FE-05": FE05})
    grade_fn = functools.partial(gr.grade_run, grader_identity=lambda _: dict(IDENTITY))
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["--run-dir", str(run_dir)], grade_fn=grade_fn, repo_root=repo_root, out=out, err=err)
    assert code == 0, err.getvalue()
    text = out.getvalue()
    assert "FE-01  TECHNICAL PASS\n" in text and "FE-05  TECHNICAL PASS\n" in text
    assert (run_dir / "grading" / "results" / "run_grade.json").is_file()
    assert (run_dir / "grading" / "results" / "scenarios" / "FE-05.json").is_file()
    assert not (run_dir / "grading" / "reviews").exists()
