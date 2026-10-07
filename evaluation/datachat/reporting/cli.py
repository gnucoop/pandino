"""Operator command for reporting on a graded DataChat Functional Evaluation run.

``python -m evaluation.datachat.reporting RUN_ID | --run-dir PATH``

Orchestration only: resolve the run → ``load_run_for_reporting`` →
``build_operator_report`` → ``render_markdown`` → ``<run>/reports/report.md``.
Offline; never grades (run ``python -m evaluation.datachat.grading`` first),
never writes anywhere else in the run package. The report is derived and is
overwritten on every run.

Exit codes: 0 report written (whatever the scenario technical outcomes —
they are content, not command failures), 2 usage/input error
or reporting refusal (missing run, ungraded or inconsistent package), 3
unexpected or persistence error, 130 interrupted.
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Sequence, TextIO

from evaluation.datachat import contract
from evaluation.datachat.evidence import _fsync_dir
from evaluation.datachat.grading.cli import RUNS_DIR_RELATIVE, _display, _UsageError, resolve_run_dir
from evaluation.datachat.reporting.operator_report import OperatorReportError, build_operator_report
from evaluation.datachat.reporting.reader import ReportingInputError, load_run_for_reporting
from evaluation.datachat.reporting.renderer import REPORT_DIR, REPORT_NAME, render_markdown

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNEXPECTED = 3
EXIT_INTERRUPTED = 130


class ReportPersistenceError(Exception):
    """The report could not be written."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.datachat.reporting",
        description="Write a Markdown report for a graded DataChat Functional Evaluation run to "
                    "<run>/reports/report.md. Offline; does not execute or grade.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("run_id", nargs="?", metavar="RUN_ID", help=f"run directory name under {RUNS_DIR_RELATIVE}/")
    target.add_argument("--run-dir", type=Path, metavar="PATH", help="explicit run directory")
    return parser


def write_report(run_dir: Path, text: str) -> Path:
    """Replace ``<run>/reports/report.md`` atomically (temp, fsync, ``os.replace``)."""
    path = Path(run_dir) / REPORT_DIR / REPORT_NAME
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(exist_ok=True)
        with open(tmp, "wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ReportPersistenceError(str(path)) from exc
    return path


def generate_report(run_dir: Path) -> tuple[str, Path]:
    """Load, project, render and persist. Returns ``(run_id, report path)``."""
    report = build_operator_report(load_run_for_reporting(run_dir))
    return report.run_id, write_report(run_dir, render_markdown(report))


def main(
    argv: Sequence[str] | None = None,
    *,
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
            run_id, path = generate_report(run_dir)
        except (ReportingInputError, OperatorReportError) as exc:
            print(f"Cannot report run: {exc}", file=err)
            if "not graded" in str(exc):
                print("Grade it first: python -m evaluation.datachat.grading <RUN_ID>", file=err)
            return EXIT_USAGE
        except ReportPersistenceError as exc:
            print(f"Cannot write report: {type(exc.__cause__ or exc).__name__}", file=err)
            print(f"Report path: {_display(Path(str(exc)), root)}", file=err)
            return EXIT_UNEXPECTED
        except Exception as exc:  # message may carry anything; only the type is shown
            print(f"Unexpected reporting error: {type(exc).__name__}", file=err)
            return EXIT_UNEXPECTED
        print("DataChat FE report generated", file=out)
        print(f"Run:    {run_id}", file=out)
        print(f"Report: {_display(path, root)}", file=out)
        return EXIT_OK
    except KeyboardInterrupt:
        print("Interrupted.", file=err)
        return EXIT_INTERRUPTED
