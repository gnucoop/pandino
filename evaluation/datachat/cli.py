"""Operator command of the DataChat Functional Evaluation harness.

``python -m evaluation.datachat FE-01 [FE-03 ...] | --all [--official]``

A thin layer over ``runner.run_suite``: it validates operator inputs,
resolves the evaluation identity, builds ``RunnerOptions``, calls the runner
and renders a short summary read from the persisted manifests. Preflight,
lifecycle, INFRA classification and evidence stay in the runner.

Identity: ``--email`` / ``--username`` win over ``MAUI_FE_EMAIL`` /
``MAUI_FE_USERNAME``. The API key comes only from ``MAUI_FE_API_KEY`` or an
interactive ``getpass`` prompt; there is deliberately no flag for it. The
``MAUI_FE_*`` variables are removed from the environment handed to Maui.

Exit codes: 0 completed, 1 aborted (INFRA), 2 usage/configuration error,
3 unexpected harness exception, 130 interrupted.
"""

import argparse
import contextlib
import getpass
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence, TextIO

from evaluation.datachat import contract, runner
from evaluation.datachat.evidence import read_json
from evaluation.datachat.model import InfraCode, SuiteStatus
from evaluation.datachat.scenarios import SCENARIOS, Scenario

DEFAULT_PORT = (
    5055  # operator default; 5000 is often taken (macOS AirPlay, make run-local)
)
EVAL_LABEL = "fe-eval-v1"
ENV_EMAIL = "MAUI_FE_EMAIL"
ENV_USERNAME = "MAUI_FE_USERNAME"
ENV_API_KEY = "MAUI_FE_API_KEY"
OPERATOR_ENV = (ENV_EMAIL, ENV_USERNAME, ENV_API_KEY)

EXIT_COMPLETED = 0
EXIT_ABORTED = 1
EXIT_USAGE = 2
EXIT_UNEXPECTED = 3
EXIT_INTERRUPTED = 130


class _UsageError(Exception):
    pass


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port out of range 1..65535: {port}")
    return port


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.datachat",
        description="Run frozen DataChat Functional Evaluation scenarios against a harness-owned Maui.",
        epilog=f"The API key is read from {ENV_API_KEY} or prompted for; it is never a command-line argument.",
    )
    parser.add_argument(
        "scenarios", nargs="*", metavar="FE-NN", help="frozen scenario IDs (e.g. FE-01)"
    )
    parser.add_argument(
        "--all", action="store_true", help="run the full frozen catalogue"
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="official run (requires --all and a clean tree)",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"Maui bind port (default {DEFAULT_PORT})",
    )
    parser.add_argument("--email", help=f"evaluation user email (default ${ENV_EMAIL})")
    parser.add_argument(
        "--username", help=f"evaluation user name (default ${ENV_USERNAME})"
    )
    return parser


def select_scenarios(ids: Sequence[str], select_all: bool) -> tuple[Scenario, ...]:
    """Frozen scenarios for the request, in catalogue order. Raises ``_UsageError``."""
    if select_all and ids:
        raise _UsageError("pass scenario IDs or --all, not both")
    if select_all:
        return SCENARIOS
    if not ids:
        raise _UsageError(
            "no scenario selected: pass scenario IDs (e.g. FE-01) or --all"
        )
    known = {s.scenario_id for s in SCENARIOS}
    unknown = sorted({i for i in ids if i not in known})
    if unknown:
        raise _UsageError(f"unknown scenario ID(s): {', '.join(unknown)}")
    wanted = set(ids)
    return tuple(s for s in SCENARIOS if s.scenario_id in wanted)


def _identity_value(
    flag: str | None, environ: Mapping[str, str], name: str, what: str
) -> str:
    value = (flag if flag is not None else environ.get(name, "")).strip()
    if not value:
        raise _UsageError(f"evaluation {what} missing: pass --{what} or set {name}")
    return value


def _api_key(
    environ: Mapping[str, str],
    getpass_fn: Callable[[str], str],
    isatty_fn: Callable[[], bool],
) -> str:
    key = environ.get(ENV_API_KEY, "")
    if not key.strip():
        if not isatty_fn():
            raise _UsageError(f"{ENV_API_KEY} not set and no TTY available")
        key = getpass_fn("Maui API key: ")
    if not key.strip():
        raise _UsageError("empty Maui API key")
    return key


def child_base_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Operator environment without the operator-only ``MAUI_FE_*`` inputs."""
    return {k: v for k, v in environ.items() if k not in OPERATOR_ENV}


# --- Rendering -------------------------------------------------------------------


def _yes(value) -> str:
    return "yes" if value is True else "no" if value is False else "unknown"


def _display(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _scenario_rows(run_dir: Path) -> list[str]:
    try:
        manifest = read_json(run_dir / "run_manifest.json")
    except (OSError, ValueError):
        return []
    rows = []
    for entry in manifest.get("scenarios", []):
        flags = {}
        try:
            flags = read_json(run_dir / entry["dir"] / "execution.json").get(
                "flags", {}
            )
        except (OSError, ValueError, KeyError):
            pass
        teardown = (
            "verified" if flags.get("teardown_verified") is True else "not verified"
        )
        rows.append(
            f"  {entry.get('scenario_id')}  {entry.get('status')}  "
            f"scoreable={_yes(flags.get('scoreable'))}  teardown={teardown}"
        )
    return rows


def render(
    result: runner.RunResult, options: runner.RunnerOptions, out: TextIO
) -> None:
    mode = "official" if options.official else "exploratory, not comparable"
    print(f"DataChat FE execution {result.status.value} ({mode})", file=out)
    print("", file=out)
    if result.run_id:
        print(f"Run:       {result.run_id}", file=out)
    if result.failure is not None:
        print(
            f"Failure:   {result.failure.code.value} — {result.failure.summary}",
            file=out,
        )
        if (
            result.failure.code is InfraCode.INFRA_PREFLIGHT
            and result.failure.summary == f"port {options.port} already in use"
        ):
            print(f"Hint:      port {options.port} in use; pass --port", file=out)
    if result.run_dir is not None:
        rows = _scenario_rows(result.run_dir)
        if rows:
            print("Scenarios:", file=out)
            for row in rows:
                print(row, file=out)
        print(f"Evidence:  {_display(result.run_dir, options.repo_root)}", file=out)
    else:
        print("Evidence:  none (failed before a run directory was created)", file=out)
    print("", file=out)
    print("Note: execution status only — grading is a separate offline step.", file=out)


# --- Entry point -----------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    getpass_fn: Callable[[str], str] = getpass.getpass,
    isatty_fn: Callable[[], bool] = sys.stdin.isatty,
    run_fn: Callable[..., runner.RunResult] = runner.run_suite,
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
            scenarios = select_scenarios(args.scenarios, args.all)
            if args.official and not args.all:
                raise _UsageError("official mode requires --all")
            email = _identity_value(args.email, environ, ENV_EMAIL, "email")
            username = _identity_value(args.username, environ, ENV_USERNAME, "username")
            api_key = _api_key(environ, getpass_fn, isatty_fn)
        except _UsageError as exc:
            print(f"error: {exc}", file=err)
            return EXIT_USAGE

        repo_root = contract.resolve_repo_root()
        options = runner.RunnerOptions(
            repo_root=repo_root,
            runs_dir=repo_root / "evaluation" / "datachat" / "runs",
            eval_label=EVAL_LABEL,
            username=username,
            email=email,
            port=args.port,
            official=args.official,
            base_env=child_base_env(environ),
        )
        try:
            result = run_fn(options, api_key, scenarios=scenarios)
        except Exception as exc:  # message may carry anything; only the type is shown
            print(f"Unexpected harness error: {type(exc).__name__}", file=err)
            print(
                "See persisted failure evidence for details (failures/ in the run directory, if one was created).",
                file=err,
            )
            return EXIT_UNEXPECTED
        render(result, options, out)
        return (
            EXIT_COMPLETED if result.status is SuiteStatus.COMPLETED else EXIT_ABORTED
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=err)
        return EXIT_INTERRUPTED
