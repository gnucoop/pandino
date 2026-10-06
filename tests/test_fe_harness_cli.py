"""Offline tests of the FE harness operator CLI (fake runner; no Maui, DB or provider)."""

import io
import json

import pytest

from evaluation.datachat import cli
from evaluation.datachat.model import Failure, InfraCode, SuiteStatus
from evaluation.datachat.runner import RunResult
from evaluation.datachat.scenarios import SCENARIOS

KEY = "super-secret-fe-key-123"
ENV = {
    "MAUI_FE_EMAIL": "fe@example.org",
    "MAUI_FE_USERNAME": "fe-user",
    "MAUI_FE_API_KEY": KEY,
    "PATH": "/usr/bin",
    "DATACHAT_MODEL": "m",
}


class FakeRunner:
    def __init__(self, result=None, raises=None):
        self.result = result or RunResult("r1", SuiteStatus.COMPLETED, None, None)
        self.raises = raises
        self.calls = []

    def __call__(self, options, api_key, scenarios):
        self.calls.append((options, api_key, scenarios))
        if self.raises is not None:
            raise self.raises
        return self.result


def run(argv, environ=ENV, runner=None, tty=False, typed=""):
    runner = runner or FakeRunner()
    out, err = io.StringIO(), io.StringIO()
    prompts = []

    def getpass_fn(prompt):
        prompts.append(prompt)
        return typed

    code = cli.main(argv, environ=environ, getpass_fn=getpass_fn, isatty_fn=lambda: tty,
                    run_fn=runner, out=out, err=err)
    return code, out.getvalue(), err.getvalue(), runner, prompts


def ids(scenarios):
    return [s.scenario_id for s in scenarios]


# Scenario selection


def test_single_scenario():
    code, _, _, r, _ = run(["FE-01"])
    assert code == 0 and ids(r.calls[0][2]) == ["FE-01"]


def test_multiple_ids_reordered_and_deduplicated():
    code, _, _, r, _ = run(["FE-03", "FE-01", "FE-03"])
    assert code == 0 and ids(r.calls[0][2]) == ["FE-01", "FE-03"]


def test_all_selects_frozen_catalogue():
    _, _, _, r, _ = run(["--all"])
    assert r.calls[0][2] == SCENARIOS


@pytest.mark.parametrize("argv", [["FE-99"], ["What is up?"], [], ["FE-01", "--all"]])
def test_bad_selection_is_usage_error(argv):
    code, _, err, r, _ = run(argv)
    assert code == 2 and r.calls == [] and "error:" in err


# Identity


def test_identity_from_env():
    options = run(["FE-01"])[3].calls[0][0]
    assert (options.email, options.username) == ("fe@example.org", "fe-user")


def test_flags_override_env():
    options = run(["FE-01", "--email", "a@b.c", "--username", "u2"])[3].calls[0][0]
    assert (options.email, options.username) == ("a@b.c", "u2")


@pytest.mark.parametrize("name", ["MAUI_FE_EMAIL", "MAUI_FE_USERNAME"])
def test_missing_identity_is_usage_error(name):
    env = {k: v for k, v in ENV.items() if k != name}
    code, _, err, r, _ = run(["FE-01"], environ=env)
    assert code == 2 and r.calls == [] and name in err


# API key


def test_env_key_used_without_prompt():
    _, _, _, r, prompts = run(["FE-01"], tty=True)
    assert r.calls[0][1] == KEY and prompts == []


@pytest.mark.parametrize("env_key", [None, "  "])
def test_getpass_used_when_env_key_missing_or_blank(env_key):
    env = {k: v for k, v in ENV.items() if k != "MAUI_FE_API_KEY"}
    if env_key is not None:
        env["MAUI_FE_API_KEY"] = env_key
    code, _, _, r, prompts = run(["FE-01"], environ=env, tty=True, typed=KEY)
    assert code == 0 and r.calls[0][1] == KEY and prompts == ["Maui API key: "]


def test_empty_getpass_is_usage_error():
    env = {k: v for k, v in ENV.items() if k != "MAUI_FE_API_KEY"}
    code, _, _, r, _ = run(["FE-01"], environ=env, tty=True, typed="   ")
    assert code == 2 and r.calls == []


def test_missing_key_without_tty_is_usage_error():
    env = {k: v for k, v in ENV.items() if k != "MAUI_FE_API_KEY"}
    code, _, err, r, prompts = run(["FE-01"], environ=env, tty=False)
    assert code == 2 and r.calls == [] and prompts == [] and "no TTY" in err


def test_no_api_key_flag():
    code, _, _, r, _ = run(["FE-01", "--api-key", KEY])
    assert code == 2 and r.calls == []


# Child environment


def test_operator_variables_stripped_from_child_env():
    options = run(["FE-01"])[3].calls[0][0]
    assert options.base_env == {"PATH": "/usr/bin", "DATACHAT_MODEL": "m"}


# Port


def test_default_port_is_operator_default():
    assert run(["FE-01"])[3].calls[0][0].port == 5055


def test_explicit_port():
    assert run(["FE-01", "--port", "6001"])[3].calls[0][0].port == 6001


@pytest.mark.parametrize("port", ["0", "65536", "abc"])
def test_invalid_port_is_usage_error(port):
    code, _, _, r, _ = run(["FE-01", "--port", port])
    assert code == 2 and r.calls == []


# Official


def test_all_official_reaches_runner():
    code, out, _, r, _ = run(["--all", "--official"])
    assert code == 0 and r.calls[0][0].official is True and "(official)" in out


def test_official_without_all_is_usage_error():
    code, _, err, r, _ = run(["FE-01", "--official"])
    assert code == 2 and r.calls == [] and "--all" in err


def test_default_is_exploratory():
    code, out, _, r, _ = run(["FE-01"])
    assert r.calls[0][0].official is False and "exploratory, not comparable" in out


# Exit codes and output


def _write_run(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "scenarios" / "FE-01").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(
        {"scenarios": [{"scenario_id": "FE-01", "status": "EXECUTED", "dir": "scenarios/FE-01"}]}))
    (run_dir / "scenarios" / "FE-01" / "execution.json").write_text(json.dumps(
        {"flags": {"scoreable": True, "teardown_verified": True}}))
    return run_dir


def test_completed_summary(tmp_path):
    run_dir = _write_run(tmp_path)
    code, out, _, _, _ = run(["FE-01"], runner=FakeRunner(RunResult("r1", SuiteStatus.COMPLETED, None, run_dir)))
    assert code == 0
    assert "DataChat FE execution COMPLETED" in out
    assert "FE-01  EXECUTED  scoreable=yes  teardown=verified" in out
    assert str(run_dir) in out
    assert "grading not implemented" in out


def test_aborted_infra_exit_one_with_port_hint():
    failure = Failure(InfraCode.INFRA_PREFLIGHT, "port 5055 already in use")
    code, out, _, _, _ = run(["FE-01"], runner=FakeRunner(RunResult(None, SuiteStatus.ABORTED_INFRA, failure, None)))
    assert code == 1
    assert "ABORTED_INFRA" in out and "INFRA_PREFLIGHT — port 5055 already in use" in out
    assert "port 5055 in use; pass --port" in out and "Evidence:  none" in out


def test_aborted_other_failure_has_no_port_hint():
    failure = Failure(InfraCode.INFRA_TOKEN_BUDGET, "token balance 1 < required 4")
    code, out, _, _, _ = run(["FE-01"], runner=FakeRunner(RunResult(None, SuiteStatus.ABORTED_INFRA, failure, None)))
    assert code == 1 and "Hint" not in out


def test_unexpected_exception_shows_type_only():
    code, out, err, _, _ = run(["FE-01"], runner=FakeRunner(raises=RuntimeError(f"boom {KEY}")))
    assert code == 3
    assert "Unexpected harness error: RuntimeError" in err
    assert "boom" not in out + err and KEY not in out + err


def test_keyboard_interrupt():
    code, _, err, _, _ = run(["FE-01"], runner=FakeRunner(raises=KeyboardInterrupt()))
    assert code == 130 and "Interrupted" in err


def test_help_exits_zero():
    code, out, _, r, _ = run(["--help"])
    assert code == 0 and "--all" in out and r.calls == []


# Secrets


def test_key_never_in_output_options_or_child_env(tmp_path):
    run_dir = _write_run(tmp_path)
    code, out, err, r, _ = run(["FE-01"], runner=FakeRunner(RunResult("r1", SuiteStatus.COMPLETED, None, run_dir)))
    options, api_key, _ = r.calls[0]
    assert api_key == KEY
    assert KEY not in out and KEY not in err
    assert KEY not in repr(options)
    assert KEY not in options.base_env.values()
    assert not any(k.startswith("MAUI_FE_") for k in options.base_env)
