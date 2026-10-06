"""Tests for evaluation.datachat.contract, the harness's frozen operating contract.

Offline: artifact checks and environment resolution run against tmp_path
fixture repositories and explicit base environments, so os.environ and the
real .env are never read or changed. The only CURRENT-code coupling is a
drift check of the copied provider → API-key map against config.py, which
has no import-time side effects.
"""

import hashlib
import os

import pytest

from evaluation.datachat import contract
from evaluation.datachat.contract import (
    EVALUATION_DIR_RELATIVE,
    GUNICORN_TIMEOUT_S,
    HTTP_CLIENT_TIMEOUT_S,
    PINNED_ARTIFACTS,
    USAGE_QUERY_SQL,
    environment_report,
    evaluation_overrides,
    gunicorn_argv,
    package_versions,
    provider_api_key_env,
    required_artifacts_match,
    resolve_child_env,
    resolve_repo_root,
    validate_env,
    verify_artifacts,
)

SECRET_KEY = "sk-provider-secret"
PG_SECRET = "pg-secret"
ENC_SECRET = "enc-secret"


def _valid_env(**overrides):
    env = {
        "DATACHAT_PROVIDER": "OpenAI",
        "DATACHAT_MODEL": "gpt-x",
        "PGHOST": "localhost",
        "PGDB": "maui",
        "PGUSER": "maui",
        "PGPWD": PG_SECRET,
        "ENCRYPTION_KEY": ENC_SECRET,
        "OPENAI_API_KEY": SECRET_KEY,
    }
    env.update(overrides)
    return env


# --- Repository root ---------------------------------------------------------


def test_repo_root_is_the_maui_checkout():
    root = resolve_repo_root()
    assert (root / "main.py").is_file()
    assert (root / "evaluation" / "datachat" / "contract.py").is_file()


def test_resolve_repo_root_rejects_non_repo(tmp_path):
    with pytest.raises(ValueError):
        resolve_repo_root(tmp_path)


# --- Artifact contract -------------------------------------------------------


def test_pinned_artifacts_are_exact():
    assert [(a.name, a.sha256, a.required) for a in PINNED_ARTIFACTS] == [
        ("evaluation_dataset.csv", "f98f610ceb21e4121ef7310883ae8e3d55496512f0a5a633f1e04aeca47f38fe", True),
        ("ground_truth.json", "25de35052d44286f05fa6cb941deab050de084ed87d0695c0fbc587f4fbd77d1", True),
        ("ground_truth.md", "cc1068b74d3004c458a3464e928da787dd3269d8761385361ef712a722db5383", True),
        ("datachat_functional_evaluation_design.md", "0d7462ea1a475339775ef6229887d1ed66cc4ec106f2567bcb7ad83663513914", True),
        ("generate_dataset.py", "0c5802b6b341466cb2b39a7b2fa721b6f5ca7794a6c53086c337bd93b904645b", False),
    ]


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A repo whose evaluation files have contents b'<name>' and pins matching them."""
    directory = tmp_path / EVALUATION_DIR_RELATIVE
    directory.mkdir(parents=True)
    pins = []
    for artifact in PINNED_ARTIFACTS:
        data = artifact.name.encode()
        (directory / artifact.name).write_bytes(data)
        pins.append(
            contract.PinnedArtifact(artifact.name, hashlib.sha256(data).hexdigest(), artifact.required)
        )
    monkeypatch.setattr(contract, "PINNED_ARTIFACTS", tuple(pins))
    return tmp_path


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir()}


def test_verify_artifacts_all_match(fake_repo):
    checks = verify_artifacts(fake_repo)
    assert [c.name for c in checks] == [a.name for a in contract.PINNED_ARTIFACTS]
    assert all(c.exists and c.matches for c in checks)
    assert required_artifacts_match(checks)
    assert checks[0].to_dict()["path"] == "docs/datachat_functional_evaluation/evaluation_dataset.csv"


def test_changed_bytes_fail_and_nothing_is_regenerated(fake_repo):
    directory = fake_repo / EVALUATION_DIR_RELATIVE
    (directory / "ground_truth.json").write_bytes(b"tampered")
    before = _snapshot(directory)

    checks = {c.name: c for c in verify_artifacts(fake_repo)}

    assert checks["ground_truth.json"].matches is False
    assert checks["ground_truth.json"].actual_sha256 == hashlib.sha256(b"tampered").hexdigest()
    assert not required_artifacts_match(tuple(checks.values()))
    assert _snapshot(directory) == before


def test_missing_required_artifact_reported(fake_repo):
    (fake_repo / EVALUATION_DIR_RELATIVE / "evaluation_dataset.csv").unlink()
    checks = {c.name: c for c in verify_artifacts(fake_repo)}
    missing = checks["evaluation_dataset.csv"]
    assert (missing.exists, missing.actual_sha256, missing.matches) == (False, None, False)
    assert not required_artifacts_match(tuple(checks.values()))
    assert not (fake_repo / EVALUATION_DIR_RELATIVE / "evaluation_dataset.csv").exists()


def test_missing_generator_is_non_blocking(fake_repo):
    (fake_repo / EVALUATION_DIR_RELATIVE / "generate_dataset.py").unlink()
    checks = {c.name: c for c in verify_artifacts(fake_repo)}
    assert checks["generate_dataset.py"].exists is False
    assert checks["generate_dataset.py"].required is False
    assert required_artifacts_match(tuple(checks.values()))


def test_missing_evaluation_directory_reports_without_raising(tmp_path):
    checks = verify_artifacts(tmp_path)
    assert not any(c.exists for c in checks)
    assert not required_artifacts_match(checks)


# --- Environment precedence --------------------------------------------------


def test_env_precedence(tmp_path):
    (tmp_path / ".env").write_text(
        "DATACHAT_MODEL=from-dotenv\n"
        "PGHOST=dotenv-host\n"
        "DATACHAT_SQL_ENABLED=true\n"
        "LOG_LEVEL=DEBUG\n"
    )
    base = {"DATACHAT_MODEL": "from-base", "DATACHAT_SQL_ENABLED": "true"}
    snapshot = dict(base)
    run_dir = tmp_path / "runs" / "r1"

    result = resolve_child_env(tmp_path, run_dir, base_env=base)

    assert result.dotenv_present is True
    assert result.env["DATACHAT_MODEL"] == "from-base"  # base wins over .env
    assert result.env["PGHOST"] == "dotenv-host"  # .env fills a gap
    assert result.env["DATACHAT_SQL_ENABLED"] == "false"  # override wins over both
    assert result.env["LOG_LEVEL"] == "INFO"
    assert result.env["AGENT_RUNS_LOG_PATH"] == str(run_dir / "agent_runs.log")
    assert result.env["DATACHAT_PLOTS_DIR"] == str(run_dir / "plots")
    assert base == snapshot


def test_env_without_dotenv(tmp_path):
    result = resolve_child_env(tmp_path, tmp_path / "run", base_env={"A": "1"})
    assert result.dotenv_present is False
    assert result.env["A"] == "1"
    assert result.env["DATACHAT_LOG_LEVEL"] == "INFO"


def test_resolve_child_env_leaves_os_environ_untouched(tmp_path, monkeypatch):
    monkeypatch.delenv("FE_ONLY_IN_DOTENV", raising=False)
    (tmp_path / ".env").write_text("FE_ONLY_IN_DOTENV=1\n")
    before = dict(os.environ)
    result = resolve_child_env(tmp_path, tmp_path / "run")
    assert result.env["FE_ONLY_IN_DOTENV"] == "1"
    assert dict(os.environ) == before


def test_evaluation_overrides_are_only_evaluation_settings(tmp_path):
    assert set(evaluation_overrides(tmp_path)) == {
        "DATACHAT_SQL_ENABLED",
        "DATACHAT_LOG_LEVEL",
        "LOG_LEVEL",
        "AGENT_RUNS_LOG_PATH",
        "DATACHAT_PLOTS_DIR",
    }


# --- Validation and report ---------------------------------------------------


def test_valid_env_has_no_problems():
    assert validate_env(_valid_env()) == []


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_model_is_invalid(value):
    assert validate_env(_valid_env(DATACHAT_MODEL=value)) == ["DATACHAT_MODEL"]


def test_missing_provider_key_is_invalid():
    env = _valid_env()
    del env["OPENAI_API_KEY"]
    assert validate_env(env) == ["OPENAI_API_KEY"]


def test_missing_required_values_are_listed():
    assert validate_env({}) == list(contract.REQUIRED_ENV)


def test_provider_key_fallback_matches_litellm_factory():
    assert provider_api_key_env("Deepinfra") == "DEEPINFRA_API_KEY"
    assert provider_api_key_env(" Anthropic ") == "ANTHROPIC_API_KEY"
    assert provider_api_key_env("Acme") == "ACME_API_KEY"


def test_provider_map_matches_current_config():
    import config

    assert contract.PROVIDER_API_KEY_MAP == config.PROVIDER_API_KEY_MAP


def test_environment_report_never_contains_secrets():
    env = _valid_env(PGPORT="5432", UNRELATED="x")
    report = environment_report(env, dotenv_present=True)

    flat = repr(report)
    for secret in (SECRET_KEY, PG_SECRET, ENC_SECRET):
        assert secret not in flat
    assert "UNRELATED" not in flat
    assert report["values"]["DATACHAT_MODEL"] == "gpt-x"
    assert report["values"]["PGPORT"] == "5432"
    assert report["values"]["MAUI_SCHEMA"] is None
    assert report["dotenv_present"] is True
    assert report["secrets_present"] == {
        "OPENAI_API_KEY": True,
        "PGPWD": True,
        "ENCRYPTION_KEY": True,
    }


def test_environment_report_presence_flags_false_when_absent():
    env = _valid_env(PGPWD="")
    del env["OPENAI_API_KEY"]
    report = environment_report(env, dotenv_present=False)
    assert report["secrets_present"] == {
        "OPENAI_API_KEY": False,
        "PGPWD": False,
        "ENCRYPTION_KEY": True,
    }


# --- Runtime constants -------------------------------------------------------


def test_gunicorn_argv():
    argv = gunicorn_argv(5123)
    assert argv[0].endswith("gunicorn")
    assert argv[1] == "main:app"
    assert argv[argv.index("-k") + 1] == "gevent"
    assert argv[argv.index("--workers") + 1] == "1"
    assert argv[argv.index("--worker-connections") + 1] == "10"
    assert argv[argv.index("--timeout") + 1] == "300"
    assert argv[argv.index("--bind") + 1] == "127.0.0.1:5123"
    assert "--preload" not in argv
    assert gunicorn_argv()[-1] == "127.0.0.1:5000"


@pytest.mark.parametrize("port, exc", [(0, ValueError), (70000, ValueError), ("5000", TypeError)])
def test_gunicorn_argv_rejects_bad_port(port, exc):
    with pytest.raises(exc):
        gunicorn_argv(port)


def test_http_timeout_exceeds_gunicorn_timeout():
    assert GUNICORN_TIMEOUT_S == 300
    assert HTTP_CLIENT_TIMEOUT_S == 330
    assert HTTP_CLIENT_TIMEOUT_S > GUNICORN_TIMEOUT_S


# --- Usage SQL ---------------------------------------------------------------


def test_usage_query_shape():
    normalized = " ".join(USAGE_QUERY_SQL.split())
    columns = "id, user_id, date, token_input, token_output, cost, model, provider, service, request_id, duration_ms, source"
    assert normalized == f"SELECT {columns} FROM logs WHERE request_id = %s ORDER BY id"


# --- Package versions --------------------------------------------------------


def test_package_versions_missing_package_is_none():
    versions = package_versions(("pytest", "fe-harness-surely-not-installed"))
    assert versions["fe-harness-surely-not-installed"] is None
    assert isinstance(versions["pytest"], str)
    assert list(versions) == ["pytest", "fe-harness-surely-not-installed"]


def test_versioned_packages_list():
    assert contract.VERSIONED_PACKAGES == (
        "smolagents", "litellm", "flask", "gunicorn", "gevent", "pandas", "matplotlib", "psycopg", "httpx",
    )
