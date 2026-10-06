"""Frozen operating contract of the DataChat Functional Evaluation harness.

Stable constants and pure helpers only:

- pinned identity (SHA-256) of the local evaluation artifacts under
  ``docs/datachat_functional_evaluation/``. That directory is git-ignored, so
  the evaluation contract is pinned by content hash; when a required
  artifact is missing or changed, preflight must refuse to run;
- the Gunicorn command line and client timeout;
- Maui's effective child environment: inherited env, then ``.env`` for
  missing keys only (``load_dotenv(override=False)``), then the
  evaluation-only overrides;
- validation and a secret-free report of that environment;
- the canonical Usage query and the package names whose versions are recorded.

Provider, model, token cost and DB settings are never fixed here: they come
from Maui's effective environment. Nothing in this module imports the Maui
application, runs Git or Gunicorn, or touches the database.
"""

import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from evaluation.datachat.evidence import sha256_file

# --- Repository and evaluation artifacts -------------------------------------

_REPO_MARKERS = ("main.py", ".git")


def resolve_repo_root(start: Path | None = None) -> Path:
    """Maui repository root (``evaluation/datachat/`` lives two levels below it)."""
    root = Path(start) if start is not None else Path(__file__).resolve().parents[2]
    missing = [marker for marker in _REPO_MARKERS if not (root / marker).exists()]
    if missing:
        raise ValueError(f"not a Maui repository root ({', '.join(missing)} missing): {root}")
    return root


REPO_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR_RELATIVE = Path("docs") / "datachat_functional_evaluation"
DATASET_NAME = "evaluation_dataset.csv"
GROUND_TRUTH_JSON_NAME = "ground_truth.json"
GROUND_TRUTH_MD_NAME = "ground_truth.md"
DESIGN_DOC_NAME = "datachat_functional_evaluation_design.md"
GENERATOR_NAME = "generate_dataset.py"


def evaluation_dir(repo_root: Path = REPO_ROOT) -> Path:
    return Path(repo_root) / EVALUATION_DIR_RELATIVE


@dataclass(frozen=True)
class PinnedArtifact:
    """An evaluation artifact pinned by hash. ``required``: MUST match; else SHOULD record."""

    name: str
    sha256: str
    required: bool


PINNED_ARTIFACTS: tuple[PinnedArtifact, ...] = (
    PinnedArtifact(DATASET_NAME, "f98f610ceb21e4121ef7310883ae8e3d55496512f0a5a633f1e04aeca47f38fe", True),
    PinnedArtifact(GROUND_TRUTH_JSON_NAME, "25de35052d44286f05fa6cb941deab050de084ed87d0695c0fbc587f4fbd77d1", True),
    PinnedArtifact(GROUND_TRUTH_MD_NAME, "cc1068b74d3004c458a3464e928da787dd3269d8761385361ef712a722db5383", True),
    PinnedArtifact(DESIGN_DOC_NAME, "279773599231b359a0a391a3be238f643128373468b77b803deab378f8b032ee", True),
    PinnedArtifact(GENERATOR_NAME, "0c5802b6b341466cb2b39a7b2fa721b6f5ca7794a6c53086c337bd93b904645b", False),
)


def pinned_artifact(name: str) -> PinnedArtifact:
    for artifact in PINNED_ARTIFACTS:
        if artifact.name == name:
            return artifact
    raise ValueError(f"unknown pinned artifact: {name!r}")


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    path: str
    required: bool
    expected_sha256: str
    exists: bool
    actual_sha256: str | None
    matches: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "required": self.required,
            "expected_sha256": self.expected_sha256,
            "exists": self.exists,
            "actual_sha256": self.actual_sha256,
            "matches": self.matches,
        }


def verify_artifacts(repo_root: Path = REPO_ROOT) -> tuple[ArtifactCheck, ...]:
    """Hash every pinned artifact as it is on disk. Read-only; never regenerates."""
    directory = evaluation_dir(repo_root)
    checks = []
    for artifact in PINNED_ARTIFACTS:
        path = directory / artifact.name
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        checks.append(
            ArtifactCheck(
                name=artifact.name,
                path=(EVALUATION_DIR_RELATIVE / artifact.name).as_posix(),
                required=artifact.required,
                expected_sha256=artifact.sha256,
                exists=exists,
                actual_sha256=actual,
                matches=actual == artifact.sha256,
            )
        )
    return tuple(checks)


def required_artifacts_match(checks: tuple[ArtifactCheck, ...]) -> bool:
    """False means preflight must stop. Non-required artifacts never block."""
    return all(check.matches for check in checks if check.required)


# --- Maui process and HTTP ---------------------------------------------------

GUNICORN_APP = "main:app"
GUNICORN_WORKER_CLASS = "gevent"
GUNICORN_WORKERS = 1  # activeEngines, the busy guard and exports are process-local
GUNICORN_WORKER_CONNECTIONS = 10
GUNICORN_TIMEOUT_S = 300
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
HTTP_CLIENT_TIMEOUT_S = 330  # must exceed GUNICORN_TIMEOUT_S


def gunicorn_executable() -> Path:
    """The ``gunicorn`` script of the running Python environment."""
    return Path(sys.executable).parent / "gunicorn"


def gunicorn_argv(port: int = DEFAULT_PORT) -> list[str]:
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError(f"port must be int, got {type(port).__name__}")
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid port: {port}")
    return [
        str(gunicorn_executable()),
        GUNICORN_APP,
        "-k", GUNICORN_WORKER_CLASS,
        "--workers", str(GUNICORN_WORKERS),
        "--worker-connections", str(GUNICORN_WORKER_CONNECTIONS),
        "--timeout", str(GUNICORN_TIMEOUT_S),
        "--bind", f"{BIND_HOST}:{port}",
    ]


# --- Effective environment ---------------------------------------------------

# Copy of config.PROVIDER_API_KEY_MAP (kept decoupled; a test guards drift).
# Unknown providers fall back to "<PROVIDER>_API_KEY", as in llm/litellm_factory.py.
PROVIDER_API_KEY_MAP: dict[str, str] = {
    "Deepinfra": "DEEPINFRA_API_KEY",
    "Mistral": "MISTRAL_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Groq": "GROQ_API_KEY",
    "Deepseek": "DEEPSEEK_API_KEY",
}

REQUIRED_ENV = (
    "DATACHAT_PROVIDER",
    "DATACHAT_MODEL",  # empty is MISSING_CONFIG for the engine
    "PGHOST",
    "PGDB",
    "PGUSER",
    "PGPWD",
    "ENCRYPTION_KEY",
)

REPORTED_ENV = (
    "DATACHAT_PROVIDER",
    "DATACHAT_MODEL",
    "DATACHAT_MAX_STEPS",
    "DATACHAT_TOKEN_COST",
    "DATACHAT_ENGINE",
    "DATACHAT_SQL_ENABLED",
    "DATACHAT_LOG_LEVEL",
    "LOG_LEVEL",
    "AGENT_RUNS_LOG_PATH",
    "DATACHAT_PLOTS_DIR",
    "PGHOST",
    "PGDB",
    "PGPORT",
    "PGUSER",
    "MAUI_SCHEMA",
    "DATACHAT_SESSION_TTL_MIN",
    "DATACHAT_RATE_LIMIT_PER_MIN",
)


def provider_api_key_env(provider: str) -> str:
    provider = provider.strip()
    return PROVIDER_API_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")


def evaluation_overrides(run_dir: Path) -> dict[str, str]:
    """Evaluation-only settings that win over both the inherited env and ``.env``."""
    run_dir = Path(run_dir)
    return {
        "DATACHAT_SQL_ENABLED": "false",
        "DATACHAT_LOG_LEVEL": "INFO",
        "LOG_LEVEL": "INFO",
        "AGENT_RUNS_LOG_PATH": str(run_dir / "agent_runs.log"),
        "DATACHAT_PLOTS_DIR": str(run_dir / "plots"),
    }


@dataclass(frozen=True)
class ChildEnv:
    env: dict[str, str]
    dotenv_present: bool


def resolve_child_env(
    repo_root: Path, run_dir: Path, base_env: Mapping[str, str] | None = None
) -> ChildEnv:
    """Maui's effective environment for the child process. Never touches ``os.environ``.

    ``base_env`` defaults to a copy of ``os.environ``.
    """
    env = dict(os.environ if base_env is None else base_env)
    dotenv_path = Path(repo_root) / ".env"
    dotenv_present = dotenv_path.is_file()
    if dotenv_present:
        for key, value in dotenv_values(dotenv_path).items():
            if value is not None and key not in env:
                env[key] = value
    env.update(evaluation_overrides(run_dir))
    return ChildEnv(env=env, dotenv_present=dotenv_present)


def validate_env(env: Mapping[str, str]) -> list[str]:
    """Names of required variables that are missing or blank (empty list = valid)."""
    problems = [name for name in REQUIRED_ENV if not (env.get(name) or "").strip()]
    provider = (env.get("DATACHAT_PROVIDER") or "").strip()
    if provider:
        key_name = provider_api_key_env(provider)
        if not (env.get(key_name) or "").strip():
            problems.append(key_name)
    return problems


def environment_report(env: Mapping[str, str], dotenv_present: bool) -> dict:
    """Run-manifest-safe view: non-secret values plus secret presence flags only."""
    provider = (env.get("DATACHAT_PROVIDER") or "").strip()
    secret_names = ["PGPWD", "ENCRYPTION_KEY"]
    if provider:
        secret_names.insert(0, provider_api_key_env(provider))
    return {
        "values": {name: env.get(name) for name in REPORTED_ENV},
        "dotenv_present": dotenv_present,
        "secrets_present": {name: bool((env.get(name) or "").strip()) for name in secret_names},
    }


# --- Usage query -------------------------------------------------------------

USAGE_QUERY_SQL = """\
SELECT
    id,
    user_id,
    date,
    token_input,
    token_output,
    cost,
    model,
    provider,
    service,
    request_id,
    duration_ms,
    source
FROM logs
WHERE request_id = %s
ORDER BY id"""


# --- Runtime provenance ------------------------------------------------------

VERSIONED_PACKAGES = (
    "smolagents",
    "litellm",
    "flask",
    "gunicorn",
    "gevent",
    "pandas",
    "matplotlib",
    "psycopg",
    "httpx",
)


def package_versions(packages: tuple[str, ...] = VERSIONED_PACKAGES) -> dict[str, str | None]:
    """Installed distribution versions; ``None`` when not installed. Imports nothing."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions
