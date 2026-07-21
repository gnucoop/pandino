from unittest.mock import patch

from routes.admin import _collect_env_vars


def _write_env(root_path, content):
    (root_path / ".env").write_text(content, encoding="utf-8")


def test_unknown_env_keys_are_not_emitted(tmp_path):
    _write_env(
        tmp_path,
        """
DATACHAT_MODEL="Qwen/Qwen2.5-Coder"
UNKNOWN_FEATURE_FLAG="enabled"
UNKNOWN_API_KEY="should-not-render"
""",
    )

    display = _collect_env_vars(str(tmp_path))

    assert display == {"DATACHAT_MODEL": "Qwen/Qwen2.5-Coder"}


def test_database_url_and_private_values_are_never_emitted(tmp_path):
    _write_env(
        tmp_path,
        """
DATABASE_URL="postgres://user:password@localhost/db"
PRIVATE_CLIENT="client-secret-value"
DATACHAT_PROVIDER="Deepinfra"
""",
    )

    display = _collect_env_vars(str(tmp_path))

    assert display == {"DATACHAT_PROVIDER": "Deepinfra"}
    rendered_values = " ".join(display.values())
    assert "postgres://user:password@localhost/db" not in rendered_values
    assert "client-secret-value" not in rendered_values


def test_known_secrets_are_status_only(tmp_path):
    _write_env(
        tmp_path,
        """
OPENAI_API_KEY="sk-openai"
PGPWD="database-password"
ADMIN_PASSWORD_HASH="bcrypt-hash"
STRIPE_SK_KEY="sk-stripe"
ENCRYPTION_KEY=""
""",
    )

    display = _collect_env_vars(str(tmp_path))

    assert display == {
        "OPENAI_API_KEY": "configured",
        "PGPWD": "configured",
        "ADMIN_PASSWORD_HASH": "configured",
        "STRIPE_SK_KEY": "configured",
        "ENCRYPTION_KEY": "not set",
    }
    assert "sk-openai" not in display.values()
    assert "database-password" not in display.values()
    assert "bcrypt-hash" not in display.values()
    assert "sk-stripe" not in display.values()


def test_known_safe_env_values_are_displayed(tmp_path):
    _write_env(
        tmp_path,
        """
DATACHAT_MODEL="Qwen/Qwen2.5-Coder"
DATACHAT_PROVIDER="Deepinfra"
DATACHAT_MAX_STEPS="12"
DATACHAT_SESSION_TTL_MIN="60"
DATACHAT_ENGINE="smolagents"
DATACHAT_LOG_LEVEL="INFO"
RAG_TOP_K="3"
RAG_MIN_SIM="0.5"
COMPLETION_TOKEN_COST="1"
AUTH_GATEWAY_URL="http://localhost:3000/validate"
OLLAMA_BASE_URL="http://localhost:11434"
""",
    )

    display = _collect_env_vars(str(tmp_path))

    assert display == {
        "DATACHAT_MODEL": "Qwen/Qwen2.5-Coder",
        "DATACHAT_PROVIDER": "Deepinfra",
        "DATACHAT_MAX_STEPS": "12",
        "DATACHAT_SESSION_TTL_MIN": "60",
        "DATACHAT_ENGINE": "smolagents",
        "DATACHAT_LOG_LEVEL": "INFO",
        "RAG_TOP_K": "3",
        "RAG_MIN_SIM": "0.5",
        "COMPLETION_TOKEN_COST": "1",
        "AUTH_GATEWAY_URL": "http://localhost:3000/validate",
        "OLLAMA_BASE_URL": "http://localhost:11434",
    }


def test_live_environment_without_dotenv_is_filtered_through_allowlists(tmp_path):
    env = {
        "DATACHAT_MODEL": "live-datachat-model",
        "RAG_TOP_K": "8",
        "OPENAI_API_KEY": "live-secret",
        "DATABASE_URL": "postgres://live-secret",
        "PRIVATE_CLIENT": "live-private-client",
        "UNLISTED_SAFE_LOOKING_NAME": "do-not-render",
    }

    with patch.dict("os.environ", env, clear=True):
        display = _collect_env_vars(str(tmp_path))

    assert display["DATACHAT_MODEL"] == "live-datachat-model"
    assert display["RAG_TOP_K"] == "8"
    assert display["OPENAI_API_KEY"] == "configured"
    assert "DATABASE_URL" not in display
    assert "PRIVATE_CLIENT" not in display
    assert "UNLISTED_SAFE_LOOKING_NAME" not in display
    assert "live-secret" not in display.values()
    assert "live-private-client" not in display.values()
