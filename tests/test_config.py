import pytest
from unittest.mock import patch

from config import load_config

REQUIRED_ENV = {
    "ENCRYPTION_KEY": "test-encryption-key",
    "PGUSER": "testuser",
    "PGPWD": "testpass",
    "PGHOST": "localhost",
    "PGDB": "testdb",
    "ADMIN_USERNAME": "admin",
    "ADMIN_PASSWORD_HASH": "hashed_password",
}


def test_load_config_success():
    env = {
        **REQUIRED_ENV,
        "DATACHAT_MODEL": "my/datachat-model",
        "RAG_TOP_K": "5",
        "DATACHAT_ENGINE": "smolagents",
        "COMPLETION_TOKEN_COST": "3",
        "PG_PORT": "5433",
    }
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()

    assert cfg.encryption_key == "test-encryption-key"
    assert cfg.database.host == "localhost"
    assert cfg.database.port == "5433"
    assert cfg.admin.username == "admin"
    assert cfg.models.datachat_model == "my/datachat-model"
    assert cfg.rag.top_k == 5
    assert cfg.datachat.engine == "smolagents"
    assert cfg.completion_token_cost == 3


def test_load_config_missing_required_raises():
    partial_env = {k: v for k, v in REQUIRED_ENV.items() if k not in ("PGHOST", "ADMIN_PASSWORD_HASH")}
    with patch.dict("os.environ", partial_env, clear=True):
        with pytest.raises(ValueError) as exc_info:
            load_config()

    message = str(exc_info.value)
    assert "PGHOST" in message
    assert "ADMIN_PASSWORD_HASH" in message


def test_load_config_defaults():
    with patch.dict("os.environ", REQUIRED_ENV, clear=True):
        cfg = load_config()

    assert cfg.database.port == "5432"
    assert cfg.rag.top_k == 3
    assert cfg.rag.min_sim == 0.5
    assert cfg.datachat.engine == "smolagents"
    assert cfg.datachat.max_steps == 12
    assert cfg.auth_gateway_url == "http://localhost:3000/validate"
