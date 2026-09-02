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
        "PGPORT": "5433",
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
    partial_env = {
        k: v
        for k, v in REQUIRED_ENV.items()
        if k not in ("PGHOST", "ADMIN_PASSWORD_HASH")
    }
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
    assert cfg.models.asr_mistral_price_per_minute_usd is None


def test_load_config_asr_mistral_price_per_minute_parsed_as_float():
    env = {**REQUIRED_ENV, "ASR_MISTRAL_PRICE_PER_MINUTE_USD": "0.003"}
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()

    assert cfg.models.asr_mistral_price_per_minute_usd == pytest.approx(0.003)


def test_load_config_asr_mistral_price_per_minute_invalid_raises():
    env = {**REQUIRED_ENV, "ASR_MISTRAL_PRICE_PER_MINUTE_USD": "not-a-number"}
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError):
            load_config()


def test_load_config_dino_legacy_usage_username_absent_is_none():
    with patch.dict("os.environ", REQUIRED_ENV, clear=True):
        cfg = load_config()

    assert cfg.dino_legacy_usage_username is None


def test_load_config_dino_legacy_usage_username_empty_is_none():
    env = {**REQUIRED_ENV, "DINO_LEGACY_USAGE_USERNAME": ""}
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()

    assert cfg.dino_legacy_usage_username is None


def test_load_config_dino_legacy_usage_username_preserved_exactly():
    env = {**REQUIRED_ENV, "DINO_LEGACY_USAGE_USERNAME": "__dino_legacy_ingestion__"}
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()

    assert cfg.dino_legacy_usage_username == "__dino_legacy_ingestion__"
