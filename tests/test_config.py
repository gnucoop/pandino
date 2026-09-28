import pytest
from unittest.mock import patch

from config import _load_datachat_sql_config, load_config

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


def test_load_config_admin_rag_usage_username_absent_is_none():
    with patch.dict("os.environ", REQUIRED_ENV, clear=True):
        cfg = load_config()

    assert cfg.admin_rag_usage_username is None


def test_load_config_admin_rag_usage_username_empty_is_none():
    env = {**REQUIRED_ENV, "ADMIN_RAG_USAGE_USERNAME": ""}
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()

    assert cfg.admin_rag_usage_username is None


def test_load_config_admin_rag_usage_username_preserved_exactly():
    env = {**REQUIRED_ENV, "ADMIN_RAG_USAGE_USERNAME": "__admin_rag_ingestion__"}
    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()

    assert cfg.admin_rag_usage_username == "__admin_rag_ingestion__"

# ---------------------------------------------------------------------------
# Datachat SQL datasource
# ---------------------------------------------------------------------------

SQL_ENV = {
    "DATACHAT_SQL_ENABLED": "true",
    "DATACHAT_DB_HOST": "sqlhost",
    "DATACHAT_DB_NAME": "analytics",
    "DATACHAT_DB_USER": "ro_user",
    "DATACHAT_DB_PASSWORD": "ro_pass",
}


def test_datachat_sql_disabled_by_default():
    with patch.dict("os.environ", REQUIRED_ENV, clear=True):
        cfg = load_config()

    assert cfg.datachat_sql.enabled is False
    assert cfg.datachat_sql.host == ""
    assert cfg.datachat_sql.db == ""
    assert cfg.datachat_sql.schema == "public"
    assert cfg.datachat_sql.port == "5432"


def test_datachat_sql_defaults_when_enabled():
    with patch.dict("os.environ", {**REQUIRED_ENV, **SQL_ENV}, clear=True):
        cfg = load_config()

    assert cfg.datachat_sql.enabled is True
    assert cfg.datachat_sql.host == "sqlhost"
    assert cfg.datachat_sql.db == "analytics"
    assert cfg.datachat_sql.max_rows == 200
    assert cfg.datachat_sql.max_columns == 25
    assert cfg.datachat_sql.max_cell_chars == 300
    assert cfg.datachat_sql.statement_timeout_ms == 10000
    assert cfg.datachat_sql.include_views is True
    assert cfg.datachat_sql.allowed_tables == ()
    assert cfg.datachat_sql.denied_tables == ()


def test_datachat_sql_never_falls_back_to_the_app_database():
    """The agent datasource is always a dedicated database."""
    with patch.dict("os.environ", {**REQUIRED_ENV, **SQL_ENV}, clear=True):
        cfg = load_config()

    assert cfg.database.host == "localhost"
    assert cfg.datachat_sql.host == "sqlhost"
    assert cfg.datachat_sql.user != cfg.database.user
    assert cfg.datachat_sql.db != cfg.database.db


@pytest.mark.parametrize("missing", sorted(SQL_ENV.keys() - {"DATACHAT_SQL_ENABLED"}))
def test_datachat_sql_enabled_requires_connection_vars(missing):
    """The datasource loader refuses an enabled-but-incomplete configuration."""
    env = {**REQUIRED_ENV, **SQL_ENV}
    del env[missing]

    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError) as exc_info:
            _load_datachat_sql_config()

    assert missing in str(exc_info.value)


@pytest.mark.parametrize("missing", sorted(SQL_ENV.keys() - {"DATACHAT_SQL_ENABLED"}))
def test_misconfigured_sql_disables_itself_instead_of_blocking_startup(missing):
    """
    A misconfigured datasource must not take the application down with it.

    Everything else — the CSV DataChat path included — works without SQL, so
    load_config() degrades to the disabled datasource rather than propagating.
    """
    env = {**REQUIRED_ENV, **SQL_ENV}
    del env[missing]

    with patch.dict("os.environ", env, clear=True):
        cfg = load_config()

    assert cfg.datachat_sql.enabled is False
    # The rest of the configuration is intact.
    assert cfg.database.host == "localhost"


@pytest.mark.parametrize("raw", ["true", "True", "1", "on", "yes", " TRUE "])
def test_datachat_sql_enabled_truthy_values(raw):
    env = {**REQUIRED_ENV, **SQL_ENV, "DATACHAT_SQL_ENABLED": raw}
    with patch.dict("os.environ", env, clear=True):
        assert load_config().datachat_sql.enabled is True


@pytest.mark.parametrize("raw", ["false", "0", "", "off", "maybe"])
def test_datachat_sql_enabled_falsy_values(raw):
    env = {**REQUIRED_ENV, "DATACHAT_SQL_ENABLED": raw}
    with patch.dict("os.environ", env, clear=True):
        assert load_config().datachat_sql.enabled is False


@pytest.mark.parametrize(
    "raw,expected",
    [("999999", 1000), ("0", 1), ("-5", 1), ("50", 50), ("not-a-number", 200)],
)
def test_datachat_sql_max_rows_is_clamped(raw, expected):
    env = {**REQUIRED_ENV, **SQL_ENV, "DATACHAT_SQL_MAX_ROWS": raw}
    with patch.dict("os.environ", env, clear=True):
        assert load_config().datachat_sql.max_rows == expected


def test_datachat_sql_table_lists_are_normalised():
    env = {**REQUIRED_ENV, **SQL_ENV, "DATACHAT_SQL_DENIED_TABLES": " A , b ,, B "}
    with patch.dict("os.environ", env, clear=True):
        assert load_config().datachat_sql.denied_tables == ("a", "b")


@pytest.mark.parametrize("schema", ["public; DROP TABLE t", "a b", "", "1bad", "-x"])
def test_datachat_sql_rejects_non_identifier_schema(schema):
    env = {**REQUIRED_ENV, **SQL_ENV, "DATACHAT_DB_SCHEMA": schema}
    with patch.dict("os.environ", env, clear=True):
        if schema == "":
            # An empty value falls back to the default rather than failing.
            assert _load_datachat_sql_config().schema == "public"
        else:
            with pytest.raises(ValueError):
                _load_datachat_sql_config()

            # And, as with any other misconfiguration, the app still boots with
            # the datasource switched off.
            assert load_config().datachat_sql.enabled is False


def test_datachat_sql_schema_is_not_validated_when_disabled():
    env = {**REQUIRED_ENV, "DATACHAT_DB_SCHEMA": "anything goes"}
    with patch.dict("os.environ", env, clear=True):
        assert load_config().datachat_sql.enabled is False


# ---------------------------------------------------------------------------
# Schema value profiling
# ---------------------------------------------------------------------------


def test_datachat_sql_profiling_defaults():
    with patch.dict("os.environ", {**REQUIRED_ENV, **SQL_ENV}, clear=True):
        sql = load_config().datachat_sql

    assert sql.schema_profile_values is True
    assert sql.schema_profile_sample_rows == 50
    assert sql.schema_profile_max_values == 3
    assert sql.schema_profile_max_value_chars == 32


def test_datachat_sql_profiling_can_be_turned_off():
    env = {**REQUIRED_ENV, **SQL_ENV, "DATACHAT_SQL_SCHEMA_PROFILE_VALUES": "false"}
    with patch.dict("os.environ", env, clear=True):
        assert load_config().datachat_sql.schema_profile_values is False


@pytest.mark.parametrize(
    "name,raw,expected",
    [
        ("DATACHAT_SQL_SCHEMA_PROFILE_SAMPLE_ROWS", "5000", 1000),
        ("DATACHAT_SQL_SCHEMA_PROFILE_SAMPLE_ROWS", "0", 1),
        ("DATACHAT_SQL_SCHEMA_PROFILE_SAMPLE_ROWS", "not-a-number", 50),
        ("DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUES", "100", 20),
        ("DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUES", "-1", 1),
        ("DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUE_CHARS", "5000", 200),
        ("DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUE_CHARS", "1", 4),
    ],
)
def test_datachat_sql_profiling_values_are_clamped(name, raw, expected):
    env = {**REQUIRED_ENV, **SQL_ENV, name: raw}
    with patch.dict("os.environ", env, clear=True):
        sql = load_config().datachat_sql

    attribute = name.replace("DATACHAT_SQL_", "").lower()
    assert getattr(sql, attribute) == expected


def test_datachat_sql_quotes_identifiers_by_default():
    with patch.dict("os.environ", {**REQUIRED_ENV, **SQL_ENV}, clear=True):
        assert load_config().datachat_sql.quote_identifiers is True


def test_datachat_sql_identifier_quoting_can_be_turned_off():
    env = {**REQUIRED_ENV, **SQL_ENV, "DATACHAT_SQL_QUOTE_IDENTIFIERS": "false"}
    with patch.dict("os.environ", env, clear=True):
        assert load_config().datachat_sql.quote_identifiers is False
