"""
Application configuration for Pandino Flask app.

All environment variable reads are centralised in load_config().
Import this module freely — it has no side effects at import time.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider → environment-variable name map
# Used by litellm_factory and agentchat handler to resolve API keys.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL connection parameters."""

    user: str
    password: str
    host: str
    db: str
    port: str
    schema: str


@dataclass(frozen=True)
class AdminConfig:
    """Admin panel credentials."""

    username: str
    password_hash: bytes


@dataclass(frozen=True)
class ModelConfig:
    """Default models and providers for each functional area."""

    datachat_model: str
    datachat_provider: str

    prompt_model: str
    prompt_provider: str

    completion_model: str
    completion_model_provider: str
    completion_model_agent_chat: str

    completion_embedding_model: str
    completion_embedding_model_provider: str

    audio_model: str
    audio_provider: str

    asr_model: str
    asr_provider: str

    vision_model: str
    vision_provider: str

    compare_docs_model: str
    compare_docs_provider: str

    # Optional[str] = None carries semantic meaning here (no override
    # configured), not an operational fallback.
    asr_base_url: Optional[str] = None

    # Governed Mistral ASR per-minute rate (USD), for Maui-side cost
    # resolution. Optional[float] = None means unconfigured, not a price of
    # zero: resolving a Mistral ASR cost without this set must fail
    # explicitly rather than silently produce cost = 0.
    asr_mistral_price_per_minute_usd: Optional[float] = None


@dataclass(frozen=True)
class ApiKeysConfig:
    """Third-party API keys — all optional at load time."""

    openai_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    deepinfra_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    # LangChain / LangSmith
    langchain_api_key: Optional[str] = None
    langchain_project: Optional[str] = None
    langchain_endpoint: Optional[str] = None
    langchain_tracing_v2: Optional[str] = None


@dataclass(frozen=True)
class RagConfig:
    """RAG retrieval parameters."""

    top_k: int
    min_sim: float
    default_namespace: str


@dataclass(frozen=True)
class DatachatConfig:
    """Datachat engine settings."""

    engine: str
    max_steps: int
    rate_limit_per_min: int
    session_ttl_min: int
    log_level: str


@dataclass(frozen=True)
class DatachatSqlConfig:
    """
    Read-only SQL datasource exposed to the Datachat agent.

    Disabled unless DATACHAT_SQL_ENABLED is truthy. This is always a dedicated
    database described by the DATACHAT_DB_* variables: it never falls back to
    the application database, which is reached through its own tools.
    """

    enabled: bool

    user: str
    password: str
    host: str
    db: str
    port: str
    schema: str

    max_rows: int
    max_columns: int
    max_cell_chars: int
    statement_timeout_ms: int

    allowed_tables: tuple[str, ...]
    denied_tables: tuple[str, ...]
    include_views: bool

    # Schema snapshot: reflected once per process and rendered into the agent's
    # system prompt. Read only when `enabled` is true.
    schema_ttl_s: int
    schema_max_chars: int
    schema_include_fks: bool

    # Repair identifier case in agent-authored SQL before it is validated and
    # run. PostgreSQL folds unquoted names to lowercase, which fails against a
    # mixed-case schema.
    quote_identifiers: bool

    # Value profiling: a bounded sample of each relation, rendered into the
    # snapshot as example values so the agent can see how a column is actually
    # written instead of guessing its format.
    schema_profile_values: bool
    schema_profile_sample_rows: int
    schema_profile_max_values: int
    schema_profile_max_value_chars: int

    pool_size: int
    pool_max_overflow: int
    pool_timeout_s: int
    pool_recycle_s: int


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration, composed from sub-configs."""

    encryption_key: str
    database: DatabaseConfig
    admin: AdminConfig
    models: ModelConfig
    api_keys: ApiKeysConfig
    rag: RagConfig
    datachat: DatachatConfig
    datachat_sql: DatachatSqlConfig

    auth_gateway_url: str
    stripe_key: Optional[str]

    # Token costs
    datachat_token_cost: int
    completion_token_cost: int
    prompt_token_cost: int
    audio_form_token_cost: int
    compare_docs_token_cost: int

    # Technical accounting identity used only by the legacy Dino fallback of
    # /storeragfile. Optional[str] = None means "no technical identity
    # configured" — the off-switch: absent configuration leaves existing
    # ingestion behaviour unchanged. The production username is a deployment
    # choice, never an implicit application fallback.
    dino_legacy_usage_username: Optional[str] = None

    # Technical accounting identity used only by POST /admin/rag-files/upload
    # embedding Usage attribution. Optional[str] = None means "no technical
    # identity configured" — the off-switch: absent configuration leaves
    # existing admin ingestion behaviour unchanged. This is an accounting
    # identity, not an admin credential: AdminConfig holds login credentials
    # and deliberately says nothing about the users table. The production
    # username is a deployment choice, never an implicit application fallback.
    admin_rag_usage_username: Optional[str] = None


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}

# The schema name is interpolated into the libpq `options` connection string,
# which cannot take bind parameters — so it must be a plain SQL identifier.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _env_flag(name: str, default: str = "false") -> bool:
    """Read a boolean environment variable."""
    return os.environ.get(name, default).strip().lower() in _TRUTHY


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an integer environment variable, clamped to [minimum, maximum]."""
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _env_table_set(name: str, default: str = "") -> tuple[str, ...]:
    """Read a comma-separated table list into a sorted, lowercased tuple."""
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    return tuple(sorted({part.strip().lower() for part in raw.split(",") if part.strip()}))


def _load_datachat_sql_config(*, force_disabled: bool = False) -> DatachatSqlConfig:
    """
    Build the Datachat SQL datasource config.

    When disabled, connection fields are left empty and never read. When
    enabled, the DATACHAT_DB_* connection variables are mandatory: this
    datasource is always a dedicated database, never the application one.

    :param force_disabled: Build the disabled configuration whatever the
                           environment says. Used by load_config() to turn the
                           SQL feature off after a misconfiguration instead of
                           taking the whole application down with it.

    Raises:
        ValueError: if the datasource is enabled and connection variables are
                    missing, or if the schema is not a plain SQL identifier.
    """
    enabled = False if force_disabled else _env_flag("DATACHAT_SQL_ENABLED")

    connection = {
        "DATACHAT_DB_HOST": os.environ.get("DATACHAT_DB_HOST"),
        "DATACHAT_DB_NAME": os.environ.get("DATACHAT_DB_NAME"),
        "DATACHAT_DB_USER": os.environ.get("DATACHAT_DB_USER"),
        "DATACHAT_DB_PASSWORD": os.environ.get("DATACHAT_DB_PASSWORD"),
    }

    if enabled:
        missing = [k for k, v in connection.items() if not v]
        if missing:
            raise ValueError(
                "DATACHAT_SQL_ENABLED is set but the following required "
                "variables are not set: " + ", ".join(missing)
            )

    schema = os.environ.get("DATACHAT_DB_SCHEMA", "public").strip() or "public"
    if enabled and not _IDENTIFIER_RE.match(schema):
        raise ValueError(
            f"DATACHAT_DB_SCHEMA must be a plain SQL identifier, got: {schema!r}"
        )

    return DatachatSqlConfig(
        enabled=enabled,
        user=connection["DATACHAT_DB_USER"] or "",
        password=connection["DATACHAT_DB_PASSWORD"] or "",
        host=connection["DATACHAT_DB_HOST"] or "",
        db=connection["DATACHAT_DB_NAME"] or "",
        port=os.environ.get("DATACHAT_DB_PORT", "5432"),
        schema=schema,
        max_rows=_env_int("DATACHAT_SQL_MAX_ROWS", 200, minimum=1, maximum=1000),
        max_columns=_env_int("DATACHAT_SQL_MAX_COLUMNS", 25, minimum=1, maximum=100),
        max_cell_chars=_env_int(
            "DATACHAT_SQL_MAX_CELL_CHARS", 300, minimum=0, maximum=2000
        ),
        statement_timeout_ms=_env_int(
            "DATACHAT_SQL_STATEMENT_TIMEOUT_MS", 10000, minimum=1000, maximum=60000
        ),
        allowed_tables=_env_table_set("DATACHAT_SQL_ALLOWED_TABLES"),
        denied_tables=_env_table_set("DATACHAT_SQL_DENIED_TABLES"),
        include_views=_env_flag("DATACHAT_SQL_INCLUDE_VIEWS", "true"),
        schema_ttl_s=_env_int(
            "DATACHAT_SQL_SCHEMA_TTL_S", 3600, minimum=0, maximum=86400
        ),
        schema_max_chars=_env_int(
            "DATACHAT_SQL_SCHEMA_MAX_CHARS", 40000, minimum=500, maximum=200000
        ),
        schema_include_fks=_env_flag("DATACHAT_SQL_SCHEMA_INCLUDE_FKS", "true"),
        quote_identifiers=_env_flag("DATACHAT_SQL_QUOTE_IDENTIFIERS", "true"),
        schema_profile_values=_env_flag("DATACHAT_SQL_SCHEMA_PROFILE_VALUES", "true"),
        schema_profile_sample_rows=_env_int(
            "DATACHAT_SQL_SCHEMA_PROFILE_SAMPLE_ROWS", 50, minimum=1, maximum=1000
        ),
        schema_profile_max_values=_env_int(
            "DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUES", 3, minimum=1, maximum=20
        ),
        schema_profile_max_value_chars=_env_int(
            "DATACHAT_SQL_SCHEMA_PROFILE_MAX_VALUE_CHARS", 32, minimum=4, maximum=200
        ),
        pool_size=_env_int("DATACHAT_SQL_POOL_SIZE", 2, minimum=1, maximum=20),
        pool_max_overflow=_env_int(
            "DATACHAT_SQL_POOL_MAX_OVERFLOW", 3, minimum=0, maximum=20
        ),
        pool_timeout_s=_env_int(
            "DATACHAT_SQL_POOL_TIMEOUT_S", 5, minimum=1, maximum=120
        ),
        pool_recycle_s=_env_int(
            "DATACHAT_SQL_POOL_RECYCLE_S", 1800, minimum=60, maximum=86400
        ),
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config() -> AppConfig:
    """
    Read all configuration from environment variables and return an AppConfig.

    Required variables (no default):
        ENCRYPTION_KEY, PGUSER, PGPWD, PGHOST, PGDB,
        ADMIN_USERNAME, ADMIN_PASSWORD_HASH

    Required only when DATACHAT_SQL_ENABLED is truthy:
        DATACHAT_DB_HOST, DATACHAT_DB_NAME, DATACHAT_DB_USER,
        DATACHAT_DB_PASSWORD

    All other variables fall back to sensible defaults matching .env.example.

    Raises:
        ValueError: if one or more required variables are missing, listing all
                    of them in a single error message.
    """
    required = {
        "ENCRYPTION_KEY": os.environ.get("ENCRYPTION_KEY"),
        "PGUSER": os.environ.get("PGUSER"),
        "PGPWD": os.environ.get("PGPWD"),
        "PGHOST": os.environ.get("PGHOST"),
        "PGDB": os.environ.get("PGDB"),
        "ADMIN_USERNAME": os.environ.get("ADMIN_USERNAME"),
        "ADMIN_PASSWORD_HASH": os.environ.get("ADMIN_PASSWORD_HASH"),
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(
            "The following required environment variables are not set: "
            + ", ".join(missing)
        )

    database = DatabaseConfig(
        user=required["PGUSER"],  # type: ignore[arg-type]
        password=required["PGPWD"],  # type: ignore[arg-type]
        host=required["PGHOST"],  # type: ignore[arg-type]
        db=required["PGDB"],  # type: ignore[arg-type]
        port=os.environ.get("PGPORT", "5432"),
        schema=os.environ.get("MAUI_SCHEMA", "public"),
    )

    admin = AdminConfig(
        username=required["ADMIN_USERNAME"],  # type: ignore[arg-type]
        password_hash=required["ADMIN_PASSWORD_HASH"].encode("utf-8"),  # type: ignore[union-attr]
    )

    models = ModelConfig(
        datachat_model=os.environ.get(
            "DATACHAT_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"
        ),
        datachat_provider=os.environ.get("DATACHAT_PROVIDER", "Deepinfra"),
        prompt_model=os.environ.get("PROMPT_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        prompt_provider=os.environ.get("PROMPT_PROVIDER", "Deepinfra"),
        completion_model=os.environ.get("COMPLETION_MODEL", "google/gemma-3-4b-it"),
        completion_model_provider=os.environ.get(
            "COMPLETION_MODEL_PROVIDER", "Deepinfra"
        ),
        completion_model_agent_chat=os.environ.get(
            "COMPLETION_MODEL_AGENT_CHAT",
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        ),
        completion_embedding_model=os.environ.get(
            "COMPLETION_EMBEDDING_MODEL", "BAAI/bge-m3"
        ),
        completion_embedding_model_provider=os.environ.get(
            "COMPLETION_EMBEDDING_MODEL_PROVIDER", "Deepinfra"
        ),
        audio_model=os.environ.get("AUDIO_MODEL", "google/gemma-3-4b-it"),
        audio_provider=os.environ.get("AUDIO_PROVIDER", "Deepinfra"),
        asr_model=os.environ.get("ASR_MODEL", "openai/whisper-large-v3"),
        asr_provider=os.environ.get("ASR_PROVIDER", "Deepinfra"),
        vision_model=os.environ.get("VISION_MODEL", "google/gemma-3-4b-it"),
        vision_provider=os.environ.get("VISION_PROVIDER", "Deepinfra"),
        compare_docs_model=os.environ.get("COMPARE_DOCS_MODEL", "google/gemma-3-4b-it"),
        compare_docs_provider=os.environ.get("COMPARE_DOCS_PROVIDER", "Google"),
        asr_base_url=os.environ.get("ASR_BASE_URL") or None,
        asr_mistral_price_per_minute_usd=(
            float(os.environ["ASR_MISTRAL_PRICE_PER_MINUTE_USD"])
            if os.environ.get("ASR_MISTRAL_PRICE_PER_MINUTE_USD")
            else None
        ),
    )

    api_keys = ApiKeysConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
        deepinfra_api_key=os.environ.get("DEEPINFRA_API_KEY"),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        langchain_api_key=os.environ.get("LANGCHAIN_API_KEY"),
        langchain_project=os.environ.get("LANGCHAIN_PROJECT"),
        langchain_endpoint=os.environ.get(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        ),
        langchain_tracing_v2=os.environ.get("LANGCHAIN_TRACING_V2", "true"),
    )

    rag = RagConfig(
        top_k=int(os.environ.get("RAG_TOP_K", "3")),
        min_sim=float(os.environ.get("RAG_MIN_SIM", "0.5")),
        default_namespace=os.environ.get("RAG_DEFAULT_NAMESPACE", "Dino"),
    )

    datachat = DatachatConfig(
        engine=os.environ.get("DATACHAT_ENGINE", "smolagents"),
        max_steps=int(os.environ.get("DATACHAT_MAX_STEPS", "12")),
        rate_limit_per_min=int(os.environ.get("DATACHAT_RATE_LIMIT_PER_MIN", "0")),
        session_ttl_min=int(os.environ.get("DATACHAT_SESSION_TTL_MIN", "60")),
        log_level=os.environ.get("DATACHAT_LOG_LEVEL", "INFO"),
    )

    # A misconfigured SQL datasource disables itself; it does not stop the
    # application from booting. The CSV DataChat path, and every other feature,
    # work without it, and failing here would take them all down over a feature
    # that is off by default.
    try:
        datachat_sql = _load_datachat_sql_config()
    except ValueError as e:
        logger.error(
            "event=datachat_sql_config_invalid detail=%s",
            e,
        )
        datachat_sql = _load_datachat_sql_config(force_disabled=True)

    return AppConfig(
        encryption_key=required["ENCRYPTION_KEY"],  # type: ignore[arg-type]
        database=database,
        admin=admin,
        models=models,
        api_keys=api_keys,
        rag=rag,
        datachat=datachat,
        datachat_sql=datachat_sql,
        auth_gateway_url=os.environ.get(
            "AUTH_GATEWAY_URL", "http://localhost:3000/validate"
        ),
        stripe_key=os.environ.get("STRIPE_SK_KEY"),
        datachat_token_cost=int(os.environ.get("DATACHAT_TOKEN_COST", "1")),
        completion_token_cost=int(os.environ.get("COMPLETION_TOKEN_COST", "1")),
        prompt_token_cost=int(os.environ.get("PROMPT_TOKEN_COST", "1")),
        audio_form_token_cost=int(os.environ.get("AUDIO_FORM_TOKEN_COST", "1")),
        compare_docs_token_cost=int(os.environ.get("COMPARE_DOCS_TOKEN_COST", "1")),
        dino_legacy_usage_username=os.environ.get("DINO_LEGACY_USAGE_USERNAME")
        or None,
        admin_rag_usage_username=os.environ.get("ADMIN_RAG_USAGE_USERNAME") or None,
    )
