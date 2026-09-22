# Pandino — Developer Documentation

Comprehensive guide for developers who need to understand, extend, run, or deploy
the Pandino Flask application.

> **Pandino** is a multi-tenant LLM gateway / backend that exposes a unified HTTP API
> for: conversational data analysis (CSV → questions), Retrieval-Augmented Generation
> (RAG) over ingested documents, an agentic "AI tutor" endpoint, document comparison,
> audio transcription / audio-to-form, and an admin panel for prompt / user / cost /
> logging management.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Project Structure](#2-project-structure)
3. [Prerequisites & Local Setup](#3-prerequisites--local-setup)
4. [Configuration System (`config.py`)](#4-configuration-system-configpy)
5. [Database Layer](#5-database-layer)
6. [Running the Application](#6-running-the-application)
7. [Layered Architecture In Depth](#7-layered-architecture-in-depth)
8. [HTTP API Reference](#8-http-api-reference)
9. [Authentication, Users & Tokens](#9-authentication-users--tokens)
10. [RAG / Vector Store](#10-rag--vector-store)
11. [DataChat Engine](#11-datachat-engine)
12. [Prompt Management (DB-driven)](#12-prompt-management-db-driven)
13. [Admin Panel](#13-admin-panel)
14. [Adding a New Endpoint (Step-by-Step)](#14-adding-a-new-endpoint-step-by-step)
15. [Testing](#15-testing)
16. [Deployment & CI/CD](#16-deployment--cicd)
17. [Conventions & Style](#17-conventions--style)

---

## 1. High-Level Architecture

Pandino is a **Flask 3** application built around a strict layered architecture:

```
                 ┌─────────────────────────────────────────────┐
   HTTP  ──────► │  routes/        Flask Blueprints (thin HTTP) │
                 └──────────────────────┬──────────────────────┘
                                        │  (validation, auth, token accounting)
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │  services/      Business / orchestration     │
                 └──────────────────────┬──────────────────────┘
                                        │  (pure logic, no Flask)
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │  infrastructure/  DB, vector store, AI, auth  │
                 │  datachat/        Data-analysis agent engine  │
                 │  llm/             LiteLLM model factory       │
                 │  utils/           Logging / serialization     │
                 └─────────────────────────────────────────────┘
```

**Core principles**

- **Routes are thin.** They validate headers, enforce auth + token budgets, delegate
  to a service, then log token usage. No business logic lives here.
- **Services are pure.** `services/*` contain orchestration with no Flask dependency,
  so they are unit-testable in isolation.
- **Infrastructure holds side-effects.** PostgreSQL access (`infrastructure/database_pg.py`),
  the PGVector store (`infrastructure/vector_store.py`), LLM/embedding factories
  (`infrastructure/ai.py`, `llm/litellm_factory.py`) and authentication gateways
  (`infrastructure/dino.py`, `infrastructure/external_auth.py`).
- **Configuration is centralized at the app boundary.** `config.load_config()`
  produces an immutable `AppConfig` dataclass that is attached to the Flask app
  (`app.config["MAUI_CONFIG"]`). A few infrastructure/admin paths keep explicit
  `os.environ` fallbacks for provider keys and runtime display.
- **Prompts are data.** Prompt templates live in code as defaults but can be
  overridden at runtime from the `prompts` DB table (see
  [§12](#12-prompt-management-db-driven)).

The application entry point is [`main.py`](main.py), which wires the 10 Blueprints,
initializes the DB and vector-store layers, configures logging, and starts the server.

---

## 2. Project Structure

```
pandino/
├── main.py                     # App entry point: creates Flask app, registers blueprints
├── config.py                   # AppConfig dataclasses + load_config() (single source of env reads)
├── requirements.txt            # Pinned dependencies
├── Dockerfile                  # Production image (gunicorn + gevent)
├── .env.example                # Template for environment variables
├── .env.variants               # Alternative model/provider presets
├── .python-version             # 3.10.13
│
├── routes/                     # HTTP layer (Flask Blueprints) — thin controllers
│   ├── system.py               #   /health, /, stub endpoints
│   ├── auth.py                 #   /checkpandinouser, /validateapikey
│   ├── users.py                #   /edittokens, /getusertokens, /feedback, /buyreport
│   ├── reporting.py            #   /prompt.txt
│   ├── documents.py            #   /compare_docs
│   ├── multimodal.py           #   /transcribe, /audioformcompilation
│   ├── ingestion.py            #   /storeragfile
│   ├── rag.py                  #   /completion.json, /agentchat
│   ├── datachat.py             #   /startdatachat, /datachat, /enddatachat
│   ├── admin.py                #   /admin/* (web UI)
│   └── utils.py                #   assert_valid_api_key() shared helper
│
├── services/                   # Business logic (Flask-free, unit-testable)
│   ├── completion_service.py   #   RAG chat completion (LangChain)
│   ├── agentchat_service.py    #   Smolagents CodeAgent "Compass AI Tutor"
│   ├── rag_ingestion_service.py#   File → chunks → embeddings (PDF/TXT/MD/audio/image)
│   ├── prompt_service.py       #   Plain prompt → LLM reply (reporting)
│   ├── audio_form_service.py   #   Transcribed audio → JSON form filling
│   ├── retrieval_service.py    #   Centralized vector retrieval
│   ├── document_text_service.py#   Local extraction: PDF/DOCX/RTF/TXT → text
│   ├── document_extraction_service.py # Local extraction + OCR fallback orchestration
│   ├── document_ocr_service.py #   PDF page → PNG rendering (provider-independent)
│   └── document_comparison_service.py # Multi-doc LLM comparison → JSON {score,summary,reasoning}
│
├── infrastructure/             # Side-effecting adapters
│   ├── database_pg.py          #   PostgreSQL access + Fernet-encrypted API keys
│   ├── database_methods.py     #   Parameterized SQL builders (psycopg.sql, injection-safe)
│   ├── vector_store.py         #   PGVector store wrapper (MauiVectorStore)
│   ├── ai.py                   #   choose_llm() / choose_emb_model() / vision / asr
│   ├── agent_manager.py        #   In-memory dict of active DataChat engines
│   ├── retriever_tool.py       #   Smolagents Tool wrapping retrieval_service
│   ├── prompt_utils.py         #   load_prompt() / render_prompt() (DB → default → env)
│   ├── dino.py                 #   Dino GraphQL auth
│   ├── external_auth.py        #   Generic auth-gateway validation
│   └── file_manager.py         #   base64 / image-path helpers
│
├── datachat/                   # Conversational CSV analysis engine
│   ├── engine_interface.py     #   DataChatEngine Protocol + bootstrap result
│   ├── engine_factory.py       #   create_engine() dispatcher (currently "smolagents")
│   ├── smolagents_engine.py    #   Smolagents CodeAgent implementation
│   ├── dataset_loader.py       #   CSV → pandas DataFrame
│   ├── bootstrap.py            #   LLM-driven bootstrap prompt builder
│   ├── bootstrap_static.py     #   Localized static HTML bootstrap (IT/EN/FR/ES)
│   ├── output_normalizer.py    #   Engine output → stable Dino response schema
│   ├── engine_output_adapter.py#   Coerce raw outputs into {kind,...} contract
│   ├── sql_guard.py            #   sqlglot validation of LLM-authored SQL
│   ├── sql_datasource.py       #   Read-only SQLAlchemy engine (optional datasource)
│   ├── schema_snapshot_loader.py #  Reflect the SQL schema once, cache, render for the prompt
│   └── tools/                  #   11 pandas-backed tools + 1 optional SQL tool
│       ├── aggregate_tool.py describe_tool.py missing_values_tool.py
│       ├── correlation_tool.py sample_rows_tool.py top_rows_tool.py
│       ├── filter_rows_tool.py row_count_tool.py plot_tool.py
│       ├── trend_tool.py unique_values_tool.py
│       ├── sql_engine_tool.py sql_tool_utils.py
│
├── llm/
│   └── litellm_factory.py      #   build_litellm_model() for Smolagents
│
├── utils/
│   ├── runtime_logging.py      #   datachat.runtime logger (stdout)
│   ├── agent_logging.py        #   Structured JSON logger → logs/agent_runs.log
│   └── agent_serialization.py  #   Smolagents RunResult → JSON payload
│
├── templates/admin/            # Jinja2 templates for the admin web UI
│   ├── base.html login.html dashboard.html users.html edit_user.html
│   ├── logs.html feedback.html prompts.html edit_prompt.html
│   ├── costs.html edit_cost.html rag_files.html
│   └── api_docs.html           #   Swagger UI page (renders project_docs/openapi.yaml)
│
├── project_docs/
│   ├── openapi.yaml            # Hand-maintained OpenAPI 3.0 spec (served in admin panel)
│   └── auth-flow.md            # Mermaid diagrams of the auth + endpoint-usage flow
│
├── docs/                       # Local ignored workspace for notes / Codex analyses
│
├── tests/                      # pytest suite
└── .github/workflows/          # CI: build_and_push.yml (multi-arch Docker)
```

---

## 3. Prerequisites & Local Setup

### Requirements

- **Python 3.10+** (`.python-version` pins `3.10.13`)
- **PostgreSQL 14+** with the **`pgvector`** extension enabled
- An **LLM provider account** with at least one API key (DeepInfra is the default;
  OpenAI, Anthropic, Google, Mistral, Groq, Deepseek, OpenRouter, Ollama are supported)
- A **Fernet-compatible `ENCRYPTION_KEY`** (used to encrypt API keys at rest)

### Step-by-step

```bash
# 1. Clone
git clone git@github.com:tulas75/pandino.git
cd pandino

# 2. Create a virtualenv (3.10+)
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate a Fernet key for ENCRYPTION_KEY
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# 5. Generate the admin password hash (bcrypt)
python -c "import bcrypt; print(bcrypt.hashpw(b'my-strong-password', bcrypt.gensalt()).decode())"

# 6. Create the PostgreSQL database & enable pgvector
createdb pandino
psql pandino -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 7. Copy and fill in environment variables
cp .env.example .env
#   → fill in ENCRYPTION_KEY, PGUSER/PGPWD/PGHOST/PGDB, ADMIN_*,
#     ADMIN_PASSWORD_HASH, and at least one provider API key.

# 8. Initialize the relational schema (from the repository root)
python3 -m infrastructure.database_pg init_db
#   → must print "Database initialized successfully."; see §5.

# 9. Run the app (from the repository root, with the virtualenv activated)
make run-local          # → http://127.0.0.1:5000
```

> **Note:** the `load_dotenv()` call in `main.py` reads `.env` automatically, so
> exports are optional in development.

---

## 4. Configuration System (`config.py`)

Application configuration is loaded through `load_config()` (`config.py:156`).
Importing `config.py` has **no side effects**; the main `AppConfig` is created when
`load_config()` is called from `main.py`.

Known direct env reads still exist and are intentional:

- `infrastructure/ai.py` and `llm/litellm_factory.py` can fall back to provider
  API-key env vars when a caller does not pass an explicit key.
- `routes/admin.py` reads `.env` or `os.environ` for the dashboard environment view,
  filtered through explicit allowlists.
- DataChat internals read selected `DATACHAT_*` variables directly.

### Required variables (no defaults — app refuses to start if missing)

| Variable                            | Purpose                                          |
| ----------------------------------- | ------------------------------------------------ |
| `ENCRYPTION_KEY`                    | Fernet key used to encrypt user API keys at rest |
| `PGUSER`, `PGPWD`, `PGHOST`, `PGDB` | PostgreSQL credentials                           |
| `ADMIN_USERNAME`                    | Admin panel login username                       |
| `ADMIN_PASSWORD_HASH`               | bcrypt hash of the admin password                |

### Optional variables (with defaults)

| Group               | Variables                                                                                                                | Default                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **DB**              | `PGPORT`, `MAUI_SCHEMA`                                                                                                  | `5432`, `public`                                       |
| **Models**          | `DATACHAT_MODEL/PROVIDER`, `PROMPT_*`, `COMPLETION_*`, `AUDIO_*`, `ASR_MODEL`, `VISION_*`, `COMPARE_DOCS_*`              | see `.env.example` (DeepInfra / Qwen / Gemma defaults) |
| **Token costs**     | `DATACHAT_TOKEN_COST`, `COMPLETION_TOKEN_COST`, `PROMPT_TOKEN_COST`, `AUDIO_FORM_TOKEN_COST`, `COMPARE_DOCS_TOKEN_COST`  | `1`                                                    |
| **RAG**             | `RAG_TOP_K`, `RAG_MIN_SIM`, `RAG_DEFAULT_NAMESPACE`                                                                      | `3`, `0.5`, `Dino`                                     |
| **DataChat engine** | `DATACHAT_ENGINE`, `DATACHAT_MAX_STEPS`, `DATACHAT_RATE_LIMIT_PER_MIN`, `DATACHAT_SESSION_TTL_MIN`, `DATACHAT_LOG_LEVEL` | `smolagents`, `12`, `0`, `60`, `INFO`                  |
| **DataChat SQL**    | `DATACHAT_SQL_ENABLED`, `DATACHAT_DB_PORT/SCHEMA`, `DATACHAT_SQL_MAX_ROWS/MAX_COLUMNS/MAX_CELL_CHARS`, `DATACHAT_SQL_STATEMENT_TIMEOUT_MS`, `DATACHAT_SQL_INCLUDE_VIEWS` (also gates view reflection), `DATACHAT_SQL_ALLOWED_TABLES/DENIED_TABLES` (tables, views and matviews alike), `DATACHAT_SQL_SCHEMA_TTL_S/SCHEMA_MAX_CHARS/SCHEMA_INCLUDE_FKS`, `DATACHAT_SQL_SCHEMA_PROFILE_VALUES/PROFILE_SAMPLE_ROWS/PROFILE_MAX_VALUES/PROFILE_MAX_VALUE_CHARS`, `DATACHAT_SQL_QUOTE_IDENTIFIERS` | `false`, `5432`/`public`, `200`/`25`/`300`, `10000`, `true`, empty, `3600`/`40000`/`true`, `true`/`50`/`3`/`32`, `true` |
| **Auth**            | `AUTH_GATEWAY_URL`, `STRIPE_SK_KEY`                                                                                      | `http://localhost:3000/validate`, `None`               |

`DATACHAT_DB_HOST`, `DATACHAT_DB_NAME`, `DATACHAT_DB_USER` and `DATACHAT_DB_PASSWORD`
have **no defaults and never fall back to the `PG*` application database**: they are
required — and the app refuses to start — when `DATACHAT_SQL_ENABLED` is truthy, and
ignored otherwise. See §11 "SQL datasource".

### How to read config inside a route/service

```python
from flask import current_app
config = current_app.config["MAUI_CONFIG"]
config.models.completion_model          # str
config.rag.top_k                        # int
config.datachat_token_cost              # int
```

The resulting `AppConfig` is a frozen dataclass composed of sub-configs:
`DatabaseConfig`, `AdminConfig`, `ModelConfig`, `ApiKeysConfig`, `RagConfig`,
`DatachatConfig` (`config.py:128`).

### Provider → env-var map

`config.PROVIDER_API_KEY_MAP` (`config.py:18`) maps a provider name (e.g.
`"Deepinfra"`) to the env var holding its key (e.g. `DEEPINFRA_API_KEY`). This is
used by both `infrastructure/ai.py` (LangChain clients) and
`llm/litellm_factory.py` (Smolagents `LiteLLMModel`).

### Secrets and local notes

- Keep real secrets in `.env` or your orchestrator secret store; `.env` is ignored.
- The admin dashboard may show safe config values, but known secrets are rendered
  only as `configured` / `not set`.
- `project_docs/` is the versioned home for maintained project documentation such as
  [`project_docs/openapi.yaml`](project_docs/openapi.yaml) and
  [`project_docs/auth-flow.md`](project_docs/auth-flow.md).
- `/docs` is ignored by `.gitignore` and should be treated as a local workspace for
  notes, investigations, and Codex analyses, not as the API documentation source.

---

## 5. Database Layer

### Relational schema

Created by `database_pg.init_db()` (`infrastructure/database_pg.py:107`):

| Table       | Purpose                                                                                         |
| ----------- | ----------------------------------------------------------------------------------------------- |
| `users`     | `id, username, api_key (encrypted), date_valid_until, tokens (≥0)`                              |
| `logs`      | Per-request token accounting: `user_id, date, token_input, token_output, cost, model, provider` |
| `costs`     | Time-bounded input/output token pricing per model+provider (used to compute `logs.cost`)        |
| `prompts`   | `title, version, message` — DB-overridable prompt templates                                     |
| `feedback`  | User thumbs-up/down on answers, optionally linked to `logs.id`                                  |
| `rag_files` | Tracking of ingested documents (`file_id`, namespace, chunk_count, language)                    |

> **Plus** one PGVector table **per namespace** (e.g. `dino`, `farm`) created
> lazily by `ensure_pgvector_namespace_ready()` (`infrastructure/vector_store.py:57`).

### API-key encryption

User API keys are encrypted with Fernet **before** storage and decrypted on read
(`database_pg.get_cipher_suite()`). The `ENCRYPTION_KEY` env var is normalized into
a URL-safe Fernet key at `init()` time. Never store or log plaintext keys.

### Query building

All SQL is built via `infrastructure/database_methods.py`, which returns
`(psycopg.sql.Composed, params)` tuples. Identifiers use `sql.Identifier` and
values use `%s` placeholders — **this is how SQL-injection safety is enforced**.
When adding a query, always add a `build_*_query()` helper here rather than
string-formatting SQL inside `database_pg.py`.

### Second datasource: `datachat/sql_datasource.py`

There are two independent database layers, and they never share a connection:

| | `infrastructure/database_pg.py` | `datachat/sql_datasource.py` |
| --- | --- | --- |
| Driver | psycopg 3 directly | SQLAlchemy 2.0 over psycopg 3 |
| Access | read-write | read-only, enforced at four layers |
| Database | the application DB (`PG*`) | a dedicated DB (`DATACHAT_DB_*`) |
| Reached by | routes and services | the DataChat agent, when enabled |

This is the first first-party use of `sqlalchemy.create_engine` in maui (SQLAlchemy
was previously only a transitive dependency of `langchain_postgres`). See §11
"SQL datasource" for the read-only guarantees.

### Database CLI

`database_pg.py` is also runnable as a CLI, for user management and for the
governed additive schema changes.

#### Supported invocation

Run it **as a module, from the repository root**:

```bash
python3 -m infrastructure.database_pg <command>
```

Use whichever interpreter the target instance is meant to run — the venv's
`python`, `python3`, or the container's interpreter; `python3` above is only a
placeholder for it. What matters is the **module form** and the **working
directory**.

`infrastructure/database_pg.py` imports top-level packages (`from config import
…`, `from infrastructure.database_methods import …`, `database_pg.py:15-16`).
Invoking it by file path puts `infrastructure/` on `sys.path` instead of the
repository root, so the import fails immediately with
`ModuleNotFoundError: No module named 'config'`. The module form is therefore the
only supported invocation.

#### Commands

Exactly these commands are accepted, per `_resolve_cli_command()`
(`database_pg.py:2448`) and `print_help()` (`database_pg.py:2424`):

| Command                                     | Arguments               | Purpose                                          |
| ------------------------------------------- | ----------------------- | ------------------------------------------------ |
| `init_db`                                   | –                       | Create any missing tables (see below)            |
| `add_user`                                  | `<username> <api_key>`  | Create a user                                    |
| `remove_user`                               | `<username>`            | Delete a user                                    |
| `get_user_by_username`                      | `<username>`            | Print one user                                   |
| `edit_tokens`                               | `<username> <quantity>` | Add/remove tokens                                |
| `list_users`                                | –                       | List all users                                   |
| `print_keys`                                | –                       | Print stored API keys                            |
| `add_usage_service_column`                  | –                       | Add `logs.service` if missing                    |
| `add_usage_request_id_column`               | –                       | Add `logs.request_id` if missing                 |
| `add_usage_duration_ms_column`              | –                       | Add `logs.duration_ms` if missing                |
| `add_user_client_column`                    | –                       | Add `users.client` if missing                    |
| `add_usage_source_column`                   | –                       | Add `logs.source` if missing                     |
| `add_usage_embedding_operation_kind_column` | –                       | Add `logs.embedding_operation_kind` if missing   |
| `add_usage_quantity_origin_column`          | –                       | Add `logs.quantity_origin` if missing            |
| `add_usage_cost_origin_column`              | –                       | Add `logs.cost_origin` if missing                |

> ⚠️ `init()` must run before any DB function. `run_cli()` wires this up via
> `load_dotenv()` + `load_config()` for you — but only for a command it
> recognises with the right argument count.

#### Read the output: a zero exit status is not success

Two current behaviours make the printed output, not the exit status, the signal:

- **An unrecognised command, or the wrong number of arguments, prints the help
  text and returns normally** — exit status `0`, nothing done, no database
  contacted (`run_cli()`, `database_pg.py:2500`; pinned by
  `tests/test_database_cli.py::test_unknown_command_shows_help_without_initializing`
  and `…::test_invalid_argument_count_shows_help_without_initializing`). A
  mistyped migration name is a silent no-op that looks like a success.
- **`init_db` catches every exception, rolls back and prints it** rather than
  re-raising (`database_pg.py:201-211`), so it exits `0` even when it did
  nothing. Confirm it printed `Database initialized successfully.` and not
  `An error occurred: …`.

The `add_*_column` commands are the exception: each raises `RuntimeError` when
the change was not committed (`database_pg.py:1043-1249`), so a genuine failure
there *does* surface as a non-zero exit.

### Initializing a new database

On a database with none of these tables, `init_db` creates the full
repository-defined schema in one go — `users`, `logs`, `costs`, `prompts`,
`feedback`, `rag_files` and `operational_events` (`init_db()`,
`database_pg.py:121-198`; the resulting shape is pinned by
`tests/test_database_schema_fresh.py`):

```bash
python3 -m infrastructure.database_pg init_db
```

No `add_*_column` command is needed afterwards: a table `init_db` has just
created already has every column.

### Upgrading an existing database

`init_db` is written entirely as `CREATE TABLE IF NOT EXISTS`. On a database
that already has a table, that statement is a **no-op for that table** — it
does not add columns to it. Re-running `init_db` on a long-lived instance
therefore creates only whole tables that are absent (this is how
`operational_events` arrives on an existing instance) and leaves every
pre-existing table exactly as it was.

The columns added to `logs` and `users` since those tables were first created
are applied by the governed `add_*_column` commands, one column each:

```bash
python3 -m infrastructure.database_pg init_db   # creates operational_events if absent

python3 -m infrastructure.database_pg add_usage_service_column
python3 -m infrastructure.database_pg add_usage_request_id_column
python3 -m infrastructure.database_pg add_usage_duration_ms_column
python3 -m infrastructure.database_pg add_user_client_column
python3 -m infrastructure.database_pg add_usage_source_column
python3 -m infrastructure.database_pg add_usage_embedding_operation_kind_column
python3 -m infrastructure.database_pg add_usage_quantity_origin_column
python3 -m infrastructure.database_pg add_usage_cost_origin_column
```

That is the complete set: these eight are the only `add_column_if_missing()`
call sites in the module, covering `logs.service`, `logs.request_id`,
`logs.duration_ms`, `logs.source`, `logs.embedding_operation_kind`,
`logs.quantity_origin`, `logs.cost_origin` and `users.client`.

The listed order is the order the commands are declared in the source and in
`print_help()`. Each command targets one fixed table and column, is
independently idempotent, and no source or test establishes a dependency
between them, so the order above is a convention for reproducibility rather
than a constraint.

Each command is safe to re-run. `add_column_if_missing()`
(`database_pg.py:965`) fails closed: it executes `ALTER TABLE` only once the
column's absence has been positively established, verifies the result before
committing, and rolls back otherwise. Per command you will see one of:

- `<table>.<column> added.` — the column was created,
- `<table>.<column> already present, no change needed.` — nothing was done,
- a `RuntimeError` and a non-zero exit — inspection, DDL or post-DDL
  verification failed; do not treat the run as complete.

### What these checks do and do not prove

The existence checks behind `init_db` and every `add_*_column` command match on
**names only**:

- `init_db` relies on `CREATE TABLE IF NOT EXISTS`, which keys on the table
  name.
- `add_column_if_missing()` calls `build_check_column_exists_query()`
  (`database_methods.py:170`), a `SELECT 1 FROM information_schema.columns
  WHERE table_schema = %s AND table_name = %s AND column_name = %s`.

So `already present, no change needed.` means **a column of that name exists on
that table** — and nothing more. It is not evidence about that column's data
type, nullability, default, constraints, indexes, or collation, and a table
reported as existing is not thereby verified against the definition in
`init_db`. Nor does a clean run of every command above establish that a
long-lived database matches the repository-defined fresh schema: columns,
constraints, defaults or indexes that drifted, or that were added outside these
commands, are invisible to a name-only check and to `CREATE TABLE IF NOT
EXISTS`.

Confirming that an existing database actually conforms requires inspecting that
database's real schema directly, against `init_db()` as the reference. The
repository provides no command or test that performs that comparison — see
[§16](#16-deployment--cicd).

---

## 6. Running the Application

### Local development

From the repository root, after activating your intended virtualenv:

```bash
make run-local      # → http://127.0.0.1:5000
```

The `run-local` target in the root `Makefile` runs:

```bash
LOG_LEVEL=INFO gunicorn main:app -k gevent --workers 1 --worker-connections 10 \
           --timeout 300 --bind 127.0.0.1:5000
```

`127.0.0.1:5000` is the local address: the server is reachable only from your own
machine. This is a convenience for starting the app; it is not deployment
configuration, and it does not change Gunicorn's shutdown behaviour — Ctrl+C still
behaves as Gunicorn normally does, tracebacks included.

**Why Gunicorn/gevent and `LOG_LEVEL=INFO` locally.** Operational Persistence
records operational events as they are emitted while the app serves requests, so
what you see locally is only trustworthy if the app runs the same way it does when
deployed. The Flask dev server differs on both points that matter here: it does not
use the gevent worker that carries concurrent, long-running agent calls, and its
reloader/debug behaviour can restart or duplicate the process underneath a run.
Running the same single gevent worker as the `Dockerfile` keeps the concurrency
model identical, and `LOG_LEVEL=INFO` keeps the operational events visible —
at a coarser level they are filtered out and a check can look clean simply because
nothing was recorded.

The Flask dev server is still available with `python main.py` (`main.py:83`,
`debug=True`, port 5000) when you want the reloader and do not need Operational
Persistence checks. Either entry point sets `MPLBACKEND=Agg` (headless matplotlib),
relaxes pandas display limits, and initializes the `datachat.runtime` +
`agent_runs` loggers.

### Production (Docker / gunicorn)

The `Dockerfile` runs:

```bash
gunicorn main:app -k gevent --workers 1 --worker-connections 10 \
           --timeout 300 --bind 0.0.0.0:5000
```

One gevent worker handles concurrent long-running LLM calls; the 300 s timeout
accommodates slow agent runs. Build & run:

```bash
docker build -t pandino .
docker run -p 5000:5000 --env-file .env pandino
```

### CORS

`CORS(app)` is enabled globally (`main.py:57`) — all origins are allowed. To lock
this down, pass `origins=[...]` instead.

### Secrets requirement

`main.py:59` raises at startup if `ENCRYPTION_KEY` is unset, and assigns it to
`app.secret_key` (used for admin session cookies).

---

## 7. Layered Architecture In Depth

### 7.1 Routes (`routes/`)

Each file defines one `Blueprint` registered in `main.py`. A typical route follows
this contract:

1. Read headers (`X-API-KEY`, `X-USER-EMAIL`, optionally `X-USER-NAME`).
2. `assert_valid_api_key(api_key, user_email)` — aborts 403 if invalid/expired
   (`routes/utils.py:7`).
3. Fetch `get_user_tokens(user_email)` and compare against the operation's
   `*_TOKEN_COST`.
4. Delegate to a `services/*` function.
5. `log_token_usage(...)` (best-effort) and `edit_tokens(user, -cost)`.
6. Return JSON.

Blueprints:

| Blueprint       | File                   | Prefix   |
| --------------- | ---------------------- | -------- |
| `system_bp`     | `routes/system.py`     | `/`      |
| `auth_bp`       | `routes/auth.py`       | `/`      |
| `users_bp`      | `routes/users.py`      | `/`      |
| `reporting_bp`  | `routes/reporting.py`  | `/`      |
| `documents_bp`  | `routes/documents.py`  | `/`      |
| `multimodal_bp` | `routes/multimodal.py` | `/`      |
| `ingestion_bp`  | `routes/ingestion.py`  | `/`      |
| `rag_bp`        | `routes/rag.py`        | `/`      |
| `datachat_bp`   | `routes/datachat.py`   | `/`      |
| `admin_bp`      | `routes/admin.py`      | `/admin` |

### 7.2 Services (`services/`)

Pure-Python orchestration. Key services:

- **`completion_service.complete_chat()`** — classic RAG: retrieves vectors, builds a
  LangChain message list, invokes the model, detects "no information" fallbacks.
- **`agentchat_service.run_agentchat()`** — builds a Smolagents `CodeAgent` with a
  single `RetrieverTool`, enforces a JSON `{answer, follow_ups}` output, serializes
  the run via `utils/agent_serialization.py`.
- **`rag_ingestion_service.process_rag_file()`** — the ingestion pipeline. Dispatches
  on mimetype (text/markdown/pdf/audio/image), chunks with LangChain splitters
  (`chunk_size=900, overlap=100`), embeds, and stores in PGVector; also writes a
  `rag_files` tracking row.
- **`document_comparison_service.compare_documents()`** — coerces a strict
  `{score(1-100), summary, reasoning}` JSON contract out of the LLM.
- **`document_extraction_service.extract_document_text()`** — local extraction first,
  then OCR fallback for scanned PDFs (`MIN_EXTRACTED_TEXT_CHARS = 50`).

### 7.3 Infrastructure (`infrastructure/`)

- **`ai.py`** — `choose_llm(provider, model, ...)` returns a LangChain
  `BaseChatModel` (Groq/OpenAI/Mistral/Google/Anthropic/Deepseek/Deepinfra/Together/
  OpenRouter/Ollama/Llama.cpp). `choose_emb_model()` returns embeddings.
  `describe_image()`, `extract_text_from_image()` (OCR), and `asr_response()`
  live here too.
- **`vector_store.py`** — `MauiVectorStore` wraps `langchain_postgres.PGVectorStore`.
  Similarity is computed as `1 - score` and filtered by `min_similarity`.
  Deduplication uses a deterministic `maui_id = f"{namespace}:{sha256(text)}"`.
- **`agent_manager.py`** — keeps active DataChat engines in an **in-memory** dict
  keyed by API key (`activeEngines`). This means sessions are **per-process** —
  relevant when scaling horizontally (see [§16](#16-deployment--cicd)).
- **`prompt_utils.py`** — `load_prompt(title, default_text=...)` resolution order:
  **DB → in-code default → env var → ""**. `render_prompt(template, **kwargs)` substitutes
  strict `{identifier}` placeholders in one regex pass; every other brace stays
  literal.

### 7.4 DataChat (`datachat/`)

See [§11](#11-datachat-engine).

### 7.5 Logging

- `utils/agent_logging.py` → `logs/agent_runs.log` (one JSON record per agent run:
  user, namespace, steps, tool_calls, token_usage, vectors_count, answer_excerpt).
- `utils/runtime_logging.py` → `datachat.runtime` logger to stdout (controlled by
  `DATACHAT_LOG_LEVEL`).
- Standard Flask logging for request errors.

---

## 8. HTTP API Reference

All protected endpoints validate the `X-API-KEY` header against
`X-USER-EMAIL` (or `username` in the body). Most deduct a token cost on success.

> **Interactive docs.** An admin-only Swagger UI is available at **`/admin/api-docs`**,
> backed by [`project_docs/openapi.yaml`](project_docs/openapi.yaml) (served as JSON via
> `/admin/openapi.json`). That spec is hand-maintained and mirrors this section — keep
> the two in sync when the API changes.

### System

| Method | Path      | Auth | Description       |
| ------ | --------- | ---- | ----------------- |
| `GET`  | `/`       | –    | Welcome string    |
| `GET`  | `/health` | –    | `{"status":"ok"}` |

### Authentication & users

| Method | Path                | Headers                                                                | Body                                                       | Description                                                                                                                                                                        |
| ------ | ------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST` | `/checkpandinouser` | `X-AUTH-TOKEN`, `X-USER-EMAIL`, `X-CLIENT`, (`X-GRAPHQL-URL` for Dino) | –                                                          | Validates the user against the external auth gateway (or Dino GraphQL), creates the user in Pandino if missing. Returns `{response:{user:{user_email, api_key, expiration_date}}}` |
| `POST` | `/validateapikey`   | `X-API-KEY`, `X-USER-EMAIL`                                            | –                                                          | 200 if the key is valid & unexpired, else 403                                                                                                                                      |
| `POST` | `/edittokens`       | `X-STRIPE-KEY` (must equal `STRIPE_SK_KEY`)                            | `{quantity, useremail}`                                    | Adds/removes tokens (Stripe-webhook style)                                                                                                                                         |
| `POST` | `/getusertokens`    | `X-API-KEY`, `X-USER-EMAIL`                                            | –                                                          | `{response:{tokens:N}}`                                                                                                                                                            |
| `POST` | `/feedback`         | `X-API-KEY`                                                            | `{username, question, answer, feedback, log_id?, source?}` | Stores positive/negative feedback; `feedback` ∈ {`positive`,`negative`}                                                                                                            |
| `POST` | `/buyreport`        | `X-API-KEY`, `X-USER-EMAIL`                                            | `{prompts:int}`                                            | Deducts `prompts` tokens                                                                                                                                                           |

### Reporting & documents

| Method | Path            | Headers                     | Body                                                                                                           | Returns                                                                     |
| ------ | --------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `POST` | `/prompt.txt`   | `X-API-KEY`                 | form: `prompt`, `username`, `language?`                                                                        | Plain-text LLM reply (content-type `text/plain`)                            |
| `POST` | `/compare_docs` | `X-API-KEY`, `X-USER-EMAIL` | multipart: `prompt`, `files[]`, `text_documents`(JSON), `file_roles`(JSON), `additional_context?`, `language?` | `{score, summary, reasoning}` JSON. Needs ≥2 documents (files and/or text). |

### Multimodal

| Method | Path                    | Headers                                     | Body                                              | Returns                                                                                        |
| ------ | ----------------------- | ------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `POST` | `/transcribe`           | `X-API-KEY`, `X-USER-EMAIL`, `X-USER-NAME`¹ | multipart `file` + `lang?`                        | Audio → Asr transcription; image → vision description; PDF/DOCX/RTF → extracted text. `{text}` |
| `POST` | `/audioformcompilation` | `X-API-KEY`, `X-USER-EMAIL`                 | `{name, exampledata, choices?, transcribedAudio}` | JSON object matching the supplied form schema                                                  |

> ¹ `/transcribe` enforces `X-USER-NAME` (400 if missing) but the handler never actually
> uses it (`routes/multimodal.py`) — unlike the DataChat endpoints, where `X-USER-NAME`
> keys the in-memory agent. It is documented as required only because the code requires it.

### RAG / Ingestion

| Method | Path               | Headers                   | Body                                                                                                 | Returns                                                                            |
| ------ | ------------------ | ------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `POST` | `/storeragfile`    | (Dino auth or `X-CLIENT`) | multipart: `file`, `url`, `namespace?`, `language?`, `authToken`, `graphqlUrl`/`userEmail`, `client` | `{status, file_id, namespace, chunk_count, language, tracking_saved}`              |
| `POST` | `/completion.json` | `X-API-KEY`               | `{chat, username, namespace?, language?, info?}`                                                     | `{answer, vectors, log_id?}` — classic RAG completion                              |
| `POST` | `/agentchat`       | `X-API-KEY`               | `{chat:[...], username, namespace?, language?}`                                                      | `{answer, follow_ups, vectors, tool_calls, metrics, debug, log_id?}` — agentic RAG |

Admin users can also manage indexed files at `/admin/rag-files`. Uploads reuse
`process_rag_file()`. Deletes submit `file_id` plus namespace to
`database_pg.delete_rag_file()`, which validates the normalized namespace before
deleting chunks from the namespace PGVector table and the `rag_files` tracking row
in one transaction.

### DataChat (conversational CSV analysis)

A session is: **start → N× chat → end**.

| Method | Path             | Headers                                    | Body                                                         | Returns                                             |
| ------ | ---------------- | ------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- |
| `POST` | `/startdatachat` | `X-API-KEY`, `X-USER-EMAIL`, `X-USER-NAME` | multipart: `file` (CSV — required unless the SQL datasource is enabled), `model_name?`, `llm_type?`, `lang?` | `{Agent active:"active", suggested_questions?}`     |
| `POST` | `/datachat`      | `X-API-KEY`, `X-USER-EMAIL`                | `{chat:"..."}`                                               | `{response:{type, value}, explanation, log_id?}`    |
| `POST` | `/enddatachat`   | `X-API-KEY`, `X-USER-EMAIL`, `X-USER-NAME` | –                                                            | Deletes the in-memory agent and cleans up plot dirs |

#### Example: full DataChat session

```bash
# 1) Start
curl -X POST http://127.0.0.1:5000/startdatachat \
  -H "X-API-KEY: $KEY" -H "X-USER-EMAIL: me@example.com" -H "X-USER-NAME: Me User" \
  -F "file=@data.csv" -F "lang=ENG"

# 2) Ask
curl -X POST http://127.0.0.1:5000/datachat \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $KEY" -H "X-USER-EMAIL: me@example.com" \
  -d '{"chat":"What is the average sales by region?"}'

# 3) End
curl -X POST http://127.0.0.1:5000/enddatachat \
  -H "X-API-KEY: $KEY" -H "X-USER-EMAIL: me@example.com" -H "X-USER-NAME: Me User"
```

#### Example: agentic RAG

```bash
curl -X POST http://127.0.0.1:5000/agentchat \
  -H "Content-Type: application/json" -H "X-API-KEY: $KEY" \
  -d '{"chat":["What is the main topic of the training material?"],
       "username":"me@example.com","namespace":"Dino","language":"ITA"}'
```

### Response shape — DataChat `response`

`normalize_datachat_response()` (`datachat/output_normalizer.py:86`) always returns:

```json
{ "type": "str|dataframe|image|dict|text_and_image", "value": <str|list|base64> }
```

The `type` tells the client how to render `value` (e.g. `image` → base64 PNG).

---

## 9. Authentication, Users & Tokens

### Authentication models

Pandino uses **four** distinct auth surfaces — not every endpoint needs `X-API-KEY`:

1. **Pandino API key** (`X-API-KEY` + a user identity): validated by
   `database_pg.validate_api_key()`. Keys are stored Fernet-encrypted; the plaintext
   must match and `date_valid_until` must be in the future. The identity is supplied via
   the `X-USER-EMAIL` header on most endpoints, or a `username`/`useremail` field in the
   JSON body on some (`/feedback`, `/completion.json`, `/agentchat`; `/edittokens` uses
   `useremail`). This covers the end-user feature endpoints.
2. **External auth gateway** (used by `/checkpandinouser` and `/storeragfile`):
   - `client == "dino"` → Dino GraphQL probe (`infrastructure/dino.py`)
   - otherwise → `AUTH_GATEWAY_URL` POST (`infrastructure/external_auth.py`)

   `/checkpandinouser` is also how a Pandino API key is first minted (you can't send a key
   you don't have yet).

3. **Stripe shared secret** (`X-STRIPE-KEY` must equal `STRIPE_SK_KEY`): guards
   `/edittokens` only. This is a machine-to-machine billing webhook, not a user call.
4. **Admin session cookie** (`admin_required`, bcrypt login at `/admin/login`): guards the
   whole `/admin/*` area, including the API-docs page. See [§13](#13-admin-panel).

Fully public endpoints (`/`, `/health`) use no auth at all.

`assert_valid_api_key()` (`routes/utils.py`) is the canonical guard for model #1 — call it
at the top of every protected route. It `abort(403)`s on failure.

### Token accounting

- Every billable operation has a `*_TOKEN_COST` config value.
- Routes check `get_user_tokens()` **before** running; if `cost > tokens`, the
  request is rejected (500 or 403 depending on endpoint).
- On success, `edit_tokens(user, -cost)` debits and `log_token_usage(...)` records
  the actual input/output tokens + computed money cost (from the `costs` table) into
  `logs`. The new `log_id` is echoed back in the response where relevant.
- `/edittokens` is the Stripe-webhook entry point (protected by `STRIPE_SK_KEY`
  instead of an API key) and is how users top up.

---

## 10. RAG / Vector Store

### Storage

Pandino uses **PGVector** (via `langchain_postgres.PGVectorStore`). Each _namespace_
maps to its own table (normalized to lowercase, `-`→`_`). Tables are created on
demand by `ensure_pgvector_namespace_ready()` the first time a namespace is ingested.

### Ingestion (`/storeragfile`)

`services/rag_ingestion_service.process_rag_file()` handles, by mimetype:

| Type              | Strategy                                                          |
| ----------------- | ----------------------------------------------------------------- |
| `text/plain`      | `RecursiveCharacterTextSplitter(900/100)`                         |
| `text/markdown`   | `MarkdownTextSplitter(900/100)`                                   |
| `application/pdf` | `pymupdf4llm.to_markdown` then markdown split (per-page metadata) |
| `audio/*`         | DeepInfra Asr → segments merged to ~900 chars                     |
| `image/*`         | Vision model → single-chunk description                           |

Each chunk gets metadata `{url, source, file_id, page?, start_time?, language?}` and
a deterministic `maui_id` so re-ingestion is idempotent (duplicates are skipped in
`store_paragraphs()`).

### Retrieval

Two consumers:

- **`completion_service`** (`/completion.json`) — retrieves vectors, injects them
  into the prompt as `RELEVANT CONTEXT`, then runs a normal LangChain chat.
- **`agentchat_service`** (`/agentchat`) — gives a Smolagents `CodeAgent` a
  `RetrieverTool` and lets the model decide when/what to retrieve.

`RAG_TOP_K` and `RAG_MIN_SIM` control result count and the similarity floor
(similarity = `1 - pgvector_distance`).

---

## 11. DataChat Engine

DataChat is the "chat with your CSV" feature. Implementation lives in `datachat/`.

### Engine interface

`DataChatEngine` (`datachat/engine_interface.py`) is a `Protocol` with three methods:

```python
def bootstrap(self, lang: str) -> EngineBootstrapResult  # suggested questions HTML
def chat(self, message: str, request_id: str | None = None) -> Any
def close(self) -> None
```

### Active implementation: `SmolagentsEngine`

`datachat/smolagents_engine.py` builds a Smolagents `CodeAgent` with **11 tools**
(`datachat/tools/`): `describe`, `missing_values`, `unique_values`, `correlation`,
`sample_rows`, `top_rows`, `filter_rows`, `row_count`, `aggregate`, `plot`, `trend`,
**plus the optional `sql_engine`** when the SQL datasource is enabled (see below).

Notable behaviors:

- **Contract enforcement.** The agent's final answer must be a JSON object with a
  `kind` ∈ `{text, table, image_path, error}`. The `final_answer_checks` guardrail
  (`_check_final_answer`) validates this; answers that still get through invalid fall
  back to a safe `{kind:"text"}` payload (`_coerce_final_payload`).
  The guard **raises** rather than returning `False`: smolagents interpolates whatever
  is raised into the step error it replays to the model, so returning `False` reaches
  the model as `"failed with error:"` and nothing it can act on. A rejected answer is
  not a failed run — the step error goes into memory and the agent gets another step.
- **Empty answers are verified, not forwarded.** A `table` payload with no rows is
  rejected once per question (`_MAX_EMPTY_FINAL_REJECTIONS`), with guidance to re-check
  the filter values against the data before concluding anything. Zero rows is far more
  often a filter written with the wrong value than data that is genuinely absent, and
  the query was valid, so no other layer sees a problem. The budget is one, deliberately:
  an empty result is a legitimate answer to "which clients lost money?", so the guard
  buys a verification round rather than forbidding the answer — and the rejection points
  at `kind="text"` as the way to report emptiness with an explanation attached.
  `sql_engine` adds the same nudge in `meta.hint` the moment a query returns nothing,
  which is earlier and usually enough on its own. Once the budget is spent the answer is
  forwarded **as it stands**: an empty `table` is returned as an empty `table`, never
  rewritten into `text`. The response `kind` is a contract with the caller, and how an
  empty result is presented is the client's decision, not the engine's.
- **Config from env.** Reads `DATACHAT_PROVIDER`, `DATACHAT_MODEL`,
  `DATACHAT_MAX_STEPS` directly (it does not receive `AppConfig`).
- **Plot isolation.** Each session gets a unique plots dir
  (`$DATACHAT_PLOTS_DIR/<user>/<session>`), cleaned up on `close()`.
- **Observability.** Emits `chat_start` / `chat_end` / `final_answer_check` /
  `cleanup_result` structured lines to the `datachat.runtime` logger.

### Output pipeline

```
engine.chat() → raw output
   → adapt_engine_output()          # coerce to {kind,...} contract
   → normalize_datachat_response()  # → stable {type, value} for Dino client
```

### Lifecycle

Engines are kept in `infrastructure/agent_manager.activeEngines` (dict keyed by API
key). `/startdatachat` creates, `/datachat` reuses, `/enddatachat` removes. Because
state is in-process memory, **sticky routing is required in production** when running
multiple gunicorn/gunicorn-gevent workers (the Dockerfile uses a single worker to
avoid this).

### SQL datasource (optional)

Off unless `DATACHAT_SQL_ENABLED` is truthy. When on, the agent gets one extra tool,
`sql_engine`, and the system prompt gains the `data_chat_sql_addendum` block, which
carries the database schema in full. With the flag off the tool list and the prompt are
identical to before the feature existed, and no connection is ever attempted.

**Schema discovery is not a tool.** `datachat/schema_snapshot_loader.py` is the SQL
counterpart of `dataset_loader.py`: just as an uploaded CSV is loaded once and its
columns rendered into the prompt, the SQL schema is reflected once and rendered into
the prompt. The agent is therefore *told* what exists instead of spending turns asking,
and `chat()`'s `reset=True` cannot throw that knowledge away between messages.

The snapshot is cached **process-wide** with a TTL (`DATACHAT_SQL_SCHEMA_TTL_S`),
following the same module-global pattern as the shared engine, because the one-shot
chat routes build and dispose an agent per request — a per-agent cache would re-reflect
the whole database every time. `invalidate_snapshot()` forces a rebuild after a schema
change.

**Identifier case is repaired, not left to the model.** PostgreSQL folds every unquoted
identifier to lowercase, and this database is overwhelmingly mixed-case: 34 of 41 relations
and 214 of 272 distinct column names only resolve when quoted. An agent writing
`FROM Trasporti` gets `42P01 relation "trasporti" does not exist` — a failure no amount of
schema detail prevents, because the name it used *was* correct.

`datachat/sql_identifiers.py` rewrites bare identifiers to their real, quoted spelling
before `sql_guard` validates the query, and the tool executes the same string it validated,
so the guard's "the AST checked is the AST that runs" property is untouched. The split
inside the rewriter is the whole design:

- **The AST decides what may be rewritten** — only an unquoted `Identifier` in relation or
  column position. This is default-deny, and it has to be: `count`, `sum` and the `YEAR` of
  `EXTRACT(YEAR FROM d)` are all bare `VAR` tokens that are not column references, and this
  schema has a `Data` column that collides with exactly that class of word. Deciding from
  tokens alone would be default-allow and would corrupt those queries.
- **The tokens decide where to cut** — only the tokenizer reports source offsets, so it
  supplies the spans. sqlglot expressions carry no positional metadata.

Because only identifier spans are replaced, string literals, comments and already-quoted
names survive byte-for-byte; none of them needs a rule of its own. The rewriter is total and
fails open: anything unparseable, multi-statement or doubtful is returned unchanged for the
guard to reject with a proper message.

Two classes of name are deliberately left alone. Names already all-lowercase need no quoting,
which keeps the rewrite off the materialized views entirely. And names that resolve to more
than one spelling are ambiguous: this schema holds both `Anno` and `anno`, both `Cliente` and
`cliente`. Columns are therefore resolved **against the relations the query actually
references**, which settles it in both directions — `anno` becomes `"Anno"` in a `Budgets`
query and stays bare in a `report_mensile` one. Only a query joining both is genuinely
ambiguous, and there the name is left for the agent to quote from the schema, which is
rendered fully quoted for that reason. `sql_tool_utils.sql_error` closes the loop by
suggesting the real spelling on a 42P01/42703.

**The snapshot carries example values, not just names and types.** Names and types say
nothing about how a column is *written*, and that is where SQL agents actually fail: a
period column holding `'202607'` rather than `'07'`, or a customer label living in
`descrizione_cliente` while `id_cliente` holds an opaque code. Guessing either wrong
returns zero rows, not an error, so nothing downstream can catch it. Each visible
relation is therefore sampled once during reflection — a single
`SELECT CAST(col AS text), ... LIMIT DATACHAT_SQL_SCHEMA_PROFILE_SAMPLE_ROWS` — and the
distinct values are rendered as an `e.g.` line under the relation.

Sampling, not `SELECT DISTINCT`: a plain `LIMIT` short-circuits, whereas `DISTINCT ...
LIMIT n` sits behind a hash aggregate that reads the relation first, and this runs across
every relation on the first request of each TTL. The cost of that choice is that the
result is a *sample*, so the rendered schema says so explicitly — without that caveat the
agent reads the sample as the column's domain and starts reporting data as missing
because it did not appear in a handful of rows. Only text, boolean and date-like columns
are sampled; a `NUMERIC` measure has no spelling to get wrong and would only spend prompt
budget. A sample that fails or times out costs that relation its examples, never the
snapshot: losing the hints is survivable, losing the schema is not.

`SqlDatasource.reflect_schema()` does the whole schema in one connection using
SQLAlchemy's `get_multi_*` API, so cost is a fixed number of round trips rather than one
connection per relation. Note `get_multi_*` defaults to `ObjectKind.TABLE`: views and
materialized views are only reflected when `DATACHAT_SQL_INCLUDE_VIEWS` widens the kind
to `ObjectKind.ANY`. Tables, views and materialized views stay three separate listings —
`list_tables()` does **not** fold views in — but share everything downstream: one
allow/denylist namespace and one `sql_engine`, since a view is read exactly like a table
in a `SELECT`.

The loader applies the allow/denylist, so a hidden relation never reaches the prompt.
A foreign key pointing at a hidden relation is dropped too — rendering it would disclose
the name the rules exist to conceal. Rendering is compact (`name(col:type PK, ...)` plus
one line per foreign key); nullability is deliberately omitted as it rarely changes the
`SELECT` the model writes. If the rendering exceeds `DATACHAT_SQL_SCHEMA_MAX_CHARS` it
degrades to relation names only, so a schema growing over time cannot silently inflate
every request.

If reflection fails, the agent is built **without** `sql_engine` and without any schema
text: one that can write SQL but was never told the schema is worse than one with no SQL
at all. The app still boots, and the CSV path is unaffected.

`REFRESH MATERIALIZED VIEW` needs no special handling: `Refresh` is already in the
guard's `_FORBIDDEN_NODE_NAMES`, so a matview can be read but never refreshed.

This is **always a dedicated database** (`DATACHAT_DB_*`). It never falls back to the
application database, which the agent reaches through its own tools.

`sql_engine` is read-only through four independent layers:

1. **Database role.** Grant the `DATACHAT_DB_USER` role nothing but `CONNECT`,
   `USAGE` and `SELECT` — see §16. Every layer below is application-level and
   therefore bypassable by a bug; grants are not.
2. **Session settings.** `datachat/sql_datasource.py` passes
   `default_transaction_read_only=on`, `statement_timeout`,
   `idle_in_transaction_session_timeout` and `search_path` through libpq's `options`,
   so they apply from connection establishment and cover schema reflection too.
3. **Transaction mode.** `execution_options={"postgresql_readonly": True}` makes
   SQLAlchemy open every transaction as `BEGIN ... READ ONLY`, leaving no window for a
   statement to run first. Never set `isolation_level="AUTOCOMMIT"` on this engine — it
   suppresses the implicit `BEGIN` and defeats this layer.
4. **Statement validation.** `datachat/sql_guard.validate_select()` parses with sqlglot
   and admits only a single `SELECT`/`WITH`/set-operation, with no write, session-changing
   or `Command` node anywhere in the tree, no filesystem/`dblink`/`pg_sleep` function, and
   no table outside the allow/denylist. Rejected queries never reach the database.

Layer 4 is not redundant: psycopg3 sends parameterless queries over the *simple* query
protocol, so `SELECT 1; DROP TABLE t` would genuinely submit both statements. It is also
the only layer that can enforce the table lists. Note that the AST check sees table
names, so a permitted view over a denied table still reads it — which is why layer 1
is the production answer.

Results are capped by `DATACHAT_SQL_MAX_ROWS` / `MAX_COLUMNS` / `MAX_CELL_CHARS`. Rows
come through a server-side cursor and are cut at the fetch, never by rewriting the SQL,
so memory is bounded whatever the query matches. Truncation is reported in
`meta.truncated`; there is deliberately no total row count, since knowing it would cost
a second `COUNT(*)`.

`DATACHAT_SQL_ALLOWED_TABLES` and `DATACHAT_SQL_DENIED_TABLES` are both empty by
default. The allowlist wins when non-empty; otherwise the denylist applies. Despite the
names, the lists are **one namespace covering tables, views and materialized views**:
the guard collects bare identifiers from `exp.Table` nodes, which is what a view name in
a `FROM` parses to, so a separate view list could not be enforced anyway. Scoping is
enforced in three places: every `extract_*` listing tool filters what it returns, so the
agent never learns a hidden relation exists; every `extract_*_info` tool refuses it; and
the guard rejects it.

The engine is a lazily built, process-wide singleton. `sql_datasource.init(config)`
does no I/O, so an unreachable database cannot block startup. **`SmolagentsEngine.close()`
must not dispose it** — the pool is shared by every session, so one user's
`/enddatachat` would drop another user's connections.

### Adding a new tool

1. Create `datachat/tools/my_tool.py` subclassing `smolagents.Tool`.
2. Declare `name`, `description`, `inputs`, `output_type`.
3. Implement `forward(...)` returning a `{kind, ...}` contract dict.
4. Instantiate it in `SmolagentsEngine._build_agent()` — or in `_sql_tools()` for a
   SQL tool.

Before adding a tool, check whether it is really a tool. A tool is for something the
model *chooses* to do, with arguments it picks. Facts the agent always needs and that
take no meaningful arguments — the dataset's columns, the database schema — belong in
the system prompt: load them once (see `dataset_loader.py` and
`schema_snapshot_loader.py`) and render them into the instructions. Making them tools
costs a turn per question and, because `chat()` runs with `reset=True`, the answer is
discarded before the next message.

---

## 12. Prompt Management (DB-driven)

Prompt templates can be customized **without redeploying**.

`infrastructure/prompt_utils.load_prompt(title, default_text=...)` resolution order:

1. **Database** — `prompts` table, highest `version` (or a specific version).
2. **In-code default** — the `default_text` argument.
3. **Environment variable** — if `fallback_env_var` is supplied.
4. Empty string.

Known prompt titles used across the codebase:

| Title                                    | Used by                          |
| ---------------------------------------- | -------------------------------- |
| `complete_chat_system`                   | `completion_service`             |
| `compass_agentchat_system`               | `agentchat_service`              |
| `reply_to_prompt_system`                 | `prompt_service`                 |
| `compare_docs_system`                    | `document_comparison_service`    |
| `audio_form_system`, `audio_form_user`   | `audio_form_service`             |
| `describe_image_user`, `vision_ocr_user` | `infrastructure/ai.py`           |
| `data_chat_system`                       | DataChat engine instructions     |
| `start_chat_system`                      | DataChat bootstrap (LLM variant) |
| `data_chat_sql_addendum`                 | DataChat SQL rules, appended to `data_chat_system` when the SQL datasource is enabled |

Manage them via the admin UI (`/admin/prompts`) or the `prompts` table directly.

### Placeholders

`render_prompt(template, **kwargs)` substitutes `{name}` placeholders. Only the keys
passed by the caller are substituted; **every other brace is literal**, so a prompt may
contain JSON examples such as `{"kind":"text"}` without being mangled. A plain
`str.format()` cannot do this — it reads `{"kind":"text"}` as a replacement field and
raises — so when editing a prompt through the admin UI there is no need to double any
braces.

A *placeholder* is defined strictly as a bare identifier in single braces,
`{name}`. `{"kind":"text"}`, `{a.b}`, `{x:>10}` and a lone `{` are therefore not
placeholders and stay literal and silent. A placeholder that **is** recognizable
but was not supplied by the caller — typically a stale name left in a DB-stored
prompt — is left visible in the rendered output and reported as
`event=prompt_placeholder_missing key=<name>`; the other placeholders still
render, so one stale name no longer discards the whole substitution.

| Placeholder    | Available in             |
| -------------- | ------------------------ |
| `{columns}`    | `data_chat_system` — the uploaded dataset's column list |
| `{sql_schema}` | `data_chat_sql_addendum` — the rendered database schema |

If an override omits `{sql_schema}`, the schema is appended after the template rather
than dropped, so an addendum stored before the placeholder existed keeps working. A
placeholder the caller does not supply is left as literal text and does not prevent the
others from rendering.

---

## 13. Admin Panel

Web UI under `/admin` (Jinja2 templates in `templates/admin/`), protected by
`admin_required` (session-based, bcrypt login at `/admin/login`).

Features:

- **Dashboard** (`/admin`) — user/token stats, CPU/memory (psutil), daily cost,
  recent activity, and a filtered env-var view. Only allowlisted safe values are
  shown; known secrets render as `configured` / `not set`.
- **Users** (`/admin/users`, `/admin/users/<id>/edit`) — paginated, searchable,
  edit token balances.
- **Logs** (`/admin/logs`) — paginated token-usage logs with date range + charts.
- **Feedback** (`/admin/feedback`) — thumbs-up/down review, filter by source/date.
- **Prompts** (`/admin/prompts`, `…/add`, `…/<id>/edit`, `…/<id>/delete`) — full
  CRUD on prompt templates.
- **Costs** (`/admin/costs`, `…/add`, `…/<id>/edit`, `…/<id>/delete`) — per-model
  input/output pricing used to compute `logs.cost`.
- **RAG files** (`/admin/rag-files`, `…/upload`, `…/delete`) — list ingested
  documents, upload new ones into a namespace, and delete a tracked file with its
  namespace chunks.
- **API Docs** (`/admin/api-docs`) — interactive Swagger UI for the HTTP API, rendered
  from [`project_docs/openapi.yaml`](project_docs/openapi.yaml). The spec is served as JSON via
  `/admin/openapi.json`; both routes are behind `admin_required`, and there is an
  "API Docs" entry in the sidebar.

Login uses `ADMIN_USERNAME` + `ADMIN_PASSWORD_HASH` from `AppConfig`.

---

## 14. Adding a New Endpoint (Step-by-Step)

To add, say, `POST /summarize_text`:

1. **Service first.** Add `services/summarize_service.py` with a pure function
   (no Flask imports). Inject the model/provider/api_key as arguments.
2. **(If needed) DB query.** Add a `build_*_query()` in
   `infrastructure/database_methods.py` and a thin wrapper in `database_pg.py`.
3. **Route.** In the appropriate Blueprint (or a new one), add the view function:
   - read `X-API-KEY` / `X-USER-EMAIL`
   - call `assert_valid_api_key(...)`
   - check tokens vs `config.<feature>_token_cost`
   - call the service
   - `log_token_usage(...)` + `edit_tokens(user, -cost)`
   - `return jsonify(...), 200`
4. **Register** any new Blueprint in `main.py`.
5. **Config.** Add new env vars + defaults to `config.py` (`load_config()` and the
   relevant sub-dataclass).
6. **Cost row.** If the endpoint logs usage, make sure a matching row exists in the
   `costs` table for the model+provider, or `log_token_usage()` will raise.
7. **Tests.** Add a test under `tests/` (mock the LLM/DB) and run `pytest`.

---

## 15. Testing

- Framework: **pytest** (`tests/`).
- No shared `conftest.py` or `pytest.ini` — run with defaults:
  ```bash
  pytest
  ```
- Existing tests cover `config.load_config()` (required/optional/defaults), document
  extraction, OCR, the documents route, and AI vision. They mock the DB and LLM
  layers, so **no live Postgres/provider is required**.
- Admin-panel regression coverage includes:
  ```bash
  pytest tests/test_admin_rag_delete.py
  pytest tests/test_admin_dashboard_env.py
  ```
  These verify namespace-aware RAG deletion and dashboard env allowlist/status-only
  behavior.
- **Security regression suite for agent-authored SQL:**
  ```bash
  pytest tests/test_sql_guard.py
  ```
  Runs without a database or mocks. It pins the cases a keyword check cannot catch —
  a write hidden in a CTE, `SELECT ... INTO`, `FOR UPDATE`, a semicolon inside a
  string literal — and doubles as the canary for a sqlglot upgrade that renames an
  expression class. Treat a failure here as a security regression, not a flaky test.
  `tests/test_sql_datasource.py` and `tests/test_sql_tools.py` cover the engine
  options and the tool contracts, also without PostgreSQL.
- When writing tests, follow the same pattern: `unittest.mock.patch.dict` for env,
  and mock `infrastructure.*` boundaries. Services are designed to be tested without
  Flask.

> There is no configured lint/typecheck command in the repo. Consider running
> `ruff` / `mypy` locally; if you adopt one, record it in this file or an
> `AGENTS.md` so it is run after edits.

---

## 16. Deployment & CI/CD

### CI (`.github/workflows/build_and_push.yml`)

On every push to any branch **and** on semver tags (`*.*.*`), GitHub Actions:

1. Builds a multi-arch (`linux/amd64`, `linux/arm64`) Docker image.
2. Pushes to **Docker Hub** (`devgnucoop/pandino`) and **GHCR**
   (`ghcr.io/gnucoop/pandino`), tagged with the semver version and `:latest`.
3. Pull requests build but do not push.

Secrets required in GitHub: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`
(`GITHUB_TOKEN` is used for GHCR automatically).

### Production notes

- **Single worker by design.** The default gunicorn command uses `--workers 1`
  because DataChat engines live in process memory. To scale horizontally you must
  add sticky sessions / externalize session state.
- **Persistent volumes:** mount `/tmp/datachat_plots` (or set `DATACHAT_PLOTS_DIR`)
  and the `logs/` directory.
- **Database:** ensure the `pgvector` extension exists, then follow [§5](#5-database-layer).
  On a **new** instance, `init_db` creates the whole schema. On an **existing**
  instance, `init_db` adds only whole missing tables (`CREATE TABLE IF NOT
  EXISTS` never alters a pre-existing table); the eight `add_*_column` commands
  in §5 apply the newer `logs` and `users` columns and must be run as part of
  the upgrade, or writes targeting those columns will fail after deploy.
- **Schema conformance is not verified by the CLI.** The commands above check
  table and column **names** only, so they cannot confirm that a long-lived
  database matches the schema in `init_db()` — types, nullability, defaults,
  constraints and indexes are outside what they inspect, and the repository
  ships no command or test that compares a live schema against the fresh one.
  Treat conformance of an existing instance as **requiring manual verification
  against that database** before relying on it. Two known cases with no
  governed command at all: `feedback.source` and `feedback.log_id` are present
  in the fresh `feedback` table created by `init_db()`, but a `feedback` table
  predating them is not upgraded by anything in the CLI.
- **DataChat SQL datasource:** create a least-privilege role for `DATACHAT_DB_USER`.
  This is the only read-only layer an application bug cannot bypass:
  ```sql
  CREATE ROLE maui_datachat LOGIN PASSWORD '…';
  GRANT CONNECT ON DATABASE analytics TO maui_datachat;
  GRANT USAGE ON SCHEMA public TO maui_datachat;
  GRANT SELECT ON ALL TABLES IN SCHEMA public TO maui_datachat;
  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO maui_datachat;
  ```
  Keep `DATACHAT_SQL_STATEMENT_TIMEOUT_MS` conservative: `psycopg-binary` reaches
  libpq through C and cannot be monkey-patched, so a slow query blocks every greenlet
  in the `-k gevent` worker, not just its own request. Agent-authored SQL makes that
  much easier to trigger than the handwritten queries elsewhere in the codebase.
- **Secrets:** provide all required env vars (see [§4](#4-configuration-system-configpy))
  via your orchestrator's secret store — never bake them into the image.
- **Admin API docs:** `/admin/api-docs` and `/admin/openapi.json` are session-protected;
  keep [`project_docs/openapi.yaml`](project_docs/openapi.yaml) current when endpoint
  contracts change.

---

## 17. Conventions & Style

- **No comments unless necessary** — the codebase is largely comment-light; prefer
  self-documenting names and docstrings (most functions have them).
- **Docstrings** follow Google/NumPy-ish style with `:param:` / `:return:`.
- **SQL safety** — always go through `infrastructure/database_methods.py` builders;
  never f-string SQL.
- **Agent SQL** — every LLM-authored SQL string goes through
  `datachat/sql_guard.validate_select()` before it reaches the database. Never call
  `SqlDatasource.run_select()` directly.
- **Configuration** — prefer `AppConfig` outside bootstrap/config code. If a direct env
  fallback is necessary, keep it explicit, narrow, and documented.
- **Documentation** — use `project_docs/` for versioned project docs. Keep `/docs` for
  local ignored notes and analyses unless the repository policy changes.
- **Prompts** — use `load_prompt(title, default_text=...)` so ops can override
  without a deploy.
- **Token discipline** — every billable route checks balance, debits, and logs.
- **Error responses** — JSON `{"error": "..."}` with an HTTP status; some legacy
  endpoints (`/prompt.txt`, `/storeragfile`) return plain text.
- **Type hints** — widely used (`Response | tuple[Response, int]`, `TypedDict`,
  `dataclass(frozen=True)`); keep new code typed.

---

_If something here drifts from the code, the code is the source of truth._
