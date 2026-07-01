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
- **Configuration is centralized.** Every environment read happens once in
  `config.load_config()` and produces an immutable `AppConfig` dataclass that is
  attached to the Flask app (`app.config["MAUI_CONFIG"]`) and available everywhere
  via `current_app`.
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
│   ├── ai.py                   #   choose_llm() / choose_emb_model() / vision / whisper
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
│   └── tools/                  #   11 pandas-backed analysis tools
│       ├── aggregate_tool.py describe_tool.py missing_values_tool.py
│       ├── correlation_tool.py sample_rows_tool.py top_rows_tool.py
│       ├── filter_rows_tool.py row_count_tool.py plot_tool.py
│       ├── trend_tool.py unique_values_tool.py
│
├── llm/
│   └── litellm_factory.py      #   build_litellm_model() for Smolagents
│
├── utils/
│   ├── runtime_logging.py      #   datachat.runtime logger (stdout)
│   ├── agent_logging.py        #   Structured JSON logger → logs/agent_runs.log
│   ├── agent_serialization.py  #   Smolagents RunResult → JSON payload
│   └── split_message.py        #   WhatsApp-style message chunking
│
├── templates/admin/            # Jinja2 templates for the admin web UI
│   ├── base.html login.html dashboard.html users.html edit_user.html
│   ├── logs.html feedback.html prompts.html edit_prompt.html
│   ├── costs.html edit_cost.html rag_files.html
│   └── api_docs.html           #   Swagger UI page (renders docs/openapi.yaml)
│
├── docs/
│   ├── openapi.yaml            # Hand-maintained OpenAPI 3.0 spec (served in admin panel)
│   └── auth-flow.md            # Mermaid diagrams of the auth + endpoint-usage flow
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

# 8. Initialize the relational schema
python -c "import infrastructure.database_pg as db; from config import load_config; \
           db.init(load_config()); db.init_db()"

# 9. Run the app
python main.py          # → http://127.0.0.1:5000
```

> **Note:** the `load_dotenv()` call in `main.py` reads `.env` automatically, so
> exports are optional in development.

---

## 4. Configuration System (`config.py`)

All environment-variable reads are centralized in `load_config()` (`config.py:156`).
Importing `config.py` has **no side effects**; the env is only read when
`load_config()` is called from `main.py`.

### Required variables (no defaults — app refuses to start if missing)

| Variable | Purpose |
|---|---|
| `ENCRYPTION_KEY` | Fernet key used to encrypt user API keys at rest |
| `PGUSER`, `PGPWD`, `PGHOST`, `PGDB` | PostgreSQL credentials |
| `ADMIN_USERNAME` | Admin panel login username |
| `ADMIN_PASSWORD_HASH` | bcrypt hash of the admin password |

### Optional variables (with defaults)

| Group | Variables | Default |
|---|---|---|
| **DB** | `PG_PORT`, `MAUI_SCHEMA` | `5432`, `public` |
| **Models** | `DATACHAT_MODEL/PROVIDER`, `PROMPT_*`, `COMPLETION_*`, `AUDIO_*`, `WHISPER_MODEL`, `VISION_*`, `COMPARE_DOCS_*` | see `.env.example` (DeepInfra / Qwen / Gemma defaults) |
| **Token costs** | `DATACHAT_TOKEN_COST`, `COMPLETION_TOKEN_COST`, `PROMPT_TOKEN_COST`, `AUDIO_FORM_TOKEN_COST`, `COMPARE_DOCS_TOKEN_COST` | `1` |
| **RAG** | `RAG_TOP_K`, `RAG_MIN_SIM`, `RAG_DEFAULT_NAMESPACE` | `3`, `0.5`, `Dino` |
| **DataChat engine** | `DATACHAT_ENGINE`, `DATACHAT_MAX_STEPS`, `DATACHAT_RATE_LIMIT_PER_MIN`, `DATACHAT_SESSION_TTL_MIN`, `DATACHAT_LOG_LEVEL` | `smolagents`, `12`, `0`, `60`, `INFO` |
| **Auth** | `AUTH_GATEWAY_URL`, `STRIPE_SK_KEY` | `http://localhost:3000/validate`, `None` |

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

---

## 5. Database Layer

### Relational schema

Created by `database_pg.init_db()` (`infrastructure/database_pg.py:107`):

| Table | Purpose |
|---|---|
| `users` | `id, username, api_key (encrypted), date_valid_until, tokens (≥0)` |
| `logs` | Per-request token accounting: `user_id, date, token_input, token_output, cost, model, provider` |
| `costs` | Time-bounded input/output token pricing per model+provider (used to compute `logs.cost`) |
| `prompts` | `title, version, message` — DB-overridable prompt templates |
| `feedback` | User thumbs-up/down on answers, optionally linked to `logs.id` |
| `rag_files` | Tracking of ingested documents (`file_id`, namespace, chunk_count, language) |

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

### Database CLI

`database_pg.py` is also runnable as a script for user management:

```bash
python infrastructure/database_pg.py init_db
python infrastructure/database_pg.py add_user <username> <api_key>
python infrastructure/database_pg.py remove_user <username>
python infrastructure/database_pg.py edit_tokens <username> <quantity>
python infrastructure/database_pg.py list_users
python infrastructure/database_pg.py print_keys
```

> ⚠️ `init()` must run before any DB function. The script wires this up via
> `load_config()` for you.

---

## 6. Running the Application

### Local development (Flask dev server)

```bash
python main.py      # debug=True, port 5000
```

`main.py:83`. Sets `MPLBACKEND=Agg` (headless matplotlib), relaxes pandas display
limits, and initializes the `datachat.runtime` + `agent_runs` loggers.

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

| Blueprint | File | Prefix |
|---|---|---|
| `system_bp` | `routes/system.py` | `/` |
| `auth_bp` | `routes/auth.py` | `/` |
| `users_bp` | `routes/users.py` | `/` |
| `reporting_bp` | `routes/reporting.py` | `/` |
| `documents_bp` | `routes/documents.py` | `/` |
| `multimodal_bp` | `routes/multimodal.py` | `/` |
| `ingestion_bp` | `routes/ingestion.py` | `/` |
| `rag_bp` | `routes/rag.py` | `/` |
| `datachat_bp` | `routes/datachat.py` | `/` |
| `admin_bp` | `routes/admin.py` | `/admin` |

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
  `describe_image()`, `extract_text_from_image()` (OCR), and `whisper_response()`
  live here too.
- **`vector_store.py`** — `MauiVectorStore` wraps `langchain_postgres.PGVectorStore`.
  Similarity is computed as `1 - score` and filtered by `min_similarity`.
  Deduplication uses a deterministic `maui_id = f"{namespace}:{sha256(text)}"`.
- **`agent_manager.py`** — keeps active DataChat engines in an **in-memory** dict
  keyed by API key (`activeEngines`). This means sessions are **per-process** —
  relevant when scaling horizontally (see [§16](#16-deployment--cicd)).
- **`prompt_utils.py`** — `load_prompt(title, default_text=...)` resolution order:
  **DB → in-code default → env var → ""**. `render_prompt(template, **kwargs)` does
  safe `.format()` substitution.

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
> backed by [`docs/openapi.yaml`](docs/openapi.yaml) (served as JSON via
> `/admin/openapi.json`). That spec is hand-maintained and mirrors this section — keep
> the two in sync when the API changes.

### System

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | – | Welcome string |
| `GET` | `/health` | – | `{"status":"ok"}` |

### Authentication & users

| Method | Path | Headers | Body | Description |
|---|---|---|---|---|
| `POST` | `/checkpandinouser` | `X-AUTH-TOKEN`, `X-USER-EMAIL`, `X-CLIENT`, (`X-GRAPHQL-URL` for Dino) | – | Validates the user against the external auth gateway (or Dino GraphQL), creates the user in Pandino if missing. Returns `{response:{user:{user_email, api_key, expiration_date}}}` |
| `POST` | `/validateapikey` | `X-API-KEY`, `X-USER-EMAIL` | – | 200 if the key is valid & unexpired, else 403 |
| `POST` | `/edittokens` | `X-STRIPE-KEY` (must equal `STRIPE_SK_KEY`) | `{quantity, useremail}` | Adds/removes tokens (Stripe-webhook style) |
| `POST` | `/getusertokens` | `X-API-KEY`, `X-USER-EMAIL` | – | `{response:{tokens:N}}` |
| `POST` | `/feedback` | `X-API-KEY` | `{username, question, answer, feedback, log_id?, source?}` | Stores positive/negative feedback; `feedback` ∈ {`positive`,`negative`} |
| `POST` | `/buyreport` | `X-API-KEY`, `X-USER-EMAIL` | `{prompts:int}` | Deducts `prompts` tokens |

### Reporting & documents

| Method | Path | Headers | Body | Returns |
|---|---|---|---|---|
| `POST` | `/prompt.txt` | `X-API-KEY` | form: `prompt`, `username`, `language?` | Plain-text LLM reply (content-type `text/plain`) |
| `POST` | `/compare_docs` | `X-API-KEY`, `X-USER-EMAIL` | multipart: `prompt`, `files[]`, `text_documents`(JSON), `file_roles`(JSON), `additional_context?`, `language?` | `{score, summary, reasoning}` JSON. Needs ≥2 documents (files and/or text). |

### Multimodal

| Method | Path | Headers | Body | Returns |
|---|---|---|---|---|
| `POST` | `/transcribe` | `X-API-KEY`, `X-USER-EMAIL`, `X-USER-NAME`¹ | multipart `file` + `lang?` | Audio → Whisper transcription; image → vision description; PDF/DOCX/RTF → extracted text. `{text}` |
| `POST` | `/audioformcompilation` | `X-API-KEY`, `X-USER-EMAIL` | `{name, exampledata, choices?, transcribedAudio}` | JSON object matching the supplied form schema |

> ¹ `/transcribe` enforces `X-USER-NAME` (400 if missing) but the handler never actually
> uses it (`routes/multimodal.py`) — unlike the DataChat endpoints, where `X-USER-NAME`
> keys the in-memory agent. It is documented as required only because the code requires it.

### RAG / Ingestion

| Method | Path | Headers | Body | Returns |
|---|---|---|---|---|
| `POST` | `/storeragfile` | (Dino auth or `X-CLIENT`) | multipart: `file`, `url`, `namespace?`, `language?`, `authToken`, `graphqlUrl`/`userEmail`, `client` | `{status, file_id, namespace, chunk_count, language, tracking_saved}` |
| `POST` | `/completion.json` | `X-API-KEY` | `{chat, username, namespace?, language?, info?}` | `{answer, vectors, log_id?}` — classic RAG completion |
| `POST` | `/agentchat` | `X-API-KEY` | `{chat:[...], username, namespace?, language?}` | `{answer, follow_ups, vectors, tool_calls, metrics, debug, log_id?}` — agentic RAG |

### DataChat (conversational CSV analysis)

A session is: **start → N× chat → end**.

| Method | Path | Headers | Body | Returns |
|---|---|---|---|---|
| `POST` | `/startdatachat` | `X-API-KEY`, `X-USER-EMAIL`, `X-USER-NAME` | multipart: `file` (CSV), `model_name?`, `llm_type?`, `lang?` | `{Agent active:"active", suggested_questions?}` |
| `POST` | `/datachat` | `X-API-KEY`, `X-USER-EMAIL` | `{chat:"..."}` | `{response:{type, value}, explanation, log_id?}` |
| `POST` | `/enddatachat` | `X-API-KEY`, `X-USER-EMAIL`, `X-USER-NAME` | – | Deletes the in-memory agent and cleans up plot dirs |

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

Pandino uses **PGVector** (via `langchain_postgres.PGVectorStore`). Each *namespace*
maps to its own table (normalized to lowercase, `-`→`_`). Tables are created on
demand by `ensure_pgvector_namespace_ready()` the first time a namespace is ingested.

### Ingestion (`/storeragfile`)

`services/rag_ingestion_service.process_rag_file()` handles, by mimetype:

| Type | Strategy |
|---|---|
| `text/plain` | `RecursiveCharacterTextSplitter(900/100)` |
| `text/markdown` | `MarkdownTextSplitter(900/100)` |
| `application/pdf` | `pymupdf4llm.to_markdown` then markdown split (per-page metadata) |
| `audio/*` | DeepInfra Whisper → segments merged to ~900 chars |
| `image/*` | Vision model → single-chunk description |

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
`sample_rows`, `top_rows`, `filter_rows`, `row_count`, `aggregate`, `plot`, `trend`.

Notable behaviors:

- **Contract enforcement.** The agent's final answer must be a JSON object with a
  `kind` ∈ `{text, table, image_path, error}`. A `final_answer_checks` guardrail
  validates this; invalid answers fall back to a safe `{kind:"text"}` payload
  (`_coerce_final_payload`, `smolagents_engine.py:138`).
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

### Adding a new tool

1. Create `datachat/tools/my_tool.py` subclassing `smolagents.Tool`.
2. Declare `name`, `description`, `inputs`, `output_type`.
3. Implement `forward(...)` returning a `{kind, ...}` contract dict.
4. Instantiate it in `SmolagentsEngine._build_agent()` (`smolagents_engine.py:314`).

---

## 12. Prompt Management (DB-driven)

Prompt templates can be customized **without redeploying**.

`infrastructure/prompt_utils.load_prompt(title, default_text=...)` resolution order:

1. **Database** — `prompts` table, highest `version` (or a specific version).
2. **In-code default** — the `default_text` argument.
3. **Environment variable** — if `fallback_env_var` is supplied.
4. Empty string.

Known prompt titles used across the codebase:

| Title | Used by |
|---|---|
| `complete_chat_system` | `completion_service` |
| `compass_agentchat_system` | `agentchat_service` |
| `reply_to_prompt_system` | `prompt_service` |
| `compare_docs_system` | `document_comparison_service` |
| `audio_form_system`, `audio_form_user` | `audio_form_service` |
| `describe_image_user`, `vision_ocr_user` | `infrastructure/ai.py` |
| `data_chat_system` | DataChat engine instructions |
| `start_chat_system` | DataChat bootstrap (LLM variant) |

Manage them via the admin UI (`/admin/prompts`) or the `prompts` table directly.

---

## 13. Admin Panel

Web UI under `/admin` (Jinja2 templates in `templates/admin/`), protected by
`admin_required` (session-based, bcrypt login at `/admin/login`).

Features:

- **Dashboard** (`/admin`) — user/token stats, CPU/memory (psutil), daily cost,
  recent activity, live env-var view.
- **Users** (`/admin/users`, `/admin/users/<id>/edit`) — paginated, searchable,
  edit token balances.
- **Logs** (`/admin/logs`) — paginated token-usage logs with date range + charts.
- **Feedback** (`/admin/feedback`) — thumbs-up/down review, filter by source/date.
- **Prompts** (`/admin/prompts`, `…/add`, `…/<id>/edit`, `…/<id>/delete`) — full
  CRUD on prompt templates.
- **Costs** (`/admin/costs`, `…/add`, `…/<id>/edit`, `…/<id>/delete`) — per-model
  input/output pricing used to compute `logs.cost`.
- **RAG files** (`/admin/rag-files`, `…/upload`) — list ingested documents and
  upload new ones into a namespace.
- **API Docs** (`/admin/api-docs`) — interactive Swagger UI for the HTTP API, rendered
  from [`docs/openapi.yaml`](docs/openapi.yaml). The spec is served as JSON via
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
- **Database:** ensure `pgvector` extension and run `init_db()` on first deploy.
- **Secrets:** provide all required env vars (see [§4](#4-configuration-system-configpy))
  via your orchestrator's secret store — never bake them into the image.

---

## 17. Conventions & Style

- **No comments unless necessary** — the codebase is largely comment-light; prefer
  self-documenting names and docstrings (most functions have them).
- **Docstrings** follow Google/NumPy-ish style with `:param:` / `:return:`.
- **SQL safety** — always go through `infrastructure/database_methods.py` builders;
  never f-string SQL.
- **Configuration** — read env only in `config.py`; consume `AppConfig` elsewhere.
- **Prompts** — use `load_prompt(title, default_text=...)` so ops can override
  without a deploy.
- **Token discipline** — every billable route checks balance, debits, and logs.
- **Error responses** — JSON `{"error": "..."}` with an HTTP status; some legacy
  endpoints (`/prompt.txt`, `/storeragfile`) return plain text.
- **Type hints** — widely used (`Response | tuple[Response, int]`, `TypedDict`,
  `dataclass(frozen=True)`); keep new code typed.

---

*If something here drifts from the code, the code is the source of truth.*
