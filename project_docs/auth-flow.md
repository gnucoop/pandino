# Pandino — Authentication & Endpoint Usage Flow

Mermaid diagrams describing the full lifecycle: obtaining an API key, using a
billable feature endpoint, and topping up the token balance. These reflect the
actual code paths in `routes/auth.py`, `routes/utils.py`, the feature routes, and
`routes/users.py`.

## 1. End-to-end lifecycle (sequence)

```mermaid
sequenceDiagram
    autonumber
    actor U as User / Client App
    participant P as Pandino API
    participant EA as External Auth Gateway<br/>(Dino GraphQL / AUTH_GATEWAY_URL)
    participant DB as PostgreSQL
    participant LLM as LLM Provider
    participant S as Stripe / Billing

    Note over U,EA: PHASE 1 — Obtain an API key (bootstrap)
    U->>P: POST /checkpandinouser<br/>X-AUTH-TOKEN, X-USER-EMAIL, X-CLIENT
    P->>EA: Validate auth token
    EA-->>P: OK / error
    alt user not in Pandino yet
        P->>DB: create user + generate api_key (valid 2y)
    end
    P->>DB: fetch user record
    P-->>U: { response: { user: { user_email, api_key, expiration_date } } }

    Note over U,LLM: PHASE 2 — Use a billable feature endpoint
    U->>P: POST /compare_docs (or /transcribe, /agentchat, ...)<br/>X-API-KEY + X-USER-EMAIL + payload
    P->>DB: assert_valid_api_key(api_key, user_email)
    DB-->>P: valid & not expired?
    alt invalid / expired
        P-->>U: 403 Forbidden
    else valid
        P->>DB: get_user_tokens(user_email)
        alt cost > balance
            P-->>U: 403 / 500 Not enough tokens
        else enough tokens
            P->>LLM: run the feature (compare / transcribe / RAG)
            LLM-->>P: result + token_usage
            P->>DB: log_token_usage(user_id, in, out, model)
            P->>DB: edit_tokens(user_email, -feature_cost)
            P-->>U: 200 result (+ log_id)
        end
    end

    Note over S,DB: PHASE 3 — Top up balance (webhook, not the user)
    S->>P: POST /edittokens<br/>X-STRIPE-KEY + { quantity, useremail }
    P->>P: X-STRIPE-KEY == STRIPE_SK_KEY ?
    alt secret mismatch
        P-->>S: 403 Invalid STRIPE KEY
    else ok
        P->>DB: edit_tokens(useremail, +quantity)
        P-->>S: 200 updated
    end
```

## 2. Per-request auth + token gate (flowchart)

```mermaid
flowchart TD
    A[Incoming request] --> B{Which auth model?}

    B -->|Public| P1["GET / , GET /health"] --> R200([200 OK])

    B -->|"Session cookie<br/>(admin panel)"| ADM["/admin/*"]
    ADM --> ADMQ{admin_logged_in<br/>in session?}
    ADMQ -->|no| ADMR[Redirect to /admin/login]
    ADMQ -->|yes| ADMOK([Render admin page])

    B -->|"External gateway"| EXT["/checkpandinouser<br/>/storeragfile"]
    EXT --> EXTQ{X-AUTH-TOKEN / authToken<br/>valid at gateway?}
    EXTQ -->|no| E403([403 / text error])
    EXTQ -->|yes| EXTOK([Issue key / ingest file])

    B -->|"Stripe secret"| STR["/edittokens"]
    STR --> STRQ{X-STRIPE-KEY == STRIPE_SK_KEY?}
    STRQ -->|no| S403([403 Invalid key])
    STRQ -->|yes| STROK([Adjust token balance])

    B -->|"API key<br/>(end-user features)"| API["/compare_docs, /transcribe,<br/>/agentchat, /completion.json,<br/>/datachat, /prompt.txt, ..."]
    API --> H{X-API-KEY +<br/>identity present?}
    H -->|no| M400([400 Missing header])
    H -->|yes| V{assert_valid_api_key<br/>key matches user &<br/>not expired?}
    V -->|no| V403([403 Forbidden])
    V -->|yes| T{feature cost<br/>&le; user tokens?}
    T -->|no| T403([403 / 500 Not enough tokens])
    T -->|yes| RUN[Run feature → call LLM]
    RUN --> LOG[log_token_usage + edit_tokens -cost]
    LOG --> OK([200 result])
```

## Notes

- The **identity** paired with `X-API-KEY` varies by endpoint — `X-USER-EMAIL` header
  on most, but a `username` field in the JSON body for `/feedback`, `/completion.json`,
  and `/agentchat` (and `useremail` for `/edittokens`).
- `X-USER-NAME` is **not** part of authentication — it is only used by DataChat to key
  the in-memory agent, and is required-but-unused on `/transcribe`.
- Not every API-key endpoint bills tokens: `/validateapikey` and `/getusertokens`
  authenticate but skip the token gate and deduction.
