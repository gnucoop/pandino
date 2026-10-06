"""Read-only Usage (``logs``) collector and preflight DB reads.

Opens its own psycopg 3 connection from Maui's effective DB settings and
never imports Maui's database module. Read-only is enforced three ways:

- startup options ``-c default_transaction_read_only=on -c search_path=<schema>``;
- ``conn.read_only = True`` before any statement (autocommit stays off,
  since psycopg ignores ``read_only`` under autocommit);
- only SELECTs are executed, nothing is committed: ``rollback()`` then
  ``close()``, and ``close()`` also on errors.

This prevents accidental writes; it is not a defence against a malicious
client. Zero, one or many Usage rows are all a successful collection
(``PRESENT``); their meaning is interpreted later against runtime evidence.
"""

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import psycopg

from evaluation.datachat.contract import USAGE_QUERY_SQL
from evaluation.datachat.model import Availability, EvidenceSourceRecord, UsageEvidence, utc_now

SOURCE = "usage"

USAGE_COLUMNS = (
    "id",
    "user_id",
    "date",
    "token_input",
    "token_output",
    "cost",
    "model",
    "provider",
    "service",
    "request_id",
    "duration_ms",
    "source",
)

# users columns verified against CURRENT infrastructure/database_pg.py schema;
# api_key is deliberately never selected.
USER_STATE_SQL = "SELECT id, username, tokens, date_valid_until FROM users WHERE username = %s"
READ_CHECK_SQL = "SELECT 1"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def validate_schema(schema: str) -> str:
    if not isinstance(schema, str) or not _IDENTIFIER.fullmatch(schema):
        raise ValueError(f"invalid schema identifier: {schema!r}")
    return schema


@dataclass(frozen=True)
class DbSettings:
    host: str
    dbname: str
    port: str
    user: str
    password: str = field(repr=False)
    schema: str = "public"

    def __post_init__(self) -> None:
        validate_schema(self.schema)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DbSettings":
        """From the resolved child env; PGPORT/MAUI_SCHEMA default as in Maui's config."""
        missing = [name for name in ("PGHOST", "PGDB", "PGUSER", "PGPWD") if not (env.get(name) or "").strip()]
        if missing:
            raise ValueError(f"missing DB settings: {', '.join(missing)}")
        return cls(
            host=env["PGHOST"],
            dbname=env["PGDB"],
            port=env.get("PGPORT") or "5432",
            user=env["PGUSER"],
            password=env["PGPWD"],
            schema=env.get("MAUI_SCHEMA") or "public",
        )

    def public(self) -> dict[str, str]:
        """Settings safe to persist (no password)."""
        return {"host": self.host, "dbname": self.dbname, "port": self.port, "user": self.user, "schema": self.schema}


def _error_reason(exc: psycopg.Error) -> str:
    # Class name plus SQLSTATE only: driver messages can echo connection details.
    sqlstate = getattr(exc, "sqlstate", None)
    return f"db_error: {type(exc).__name__}" + (f" sqlstate={sqlstate}" if sqlstate else "")


@contextmanager
def read_only_connection(settings: DbSettings) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(
        host=settings.host,
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        port=settings.port,
        options=f"-c default_transaction_read_only=on -c search_path={validate_schema(settings.schema)}",
    )
    try:
        conn.read_only = True
        yield conn
        conn.rollback()
    finally:
        conn.close()


@dataclass(frozen=True)
class UsageCollection:
    record: EvidenceSourceRecord
    evidence: UsageEvidence | None  # rows for usage/rows.json; None when unavailable


def collect_usage(
    settings: DbSettings, request_id: str, primary_log_id: int | None = None
) -> UsageCollection:
    """Usage rows for ``request_id`` in ``id`` order. Query failure → ``UNAVAILABLE``."""
    executed_at = utc_now()
    locator: dict[str, Any] = {"request_id": request_id, "query_sql": USAGE_QUERY_SQL, "schema": settings.schema}
    try:
        with read_only_connection(settings) as conn:
            with conn.cursor() as cursor:
                cursor.execute(USAGE_QUERY_SQL, (request_id,))
                fetched = cursor.fetchall()
    except psycopg.Error as exc:
        record = EvidenceSourceRecord(SOURCE, Availability.UNAVAILABLE, _error_reason(exc), locator=locator)
        return UsageCollection(record, None)

    rows = tuple(dict(zip(USAGE_COLUMNS, row)) for row in fetched)
    locator["row_count"] = len(rows)
    evidence = UsageEvidence(
        request_id=request_id,
        query_sql=USAGE_QUERY_SQL,
        schema=settings.schema,
        executed_at=executed_at,
        rows=rows,
        primary_log_id=primary_log_id,
        primary_row_present=primary_log_id is not None and any(row["id"] == primary_log_id for row in rows),
    )
    return UsageCollection(EvidenceSourceRecord(SOURCE, Availability.PRESENT, locator=locator), evidence)


# --- Preflight reads ---------------------------------------------------------


def check_read_access(settings: DbSettings) -> str | None:
    """``None`` when a read-only SELECT succeeds, else a password-free reason."""
    try:
        with read_only_connection(settings) as conn:
            with conn.cursor() as cursor:
                cursor.execute(READ_CHECK_SQL)
                cursor.fetchone()
    except psycopg.Error as exc:
        return _error_reason(exc)
    return None


@dataclass(frozen=True)
class UserTokenState:
    user_id: int
    username: str
    tokens: int
    date_valid_until: str


def fetch_user_token_state(settings: DbSettings, username: str) -> UserTokenState | None:
    """The evaluation user's id and token balance; ``None`` when the user does not exist.

    ``psycopg.Error`` propagates: the caller decides how a failed lookup is classified.
    """
    with read_only_connection(settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(USER_STATE_SQL, (username,))
            row = cursor.fetchone()
    if row is None:
        return None
    return UserTokenState(user_id=row[0], username=row[1], tokens=row[2], date_valid_until=row[3])
