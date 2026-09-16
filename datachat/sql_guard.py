"""
sql_guard.py
------------
Validation of LLM-authored SQL before it reaches the database.

This module is deliberately free of any database or SQLAlchemy dependency: it
parses with sqlglot only, so the whole security surface is unit-testable
without a live PostgreSQL. Every rule fails closed — anything that cannot be
proven to be a single read-only SELECT is rejected.

The validated string is passed to the database verbatim. It is never re-emitted
through sqlglot's generator: that round-trip is not guaranteed lossless, and
silently rewriting a query would be a correctness hazard for no security gain,
since the checks below run on the AST of the exact text that gets executed.

"Verbatim" is relative to the string handed to validate_select(). A caller may
normalise the query first — datachat.sql_identifiers repairs identifier case —
but must then execute the same string it validated, so that the AST checked here
remains the AST of what runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError

# Statement kinds allowed as the root of the parsed query.
# exp.SetOperation is the base class of Union / Except / Intersect.
_ALLOWED_ROOTS = (exp.Select, exp.SetOperation, exp.Subquery)

# Node types that must not appear anywhere in the tree, not just at the root.
# A CTE such as `WITH x AS (INSERT ... RETURNING *) SELECT * FROM x` parses
# with a Select root, so a root-only check would let it through.
# Resolved defensively by name: a sqlglot upgrade that renames or drops a class
# then surfaces as a test failure rather than a silent hole.
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Grant",
    "Copy",
    "Command",
    "Transaction",
    "Commit",
    "Rollback",
    "Set",
    "Use",
    "Into",
    "Lock",
    "Attach",
    "Detach",
    "Refresh",
    "Pragma",
)

_FORBIDDEN_NODES = tuple(
    node
    for node in (getattr(exp, name, None) for name in _FORBIDDEN_NODE_NAMES)
    if node is not None
)

# Functions that read the filesystem, reach other servers, stall the backend or
# change session state. All of them are legal inside a read-only transaction.
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "set_config",
        "query_to_xml",
        "pg_logical_emit_message",
    }
)


@dataclass(frozen=True)
class SqlGuardResult:
    """Outcome of validating one query. `code` is None when `ok` is True."""

    ok: bool
    code: Optional[str]
    message: str
    tables: tuple[str, ...]


def _rejected(code: str, message: str) -> SqlGuardResult:
    return SqlGuardResult(ok=False, code=code, message=message, tables=())


def _function_name(node: exp.Expression) -> str:
    """Best-effort lowercase name of a function node."""
    if isinstance(node, exp.Anonymous):
        return str(node.name or "").lower()
    try:
        return str(node.sql_name() or "").lower()
    except Exception:
        return ""


def validate_select(
    query: str,
    *,
    allowed_tables: frozenset[str] = frozenset(),
    denied_tables: frozenset[str] = frozenset(),
) -> SqlGuardResult:
    """
    Validate that `query` is a single read-only SELECT statement.

    :param query: The SQL text authored by the agent.
    :param allowed_tables: When non-empty, only these tables may be referenced;
                           `denied_tables` is then ignored.
    :param denied_tables: Tables that may not be referenced.
    :return: A SqlGuardResult; on success `tables` holds the bare, lowercased
             names of every table referenced by the query.
    """
    text = (query or "").strip()
    if not text:
        return _rejected("MISSING_QUERY", "No SQL query was provided.")

    try:
        parsed = [stmt for stmt in sqlglot.parse(text, read="postgres") if stmt]
    except ParseError as e:
        return _rejected("INVALID_SQL", f"The SQL could not be parsed: {e}")
    except SqlglotError as e:
        return _rejected("INVALID_SQL", f"The SQL could not be parsed: {e}")

    if not parsed:
        return _rejected("MISSING_QUERY", "No SQL statement was found.")

    if len(parsed) > 1:
        return _rejected(
            "MULTIPLE_STATEMENTS",
            "Only one statement is allowed. Send a single SELECT, without "
            "semicolon-separated follow-ups.",
        )

    root = parsed[0]
    if not isinstance(root, _ALLOWED_ROOTS):
        return _rejected(
            "NON_SELECT_STATEMENT",
            "Only SELECT queries are allowed (optionally preceded by WITH). "
            "Rewrite the request as a plain SELECT.",
        )

    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return _rejected(
                "FORBIDDEN_STATEMENT",
                "The query contains a write or session-changing operation "
                f"({type(node).__name__.lower()}), which is not allowed. "
                "Use a plain read-only SELECT.",
            )
        if isinstance(node, exp.Func) and _function_name(node) in _FORBIDDEN_FUNCTIONS:
            return _rejected(
                "FORBIDDEN_FUNCTION",
                f"The function '{_function_name(node)}' is not allowed. "
                "Query the tables directly instead.",
            )

    tables = tuple(sorted({t.name.lower() for t in root.find_all(exp.Table) if t.name}))

    if allowed_tables:
        refused = [t for t in tables if t not in allowed_tables]
        if refused:
            return _rejected(
                "TABLE_NOT_ALLOWED",
                f"These tables are not available: {', '.join(refused)}. "
                "Use only the relations listed in the schema in your instructions.",
            )
    elif denied_tables:
        refused = [t for t in tables if t in denied_tables]
        if refused:
            return _rejected(
                "TABLE_NOT_ALLOWED",
                f"These tables are not available: {', '.join(refused)}. "
                "Use only the relations listed in the schema in your instructions.",
            )

    return SqlGuardResult(ok=True, code=None, message="OK", tables=tables)
