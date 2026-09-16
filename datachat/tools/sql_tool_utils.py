"""
sql_tool_utils.py
-----------------
Helpers shared by the SQL tools: JSON-safe cell coercion and the mapping from
database exceptions to the {"kind": "error", ...} tool contract.
"""

import logging
import re
from collections.abc import Iterable
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy.exc import InterfaceError, OperationalError

from datachat.sql_identifiers import suggest_identifier

# Database messages are surfaced to the model to let it fix its own query, so
# they are trimmed to keep the agent context small.
_MAX_ERROR_CHARS = 300

# PostgreSQL names the offending identifier in double quotes:
#   relation "<relation_name>" does not exist / column "<column_name>" does not exist
_QUOTED_NAME_RE = re.compile(r'"([^"]+)"')

# PostgreSQL SQLSTATE -> tool error code.
_SQLSTATE_CODES: dict[str, str] = {
    "57014": "QUERY_TIMEOUT",  # query_canceled
    "25006": "READ_ONLY_VIOLATION",  # read_only_sql_transaction
    "42501": "PERMISSION_DENIED",  # insufficient_privilege
    "42P01": "INVALID_TABLE",  # undefined_table
    "42703": "INVALID_COLUMN",  # undefined_column
    "42601": "INVALID_SQL",  # syntax_error
    "53300": "DB_UNAVAILABLE",  # too_many_connections
}


def to_json_scalar(value: Any) -> Any:
    """
    Convert a database value to a JSON scalar: str | int | float | bool | None.

    Types that psycopg returns but json cannot serialise (Decimal, date/time,
    UUID, memoryview, ...) become strings rather than failing the whole result.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (str, int, float)):
        return value

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"

    if isinstance(value, dict):
        return "; ".join(
            f"{k}={str(v).replace('{', '').replace('}', '')}" for k, v in value.items()
        )

    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(v).replace("{", "").replace("}", "") for v in value)

    return str(value)


def truncate_cell(value: Any, max_chars: int) -> Any:
    """Truncate long strings so one wide cell cannot flood the agent context."""
    if max_chars <= 0:
        return value
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…"
    return value


def _case_suggestion(
    message: str, identifiers: Iterable[str] | None
) -> str | None:
    """
    Turn "relation \"<relation_name>\" does not exist" into a name the agent can use.

    PostgreSQL reports the *folded* name it failed to find, which is exactly the
    form that matches case-insensitively against the real spelling. This is the
    fallback for the names the rewriter deliberately declines to touch — the ones
    that exist in more than one casing.
    """
    if not identifiers:
        return None

    found = _QUOTED_NAME_RE.search(message or "")
    if not found:
        return None

    real = suggest_identifier(found.group(1), identifiers)
    if real is None:
        return None

    return (
        f' Did you mean "{real}"? Identifiers in this database are case-sensitive: '
        "PostgreSQL folds an unquoted name to lowercase, so write it double-quoted "
        "exactly as shown in your instructions."
    )


def sql_error(
    exc: Exception,
    *,
    query: str | None = None,
    identifiers: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Map a database exception onto the tool error contract.

    Messages from a failed *statement* (unknown column, syntax error, ...) are
    passed through, because they are what lets the agent repair its own query.
    Messages from a failed *connection* are replaced by a fixed string: libpq
    echoes the connection parameters, password included, into those.
    """
    sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
    code = _SQLSTATE_CODES.get(sqlstate or "")

    if code is None:
        code = (
            "DB_UNAVAILABLE"
            if isinstance(exc, (OperationalError, InterfaceError))
            else "TOOL_FAILED"
        )

    if code == "READ_ONLY_VIOLATION":
        # The guard should have rejected this before it reached the database.
        logging.error(
            "[datachat][sql] read-only violation reached the database, "
            "sql_guard has a gap: query=%s",
            query,
        )

    if code == "DB_UNAVAILABLE":
        return {
            "kind": "error",
            "message": "The SQL database is currently unreachable.",
            "code": code,
        }

    detail = getattr(exc, "orig", None)
    message = str(detail) if detail is not None else str(exc)
    message = message.strip().splitlines()[0] if message.strip() else type(exc).__name__
    message = message[:_MAX_ERROR_CHARS]

    # Appended after the truncation: the suggestion is the actionable half of the
    # message, and trimming the database's prose must never eat it.
    if code in {"INVALID_TABLE", "INVALID_COLUMN"}:
        suggestion = _case_suggestion(message, identifiers)
        if suggestion:
            message += suggestion

    return {
        "kind": "error",
        "message": message,
        "code": code,
    }
