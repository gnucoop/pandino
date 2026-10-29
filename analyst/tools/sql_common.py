"""
sql_common.py
-------------

What the analyst's SQL tools share: turning the agent's `table_or_query` and
column names into SQL, and running the result through the same guard as
DataChat's sql_engine.

The statistics tools (sql_stats, sql_outliers, sql_correlation,
sql_group_stats) build their SQL from the agent's arguments. 
Every name is resolved against the reflected schema and quoted, a query source is wrapped as a
subquery, and the *final* statement - exactly the text that runs - goes
through validate_select(). Anything that is not a single read-only SELECT on
visible relations is refused before reaching the database, whose session is
read-only and time-limited anyway (sql_datasource).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from smolagents import Tool

from datachat.sql_datasource import SqlDatasource
from datachat.sql_guard import validate_select
from datachat.sql_identifiers import quote_ident, quote_identifiers
from datachat.tools.sql_tool_utils import sql_error, to_json_scalar, truncate_cell

logger = logging.getLogger(__name__)

SOURCE_ALIAS = "source_data"

_QUERY_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class SqlToolError(Exception):
    """A problem with the agent's arguments, reported back as a tool error."""

    def __init__(self, message: str, code: str = "INVALID_ARGUMENT") -> None:
        super().__init__(message)
        self.code = code


def _lookup(name: str, candidates: Sequence[str]) -> Optional[str]:
    """The real name for `name`: exact match first, then a unique case-insensitive one."""
    if name in candidates:
        return name
    folded = [c for c in candidates if c.casefold() == name.casefold()]
    return folded[0] if len(folded) == 1 else None


def split_names(names: str) -> list[str]:
    """'a, "B" ,c' -> ['a', 'B', 'c']"""
    return [part.strip().strip('"') for part in str(names or "").split(",") if part.strip()]


def round_number(value: Any) -> Any:
    value = to_json_scalar(value)
    return round(value, 4) if isinstance(value, float) else value


class SqlToolBase(Tool):
    """Base of the analyst's SQL tools; subclasses implement forward()."""

    def __init__(
        self,
        datasource: SqlDatasource,
        identifiers: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__()
        self._datasource = datasource
        # Real relation -> real columns, from the snapshot shown to the agent.
        self._identifiers = {k: tuple(v) for k, v in (identifiers or {}).items()}
        self._known_names = frozenset(
            list(self._identifiers) + [c for cols in self._identifiers.values() for c in cols]
        )

    # --- building ---

    def source(self, table_or_query: str) -> tuple[str, Optional[tuple[str, ...]]]:
        """
        The FROM item for `table_or_query`, and the real columns when it is a
        relation (None for a query, whose columns are only known by running it).
        """
        text = str(table_or_query or "").strip().rstrip(";")
        if not text:
            raise SqlToolError("table_or_query is empty.")

        if _QUERY_START_RE.match(text):
            return f"({text}) AS {SOURCE_ALIAS}", None

        relation = _lookup(text.strip('"'), list(self._identifiers))
        if relation is None:
            raise SqlToolError(
                f"Unknown relation {text!r}: use a relation from the schema, or a SELECT query.",
                "INVALID_TABLE",
            )
        return f"{quote_ident(relation)} AS {SOURCE_ALIAS}", self._identifiers[relation]

    def column(self, name: str, columns: Optional[Sequence[str]]) -> str:
        """A quoted column reference, resolved against the relation's real columns."""
        if columns is None:
            # A query's output columns are its aliases, spelled as the agent wrote them.
            return quote_ident(name)
        real = _lookup(name, columns)
        if real is None:
            raise SqlToolError(
                f"Unknown column {name!r}. Available: {', '.join(columns)}.", "INVALID_COLUMN"
            )
        return quote_ident(real)

    # --- running ---

    def run(self, sql: str, max_rows: int) -> tuple[list[str], list[Sequence[Any]], bool]:
        """
        Guard and run a statement. Raises SqlToolError when it is refused.

        Identifier case is repaired before validating, so that the validated
        text is the text that runs (see SqlEngineTool).
        """
        if self._identifiers and self._datasource.quote_identifiers:
            sql, _ = quote_identifiers(sql, self._identifiers)

        guard = validate_select(
            sql,
            allowed_tables=self._datasource.allowed_tables,
            denied_tables=self._datasource.denied_tables,
        )
        if not guard.ok:
            raise SqlToolError(guard.message, guard.code or "INVALID_SQL")
        return self._datasource.run_select(sql, max_rows)

    def records(self, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
        """Rows as JSON records, capped in columns and cell size like sql_engine."""
        kept = list(columns)[: self._datasource.max_columns]
        limit = self._datasource.max_cell_chars
        return [
            {str(c): truncate_cell(round_number(v), limit) for c, v in zip(kept, row)}
            for row in rows
        ]

    def failure(self, e: Exception, sql: Optional[str] = None) -> dict[str, Any]:
        """The tool error for an exception raised while building or running."""
        if isinstance(e, SqlToolError):
            return {"kind": "error", "message": str(e), "code": e.code}
        logger.exception("event=analyst_sql_tool_failed tool=%s", self.name)
        return sql_error(e, query=sql, identifiers=self._known_names)
