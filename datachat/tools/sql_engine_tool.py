import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Optional

from smolagents import Tool

from datachat.output_normalizer import replace_nan
from datachat.sql_datasource import SqlDatasource
from datachat.sql_guard import validate_select
from datachat.sql_identifiers import quote_identifiers
from datachat.tools.sql_tool_utils import sql_error, to_json_scalar, truncate_cell
from utils.logging_config import get_request_id

# Agent-visible SQL behaviour - the tool being invoked, a query accepted or
# rejected, its result and truncation - describes what the DataChat agent did
# during a run, so it belongs to the datachat.runtime channel rather than to a
# per-module logger. Internal SQL infrastructure (datasource lifecycle, schema
# reflection) keeps its own module logger.
runtime_logger = logging.getLogger("datachat.runtime")


class SqlEngineTool(Tool):
    """
    Run one read-only SELECT against the connected SQL database.

    Every query is validated by datachat.sql_guard before it reaches the
    database: anything that is not a single read-only SELECT is rejected
    without a round trip. Results are capped in rows, columns and cell size.
    """

    name = "sql_engine"
    description = (
        "Run one read-only SQL SELECT query against the connected database and return "
        "the rows as a table. Only a single SELECT (or WITH ... SELECT) statement is "
        "accepted: multiple statements, INSERT/UPDATE/DELETE and any DDL are rejected. "
        "The database schema is already given to you in full in your instructions: use "
        "exactly the table, view and column names listed there. Add your own LIMIT when "
        "exploring: results are capped and the response reports whether they were truncated."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "query": {
            "type": "string",
            "description": "A single read-only SQL SELECT statement (PostgreSQL dialect).",
        },
        "max_rows": {
            "type": "integer",
            "description": (
                "Maximum number of rows to return. Capped by the server-side limit; "
                "it can only lower it, never raise it."
            ),
            "nullable": True,
        },
    }

    def __init__(
        self,
        datasource: SqlDatasource,
        identifiers: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__()
        self._datasource = datasource
        # Real relation -> real columns, from the same snapshot the prompt was
        # rendered from. None means no case repair, which is how every caller
        # that does not care about it behaves.
        self._identifiers = identifiers or {}
        # Flat set of every real name, for suggesting a spelling when the
        # database rejects one the rewriter could not resolve on its own.
        self._known_names = frozenset(
            [name for name in self._identifiers]
            + [column for columns in self._identifiers.values() for column in columns]
        )

    def forward(
        self,
        query: str,
        max_rows: Optional[int] = None,
    ) -> dict[str, Any]:
        try:
            sql = (query or "").strip()

            # Repair identifier case BEFORE validating. The guard's contract is
            # that it checks the AST of the exact text that executes, so the
            # rewrite has to land first and `sql` must stay the single variable
            # that is both validated and run.
            if self._identifiers and self._datasource.quote_identifiers:
                sql, quoted = quote_identifiers(sql, self._identifiers)
                if quoted:
                    runtime_logger.info(
                        "event=sql_tool_identifiers_quoted request_id=%s identifiers=%s",
                        get_request_id(),
                        ",".join(quoted),
                    )

            guard = validate_select(
                sql,
                allowed_tables=self._datasource.allowed_tables,
                denied_tables=self._datasource.denied_tables,
            )
            if not guard.ok:
                runtime_logger.info(
                    "event=sql_tool_rejected request_id=%s code=%s tables=%s",
                    get_request_id(),
                    guard.code,
                    ",".join(guard.tables) or "none",
                )
                return {"kind": "error", "message": guard.message, "code": guard.code}

            limit = self._datasource.max_rows
            if max_rows is not None:
                try:
                    limit = max(1, min(int(max_rows), limit))
                except (TypeError, ValueError):
                    limit = self._datasource.max_rows

            started = time.time()
            try:
                columns, rows, truncated = self._datasource.run_select(sql, limit)
            except Exception as e:
                runtime_logger.exception(
                    "event=sql_tool_query_failed request_id=%s", get_request_id()
                )
                return sql_error(e, query=sql, identifiers=self._known_names)
            duration_ms = round((time.time() - started) * 1000, 2)

            max_columns = self._datasource.max_columns
            kept_columns = columns[:max_columns]
            columns_truncated = len(columns) > max_columns
            max_cell_chars = self._datasource.max_cell_chars

            records = [
                {
                    str(name): truncate_cell(to_json_scalar(value), max_cell_chars)
                    for name, value in zip(kept_columns, row)
                }
                for row in rows
            ]
            records = replace_nan(records)

            runtime_logger.info(
                "event=sql_tool_result request_id=%s returned=%s truncated=%s "
                "columns=%s duration_ms=%s tables=%s",
                get_request_id(),
                len(records),
                truncated,
                len(kept_columns),
                duration_ms,
                ",".join(guard.tables) or "none",
            )

            meta: dict[str, Any] = {
                "returned": len(records),
                "truncated": truncated,
                "max_rows": limit,
                "columns": kept_columns,
                "column_count": len(kept_columns),
                "columns_truncated": columns_truncated,
                "duration_ms": duration_ms,
            }

            if not records:
                # The earliest point at which a wrong filter value can be caught.
                # The query was valid, so nothing else in the stack will flag it.
                meta["hint"] = (
                    "No rows matched. This is usually a filter value that does not "
                    "match how the data is actually written, not missing data. Check "
                    "the real values of the columns you filtered on (for example "
                    "SELECT DISTINCT on them) before concluding the data is absent."
                )

            return {"kind": "table", "data": records, "meta": meta}

        except Exception as e:
            runtime_logger.exception(
                "event=sql_tool_failed request_id=%s", get_request_id()
            )
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
