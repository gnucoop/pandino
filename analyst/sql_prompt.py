"""
sql_prompt.py
-------------

The database part of a task agent's instructions, scoped to the brief.

DataChat and the interviewer show the agent the whole schema
(datachat.sql_prompt). Task and chart agents work on the relations the plan
names (chosen by the planner against the schema), and a task agent runs once
per task: rendering only those relations in full, and the others by name,
keeps every agent's instructions short.

The identifier index still covers the whole visible schema, so a query on a
relation outside the scope (a data-quality check, say) is quoted correctly.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Optional

import datachat.schema_snapshot_loader as schema_snapshot_loader
from datachat.sql_datasource import SqlDatasource
from datachat.sql_identifiers import quote_ident
from datachat.sql_prompt import SQL_SCHEMA_PLACEHOLDER, SqlPromptResult
from infrastructure.prompt_utils import render_prompt

# The planner's estimate, kept in pg_class by ANALYZE/autovacuum: reading it
# scans no table. -1 (never analysed) never passes the threshold.
_ESTIMATED_ROWS_SQL = (
    "SELECT c.relname, c.reltuples::bigint FROM pg_class c "
    "JOIN pg_namespace n ON n.oid = c.relnamespace "
    "WHERE n.nspname = '{schema}' AND c.relkind IN ('r', 'p', 'm') AND c.reltuples > {max_rows}"
)


def _approx(rows: int) -> str:
    if rows >= 1_000_000:
        return f"~{rows / 1_000_000:.1f}M"
    return f"~{rows // 1000}k"


def large_relations_note(
    datasource: SqlDatasource, relation_names: Iterable[str], *, max_rows: int
) -> str:
    """
    The relations load_dataset cannot load whole (more than max_rows rows,
    estimated), and how to analyse them; "" when there are none or the
    estimate cannot be read.

    Without it the agents learn a table is huge from timeouts, or worse, compute
    totals on the first max_rows rows of a truncated load. Built in code, not in
    a prompt, so a prompt stored in the database cannot drop it.
    """
    names = {name.casefold(): name for name in relation_names}
    sql = _ESTIMATED_ROWS_SQL.format(schema=datasource.schema.replace("'", "''"), max_rows=int(max_rows))
    try:
        _, rows, _ = datasource.run_select(sql, 500)
        # Only names of the snapshot: what the agent sees is what it may query.
        large = sorted(
            ((names[str(relname).casefold()], int(estimate)) for relname, estimate in rows
             if str(relname).casefold() in names),
            key=lambda item: -item[1],
        )
    except Exception:
        # Only a hint: the agents work without it, as they did before.
        return ""
    if not large:
        return ""
    return (
        "LARGE TABLES (rows estimated from PostgreSQL statistics): "
        + ", ".join(f"{quote_ident(name)} {_approx(estimate)}" for name, estimate in large)
        + ".\n"
        f"- load_dataset keeps only the first {max_rows} rows of a result: never compute totals, "
        "averages or rankings on a load of these tables. Aggregate in SQL (GROUP BY, the sql_* tools).\n"
        "- When a method needs rows (detect_anomalies, run_regression, run_forecast), load a random "
        'sample: ... FROM "<table>" TABLESAMPLE SYSTEM (<percent>) REPEATABLE (42), sized to stay '
        f"under {max_rows} rows, and say in your summary that it is a sample.\n"
        "- Statistics over a whole large table are slow: filter to the brief's period or scope first."
    )


def scope_relations(plan: dict[str, Any]) -> list[str]:
    """The relations the plan says the analysis reads."""
    return [str(r) for r in plan.get("relations") or [] if isinstance(r, str)]


def build_scoped_sql_addendum(
    datasource: Optional[SqlDatasource],
    *,
    template: str,
    relations: Iterable[str],
    max_rows: int,
) -> Optional[SqlPromptResult]:
    """
    Render the SQL addendum with the scope's relations in full, or None when SQL
    must stay out of the agent (see datachat.sql_prompt.build_sql_addendum).

    A scope matching no visible relation falls back to the whole schema. The
    large relations note (any relation, in scope or not) closes it.
    """
    if datasource is None:
        return None
    snapshot = schema_snapshot_loader.get_schema_snapshot(datasource)
    if snapshot is None:
        return None

    wanted = {name.casefold() for name in relations}
    in_scope = tuple(r for r in snapshot.relations if r.name.casefold() in wanted)
    others = [r.name for r in snapshot.relations if r.name.casefold() not in wanted]

    shown = dataclasses.replace(snapshot, relations=in_scope) if in_scope else snapshot
    rendered = schema_snapshot_loader.render_schema(shown, datasource.schema_max_chars)
    if in_scope and others:
        rendered += (
            "\n\nThe relations above are the ones this analysis is about. These others "
            "also exist, outside its scope; query them only if your sub-task needs it: "
            + ", ".join(quote_ident(name) for name in others)
        )

    if SQL_SCHEMA_PLACEHOLDER in template:
        text = render_prompt(template, sql_schema=rendered)
    else:
        text = template + "\n\n" + rendered

    note = large_relations_note(datasource, (r.name for r in snapshot.relations), max_rows=max_rows)
    if note:
        text += "\n\n" + note

    return SqlPromptResult(
        text=text,
        identifiers=schema_snapshot_loader.build_identifier_index(snapshot),
    )
