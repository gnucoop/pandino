"""
sql_prompt.py
-------------

Turns the SQL datasource into the part of an agent's instructions that
describes the database, shared by every engine that reads it.

The flow is always the same: take the process-wide schema snapshot, render it
within the configured character budget, place it into the engine's SQL
addendum, and derive — from that very snapshot — the relation -> columns index
that sql_engine uses to quote identifiers. Keeping both in one result is the
point: a TTL expiry mid-session can then never have the tool quoting names the
agent was not shown.

The addendum template is passed in already loaded rather than looked up here,
so that each engine keeps its own prompt title and its own in-code default,
and a DB-stored override is resolved where the engine resolves its prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import datachat.schema_snapshot_loader as schema_snapshot_loader
from datachat.sql_datasource import SqlDatasource
from infrastructure.prompt_utils import render_prompt

runtime_logger = logging.getLogger("datachat.runtime")

SQL_SCHEMA_PLACEHOLDER = "{sql_schema}"


@dataclass(frozen=True)
class SqlPromptResult:
    """The rendered SQL addendum and the identifier index it was built from."""

    text: str
    identifiers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relation_count: int = 0
    schema_chars: int = 0


def build_sql_addendum(
    datasource: Optional[SqlDatasource],
    *,
    template: str,
    engine_name: str,
    user_name: str,
) -> Optional[SqlPromptResult]:
    """
    Render the SQL addendum for an agent, or None when SQL must stay out of it.

    None means either that the datasource is disabled or that the schema could
    not be reflected. An agent that can write SQL but was never told the schema
    is worse than one without SQL, so callers drop sql_engine as well.

    :param datasource: the SQL datasource, or None when it is disabled.
    :param template: the addendum text, normally carrying {sql_schema}. An
        override stored before the placeholder existed has the schema appended
        instead of silently dropped.
    :param engine_name: engine label for the runtime log lines.
    :param user_name: session user, for the runtime log lines.
    :return: the rendered addendum and identifier index, or None.
    """
    # Gate: with no datasource nothing is reflected at all.
    if datasource is None:
        return None

    snapshot = schema_snapshot_loader.get_schema_snapshot(datasource)
    if snapshot is None:
        runtime_logger.info(
            "engine_init_sql engine=%s user=%s status=error reason=SCHEMA_UNAVAILABLE",
            engine_name,
            user_name,
        )
        return None

    rendered = schema_snapshot_loader.render_schema(
        snapshot, datasource.schema_max_chars
    )

    if SQL_SCHEMA_PLACEHOLDER in template:
        text = render_prompt(template, sql_schema=rendered)
    else:
        text = template + "\n\n" + rendered

    runtime_logger.info(
        "engine_init_sql engine=%s user=%s status=ready relations=%s schema_chars=%s",
        engine_name,
        user_name,
        len(snapshot.relations),
        len(rendered),
    )
    return SqlPromptResult(
        text=text,
        identifiers=schema_snapshot_loader.build_identifier_index(snapshot),
        relation_count=len(snapshot.relations),
        schema_chars=len(rendered),
    )
