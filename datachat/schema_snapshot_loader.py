"""
schema_snapshot_loader.py
-------------------------
The SQL counterpart of dataset_loader.py.

dataset_loader turns an uploaded CSV into a DataFrame once, and the engine
renders its column list into the agent's system prompt. This module does the
same for the SQL database: it reflects the schema once, caches it for the
process, and renders it compactly so the agent is told what exists instead of
spending turns discovering it.

The snapshot is shared process-wide and rebuilt when older than the configured
TTL, because the one-shot chat routes create and dispose an agent per request:
a per-agent cache would re-reflect the whole database every time.

Nothing here runs unless the SQL datasource is enabled. There are no
import-time side effects, no module-level configuration access, and
get_schema_snapshot(None) returns None without touching the cache, so a caller
that forgets the enabled check still cannot populate it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from datachat.sql_datasource import SqlDatasource
from datachat.sql_identifiers import quote_ident

logger = logging.getLogger(__name__)

# Process-wide cache, following the same module-global + lock pattern as the
# shared engine in sql_datasource.py.
_snapshot: Optional["SchemaSnapshot"] = None
_snapshot_at: float = 0.0
_lock = threading.Lock()

_KIND_HEADINGS: tuple[tuple[str, str], ...] = (
    ("table", "TABLES"),
    ("view", "VIEWS"),
    (
        "materialized_view",
        "MATERIALIZED VIEWS (precomputed and fast; data is as fresh as the last refresh)",
    ),
)


# Types worth sampling for example values. Value-format mistakes happen on the
# columns a query filters on — codes, labels, period keys, dates — so those are
# profiled and measures are not: a NUMERIC has no spelling to get wrong, and
# sampling every one of them would spend the prompt budget to say nothing. UUID,
# JSON and binary are excluded for the same reason. A custom enum or domain type
# reflects under its own name and simply goes unprofiled.
_PROFILABLE_TYPE_PREFIXES: tuple[str, ...] = (
    "TEXT",
    "VARCHAR",
    "CHAR",
    "NCHAR",
    "NVARCHAR",
    "STRING",
    "ENUM",
    "BOOL",
    "DATE",
    "TIME",
)

# Rendering tiers, richest first. render_schema() walks these until one fits the
# character budget, so a growing schema loses example values before it loses
# column names, and column names before it loses nothing but the relation list.
_RENDER_TIERS: tuple[tuple[str, bool, bool], ...] = (
    # (tier name, names_only, with_examples)
    ("full", False, True),
    ("no_examples", False, False),
    ("names_only", True, False),
)


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool
    # Example values sampled from the relation. Empty when profiling is off, the
    # type is not worth profiling, or the sample failed. Never exhaustive — see
    # SqlDatasource.sample_column_values.
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    referred_table: str
    referred_columns: tuple[str, ...]


@dataclass(frozen=True)
class Relation:
    name: str
    kind: str  # "table" | "view" | "materialized_view"
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]


@dataclass(frozen=True)
class SchemaSnapshot:
    """
    What the agent is allowed to know about the database.

    Already filtered by the allow/deny configuration: a relation absent from
    `relations` is one the agent must never learn exists.
    """

    schema: str
    relations: tuple[Relation, ...]
    loaded_at: float

    def relation_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.relations)


def _is_profilable(column_type: str) -> bool:
    """Whether a column of this reflected type is worth sampling values for."""
    return str(column_type or "").strip().upper().startswith(_PROFILABLE_TYPE_PREFIXES)


def _truncate(value: str, max_chars: int) -> str:
    """Cap one example value so a wide column cannot dominate the prompt."""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "…"


def _profile_columns(
    datasource: SqlDatasource,
    relation: str,
    columns: list[dict],
) -> dict[str, tuple[str, ...]]:
    """
    Sample example values for the columns of `relation` worth profiling.

    Never raises. One relation whose sample times out, is refused by the role, or
    disappears mid-reflection is rendered without examples; failing the whole
    snapshot over it would cost the agent the schema itself, which is far worse
    than the missing hints.
    """
    if not datasource.schema_profile_values:
        return {}

    names = [
        col["column"] for col in columns if _is_profilable(str(col.get("type") or ""))
    ]
    if not names:
        return {}

    try:
        sampled = datasource.sample_column_values(
            relation, names, datasource.schema_profile_sample_rows
        )
    except Exception:
        logger.warning(
            "event=schema_value_sampling_failed relation=%s",
            relation,
            exc_info=True,
        )
        return {}

    max_values = datasource.schema_profile_max_values
    max_chars = datasource.schema_profile_max_value_chars

    examples: dict[str, tuple[str, ...]] = {}
    for name, values in (sampled or {}).items():
        kept = tuple(
            _truncate(str(value), max_chars) for value in list(values)[:max_values]
        )
        if kept:
            examples[name] = kept
    return examples


def load_schema_snapshot(datasource: SqlDatasource) -> SchemaSnapshot:
    """
    Reflect the schema and filter it down to what the agent may see.

    The direct analogue of dataset_loader.load_csv_to_dataframe: one call, out
    of band, producing the data the prompt is built from. Performs database I/O
    and propagates any failure to the caller.
    """
    reflected = datasource.reflect_schema()

    visible_names = {
        record["name"]
        for record in reflected
        if datasource.is_table_visible(record["name"])
    }

    relations: list[Relation] = []
    profiled = 0
    for record in reflected:
        name = record["name"]
        if name not in visible_names:
            continue

        # Only visible relations are sampled: a hidden one must not be queried
        # at all, not even to describe it.
        examples = _profile_columns(datasource, name, record["columns"])
        if examples:
            profiled += 1

        relations.append(
            Relation(
                name=name,
                kind=record["kind"],
                columns=tuple(
                    Column(
                        name=col["column"],
                        type=col["type"],
                        primary_key=bool(col["primary_key"]),
                        examples=examples.get(col["column"], ()),
                    )
                    for col in record["columns"]
                ),
                # A foreign key pointing at a hidden relation is dropped: naming
                # it in the prompt would disclose a table the allow/deny rules
                # exist to conceal.
                foreign_keys=tuple(
                    ForeignKey(
                        columns=tuple(fk["columns"]),
                        referred_table=fk["referred_table"],
                        referred_columns=tuple(fk["referred_columns"]),
                    )
                    for fk in record["foreign_keys"]
                    if fk["referred_table"] in visible_names
                ),
            )
        )

    snapshot = SchemaSnapshot(
        schema=datasource.schema,
        relations=tuple(relations),
        loaded_at=time.time(),
    )

    logger.info(
        "event=schema_snapshot_loaded schema=%s visible=%s hidden=%s profiled=%s",
        snapshot.schema,
        len(relations),
        len(reflected) - len(relations),
        profiled,
    )
    return snapshot


def get_schema_snapshot(
    datasource: Optional[SqlDatasource],
) -> Optional[SchemaSnapshot]:
    """
    Return the shared snapshot, building it on first use and after the TTL.

    Returns None when the datasource is disabled (the caller's gate failed
    open) or when reflection fails, so an unreachable database degrades to an
    agent without SQL rather than one that writes SQL against a schema it was
    never told.
    """
    global _snapshot, _snapshot_at

    if datasource is None:
        return None

    ttl = datasource.schema_ttl_s
    now = time.time()

    snapshot = _snapshot
    if snapshot is not None and ttl > 0 and (now - _snapshot_at) < ttl:
        return snapshot

    with _lock:
        # Re-check inside the lock: concurrent agent creations must reflect once,
        # not once each.
        now = time.time()
        if _snapshot is not None and ttl > 0 and (now - _snapshot_at) < ttl:
            return _snapshot

        try:
            built = load_schema_snapshot(datasource)
        except Exception:
            logger.exception("event=schema_snapshot_reflection_failed")
            return None

        _snapshot = built
        _snapshot_at = time.time()
        return built


def invalidate_snapshot() -> None:
    """
    Drop the cached snapshot so the next call reflects again.

    For tests, and for a future admin refresh endpoint after a schema change.
    """
    global _snapshot, _snapshot_at
    with _lock:
        _snapshot = None
        _snapshot_at = 0.0


def build_identifier_index(
    snapshot: Optional[SchemaSnapshot],
) -> dict[str, tuple[str, ...]]:
    """
    Real relation name -> its real column names, for the identifier rewriter.

    Built from the snapshot rather than from the database so that the names the
    rewriter may substitute are exactly the names the prompt was rendered from —
    allow/deny filtering included. A hidden relation is absent here for the same
    reason it is absent from the prompt: quoting its name would disclose that it
    exists.
    """
    if snapshot is None:
        return {}
    return {
        relation.name: tuple(col.name for col in relation.columns)
        for relation in snapshot.relations
    }


def _sql_literal(value: str) -> str:
    """Render an example the way it would be written in a WHERE clause."""
    return "'" + str(value).replace("'", "''") + "'"


def _render_relation(
    relation: Relation, *, names_only: bool, with_examples: bool
) -> list[str]:
    # Every identifier is rendered double-quoted, lowercase ones included.
    # PostgreSQL folds an unquoted name to lowercase, so a mixed-case relation
    # only resolves when quoted; quoting uniformly gives the agent one rule to
    # copy rather than a per-name judgement call, and "report_mensile" means
    # exactly what report_mensile means.
    name = quote_ident(relation.name)
    if names_only:
        return [f"  {name}"]

    columns = ", ".join(
        f"{quote_ident(col.name)}:{col.type}" + (" PK" if col.primary_key else "")
        for col in relation.columns
    )
    lines = [f"  {name}({columns})"]
    for fk in relation.foreign_keys:
        local = ", ".join(quote_ident(c) for c in fk.columns)
        referred = ", ".join(quote_ident(c) for c in fk.referred_columns)
        lines.append(
            f"    -> {name}.{local} "
            f"references {quote_ident(fk.referred_table)}.{referred}"
        )

    if with_examples:
        # One line per relation rather than one per column: the whole point is to
        # be scannable next to the column list without doubling its height.
        rendered = "; ".join(
            f"{quote_ident(col.name)}={', '.join(_sql_literal(v) for v in col.examples)}"
            for col in relation.columns
            if col.examples
        )
        if rendered:
            lines.append(f"    e.g. {rendered}")

    return lines


def _render(snapshot: SchemaSnapshot, *, names_only: bool, with_examples: bool) -> str:
    lines = [
        f"AVAILABLE RELATIONS (schema: {snapshot.schema})",
        "The relations below are the complete and authoritative set available to you.",
        "Every name and type is exact. Nothing that is not listed here exists.",
        "Every name is shown double-quoted and must be written that way in your SQL. "
        'PostgreSQL folds an unquoted name to lowercase, so FROM Trasporti looks for '
        '"trasporti" and fails, while FROM "Trasporti" works. Copy each name exactly '
        "as shown, quotes included.",
    ]

    has_examples = any(
        col.examples for relation in snapshot.relations for col in relation.columns
    )

    if with_examples and not names_only and has_examples:
        # Without this caveat the sample reads as the column's domain, and the
        # agent starts reporting that data is missing because a value it needs
        # did not happen to appear in a handful of rows. It is only worth its
        # tokens when something was actually sampled.
        lines.append(
            'Values after "e.g." are examples taken from a few sample rows, NOT the '
            "complete set of values a column holds. Use them to see how a column is "
            "written — its format, its spelling, whether it holds a code or a label. "
            "Never conclude that a value is absent from the database because it is "
            "not listed there."
        )

    for kind, heading in _KIND_HEADINGS:
        of_kind = [r for r in snapshot.relations if r.kind == kind]
        if not of_kind:
            continue
        lines.append("")
        lines.append(heading)
        for relation in of_kind:
            lines.extend(
                _render_relation(
                    relation, names_only=names_only, with_examples=with_examples
                )
            )

    return "\n".join(lines)


def render_schema(snapshot: SchemaSnapshot, max_chars: int) -> str:
    """
    Render the snapshot as compact text for the system prompt.

    Nullability is deliberately omitted: it costs a token per column and rarely
    changes the SELECT the model writes. Types and primary keys are kept
    because they drive casts and joins.

    If the full rendering exceeds `max_chars` it degrades a tier at a time: first
    the example values go, then the column lists, leaving bare relation names.
    Examples are shed first because a column list the agent cannot see at all is
    a worse failure than one it can see without sample values. At a schema of a
    few dozen relations this never fires; it exists so that a schema growing over
    time cannot silently inflate every request.
    """
    if not snapshot.relations:
        # An empty schema is not an error, but the agent must be told plainly
        # rather than shown an empty heading it might try to fill in.
        return (
            f"AVAILABLE RELATIONS (schema: {snapshot.schema})\n"
            "The database is reachable but exposes no relations you may query."
        )

    rendered = ""
    chosen_tier = ""
    for index, (tier, names_only, with_examples) in enumerate(_RENDER_TIERS):
        rendered = _render(
            snapshot, names_only=names_only, with_examples=with_examples
        )
        chosen_tier = tier
        if len(rendered) <= max_chars:
            break
        if index < len(_RENDER_TIERS) - 1:
            logger.warning(
                "event=schema_render_budget_exceeded chars=%s tier=%s "
                "max_chars=%s relations=%s degrading=true",
                len(rendered),
                tier,
                max_chars,
                len(snapshot.relations),
            )

    logger.info(
        "event=schema_rendered schema=%s relations=%s tier=%s chars=%s",
        snapshot.schema,
        len(snapshot.relations),
        chosen_tier,
        len(rendered),
    )
    return rendered
