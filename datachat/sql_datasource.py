"""
sql_datasource.py
-----------------
Read-only SQL datasource for the Datachat agent.

This is a second, separate database layer: infrastructure/database_pg.py owns
the read-write application database over raw psycopg, while this module owns a
read-only SQLAlchemy engine over a dedicated database that the agent may query.
The two never share a connection.

init() only stores configuration — no I/O happens at import or at init, so an
unreachable database cannot prevent the application from booting. The engine is
built lazily on first use and shared process-wide.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any, Optional
from urllib.parse import quote_plus

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine.reflection import ObjectKind

from config import AppConfig, DatachatSqlConfig
from datachat.sql_identifiers import quote_ident as _quote_ident

logger = logging.getLogger(__name__)

_config: Optional[DatachatSqlConfig] = None
_engine: Optional[Engine] = None
_lock = threading.Lock()


def init(config: AppConfig) -> None:
    """Store the datasource configuration. Performs no database I/O."""
    global _config
    _config = config.datachat_sql
    logger.info(
        "event=sql_datasource_init enabled=%s host=%s db=%s schema=%s",
        _config.enabled,
        _config.host or "unset",
        _config.db or "unset",
        _config.schema,
    )


def get_datasource() -> Optional["SqlDatasource"]:
    """
    Return the shared datasource, or None when it is disabled or init() was
    never called. Constructing it performs no database I/O.
    """
    if _config is None or not _config.enabled:
        return None
    return SqlDatasource(_config)


def dispose() -> None:
    """Dispose the shared engine and its pool.

    The pool is shared by every Datachat session, so this must not be called
    when a single session ends — only at process shutdown or from tests.
    """
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None


def _build_engine(cfg: DatachatSqlConfig) -> Engine:
    """
    Build the read-only engine.

    Read-only is enforced twice at this layer:
    - `default_transaction_read_only=on` is applied by libpq at connection
      establishment, so it covers every transaction on the connection,
      including schema reflection.
    - `postgresql_readonly` makes SQLAlchemy open each transaction as
      `BEGIN ... READ ONLY`, which leaves no window for a statement to run
      before the read-only mode is in effect.

    `statement_timeout` bounds how long an agent-authored query can occupy a
    worker; the connection also drops out of an abandoned transaction.
    """
    url = (
        f"postgresql+psycopg://{quote_plus(cfg.user)}:{quote_plus(cfg.password)}"
        f"@{cfg.host}:{cfg.port}/{cfg.db}"
    )
    options = " ".join(
        [
            f"-c search_path={cfg.schema}",
            "-c default_transaction_read_only=on",
            f"-c statement_timeout={cfg.statement_timeout_ms}",
            f"-c idle_in_transaction_session_timeout={cfg.statement_timeout_ms * 2}",
        ]
    )
    return create_engine(
        url,
        connect_args={"options": options, "application_name": "maui-datachat-sql"},
        execution_options={"postgresql_readonly": True},
        pool_size=cfg.pool_size,
        max_overflow=cfg.pool_max_overflow,
        pool_timeout=cfg.pool_timeout_s,
        pool_recycle=cfg.pool_recycle_s,
        pool_pre_ping=True,
    )


def _get_engine(cfg: DatachatSqlConfig) -> Engine:
    """Return the process-wide engine, building it on first use."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = _build_engine(cfg)
    return _engine


class SqlDatasource:
    """
    Thin façade over the shared read-only engine.

    Instances are cheap and stateless: they are created per tool call and hold
    only the frozen config. All connection parameters come from configuration,
    so the agent cannot influence what is connected to.
    """

    def __init__(self, cfg: DatachatSqlConfig) -> None:
        self._cfg = cfg

    @property
    def schema(self) -> str:
        return self._cfg.schema

    @property
    def max_rows(self) -> int:
        return self._cfg.max_rows

    @property
    def max_columns(self) -> int:
        return self._cfg.max_columns

    @property
    def max_cell_chars(self) -> int:
        return self._cfg.max_cell_chars

    @property
    def include_views(self) -> bool:
        return self._cfg.include_views

    @property
    def schema_ttl_s(self) -> int:
        return self._cfg.schema_ttl_s

    @property
    def schema_max_chars(self) -> int:
        return self._cfg.schema_max_chars

    @property
    def schema_include_fks(self) -> bool:
        return self._cfg.schema_include_fks

    @property
    def quote_identifiers(self) -> bool:
        return self._cfg.quote_identifiers

    @property
    def schema_profile_values(self) -> bool:
        return self._cfg.schema_profile_values

    @property
    def schema_profile_sample_rows(self) -> int:
        return self._cfg.schema_profile_sample_rows

    @property
    def schema_profile_max_values(self) -> int:
        return self._cfg.schema_profile_max_values

    @property
    def schema_profile_max_value_chars(self) -> int:
        return self._cfg.schema_profile_max_value_chars

    @property
    def allowed_tables(self) -> frozenset[str]:
        return frozenset(self._cfg.allowed_tables)

    @property
    def denied_tables(self) -> frozenset[str]:
        return frozenset(self._cfg.denied_tables)

    def is_table_visible(self, table: str) -> bool:
        """Whether the allow/deny configuration lets the agent see `table`."""
        name = (table or "").strip().lower()
        if not name:
            return False
        if self._cfg.allowed_tables:
            return name in self.allowed_tables
        return name not in self.denied_tables

    def _list_relations(self, inspector_method: str) -> list[str]:
        """Reflect one kind of relation name from the configured schema."""
        with _get_engine(self._cfg).connect() as conn:
            # Reflect from the connection, not the engine: inspect(engine)
            # opens a connection of its own just to be constructed.
            inspector = inspect(conn)
            names = getattr(inspector, inspector_method)(schema=self._cfg.schema)
        return sorted(set(names))

    def list_tables(self) -> list[str]:
        """Return the base table names in the schema. Views are excluded."""
        return self._list_relations("get_table_names")

    def list_views(self) -> list[str]:
        """Return the plain (non-materialized) view names in the schema."""
        return self._list_relations("get_view_names")

    def list_materialized_views(self) -> list[str]:
        """Return the materialized view names in the schema."""
        return self._list_relations("get_materialized_view_names")

    def describe_table(self, table: str) -> list[dict[str, Any]]:
        """
        Return one record per column of `table`.

        Works for base tables, views and materialized views alike: the
        PostgreSQL dialect reflects columns for every table-like relkind, and a
        relation without a primary key falls back to the empty set below.
        """
        with _get_engine(self._cfg).connect() as conn:
            inspector = inspect(conn)
            columns = inspector.get_columns(table, schema=self._cfg.schema)
            try:
                primary_key = set(
                    inspector.get_pk_constraint(table, schema=self._cfg.schema).get(
                        "constrained_columns"
                    )
                    or []
                )
            except Exception:
                primary_key = set()

        return [
            {
                "column": col["name"],
                "type": str(col.get("type")),
                "nullable": bool(col.get("nullable", True)),
                "primary_key": col["name"] in primary_key,
            }
            for col in columns
        ]

    def reflect_schema(self) -> list[dict[str, Any]]:
        """
        Reflect every relation in the schema in a single pass.

        Returns one record per relation, ordered tables then views then
        materialized views, each sorted by name:

            {"name", "kind", "columns": [...], "foreign_keys": [...]}

        Unlike describe_table(), which opens a connection per relation, this
        uses one connection and one Inspector for the whole schema: the
        get_multi_* reflection API fetches the catalog for every relation at
        once, so a schema of any size costs a fixed number of round trips.

        Views and materialized views are reflected only when the configuration
        includes them, and foreign keys only when schema_include_fks is set.
        Allow/deny filtering is NOT applied here — that is the caller's job, so
        this method reports what the database actually contains.
        """
        include_views = self._cfg.include_views
        kind = ObjectKind.ANY if include_views else ObjectKind.TABLE
        schema = self._cfg.schema

        with _get_engine(self._cfg).connect() as conn:
            inspector = inspect(conn)

            # Names first: these classify each relation. The multi-dicts below
            # are keyed by (schema, name) and do not say what kind a relation is.
            by_kind: list[tuple[str, list[str]]] = [
                ("table", sorted(set(inspector.get_table_names(schema=schema))))
            ]
            if include_views:
                by_kind.append(
                    ("view", sorted(set(inspector.get_view_names(schema=schema))))
                )
                by_kind.append(
                    (
                        "materialized_view",
                        sorted(set(inspector.get_materialized_view_names(schema=schema))),
                    )
                )

            all_columns = inspector.get_multi_columns(schema=schema, kind=kind)
            all_pks = inspector.get_multi_pk_constraint(schema=schema, kind=kind)
            if self._cfg.schema_include_fks:
                all_fks = inspector.get_multi_foreign_keys(schema=schema, kind=kind)
            else:
                all_fks = {}

        relations: list[dict[str, Any]] = []
        for relation_kind, names in by_kind:
            for name in names:
                key = (schema, name)
                columns = all_columns.get(key)
                if columns is None:
                    # Reflected by name but not by the multi-call: a relation
                    # dropped mid-reflection, or one the role cannot read.
                    logger.warning(
                        "event=sql_relation_skipped_no_columns schema=%s relation=%s",
                        schema,
                        name,
                    )
                    continue

                primary_key = set(
                    (all_pks.get(key) or {}).get("constrained_columns") or []
                )

                relations.append(
                    {
                        "name": name,
                        "kind": relation_kind,
                        "columns": [
                            {
                                "column": col["name"],
                                "type": str(col.get("type")),
                                "nullable": bool(col.get("nullable", True)),
                                "primary_key": col["name"] in primary_key,
                            }
                            for col in columns
                        ],
                        "foreign_keys": [
                            {
                                "columns": list(fk.get("constrained_columns") or []),
                                "referred_table": fk.get("referred_table") or "",
                                "referred_columns": list(
                                    fk.get("referred_columns") or []
                                ),
                            }
                            for fk in (all_fks.get(key) or [])
                            if fk.get("referred_table")
                        ],
                    }
                )

        logger.info(
            "event=sql_schema_reflected schema=%s relations=%s include_views=%s include_fks=%s",
            schema,
            len(relations),
            include_views,
            self._cfg.schema_include_fks,
        )
        return relations

    def sample_column_values(
        self, relation: str, columns: Sequence[str], sample_rows: int
    ) -> dict[str, list[str]]:
        """
        Return the distinct values seen for `columns` in a sample of `relation`.

        One query per relation: `SELECT CAST(col AS text), ... LIMIT n`. A plain
        LIMIT over a scan stops as soon as it has its rows, which is why this
        samples rather than running `SELECT DISTINCT ... LIMIT n` — DISTINCT sits
        behind a hash aggregate that reads the whole relation first, and this runs
        across every relation in the schema on the first request of each TTL.

        The cost of that choice is that the result is a sample, not the column's
        domain: values are deduplicated in first-seen order but a value missing
        here is not a value missing from the table. The renderer is responsible
        for saying so.

        Values come back already cast to text so the caller never has to know the
        column's type. Errors propagate: profiling one relation may fail without
        taking down the snapshot, and that is the caller's decision to make.
        """
        wanted = [str(c) for c in columns if str(c or "").strip()]
        if not wanted or sample_rows < 1:
            return {}

        selected = ", ".join(f"CAST({_quote_ident(c)} AS text)" for c in wanted)
        query = (
            f"SELECT {selected} "
            f"FROM {_quote_ident(self._cfg.schema)}.{_quote_ident(relation)} "
            f"LIMIT {int(sample_rows)}"
        )

        with _get_engine(self._cfg).connect() as conn:
            rows = conn.execute(text(query)).fetchall()

        values: dict[str, list[str]] = {name: [] for name in wanted}
        for row in rows:
            for name, value in zip(wanted, row):
                if value is None:
                    continue
                bucket = values[name]
                if value not in bucket:
                    bucket.append(value)
        return values

    def run_select(
        self, query: str, max_rows: int
    ) -> tuple[list[str], list[Sequence[Any]], bool]:
        """
        Execute a validated SELECT and return (columns, rows, truncated).

        Rows are fetched through a server-side cursor and capped at
        `max_rows`, so memory stays bounded whatever the query matches. One
        extra row is fetched purely to detect truncation; the remainder of the
        result is discarded server-side.

        The query text must already have passed sql_guard.validate_select().
        """
        with _get_engine(self._cfg).connect() as conn:
            result = conn.execution_options(
                stream_results=True, max_row_buffer=max_rows + 1
            ).execute(text(query))
            columns = list(result.keys())
            rows = result.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            result.close()

        return columns, list(rows[:max_rows]), truncated
