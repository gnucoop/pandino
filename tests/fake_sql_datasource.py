"""
A duck-typed stand-in for SqlDatasource, shared by the SQL tests.

It implements the surface the tools and the schema snapshot loader actually
use, so no test in this suite needs a PostgreSQL instance.
"""

from sqlalchemy.exc import OperationalError


class FakeDatasource:
    """Implements the SqlDatasource surface the SQL code paths actually use."""

    def __init__(
        self,
        *,
        tables=(),
        views=(),
        materialized_views=(),
        columns=(),
        foreign_keys=None,
        rows=(),
        result_columns=(),
        truncated=False,
        allowed_tables=frozenset(),
        denied_tables=frozenset(),
        include_views=True,
        max_rows=200,
        max_columns=25,
        max_cell_chars=300,
        schema_ttl_s=3600,
        schema_max_chars=20000,
        schema_include_fks=True,
        quote_identifiers=True,
        schema_profile_values=True,
        schema_profile_sample_rows=50,
        schema_profile_max_values=5,
        schema_profile_max_value_chars=32,
        samples=None,
        samples_raise=None,
        raises=None,
    ):
        self.schema = "public"
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_cell_chars = max_cell_chars
        self.allowed_tables = allowed_tables
        self.denied_tables = denied_tables
        self.include_views = include_views
        self.schema_ttl_s = schema_ttl_s
        self.schema_max_chars = schema_max_chars
        self.schema_include_fks = schema_include_fks
        self.quote_identifiers = quote_identifiers
        self.schema_profile_values = schema_profile_values
        self.schema_profile_sample_rows = schema_profile_sample_rows
        self.schema_profile_max_values = schema_profile_max_values
        self.schema_profile_max_value_chars = schema_profile_max_value_chars
        # {column_name: [value, ...]} returned for every relation sampled.
        self._samples = dict(samples or {})
        self._samples_raise = samples_raise
        self.sample_calls = []
        self._tables = list(tables)
        self._views = list(views)
        self._materialized_views = list(materialized_views)
        self._columns = list(columns)
        # {relation_name: [{"columns", "referred_table", "referred_columns"}]}
        self._foreign_keys = dict(foreign_keys or {})
        self._rows = list(rows)
        self._result_columns = list(result_columns)
        self._truncated = truncated
        self._raises = raises
        self.run_select_calls = []
        self.reflect_calls = 0

    def is_table_visible(self, table):
        name = (table or "").strip().lower()
        if not name:
            return False
        if self.allowed_tables:
            return name in self.allowed_tables
        return name not in self.denied_tables

    def list_tables(self):
        if self._raises:
            raise self._raises
        return list(self._tables)

    def list_views(self):
        if self._raises:
            raise self._raises
        return list(self._views)

    def list_materialized_views(self):
        if self._raises:
            raise self._raises
        return list(self._materialized_views)

    def describe_table(self, table):
        if self._raises:
            raise self._raises
        return list(self._columns)

    def reflect_schema(self):
        """
        Mirrors SqlDatasource.reflect_schema: one record per relation, ordered
        tables then views then materialized views, with no allow/deny filtering
        applied — that is the loader's job.
        """
        self.reflect_calls += 1
        if self._raises:
            raise self._raises

        groups = [("table", self._tables)]
        if self.include_views:
            groups.append(("view", self._views))
            groups.append(("materialized_view", self._materialized_views))

        records = []
        for kind, names in groups:
            for name in sorted(names):
                records.append(
                    {
                        "name": name,
                        "kind": kind,
                        "columns": [dict(col) for col in self._columns],
                        "foreign_keys": (
                            [dict(fk) for fk in self._foreign_keys.get(name, [])]
                            if self.schema_include_fks
                            else []
                        ),
                    }
                )
        return records

    def sample_column_values(self, relation, columns, sample_rows):
        """
        Mirrors SqlDatasource.sample_column_values: one entry per requested
        column, holding the distinct values seen in a bounded sample.
        """
        self.sample_calls.append((relation, list(columns), sample_rows))
        if self._samples_raise:
            raise self._samples_raise
        return {
            name: list(self._samples.get(name, []))
            for name in columns
            if name in self._samples
        }

    def run_select(self, query, max_rows):
        self.run_select_calls.append((query, max_rows))
        if self._raises:
            raise self._raises
        return list(self._result_columns), list(self._rows), self._truncated


def operational_error(sqlstate=None):
    class Orig(Exception):
        pass

    orig = Orig("connection failed to host=secret password=hunter2")
    if sqlstate is not None:
        orig.sqlstate = sqlstate
    return OperationalError("SELECT 1", {}, orig)
