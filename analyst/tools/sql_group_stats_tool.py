from typing import Any, ClassVar, Mapping, Optional, Sequence

from analyst.tools.sql_common import SqlToolBase, SqlToolError, round_number, split_names
from datachat.sql_datasource import SqlDatasource
from datachat.sql_identifiers import quote_ident

_AGGREGATES = {
    "sum": "SUM",
    "count": "COUNT",
    "avg": "AVG",
    "min": "MIN",
    "max": "MAX",
    "stddev": "STDDEV",
}


class SqlGroupStatsTool(SqlToolBase):
    """
    Aggregates per group, computed in the database.

    Groups are ordered by the first metric and capped;
    the total number of groups comes from a window count in the same query.
    """

    name = "sql_group_stats"
    description = (
        "Aggregate metrics per group in the database: group_by is a comma-separated list "
        "of columns, metrics a comma-separated list of 'column:aggregate' with aggregate "
        "one of sum, count, avg, min, max, stddev (e.g. 'ricavo:sum,ricavo:avg'). Groups "
        "are ordered by the first metric, largest first, and capped; total_groups tells "
        "how many there are."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "table_or_query": {
            "type": "string",
            "description": "A relation name from the schema, or a read-only SELECT query.",
        },
        "group_by": {"type": "string", "description": "Comma-separated grouping columns."},
        "metrics": {
            "type": "string",
            "description": "Comma-separated 'column:aggregate' pairs.",
        },
    }

    def __init__(
        self,
        datasource: SqlDatasource,
        *,
        max_rows: int,
        identifiers: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__(datasource, identifiers)
        self._max_rows = max(1, min(max_rows, datasource.max_rows))

    def forward(self, table_or_query: str, group_by: str, metrics: str) -> dict[str, Any]:
        sql = None
        try:
            source, real_columns = self.source(table_or_query)
            groups = split_names(group_by)
            if not groups:
                raise SqlToolError("group_by is empty.")

            selected = [f"{self.column(g, real_columns)} AS {quote_ident(g)}" for g in groups]
            first_metric = None
            for spec in split_names(metrics):
                column, _, aggregate = spec.partition(":")
                function = _AGGREGATES.get(aggregate.strip().lower())
                if function is None:
                    raise SqlToolError(
                        f"Invalid metric {spec!r}: use 'column:aggregate' with aggregate one "
                        f"of {', '.join(_AGGREGATES)}."
                    )
                alias = quote_ident(f"{column.strip()}_{aggregate.strip().lower()}")
                selected.append(f"{function}({self.column(column.strip(), real_columns)}) AS {alias}")
                first_metric = first_metric or alias
            if first_metric is None:
                raise SqlToolError("metrics is empty.")

            # The window count comes first, so the grouping columns are positions 2..n+1.
            positions = ", ".join(str(i + 2) for i in range(len(groups)))
            sql = (
                f"SELECT COUNT(*) OVER () AS total_groups, {', '.join(selected)} "
                f"FROM {source} GROUP BY {positions} "
                f"ORDER BY {first_metric} DESC NULLS LAST"
            )
            columns, rows, truncated = self.run(sql, self._max_rows)

            total = round_number(rows[0][0]) if rows else 0
            return {
                "kind": "group_stats",
                "total_groups": total,
                "returned": len(rows),
                "truncated": truncated,
                "groups": self.records(columns[1:], [row[1:] for row in rows]),
            }
        except Exception as e:
            return self.failure(e, sql)
