from typing import Any, ClassVar

from analyst.tools.sql_common import SqlToolBase, SqlToolError, round_number, split_names

_STATS = ("count", "mean", "std", "min", "max", "p25", "median", "p75")


class SqlStatsTool(SqlToolBase):
    """
    Descriptive statistics of numeric columns, computed in the database.

    One aggregate query, however large the table, and a few numbers back per column.
    """

    name = "sql_stats"
    description = (
        "Descriptive statistics computed in the database (no rows are loaded): count, "
        "mean, std, min, max, p25, median, p75 for each numeric column. table_or_query is "
        "a relation name or a SELECT query (e.g. one filtering a period)."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "table_or_query": {
            "type": "string",
            "description": "A relation name from the schema, or a read-only SELECT query.",
        },
        "columns": {
            "type": "string",
            "description": "Comma-separated numeric column names.",
        },
    }

    def forward(self, table_or_query: str, columns: str) -> dict[str, Any]:
        sql = None
        try:
            source, real_columns = self.source(table_or_query)
            names = split_names(columns)
            if not names:
                raise SqlToolError("columns is empty.")

            parts = []
            for i, name in enumerate(names):
                col = self.column(name, real_columns)
                parts += [
                    f"COUNT({col}) AS c{i}_count",
                    f"AVG({col}) AS c{i}_mean",
                    f"STDDEV({col}) AS c{i}_std",
                    f"MIN({col}) AS c{i}_min",
                    f"MAX({col}) AS c{i}_max",
                    f"PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS c{i}_p25",
                    f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}) AS c{i}_median",
                    f"PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) AS c{i}_p75",
                ]
            sql = f"SELECT {', '.join(parts)} FROM {source}"

            result_columns, rows, _ = self.run(sql, 1)
            row = dict(zip(result_columns, rows[0])) if rows else {}
            stats = {
                name: {stat: round_number(row.get(f"c{i}_{stat}")) for stat in _STATS}
                for i, name in enumerate(names)
            }
            return {"kind": "stats", "stats": stats}
        except Exception as e:
            return self.failure(e, sql)
