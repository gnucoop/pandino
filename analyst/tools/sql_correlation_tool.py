from typing import Any, ClassVar

from analyst.tools.sql_common import SqlToolBase, SqlToolError, round_number, split_names


class SqlCorrelationTool(SqlToolBase):
    """
    Pearson correlation matrix of numeric columns, computed in the database.

    One CORR() per pair, in a
    single aggregate query.
    """

    name = "sql_correlation"
    description = (
        "Pearson correlation matrix of two or more numeric columns, computed in the "
        "database (no rows are loaded). table_or_query is a relation name or a SELECT query."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "table_or_query": {
            "type": "string",
            "description": "A relation name from the schema, or a read-only SELECT query.",
        },
        "columns": {
            "type": "string",
            "description": "Comma-separated numeric column names (at least two).",
        },
    }

    def forward(self, table_or_query: str, columns: str) -> dict[str, Any]:
        sql = None
        try:
            source, real_columns = self.source(table_or_query)
            names = split_names(columns)
            if len(names) < 2:
                raise SqlToolError("columns must name at least two columns.")

            quoted = [self.column(name, real_columns) for name in names]
            pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
            parts = [f"CORR({quoted[i]}, {quoted[j]}) AS r_{i}_{j}" for i, j in pairs]
            sql = f"SELECT COUNT(*) AS n, {', '.join(parts)} FROM {source}"

            result_columns, rows, _ = self.run(sql, 1)
            row = dict(zip(result_columns, rows[0])) if rows else {}

            matrix: dict[str, dict[str, Any]] = {name: {other: None for other in names} for name in names}
            for i, name in enumerate(names):
                matrix[name][name] = 1.0
            for i, j in pairs:
                value = round_number(row.get(f"r_{i}_{j}"))
                matrix[names[i]][names[j]] = value
                matrix[names[j]][names[i]] = value

            return {"kind": "correlation", "rows": round_number(row.get("n")), "matrix": matrix}
        except Exception as e:
            return self.failure(e, sql)
