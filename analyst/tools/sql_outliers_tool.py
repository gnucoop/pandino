from typing import Any, ClassVar, Mapping, Optional, Sequence

from analyst.tools.sql_common import SOURCE_ALIAS, SqlToolBase, SqlToolError, round_number
from datachat.sql_datasource import SqlDatasource

_DEFAULT_THRESHOLDS = {"zscore": 3.0, "iqr": 1.5}


class SqlOutliersTool(SqlToolBase):
    """
    Outliers of one numeric column, found in the database.

    The total comes from a window count in the same query,
    so the sample is capped without a second round trip.
    """

    name = "sql_outliers"
    description = (
        "Find outliers of one numeric column in the database: method 'zscore' (|z| above "
        "threshold, default 3) or 'iqr' (outside Q1/Q3 -/+ threshold x IQR, default 1.5). "
        "Returns the total number of outliers, the bounds used and a sample of the most "
        "extreme rows."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "table_or_query": {
            "type": "string",
            "description": "A relation name from the schema, or a read-only SELECT query.",
        },
        "column": {"type": "string", "description": "The numeric column to check."},
        "method": {
            "type": "string",
            "description": "'zscore' (default) or 'iqr'.",
            "nullable": True,
        },
        "threshold": {
            "type": "number",
            "description": "z-score limit, or IQR multiplier. Defaults: 3 for zscore, 1.5 for iqr.",
            "nullable": True,
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

    def forward(
        self,
        table_or_query: str,
        column: str,
        method: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        sql = None
        try:
            method = str(method or "zscore").strip().lower()
            if method not in _DEFAULT_THRESHOLDS:
                raise SqlToolError("method must be 'zscore' or 'iqr'.")
            limit = float(threshold) if threshold is not None else _DEFAULT_THRESHOLDS[method]

            source, real_columns = self.source(table_or_query)
            col = f"{SOURCE_ALIAS}.{self.column(column, real_columns)}"

            if method == "zscore":
                bounds = f"SELECT AVG({col}) AS center, STDDEV({col}) AS spread FROM {source}"
                score = f"({col} - b.center) / b.spread"
                condition = f"b.spread > 0 AND ABS({score}) > {limit}"
                order = f"ABS({score}) DESC"
                low_high = f"b.center - {limit} * b.spread, b.center + {limit} * b.spread"
            else:
                bounds = (
                    f"SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) AS q1, "
                    f"PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) AS q3 FROM {source}"
                )
                low = f"(b.q1 - {limit} * (b.q3 - b.q1))"
                high = f"(b.q3 + {limit} * (b.q3 - b.q1))"
                condition = f"({col} < {low} OR {col} > {high})"
                order = f"GREATEST({low} - {col}, {col} - {high}) DESC"
                low_high = f"{low}, {high}"

            sql = (
                f"WITH b AS ({bounds}) "
                # The extra columns come first: records() keeps the leading columns.
                f"SELECT COUNT(*) OVER (), {low_high}, {SOURCE_ALIAS}.* "
                f"FROM {source} CROSS JOIN b WHERE {condition} ORDER BY {order}"
            )
            columns, rows, _ = self.run(sql, self._max_rows)

            total, low_bound, high_bound = (rows[0][:3] if rows else (0, None, None))
            records = self.records(columns[3:], [row[3:] for row in rows])

            return {
                "kind": "outliers",
                "method": method,
                "threshold": round_number(limit),
                "total_outliers": round_number(total),
                "bounds": {"low": round_number(low_bound), "high": round_number(high_bound)}
                if rows
                else None,
                "sample": records,
            }
        except Exception as e:
            return self.failure(e, sql)
