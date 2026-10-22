import re
from decimal import Decimal
from typing import Any, ClassVar, Mapping, Optional, Sequence

import pandas as pd

from analyst.tools.sql_common import SqlToolBase, SqlToolError
from datachat.sql_datasource import SqlDatasource

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _to_frame(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame([tuple(r) for r in rows], columns=list(columns))
    for name in frame.columns:
        values = frame[name].dropna()
        # NUMERIC columns arrive as Decimal, which pandas keeps as objects.
        if not values.empty and all(isinstance(v, Decimal) for v in values):
            frame[name] = frame[name].astype(float)
    return frame


class LoadDatasetTool(SqlToolBase):
    """
    Load the result of a SELECT into a named DataFrame, out of the agent's context.

    The DataFrame goes into the run's `datasets` dict, which is in every task agent's namespace,
    and only its shape and three rows come back to the model.
    The rows never pass through the context, and a later sub-task can reuse the dataset by name.
    """

    name = "load_dataset"
    description = (
        "Run one read-only SQL SELECT and load its result into the DataFrame "
        "datasets[name], available to your code in this and later steps (and to later "
        "sub-tasks). Returns only the row count, the column types and 3 sample rows. "
        "Aggregate in SQL whenever you can: rows beyond the server limit are dropped "
        "and the response says so. Use it for data you need to compute on; use "
        "sql_engine for a quick look at a few rows."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "name": {
            "type": "string",
            "description": "snake_case dataset name, e.g. 'monthly_trips'. An existing name is replaced.",
        },
        "query": {
            "type": "string",
            "description": "A single read-only SQL SELECT statement (PostgreSQL dialect).",
        },
    }

    def __init__(
        self,
        datasource: SqlDatasource,
        datasets: dict[str, pd.DataFrame],
        *,
        max_rows: int,
        identifiers: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__(datasource, identifiers)
        self._datasets = datasets
        self._max_rows = max_rows

    def forward(self, name: str, query: str) -> dict[str, Any]:
        sql = str(query or "").strip()
        try:
            key = str(name or "").strip()
            if not _NAME_RE.match(key):
                raise SqlToolError("name must be snake_case (letters, digits, underscores).")

            columns, rows, truncated = self.run(sql, self._max_rows)
            frame = _to_frame(columns, rows)
            self._datasets[key] = frame

            result: dict[str, Any] = {
                "kind": "dataset",
                "name": key,
                "rows": len(frame),
                "truncated": truncated,
                "columns": {str(c): str(t) for c, t in frame.dtypes.items()},
                "sample": self.records(columns, rows[:3]),
            }
            if truncated:
                result["warning"] = (
                    f"Only the first {self._max_rows} rows were loaded, not all the matching "
                    "rows: totals, averages and rankings computed on it are wrong. Aggregate in "
                    "SQL (GROUP BY), or load a random sample with TABLESAMPLE SYSTEM (<percent>) "
                    "REPEATABLE (42)."
                )
            elif frame.empty:
                result["warning"] = (
                    "No rows matched: check the real values of the columns you filtered on."
                )
            return result
        except Exception as e:
            return self.failure(e, sql)
