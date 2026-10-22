from typing import Any, Mapping, Optional, Sequence

from datachat.sql_datasource import SqlDatasource
from datachat.tools.sql_engine_tool import SqlEngineTool


class AnalystSqlEngineTool(SqlEngineTool):
    """
    DataChat's sql_engine, with the analyst's own row cap.

    Its rows land in the agent's context, so the analyst keeps them few
    (ANALYST_SQL_MAX_ROWS): data to compute on goes through load_dataset.
    """

    def __init__(
        self,
        datasource: SqlDatasource,
        *,
        max_rows: int,
        identifiers: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__(datasource, identifiers=identifiers)
        self._max_rows = max_rows

    def forward(self, query: str, max_rows: Optional[int] = None) -> dict[str, Any]:
        try:
            limit = min(int(max_rows), self._max_rows) if max_rows is not None else self._max_rows
        except (TypeError, ValueError):
            limit = self._max_rows
        return super().forward(query, max_rows=limit)
