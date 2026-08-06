import logging
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan
from datachat.tools.limits import (
    MIN_RELIABLE_SAMPLE,
    join_notes,
    sample_warning,
)


def _to_json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


# Language-neutral sentinel for missing categories; see DINO_CLIENT_SPEC.md.
_BLANK_LABEL = "(empty)"

# What pandas produces when a missing value has already been stringified upstream.
_NULL_TEXTS = {"", "nan", "none", "<na>", "nat", "null"}


def _labelled(series: pd.Series) -> pd.Series:
    """
    Categories as strings, with every flavour of missing collapsed to one sentinel.

    isna() on the original series is authoritative; the string check catches values that were
    already converted before reaching us -- astype(str) turns None into the literal "None",
    which would otherwise appear as a category of its own.
    """
    text = series.astype(str).str.strip()
    blank = series.isna() | text.str.lower().isin(_NULL_TEXTS)
    return text.mask(blank, _BLANK_LABEL)


class CrosstabTool(Tool):
    """
    Two-dimensional breakdown: one column down the rows, another across the columns.

    This is the shape most survey questions actually have ("satisfaction by course AND by
    role"). `aggregate` returns a long table and handles at most two grouping columns;
    crosstab returns the wide table people expect to read, and can express each cell as a
    percentage of its row or column.
    """

    name = "crosstab"
    description = (
        "Cross-tabulate two columns: one down the rows, one across the columns, with counts "
        "or an aggregated metric in the cells. Use for questions with two dimensions, e.g. "
        "'satisfaction by course and by role' or 'answer distribution per group'. "
        "Set normalize='rows' or 'columns' for percentages. "
        "For a single dimension use 'aggregate' instead."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "rows": {
            "type": "string",
            "description": "Column whose values become the table rows.",
        },
        "columns": {
            "type": "string",
            "description": "Column whose values become the table columns.",
        },
        "metric": {
            "type": "string",
            "description": (
                "Optional numeric column to aggregate in each cell. "
                "Omit to count rows instead."
            ),
            "nullable": True,
        },
        "op": {
            "type": "string",
            "description": (
                "Aggregation for the cells: 'count' (default), 'mean', 'sum', 'min', 'max'. "
                "Anything other than 'count' requires 'metric'."
            ),
            "enum": ["count", "mean", "sum", "min", "max"],
            "nullable": True,
        },
        "normalize": {
            "type": "string",
            "description": (
                "Express cells as percentages: 'rows' (each row sums to 100), "
                "'columns' (each column sums to 100), 'all', or 'none' (default). "
                "Only valid with op='count'."
            ),
            "enum": ["none", "rows", "columns", "all"],
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the crosstab runs on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        rows: str,
        columns: str,
        metric: Optional[str] = None,
        op: Optional[str] = None,
        normalize: Optional[str] = None,
        data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")
                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "table", "data": []}
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            row_col = (rows or "").strip()
            col_col = (columns or "").strip()

            if not row_col or not col_col:
                return {
                    "kind": "error",
                    "message": "Both 'rows' and 'columns' are required.",
                    "code": "MISSING_DIMENSION",
                }

            missing = [c for c in (row_col, col_col) if c not in df.columns]
            if missing:
                return {
                    "kind": "error",
                    "message": f"Column not found: {', '.join(missing)}",
                    "code": "INVALID_COLUMN",
                }

            if row_col == col_col:
                return {
                    "kind": "error",
                    "message": "'rows' and 'columns' must be different columns.",
                    "code": "SAME_DIMENSION",
                }

            op_clean = (op or "count").strip().lower()
            allowed_ops = {"count", "mean", "sum", "min", "max"}
            if op_clean not in allowed_ops:
                return {
                    "kind": "error",
                    "message": f"Invalid op '{op_clean}'. Allowed: {sorted(allowed_ops)}",
                    "code": "INVALID_OP",
                }

            metric_clean = (metric or "").strip() or None
            if op_clean != "count":
                if not metric_clean:
                    return {
                        "kind": "error",
                        "message": f"op='{op_clean}' requires a 'metric' column.",
                        "code": "MISSING_METRIC",
                    }
                if metric_clean not in df.columns:
                    return {
                        "kind": "error",
                        "message": f"Invalid metric column: {metric_clean}",
                        "code": "INVALID_METRIC",
                    }

            norm_clean = (normalize or "none").strip().lower()
            allowed_norm = {"none", "rows", "columns", "all"}
            if norm_clean not in allowed_norm:
                return {
                    "kind": "error",
                    "message": f"Invalid normalize '{norm_clean}'. Allowed: {sorted(allowed_norm)}",
                    "code": "INVALID_NORMALIZE",
                }
            if norm_clean != "none" and op_clean != "count":
                return {
                    "kind": "error",
                    "message": "normalize only applies to op='count'.",
                    "code": "INVALID_NORMALIZE_OP",
                }

            # Blank cells become an explicit label rather than vanishing: a crosstab that
            # quietly drops the non-responders misstates every percentage in the table.
            row_series = _labelled(df[row_col])
            col_series = _labelled(df[col_col])

            if op_clean == "count":
                normalize_arg: Any = {
                    "none": False,
                    "rows": "index",
                    "columns": "columns",
                    "all": "all",
                }[norm_clean]
                table = pd.crosstab(row_series, col_series, normalize=normalize_arg)
                if norm_clean != "none":
                    table = (table * 100).round(2)
            else:
                metric_num = pd.to_numeric(df[metric_clean], errors="coerce")
                tmp = pd.DataFrame(
                    {row_col: row_series, col_col: col_series, metric_clean: metric_num}
                )
                table = tmp.pivot_table(
                    index=row_col,
                    columns=col_col,
                    values=metric_clean,
                    aggfunc=op_clean,
                    dropna=False,
                )
                table = table.round(4)

            total_rows = int(table.shape[0])
            total_cols = int(table.shape[1])

            # Flatten to records: the row label plus one field per column value.
            table = table.reset_index()
            table.columns = [str(c) for c in table.columns]
            records = [
                {str(k): _to_json_scalar(v) for k, v in row.items()}
                for row in table.to_dict(orient="records")
            ]
            records = replace_nan(records)

            # Cell counts drive the caveat: a mean over a 2-row cell is not a finding.
            cell_counts = pd.crosstab(row_series, col_series)
            smallest_cell = int(cell_counts.to_numpy().min()) if cell_counts.size else 0
            thin_cells = int((cell_counts.to_numpy() < MIN_RELIABLE_SAMPLE).sum()) if cell_counts.size else 0

            small_note = None
            if op_clean != "count" and thin_cells:
                small_note = sample_warning(smallest_cell, label="cell", count=thin_cells)

            logging.info(
                "[datachat][crosstab_tool] rows=%s columns=%s op=%s metric=%s normalize=%s "
                "shape=%sx%s smallest_cell=%s",
                row_col, col_col, op_clean, metric_clean, norm_clean,
                total_rows, total_cols, smallest_cell,
            )

            payload: dict[str, Any] = {
                "kind": "table",
                "data": records,
                "export_name": f"{op_clean}_{metric_clean or 'count'}_{row_col}_x_{col_col}",
                "meta": {
                    "rows": row_col,
                    "columns": col_col,
                    "op": op_clean,
                    "metric": metric_clean,
                    "normalize": norm_clean,
                    "row_values": total_rows,
                    "column_values": total_cols,
                },
            }
            note = join_notes(small_note)
            if note:
                payload["note"] = note
            return payload

        except Exception as e:
            logging.exception("[datachat][crosstab_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
