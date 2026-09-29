import datetime as dt
import logging
import numbers
from collections import Counter
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from smolagents import Tool

logger = logging.getLogger(__name__)

MISSING_LABEL = "(empty)"

_ALLOWED_NORMALIZE = ("none", "rows", "columns", "all")

# Category classes, in the order their categories are listed; missing is last.
_BOOLEAN, _NUMBER, _DATETIME, _STRING, _OTHER, _MISSING = range(6)
_CLASS_NAMES = {
    _BOOLEAN: "boolean",
    _NUMBER: "number",
    _DATETIME: "datetime",
    _STRING: "string",
    _OTHER: "string",
    _MISSING: "missing",
}

_Key = tuple[int, Any]


def _category_key(value: Any) -> _Key:
    """
    Typed identity of an observed value.

    The class keeps categories apart that Python equality would merge
    (True == 1) or that share a display text (1 and "1"). Within a class,
    ordinary equality applies, as in pandas grouping.
    """
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return (_MISSING, None)
    if isinstance(value, (bool, np.bool_)):
        return (_BOOLEAN, bool(value))
    if isinstance(value, (dt.datetime, dt.date, np.datetime64)):
        return (_DATETIME, pd.Timestamp(value))
    if isinstance(value, numbers.Number):
        return (_NUMBER, value.item() if isinstance(value, np.generic) else value)
    if isinstance(value, str):
        return (_STRING, value)
    return (_OTHER, str(value))


def _sort_categories(keys: set[_Key]) -> list[_Key]:
    ordered: list[_Key] = []
    for cls in sorted({k[0] for k in keys}):
        group = [k for k in keys if k[0] == cls]
        try:
            group.sort(key=lambda k: k[1])
        except TypeError:
            # e.g. tz-aware and naive timestamps, or complex numbers
            group.sort(key=lambda k: str(k[1]))
        ordered.extend(group)
    return ordered


def _json_value(key: _Key) -> Any:
    cls, value = key
    if cls == _MISSING:
        return MISSING_LABEL
    if cls == _DATETIME:
        return value.isoformat()
    return value


def _text(key: _Key) -> str:
    value = _json_value(key)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _labels(keys: list[_Key], reserved: frozenset[str] = frozenset()) -> dict[_Key, str]:
    """
    Human-readable labels, type-qualified only where distinct categories
    (or a reserved structural name) would otherwise share the same text.
    """
    by_text: dict[str, list[_Key]] = {}
    for key in keys:
        by_text.setdefault(_text(key), []).append(key)

    labels: dict[_Key, str] = {}
    for text, group in by_text.items():
        qualify = len(group) > 1 or text in reserved
        for key in group:
            labels[key] = f"{text} [{_CLASS_NAMES[key[0]]}]" if qualify else text

    # A qualified label can still match another literal value; number the rare leftovers.
    taken = set(reserved)
    for key in keys:
        label, n = labels[key], 2
        while label in taken:
            label = f"{labels[key]} ({n})"
            n += 1
        labels[key] = label
        taken.add(label)
    return labels


class CrosstabTool(Tool):
    """
    Two-dimensional count matrix: one column down the rows, another across
    the columns, optionally normalised to proportions.
    """

    name = "crosstab"
    description = (
        "Cross-tabulate two columns: the values of `rows` down the table, the values of `columns` "
        "across it, and the number of observed rows in each cell (wide table). "
        "Set normalize='rows', 'columns' or 'all' to get proportions (0..1) of the row total, "
        "column total or grand total instead of counts. Missing values are counted as the "
        "category '(empty)'. Counts only: for sums, means or other metrics use 'aggregate'."
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
        "normalize": {
            "type": "string",
            "description": (
                "'none' (default, counts), 'rows' (each row sums to 1), "
                "'columns' (each column sums to 1) or 'all' (the whole table sums to 1)."
            ),
            "enum": list(_ALLOWED_NORMALIZE),
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

            row_col = rows.strip() if isinstance(rows, str) else ""
            col_col = columns.strip() if isinstance(columns, str) else ""
            if not row_col or not col_col:
                return {
                    "kind": "error",
                    "message": "Both 'rows' and 'columns' are required.",
                    "code": "MISSING_DIMENSION",
                }

            unknown = [c for c in (row_col, col_col) if c not in df.columns]
            if unknown:
                return {
                    "kind": "error",
                    "message": f"Invalid column: {', '.join(unknown)}",
                    "code": "INVALID_COLUMN",
                }

            if row_col == col_col:
                return {
                    "kind": "error",
                    "message": "'rows' and 'columns' must be different columns.",
                    "code": "SAME_DIMENSION",
                }

            if normalize is None:
                norm = "none"
            elif isinstance(normalize, str):
                norm = normalize.strip().lower()
            else:
                norm = ""
            if norm not in _ALLOWED_NORMALIZE:
                return {
                    "kind": "error",
                    "message": f"Invalid normalize '{normalize}'. Allowed: {list(_ALLOWED_NORMALIZE)}",
                    "code": "INVALID_NORMALIZE",
                }

            # Iterating the values (not the dtype's categories) keeps only observed categories.
            cells = Counter(
                (_category_key(r), _category_key(c))
                for r, c in zip(df[row_col].tolist(), df[col_col].tolist())
            )
            row_keys = _sort_categories({r for r, _ in cells})
            col_keys = _sort_categories({c for _, c in cells})

            row_totals: Counter = Counter()
            col_totals: Counter = Counter()
            for (r, c), n in cells.items():
                row_totals[r] += n
                col_totals[c] += n
            grand_total = sum(cells.values())

            col_labels = _labels(col_keys, reserved=frozenset({row_col}))
            row_labels = _labels(row_keys)

            records: list[dict[str, Any]] = []
            for r in row_keys:
                # Keep the native JSON value unless its text had to be disambiguated.
                row_value = _json_value(r) if row_labels[r] == _text(r) else row_labels[r]
                record: dict[str, Any] = {row_col: row_value}
                for c in col_keys:
                    n = cells.get((r, c), 0)
                    if norm == "rows":
                        record[col_labels[c]] = n / row_totals[r]
                    elif norm == "columns":
                        record[col_labels[c]] = n / col_totals[c]
                    elif norm == "all":
                        record[col_labels[c]] = n / grand_total
                    else:
                        record[col_labels[c]] = n
                records.append(record)

            logger.info(
                "event=tool_call_result rows=%s columns=%s normalize=%s shape=%sx%s",
                row_col,
                col_col,
                norm,
                len(row_keys),
                len(col_keys),
            )
            return {"kind": "table", "data": records}

        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
