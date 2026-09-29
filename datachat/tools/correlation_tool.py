import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = {"pearson", "spearman"}


class CorrelationTool(Tool):
    """
    Compute correlation between two numeric columns, or rank correlations
    against one column / across all numeric column pairs.
    """

    name = "correlation"
    description = (
        "Compute the correlation coefficient between numeric columns. "
        "Give col_x and col_y for a single pair (single-row table). "
        "Give only col_x to rank every other numeric column against it, or neither "
        "for all numeric column pairs; rankings are ordered by absolute correlation. "
        "method='pearson' (default) or 'spearman' (rank-based, suited to ordinal rating scales)."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "col_x": {
            "type": "string",
            "description": "First numeric column. Omit together with col_y for all numeric pairs.",
            "nullable": True,
        },
        "col_y": {
            "type": "string",
            "description": "Second numeric column. Omit to rank all numeric columns against col_x.",
            "nullable": True,
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, correlation will be computed on this data instead of the session dataset."
            ),
            "items": {"type": "object"},
            "nullable": True,
        },
        "method": {
            "type": "string",
            "description": "Correlation method: 'pearson' (default) or 'spearman'.",
            "nullable": True,
        },
    }

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df

    def forward(
        self,
        col_x: str | None = None,
        col_y: str | None = None,
        data: list[dict[str, Any]] | None = None,
        method: str | None = None
    ) -> dict[str, Any]:
        
        try:
            if data is not None:
                if isinstance(data, dict) and "data" in data:
                    data = data.get("data")

                if not isinstance(data, list):
                    return {"kind": "error", "message": "Invalid data: expected a list of records.", "code": "INVALID_DATA"}
                if len(data) == 0:
                    return {"kind": "error", "message": "Not enough data to compute correlation.", "code": "INSUFFICIENT_DATA"}

                try:
                    df = pd.DataFrame(data)
                except Exception:
                    return {"kind": "error", "message": "Invalid data: could not build a table from records.", "code": "INVALID_DATA"}
            else:
                df = self._df

            x = (col_x or "").strip()
            y = (col_y or "").strip()

            method_clean = (method or "pearson").strip().lower()
            if method_clean not in _ALLOWED_METHODS:
                return {"kind": "error", "message": f"Invalid method: {method_clean}", "code": "INVALID_METHOD"}

            if not y:
                if x and x not in df.columns and data is not None:
                    x = {c.lower(): c for c in df.columns}.get(x.lower(), x)
                return self._ranking(df, anchor=x or None, method=method_clean)

            if not x:
                return {"kind": "error", "message": "Missing col_x or col_y.", "code": "MISSING_COLUMNS"}

            if x not in df.columns or y not in df.columns:
                if data is not None:
                    lowered = {c.lower(): c for c in df.columns}
                    if x not in df.columns:
                        hit = lowered.get(x.lower())
                        if hit:
                            x = hit
                    if y not in df.columns:
                        hit = lowered.get(y.lower())
                        if hit:
                            y = hit

            if x not in df.columns:
                return {"kind": "error", "message": f"Invalid col_x: {x}", "code": "INVALID_COLUMN"}
            if y not in df.columns:
                return {"kind": "error", "message": f"Invalid col_y: {y}", "code": "INVALID_COLUMN"}

            s_x = pd.to_numeric(df[x], errors="coerce")
            s_y = pd.to_numeric(df[y], errors="coerce")

            x_valid = int(s_x.notna().sum())
            y_valid = int(s_y.notna().sum())

            # If one column has (almost) no numeric values, correlation is not the right operation
            if x_valid < 2:
                return {
                    "kind": "error",
                    "message": f"Column '{x}' has not enough numeric values to compute correlation.",
                    "code": "NO_NUMERIC_DATA",
                }
            if y_valid < 2:
                return {
                    "kind": "error",
                    "message": f"Column '{y}' has not enough numeric values to compute correlation.",
                    "code": "NO_NUMERIC_DATA",
                }

            tmp = pd.DataFrame({x: s_x, y: s_y}).dropna()

            if len(tmp) < 2:
                return {
                    "kind": "error",
                    "message": "Not enough valid numeric pairs to compute correlation.",
                    "code": "INSUFFICIENT_DATA",
                }

            if tmp[x].nunique(dropna=True) < 2 or tmp[y].nunique(dropna=True) < 2:
                return {
                    "kind": "error",
                    "message": "One of the columns has zero variance; correlation is undefined.",
                    "code": "ZERO_VARIANCE",
                }

            corr = float(tmp[x].corr(tmp[y], method=method_clean))
            row = {
                "col_x": x,
                "col_y": y,
                "method": method_clean,
                "correlation": corr,
                "n": int(len(tmp)),
            }
            records = replace_nan([row])

            logger.info("event=tool_call_result x=%s y=%s n=%s corr=%.6f", x, y, len(tmp), corr)
            return {"kind": "table", "data": records}

        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    def _ranking(self, df: pd.DataFrame, anchor: str | None, method: str) -> dict[str, Any]:
        """
        Long-format correlations: every numeric column against `anchor`, or every
        unordered pair of numeric columns. Self-pairs are excluded; each pair appears
        once. Ordered by absolute correlation (desc), then col_x, col_y.
        """
        numeric: dict[str, pd.Series] = {}
        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().sum() >= 2 and series.nunique(dropna=True) >= 2:
                numeric[str(col)] = series

        if anchor is not None and anchor not in numeric:
            if anchor not in df.columns:
                return {"kind": "error", "message": f"Invalid col_x: {anchor}", "code": "INVALID_COLUMN"}
            return {
                "kind": "error",
                "message": f"Column '{anchor}' has not enough numeric variation to compute correlation.",
                "code": "NO_NUMERIC_DATA",
            }

        names = list(numeric)
        if anchor is not None:
            pairs = [(anchor, other) for other in names if other != anchor]
        else:
            pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

        rows: list[dict[str, Any]] = []
        for a, b in pairs:
            tmp = pd.DataFrame({"a": numeric[a], "b": numeric[b]}).dropna()
            if len(tmp) < 2 or tmp["a"].nunique() < 2 or tmp["b"].nunique() < 2:
                continue
            corr = tmp["a"].corr(tmp["b"], method=method)
            if pd.isna(corr):
                continue
            rows.append(
                {"col_x": a, "col_y": b, "method": method, "correlation": float(corr), "n": int(len(tmp))}
            )

        if not rows:
            return {
                "kind": "error",
                "message": "Not enough valid numeric pairs to compute correlation.",
                "code": "INSUFFICIENT_DATA",
            }

        rows.sort(key=lambda r: (-abs(r["correlation"]), r["col_x"], r["col_y"]))

        logger.info(
            "event=tool_call_result mode=ranking anchor=%s method=%s numeric_cols=%s pairs=%s",
            anchor, method, len(numeric), len(rows),
        )
        return {"kind": "table", "data": replace_nan(rows)}
