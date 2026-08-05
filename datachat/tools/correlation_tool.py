import logging
from typing import Any, ClassVar

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan


def _strength_label(corr: float) -> str:
    """Plain-language magnitude, so a coefficient is not mistaken for a big finding."""
    magnitude = abs(corr)
    if magnitude < 0.1:
        return "none"
    if magnitude < 0.3:
        return "weak"
    if magnitude < 0.5:
        return "moderate"
    if magnitude < 0.7:
        return "strong"
    return "very strong"


class CorrelationTool(Tool):
    """
    Compute correlation between two numeric columns.
    """

    name = "correlation"
    description = (
        "Measure how strongly numeric columns move together. "
        "Give col_x and col_y for a single pair; give only col_x to rank every other "
        "numeric column against it (e.g. 'which questions relate to overall satisfaction?'); "
        "give neither for all pairs. "
        "Use method='spearman' for rating scales and any 1-5 answer -- 'pearson' assumes "
        "evenly spaced values, which ordinal scales are not."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "col_x": {
            "type": "string",
            "description": (
                "Numeric column. Omit together with col_y to get every pair of numeric "
                "columns."
            ),
            "nullable": True,
        },
        "col_y": {
            "type": "string",
            "description": (
                "Second numeric column. Omit to compare col_x against every other numeric "
                "column, ranked by strength."
            ),
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
            "description": (
                "'pearson' (default, linear, for continuous measures) or 'spearman' "
                "(rank-based, correct for ordinal rating scales)."
            ),
            "enum": ["pearson", "spearman"],
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
            allowed_methods = {"pearson", "spearman"}
            if method_clean not in allowed_methods:
                return {
                    "kind": "error",
                    "message": f"Invalid method '{method_clean}'. Allowed: {sorted(allowed_methods)}",
                    "code": "INVALID_METHOD",
                }

            # One column, or none: return a ranked matrix instead of a single pair.
            if not y:
                return self._correlation_matrix(df, anchor=x or None, method=method_clean)

            if not x:
                return {"kind": "error", "message": "Missing col_x.", "code": "MISSING_COLUMNS"}

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
                "correlation": round(corr, 6),
                "strength": _strength_label(corr),
                "n": int(len(tmp)),
            }
            records = replace_nan([row])

            logging.info(
                "[datachat][correlation_tool] x=%s y=%s method=%s n=%s corr=%.6f",
                x, y, method_clean, len(tmp), corr,
            )
            return {
                "kind": "table",
                "data": records,
                "export_name": f"correlation_{x}_{y}",
            }

        except Exception as e:
            logging.exception("[datachat][correlation_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    # ------------------------------------------------------------------
    # Matrix mode
    # ------------------------------------------------------------------

    def _correlation_matrix(
        self,
        df: pd.DataFrame,
        anchor: str | None,
        method: str,
    ) -> dict[str, Any]:
        """
        Long-format correlations: every numeric column against `anchor`, or all pairs.

        Long rather than wide so the result stays a normal table -- previewable, exportable
        and sortable by strength, which is what "which questions matter most?" needs.
        """
        numeric: dict[str, pd.Series] = {}
        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            # Require some spread: a constant column correlates with nothing.
            if series.notna().sum() >= 2 and series.nunique(dropna=True) >= 2:
                numeric[str(col)] = series

        if anchor is not None and anchor not in numeric:
            if anchor not in df.columns:
                return {"kind": "error", "message": f"Invalid col_x: {anchor}", "code": "INVALID_COLUMN"}
            return {
                "kind": "error",
                "message": f"Column '{anchor}' has not enough numeric variation to correlate.",
                "code": "NO_NUMERIC_DATA",
            }

        if len(numeric) < 2:
            return {
                "kind": "error",
                "message": "Need at least two numeric columns with variation to build a correlation matrix.",
                "code": "INSUFFICIENT_DATA",
            }

        names = list(numeric)
        pairs = (
            [(anchor, other) for other in names if other != anchor]
            if anchor is not None
            else [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        )

        rows: list[dict[str, Any]] = []
        for a, b in pairs:
            tmp = pd.DataFrame({a: numeric[a], b: numeric[b]}).dropna()
            if len(tmp) < 2 or tmp[a].nunique() < 2 or tmp[b].nunique() < 2:
                continue
            corr = tmp[a].corr(tmp[b], method=method)
            if pd.isna(corr):
                continue
            rows.append(
                {
                    "col_x": a,
                    "col_y": b,
                    "method": method,
                    "correlation": round(float(corr), 6),
                    "strength": _strength_label(float(corr)),
                    "n": int(len(tmp)),
                }
            )

        if not rows:
            return {
                "kind": "error",
                "message": "No column pair had enough overlapping numeric values.",
                "code": "INSUFFICIENT_DATA",
            }

        # Strongest relationship first, sign ignored: a strong negative matters as much.
        rows.sort(key=lambda r: -abs(r["correlation"]))

        logging.info(
            "[datachat][correlation_tool] matrix anchor=%s method=%s numeric_cols=%s pairs=%s",
            anchor or "all", method, len(numeric), len(rows),
        )
        return {
            "kind": "table",
            "data": replace_nan(rows),
            "export_name": f"correlation_{anchor}" if anchor else "correlation_matrix",
            "meta": {"method": method, "anchor": anchor, "numeric_columns": len(numeric)},
        }
