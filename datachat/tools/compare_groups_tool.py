import datetime as dt
import logging
import math
import numbers
import warnings
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from smolagents import Tool

from datachat.result_provenance import record_trusted_result

logger = logging.getLogger(__name__)

TEST_NAME = "welch_t"


class _ToolError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _is_missing(value: Any) -> bool:
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _group_class(value: Any) -> str:
    """
    Type class of a group key. Keys of different classes are never the same
    group, so True, 1 and "1" stay apart; within a class ordinary equality
    applies, as in pandas grouping (1 and 1.0 are one number).
    """
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (dt.datetime, dt.date, np.datetime64)):
        return "datetime"
    if isinstance(value, numbers.Number):
        return "number"
    if isinstance(value, str):
        return "string"
    return "other"


def _normalized(value: Any) -> Any:
    cls = _group_class(value)
    if cls == "boolean":
        return bool(value)
    if cls == "datetime":
        return pd.Timestamp(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _matches(value: Any, requested: Any) -> bool:
    """Typed equality between an observed group key and the requested one."""
    if _group_class(value) != _group_class(requested):
        return False
    try:
        return bool(_normalized(value) == _normalized(requested))
    except Exception:
        return False


def _select_group(keys: pd.Series, requested: Any) -> pd.Series:
    """
    Boolean mask of the rows whose (non-missing) group key is ``requested``.

    Matching is strict and typed: a string selects only an identical string
    key (whitespace included), never a number, boolean or date.
    """
    # Iterate the values: Series.map on a categorical column returns a categorical.
    values = keys.tolist()
    present = [not _is_missing(v) for v in values]

    def mask(test: Any) -> pd.Series:
        return pd.Series([p and test(v) for v, p in zip(values, present)], index=keys.index, dtype=bool)

    return mask(lambda v: _matches(v, requested))


def _json_group(value: Any) -> Any:
    value = _normalized(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _usable_number(value: Any) -> Optional[float]:
    """
    The finite numeric value of a metric cell, or None when it is unusable:
    missing, boolean, date/time, non-finite or not convertible. Numeric
    strings are converted explicitly.
    """
    if _is_missing(value) or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (dt.datetime, dt.date, dt.time, dt.timedelta, np.datetime64, np.timedelta64)):
        return None
    if isinstance(value, numbers.Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _finite_or_none(value: Any) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


class CompareGroupsTool(Tool):
    """
    Welch two-sample comparison of the mean of one numeric column between two
    explicitly named groups. Factual statistics only: no verdicts or labels.
    """

    name = "compare_groups"
    description = (
        "Compare the mean of a numeric column between two named groups of another column, "
        "using Welch's unpaired two-sample t-test (unequal variances). Returns one row: "
        "group sizes and means, difference = mean_a - mean_b, the t statistic, the two-sided "
        "p-value and the 95% confidence interval of the difference. Both groups must be named "
        "explicitly. Metric values that are missing, non-numeric, boolean, dates or infinite "
        "are excluded. It reports evidence only; a high p-value does not show the groups are equal."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "metric": {
            "type": "string",
            "description": "Numeric column whose mean is compared.",
        },
        "group_col": {
            "type": "string",
            "description": "Column that identifies the groups.",
        },
        "group_a": {
            "type": "any",
            "description": "Value of group_col selecting the first group (exact value, with its type).",
        },
        "group_b": {
            "type": "any",
            "description": "Value of group_col selecting the second group (exact value, with its type).",
        },
        "data": {
            "type": "array",
            "description": (
                "Optional table records (list of objects) produced by another tool. "
                "If provided, the comparison runs on this data instead of the session dataset."
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
        metric: str,
        group_col: str,
        group_a: Any,
        group_b: Any,
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

            return self._compare(df, metric, group_col, group_a, group_b)

        except _ToolError as e:
            return {"kind": "error", "message": e.message, "code": e.code}
        except Exception as e:
            logger.exception("event=tool_call_failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}

    def _compare(self, df: pd.DataFrame, metric: Any, group_col: Any, group_a: Any, group_b: Any) -> dict[str, Any]:
        from scipy import stats  # noqa: PLC0415

        metric_col = metric.strip() if isinstance(metric, str) else ""
        grp_col = group_col.strip() if isinstance(group_col, str) else ""
        if not metric_col or not grp_col:
            raise _ToolError("Both 'metric' and 'group_col' are required.", "MISSING_PARAMS")
        for param, value in (("group_a", group_a), ("group_b", group_b)):
            if _is_missing(value) or (isinstance(value, str) and value == ""):
                raise _ToolError(f"'{param}' is required: name both groups explicitly.", "MISSING_PARAMS")

        unknown = [c for c in (metric_col, grp_col) if c not in df.columns]
        if unknown:
            raise _ToolError(f"Invalid column: {', '.join(unknown)}", "INVALID_COLUMN")
        if metric_col == grp_col:
            raise _ToolError("'metric' and 'group_col' must be different columns.", "SAME_COLUMN")

        keys = df[grp_col]
        mask_a = _select_group(keys, group_a)
        mask_b = _select_group(keys, group_b)
        absent = [p for p, m in (("group_a", mask_a), ("group_b", mask_b)) if not m.any()]
        if absent:
            raise _ToolError(
                f"Group not found in '{grp_col}': {', '.join(absent)}.",
                "INVALID_GROUP",
            )
        if (mask_a & mask_b).any() or _matches(group_a, group_b):
            raise _ToolError("group_a and group_b must be two different groups.", "SAME_GROUP")
        label_a = _json_group(keys[mask_a].iloc[0])
        label_b = _json_group(keys[mask_b].iloc[0])

        values = df[metric_col]
        clean_a = [v for v in map(_usable_number, values[mask_a].tolist()) if v is not None]
        clean_b = [v for v in map(_usable_number, values[mask_b].tolist()) if v is not None]
        excluded_a = int(mask_a.sum()) - len(clean_a)
        excluded_b = int(mask_b.sum()) - len(clean_b)

        if not clean_a and not clean_b:
            raise _ToolError(
                f"No usable numeric values in '{metric_col}' for the selected groups.",
                "NO_NUMERIC_DATA",
            )
        if len(clean_a) < 2 or len(clean_b) < 2:
            raise _ToolError(
                "Each group needs at least 2 usable numeric values: "
                f"group_a has {len(clean_a)}, group_b has {len(clean_b)}.",
                "GROUP_TOO_SMALL",
            )

        a = np.asarray(clean_a, dtype=float)
        b = np.asarray(clean_b, dtype=float)
        mean_a, mean_b = float(a.mean()), float(b.mean())

        # Both groups constant: the standard error is zero and scipy would report
        # an infinite statistic with p=0 (or NaN), which is not a measurement.
        zero_variance = bool(np.ptp(a) == 0 and np.ptp(b) == 0)
        precision_warning = False
        statistic = p_value = ci_low = ci_high = None
        if not zero_variance:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = stats.ttest_ind(a, b, equal_var=False)
                ci = result.confidence_interval(confidence_level=0.95)
            precision_warning = any(issubclass(w.category, RuntimeWarning) for w in caught)
            statistic = _finite_or_none(result.statistic)
            p_value = _finite_or_none(result.pvalue)
            ci_low = _finite_or_none(ci.low)
            ci_high = _finite_or_none(ci.high)
        non_finite = not zero_variance and None in (statistic, p_value, ci_low, ci_high)
        if non_finite:
            statistic = p_value = ci_low = ci_high = None

        record = {
            "group_a": label_a,
            "group_b": label_b,
            "n_a": len(clean_a),
            "n_b": len(clean_b),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "difference": mean_a - mean_b,
            "test": TEST_NAME,
            "statistic": statistic,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

        notes: list[str] = []
        if excluded_a or excluded_b:
            notes.append(
                f"Excluded rows without a usable numeric '{metric_col}' value: "
                f"{excluded_a} in group_a, {excluded_b} in group_b."
            )
        if zero_variance:
            notes.append(
                "The t statistic, p-value and confidence interval are undefined "
                "because both groups have zero variance."
            )
        elif non_finite:
            notes.append(
                "The t-test did not return finite values; the t statistic, p-value "
                "and confidence interval are not reported."
            )
        elif precision_warning:
            notes.append(
                "The t-test reported a numerical precision warning (values nearly identical); "
                "the statistic and p-value may be inaccurate."
            )
        note = " ".join(notes) or None

        logger.info(
            "event=tool_call_result metric=%s group_col=%s n_a=%s n_b=%s excluded_a=%s excluded_b=%s "
            "zero_variance=%s non_finite=%s precision_warning=%s",
            metric_col,
            grp_col,
            len(clean_a),
            len(clean_b),
            excluded_a,
            excluded_b,
            zero_variance,
            non_finite,
            precision_warning,
        )

        payload: dict[str, Any] = {"kind": "table", "data": [record]}
        if note:
            payload["note"] = note
        return record_trusted_result(payload, note=note)
