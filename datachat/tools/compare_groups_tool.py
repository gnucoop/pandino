import logging
import math
from typing import Any, ClassVar, Optional

import pandas as pd
from smolagents import Tool

from datachat.output_normalizer import replace_nan
from datachat.tools.limits import MIN_RELIABLE_SAMPLE, join_notes, sample_warning


_P_FLOOR = 1e-6


def _format_p(p: float) -> str:
    """Never render a p-value as 0: below the floor it is a bound, not a measurement."""
    return f"<{_P_FLOOR:g}" if p < _P_FLOOR else f"{p:.4g}"


def _p_phrase(p: float) -> str:
    """'p<1e-06' or 'p=0.0423' -- avoids writing 'p=<1e-06' with two operators."""
    return f"p<{_P_FLOOR:g}" if p < _P_FLOOR else f"p={p:.4g}"


def _cohens_d(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Standardised mean difference, using the pooled SD."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled <= 0 or math.isnan(pooled):
        return None
    return float((a.mean() - b.mean()) / math.sqrt(pooled))


def _effect_label(d: Optional[float]) -> str:
    """Conventional Cohen thresholds, for a plain-language summary."""
    if d is None:
        return "unknown"
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


class CompareGroupsTool(Tool):
    """
    Compare a numeric measure between two groups and say whether the gap is meaningful.

    Ranking group averages is easy and misleading: 800 responses over 80 courses leaves
    ~10 each, so the top of any ranking is mostly noise. This tool reports the difference
    together with a confidence interval, an effect size and an explicit verdict, so the
    agent can state that a gap is *not* a real finding.
    """

    name = "compare_groups"
    description = (
        "Compare a numeric column between two groups and report whether the difference is "
        "statistically meaningful: n, mean and median per group, the difference, a 95% "
        "confidence interval, effect size and a verdict. "
        "Use this before saying one group scores higher than another -- 'aggregate' gives "
        "averages but cannot tell a real gap from noise."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "metric": {
            "type": "string",
            "description": "Numeric column to compare (e.g. a 1-5 rating).",
        },
        "group_col": {
            "type": "string",
            "description": "Column that identifies the groups.",
        },
        "group_a": {
            "type": "string",
            "description": "First group value. Omit to use the two largest groups.",
            "nullable": True,
        },
        "group_b": {
            "type": "string",
            "description": "Second group value. Omit to use the two largest groups.",
            "nullable": True,
        },
        "ordinal": {
            "type": "boolean",
            "description": (
                "True (default) for rating scales: adds a Mann-Whitney test, which does not "
                "assume the 1-5 steps are evenly spaced. Set False for genuinely continuous "
                "measures."
            ),
            "nullable": True,
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
        group_a: Optional[str] = None,
        group_b: Optional[str] = None,
        ordinal: Optional[bool] = True,
        data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            try:
                from scipy import stats
            except ImportError:
                return {
                    "kind": "error",
                    "message": "scipy is required for group comparison. Install it: pip install scipy",
                    "code": "MISSING_SCIPY",
                }

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

            metric_clean = (metric or "").strip()
            group_clean = (group_col or "").strip()

            if not metric_clean or not group_clean:
                return {
                    "kind": "error",
                    "message": "Both 'metric' and 'group_col' are required.",
                    "code": "MISSING_PARAMS",
                }

            missing = [c for c in (metric_clean, group_clean) if c not in df.columns]
            if missing:
                return {"kind": "error", "message": f"Column not found: {', '.join(missing)}", "code": "INVALID_COLUMN"}

            work = pd.DataFrame(
                {
                    "group": df[group_clean].astype(str).str.strip(),
                    "value": pd.to_numeric(df[metric_clean], errors="coerce"),
                }
            ).dropna(subset=["value"])
            work = work[work["group"] != ""]

            if work.empty:
                return {
                    "kind": "error",
                    "message": f"No numeric values found in '{metric_clean}'.",
                    "code": "NO_NUMERIC_DATA",
                }

            # Default to the two best-supported groups: comparing the two thinnest is the
            # least informative thing the agent could do by accident.
            sizes = work["group"].value_counts()
            a_label = (group_a or "").strip()
            b_label = (group_b or "").strip()

            if not a_label or not b_label:
                if len(sizes) < 2:
                    return {
                        "kind": "error",
                        "message": f"'{group_clean}' has fewer than two groups with data.",
                        "code": "NOT_ENOUGH_GROUPS",
                    }
                a_label, b_label = str(sizes.index[0]), str(sizes.index[1])

            if a_label == b_label:
                return {"kind": "error", "message": "group_a and group_b must differ.", "code": "SAME_GROUP"}

            unknown = [g for g in (a_label, b_label) if g not in set(sizes.index)]
            if unknown:
                available = ", ".join(str(g) for g in list(sizes.index)[:10])
                return {
                    "kind": "error",
                    "message": f"Group not found: {', '.join(unknown)}. Available include: {available}",
                    "code": "INVALID_GROUP",
                }

            a = work.loc[work["group"] == a_label, "value"]
            b = work.loc[work["group"] == b_label, "value"]

            if len(a) < 2 or len(b) < 2:
                return {
                    "kind": "error",
                    "message": (
                        f"Need at least 2 values per group: '{a_label}' has {len(a)}, "
                        f"'{b_label}' has {len(b)}."
                    ),
                    "code": "GROUP_TOO_SMALL",
                }

            diff = float(a.mean() - b.mean())

            # Welch's t-test: does not assume the two groups share a variance.
            t_stat, p_welch = stats.ttest_ind(a, b, equal_var=False)

            # Welch-Satterthwaite CI for the difference of means.
            se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            if se > 0:
                dof_num = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2
                dof_den = (
                    (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                    + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
                )
                dof = dof_num / dof_den if dof_den > 0 else len(a) + len(b) - 2
                margin = float(stats.t.ppf(0.975, dof) * se)
                ci_low, ci_high = diff - margin, diff + margin
            else:
                ci_low = ci_high = diff

            p_ordinal = None
            use_ordinal = bool(ordinal) if ordinal is not None else True
            if use_ordinal:
                try:
                    _, p_ordinal = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    # Raised when both groups are constant and identical.
                    p_ordinal = None

            # The ordinal test is the one to trust on rating scales when both are available.
            p_value = float(p_ordinal) if p_ordinal is not None else float(p_welch)
            significant = bool(p_value < 0.05)
            d = _cohens_d(a, b)
            effect = _effect_label(d)

            smallest = min(len(a), len(b))
            reliable = smallest >= MIN_RELIABLE_SAMPLE

            p_display = _format_p(p_value)
            p_phrase = _p_phrase(p_value)

            if not significant:
                verdict = (
                    f"No meaningful difference: '{a_label}' and '{b_label}' are "
                    f"statistically indistinguishable ({p_phrase}). Do not report one "
                    f"as better than the other."
                )
            else:
                higher, lower = (a_label, b_label) if diff > 0 else (b_label, a_label)
                verdict = (
                    f"'{higher}' scores higher than '{lower}' by {abs(diff):.2f} "
                    f"({p_phrase}, {effect} effect)."
                )
                if not reliable:
                    verdict += " Treat with caution given the small samples."

            # Exactly 10 fields, in order of what a reader needs first: the client previews
            # only the first 10 columns, so a wider record would hide the conclusion. The
            # supporting detail goes in `meta` and the verdict in `note`, both of which
            # reach the client whole.
            record = {
                "group_a": a_label,
                "n_a": int(len(a)),
                "mean_a": round(float(a.mean()), 4),
                "group_b": b_label,
                "n_b": int(len(b)),
                "mean_b": round(float(b.mean()), 4),
                "difference": round(diff, 4),
                "ci95": f"[{ci_low:.2f}, {ci_high:.2f}]",
                "p_value": p_display,
                "significant": significant,
            }

            logging.info(
                "[datachat][compare_groups_tool] metric=%s group_col=%s a=%s(n=%s) b=%s(n=%s) "
                "diff=%.4f p=%.4f test=%s significant=%s",
                metric_clean, group_clean, a_label, len(a), b_label, len(b),
                diff, p_value, "mann-whitney" if p_ordinal is not None else "welch t-test", significant,
            )

            payload: dict[str, Any] = {
                "kind": "table",
                "data": replace_nan([record]),
                "export_name": f"compare_{a_label}_vs_{b_label}",
                "meta": {
                    "metric": metric_clean,
                    "group_col": group_clean,
                    "median_a": round(float(a.median()), 4),
                    "median_b": round(float(b.median()), 4),
                    # Floored, not rounded: round(1e-30, 6) would report exactly 0.
                    "p_value_exact": max(p_value, _P_FLOOR) if p_value < _P_FLOOR else round(p_value, 6),
                    "ci95_low": round(float(ci_low), 4),
                    "ci95_high": round(float(ci_high), 4),
                    "test": "mann-whitney" if p_ordinal is not None else "welch t-test",
                    "effect": effect,
                    "effect_size_d": round(d, 4) if d is not None else None,
                },
            }
            # The verdict is the answer, so it goes where nothing can trim it.
            payload["note"] = join_notes(
                verdict,
                sample_warning(smallest, label="group", count=2 if not reliable else 1),
            )
            return payload

        except Exception as e:
            logging.exception("[datachat][compare_groups_tool] failed")
            return {"kind": "error", "message": str(e), "code": "TOOL_FAILED"}
