"""
The aggregate small-sample note counts the metric values each operation actually
used, not the rows carrying the group key.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.limits import MIN_RELIABLE_SAMPLE


def _groups(out, cols, value_col):
    return {tuple(r[c] for c in cols): r[value_col] for r in out["data"]}


def _full(n, value=1.0):
    return [value] * n


def test_mean_counts_only_usable_metric_values():
    data = [{"service": "B", "score": 0.9}] + [{"service": "B", "score": None}] * 3
    out = AggregateTool(pd.DataFrame()).forward(group_by="service", op="mean", metric="score", data=data)
    assert _groups(out, ["service"], "mean_score") == {("B",): 0.9}
    assert out["note"] == f"1 group(s) are based on fewer than {MIN_RELIABLE_SAMPLE} rows (the smallest has 1)."


def test_multiple_groups_report_thin_count_and_smallest_effective_sample():
    df = pd.DataFrame(
        {
            "g": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            # A: 20 usable; B: 5 usable of 20; C: 3 usable of 20.
            "v": _full(20) + _full(5) + [None] * 15 + _full(3) + [np.nan] * 17,
        }
    )
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v")
    assert out["note"].startswith("2 group(s)")
    assert "smallest has 3" in out["note"]


def test_sum_ignores_missing_and_non_numeric_metric_values():
    df = pd.DataFrame({"g": ["A"] * 4, "v": [2, None, "n/a", 3]})
    out = AggregateTool(df).forward(group_by="g", op="sum", metric="v")
    assert _groups(out, ["g"], "sum_v") == {("A",): 5.0}
    assert "smallest has 2" in out["note"]


@pytest.mark.parametrize("op, expected", [("min", 1.0), ("max", 7.0)])
def test_min_max_count_only_usable_metric_values(op, expected):
    df = pd.DataFrame({"g": ["A"] * 5, "v": [1, 7, None, None, None]})
    out = AggregateTool(df).forward(group_by="g", op=op, metric="v")
    assert _groups(out, ["g"], f"{op}_v") == {("A",): expected}
    assert "smallest has 2" in out["note"]


def test_min_max_string_fallback_keeps_every_row():
    # An entirely non-numeric metric is compared as strings, so every row takes part.
    df = pd.DataFrame({"g": ["A"] * 3, "v": ["b", "a", None]})
    # Only the sample accounting is pinned here, not the fallback's result value.
    out = AggregateTool(df).forward(group_by="g", op="min", metric="v")
    assert "smallest has 3" in out["note"]


def test_group_with_no_usable_values_reports_zero():
    df = pd.DataFrame({"g": ["A"] * 20 + ["B"] * 4, "v": _full(20) + [None] * 4})
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v")
    groups = _groups(out, ["g"], "mean_v")
    assert groups[("A",)] == 1.0 and groups[("B",)] is None
    assert out["note"].startswith("1 group(s)") and "smallest has 0" in out["note"]


def test_two_column_group_by_counts_per_combination():
    df = pd.DataFrame(
        {
            "a": ["x"] * 20 + ["y"] * 20,
            "b": ["p"] * 20 + ["q"] * 20,
            "v": _full(20) + _full(4) + [None] * 16,
        }
    )
    out = AggregateTool(df).forward(group_by=["a", "b"], op="mean", metric="v")
    assert out["note"].startswith("1 group(s)") and "smallest has 4" in out["note"]


def test_explicit_n_judges_only_returned_groups():
    df = pd.DataFrame(
        {
            "g": ["big"] * 20 + ["thin"] * 20,
            # 'thin' has 2 usable values and a lower mean, so n=1 leaves it out.
            "v": _full(20, 10.0) + _full(2, 1.0) + [None] * 18,
        }
    )
    tool = AggregateTool(df)
    assert "note" not in tool.forward(group_by="g", op="mean", metric="v", n=1)
    assert "smallest has 2" in tool.forward(group_by="g", op="mean", metric="v")["note"]


def test_categorical_group_keys_with_missing_and_unobserved_categories():
    df = pd.DataFrame(
        {
            "g": pd.Categorical(["A"] * 20 + ["B"] * 3 + [None] * 2, categories=["A", "B", "C"]),
            "v": _full(20) + [1.0, None, None] + [5.0, None],
        }
    )
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v")
    groups = _groups(out, ["g"], "mean_v")
    assert set(groups) == {("A",), ("B",), (None,)}
    assert out["note"].startswith("2 group(s)") and "smallest has 1" in out["note"]


def test_prefilter_population_drives_the_sample():
    df = pd.DataFrame(
        {
            "g": ["A"] * 30,
            "k": [1] * 20 + [2] * 10,
            # Under k == 1 only 3 metric values are usable; unfiltered there would be 13.
            "v": _full(3) + [None] * 17 + _full(10),
        }
    )
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v", where_col="k", value=1)
    assert "smallest has 3" in out["note"]


def test_count_still_has_no_note():
    df = pd.DataFrame({"g": ["A", "A", "B"], "v": [None, None, None]})
    out = AggregateTool(df).forward(group_by="g", op="count")
    assert _groups(out, ["g"], "count") == {("A",): 2, ("B",): 1}
    assert "note" not in out


def test_fully_usable_group_keeps_row_count():
    df = pd.DataFrame({"g": ["A"] * 20 + ["B"] * 5, "v": _full(25)})
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v")
    assert out["note"].startswith("1 group(s)") and "smallest has 5" in out["note"]
