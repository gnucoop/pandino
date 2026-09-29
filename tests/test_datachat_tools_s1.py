"""S1 behaviour of the aggregate, filter_rows and correlation DataChat tools."""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.correlation_tool import CorrelationTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.limits import MIN_RELIABLE_SAMPLE


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    return pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S", "S"],
            "channel": ["web", "web", "shop", "web", "shop", "shop"],
            "amount": [10, 20, 30, 40, 50, 60],
        }
    )


def test_aggregate_single_column_unchanged(sales_df):
    out = AggregateTool(sales_df).forward(group_by="region", op="sum", metric="amount")
    assert out["kind"] == "table"
    assert out["data"] == [
        {"region": "S", "sum_amount": 150},
        {"region": "N", "sum_amount": 60},
    ]


def test_aggregate_single_item_list_is_single_column(sales_df):
    out = AggregateTool(sales_df).forward(group_by=["region"], op="count")
    assert {tuple(r.keys()) for r in out["data"]} == {("region", "count")}


def test_aggregate_two_columns(sales_df):
    out = AggregateTool(sales_df).forward(group_by=["region", "channel"], op="sum", metric="amount")
    got = {(r["region"], r["channel"]): r["sum_amount"] for r in out["data"]}
    assert got == {("N", "web"): 30, ("N", "shop"): 30, ("S", "web"): 40, ("S", "shop"): 110}


def test_aggregate_two_columns_count(sales_df):
    out = AggregateTool(sales_df).forward(group_by=["region", "channel"], op="count")
    got = {(r["region"], r["channel"]): r["count"] for r in out["data"]}
    assert got == {("N", "web"): 2, ("N", "shop"): 1, ("S", "web"): 1, ("S", "shop"): 2}


def test_aggregate_more_than_two_columns_is_an_error(sales_df):
    out = AggregateTool(sales_df).forward(group_by=["region", "channel", "amount"], op="count")
    assert out["kind"] == "error"
    assert out["code"] == "TOO_MANY_GROUP_BY"


def test_aggregate_invalid_second_column(sales_df):
    out = AggregateTool(sales_df).forward(group_by=["region", "nope"], op="count")
    assert out["code"] == "INVALID_GROUP_BY"


def test_small_sample_note_for_mean(sales_df):
    out = AggregateTool(sales_df).forward(group_by="region", op="mean", metric="amount")
    note = out["note"]
    assert f"fewer than {MIN_RELIABLE_SAMPLE} rows" in note
    assert "2 group(s)" in note
    assert "smallest has 3" in note


def test_no_small_sample_note_for_count(sales_df):
    out = AggregateTool(sales_df).forward(group_by="region", op="count")
    assert "note" not in out


def test_no_small_sample_note_when_groups_are_large():
    df = pd.DataFrame({"g": ["a"] * 20 + ["b"] * 15, "v": range(35)})
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v")
    assert "note" not in out


def test_small_sample_note_only_counts_returned_groups():
    df = pd.DataFrame({"g": ["big"] * 20 + ["tiny"] * 2, "v": [100] * 20 + [1] * 2})
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v", n=1)
    assert [r["g"] for r in out["data"]] == ["big"]
    assert "note" not in out


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


@pytest.fixture
def text_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "comment": ["Orario scomodo", "  ", None, "ottimo (orario)", "prezzo a*b", "nan"],
            "score": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _ids(out):
    return [r["id"] for r in out["data"]]


def test_is_empty_text_column(text_df):
    out = FilterRowsTool(text_df).forward(where_col="comment", op="is_empty")
    assert _ids(out) == [2, 3]


def test_is_not_empty_text_column(text_df):
    out = FilterRowsTool(text_df).forward(where_col="comment", op="is_not_empty", n=50)
    assert _ids(out) == [1, 4, 5, 6]


def test_is_empty_numeric_column(text_df):
    out = FilterRowsTool(text_df).forward(where_col="score", op="is_empty")
    assert _ids(out) == [2]


def test_contains_is_case_insensitive(text_df):
    out = FilterRowsTool(text_df).forward(where_col="comment", op="contains", value="ORARIO")
    assert _ids(out) == [1, 4]


def test_contains_is_literal_not_regex(text_df):
    tool = FilterRowsTool(text_df)
    assert _ids(tool.forward(where_col="comment", op="contains", value="(orario)")) == [4]
    assert _ids(tool.forward(where_col="comment", op="contains", value="a*b")) == [5]
    assert _ids(tool.forward(where_col="comment", op="contains", value=".*")) == []


def test_contains_never_matches_missing_cells(text_df):
    # A missing cell must not match via its "nan"/"None" string form.
    out = FilterRowsTool(text_df).forward(where_col="comment", op="contains", value="n")
    assert 3 not in _ids(out)
    assert 6 in _ids(out)  # the literal string "nan" is a real value


def test_not_contains_excludes_empty_cells(text_df):
    out = FilterRowsTool(text_df).forward(where_col="comment", op="not_contains", value="orario")
    assert _ids(out) == [5, 6]


@pytest.mark.parametrize("op", ["contains", "not_contains"])
@pytest.mark.parametrize(
    "column",
    [
        [1.0, 4.0, None],  # numeric
        [True, False, True],  # boolean
        [True, None, False],  # boolean with missing (object dtype)
        pd.to_datetime(["2024-01-04", "2024-02-01", None]),  # datetime
        ["a4", 4, None],  # mixed text/number (object dtype)
    ],
)
def test_text_ops_reject_non_text_columns(op, column):
    df = pd.DataFrame({"id": [1, 2, 3], "col": column})
    out = FilterRowsTool(df).forward(where_col="col", op=op, value="4")
    assert out["kind"] == "error"
    assert out["code"] == "NON_TEXT_COLUMN"


def test_text_op_rejects_non_text_second_column(text_df):
    out = FilterRowsTool(text_df).forward(
        where_col="comment", op="is_not_empty", where_col2="score", op2="contains", value2="4"
    )
    assert out["code"] == "NON_TEXT_COLUMN_2"


def test_contains_on_all_missing_column_matches_nothing():
    df = pd.DataFrame({"id": [1, 2], "col": [None, None]})
    assert _ids(FilterRowsTool(df).forward(where_col="col", op="contains", value="x")) == []


@pytest.mark.parametrize("op", ["eq", "lt", "lte", "gt", "gte", "contains", "not_contains"])
def test_omitted_value_is_rejected_for_value_ops(text_df, op):
    out = FilterRowsTool(text_df).forward(where_col="comment", op=op)
    assert out["kind"] == "error"
    assert out["code"] == "MISSING_FILTER_VALUE"


def test_omitted_value_is_rejected_for_default_eq(text_df):
    assert FilterRowsTool(text_df).forward(where_col="comment")["code"] == "MISSING_FILTER_VALUE"


def test_contains_requires_a_value(text_df):
    tool = FilterRowsTool(text_df)
    assert tool.forward(where_col="comment", op="contains", value=None)["code"] == "MISSING_FILTER_VALUE"
    assert tool.forward(where_col="comment", op="not_contains", value=" ")["code"] == "MISSING_FILTER_VALUE"


def test_second_condition_is_not_empty_without_value(text_df):
    out = FilterRowsTool(text_df).forward(
        where_col="score", op="gte", value=3, where_col2="comment", op2="is_not_empty"
    )
    assert _ids(out) == [4, 5, 6]  # id 3 has score 3 but no comment


def test_eq_none_and_empty_string_semantics_unchanged(text_df):
    tool = FilterRowsTool(text_df)
    # CURRENT: eq compares string forms; None matches the "none" string form, "" matches "".
    assert _ids(tool.forward(where_col="comment", value=None)) == [3]
    assert _ids(tool.forward(where_col="comment", value="")) == [2]


def test_eq_existing_behaviour(text_df):
    out = FilterRowsTool(text_df).forward(where_col="comment", value="orario scomodo")
    assert _ids(out) == [1]


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------


@pytest.fixture
def corr_df():
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "lin": [2, 4, 6, 8, 10, 12],
            "mono": [1, 4, 9, 16, 25, 1000],
            "neg": [6, 5, 4, 3, 2, 1],
            "noise": [3, 1, 4, 1, 5, 9],
            "const": [7, 7, 7, 7, 7, 7],
            "label": ["a", "b", "c", "d", "e", "f"],
        }
    )


def test_pearson_pair_unchanged(corr_df):
    out = CorrelationTool(corr_df).forward(col_x="x", col_y="lin")
    assert out["kind"] == "table"
    (row,) = out["data"]
    assert set(row) == {"col_x", "col_y", "method", "correlation", "n"}
    assert row["method"] == "pearson"
    assert row["correlation"] == pytest.approx(1.0)
    assert row["n"] == 6


def test_spearman_pair(corr_df):
    tool = CorrelationTool(corr_df)
    spearman = tool.forward(col_x="x", col_y="mono", method="spearman")["data"][0]
    pearson = tool.forward(col_x="x", col_y="mono")["data"][0]
    assert spearman["method"] == "spearman"
    assert spearman["correlation"] == pytest.approx(1.0)
    assert pearson["correlation"] < 0.9


def test_invalid_method(corr_df):
    out = CorrelationTool(corr_df).forward(col_x="x", col_y="lin", method="kendall")
    assert out["code"] == "INVALID_METHOD"


def test_anchor_ranking(corr_df):
    out = CorrelationTool(corr_df).forward(col_x="x")
    rows = out["data"]
    assert all(r["col_x"] == "x" for r in rows)
    ys = [r["col_y"] for r in rows]
    assert "x" not in ys  # no self-correlation
    assert "const" not in ys and "label" not in ys  # no variation / not numeric
    assert ys[:2] == ["lin", "neg"]  # |1.0| ties broken by name
    mags = [abs(r["correlation"]) for r in rows]
    assert mags == sorted(mags, reverse=True)


def test_all_pairs_ranking(corr_df):
    rows = CorrelationTool(corr_df).forward()["data"]
    numeric = ["x", "lin", "mono", "neg", "noise"]
    assert len(rows) == len(numeric) * (len(numeric) - 1) // 2
    assert len({frozenset((r["col_x"], r["col_y"])) for r in rows}) == len(rows)
    keys = [(-abs(r["correlation"]), r["col_x"], r["col_y"]) for r in rows]
    assert keys == sorted(keys)


def test_ranking_is_deterministic(corr_df):
    tool = CorrelationTool(corr_df)
    assert tool.forward() == tool.forward()


def test_ranking_handles_missing_values():
    df = pd.DataFrame({"a": [1, 2, None, 4, 5], "b": [2, 4, 6, None, 10], "c": [None] * 4 + [1]})
    rows = CorrelationTool(df).forward(col_x="a")["data"]
    assert [(r["col_y"], r["n"]) for r in rows] == [("b", 3)]


def test_anchor_without_numeric_variation(corr_df):
    out = CorrelationTool(corr_df).forward(col_x="const")
    assert out["code"] == "NO_NUMERIC_DATA"


def test_anchor_invalid_column(corr_df):
    assert CorrelationTool(corr_df).forward(col_x="nope")["code"] == "INVALID_COLUMN"


def test_col_y_without_col_x(corr_df):
    assert CorrelationTool(corr_df).forward(col_y="x")["code"] == "MISSING_COLUMNS"


def test_ranking_has_no_strength_labels(corr_df):
    for row in CorrelationTool(corr_df).forward()["data"]:
        assert set(row) == {"col_x", "col_y", "method", "correlation", "n"}
