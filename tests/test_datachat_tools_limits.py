"""Result limits of the DataChat DataFrame tools: complete by default, bounded only on request."""

import pandas as pd
import pytest

from datachat.tools import plot_tool
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.limits import MIN_RELIABLE_SAMPLE, InvalidLimit, optional_limit
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.plot_tool import PlotTool, _pie_slices
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.trend_tool import TrendTool
from datachat.tools.unique_values_tool import UniqueValuesTool

WIDE_COLS = [f"c{i}" for i in range(15)]


@pytest.fixture
def wide_df():
    return pd.DataFrame({c: range(100) for c in WIDE_COLS})


# ---------------------------------------------------------------------------
# optional_limit
# ---------------------------------------------------------------------------


def test_optional_limit():
    assert optional_limit(None) is None
    assert optional_limit(7) == 7
    assert optional_limit("3") == 3
    for bad in (0, -1, "abc"):
        with pytest.raises(InvalidLimit):
            optional_limit(bad)


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------


def test_filter_returns_all_matches_and_columns_by_default(wide_df):
    out = FilterRowsTool(wide_df).forward(where_col="c0", op="gte", value=20)
    assert len(out["data"]) == 80
    assert list(out["data"][0].keys()) == WIDE_COLS
    assert out["meta"] == {"offset": 0, "returned": 80, "total_matches": 80}


def test_filter_explicit_n_limits(wide_df):
    out = FilterRowsTool(wide_df).forward(where_col="c0", op="is_not_empty", n=500)
    assert len(out["data"]) == 100
    out = FilterRowsTool(wide_df).forward(where_col="c0", op="is_not_empty", n=7, offset=90)
    assert [r["c0"] for r in out["data"]] == list(range(90, 97))
    assert out["meta"] == {"offset": 90, "returned": 7, "total_matches": 100}


def test_filter_offset_without_n_returns_the_rest(wide_df):
    out = FilterRowsTool(wide_df).forward(where_col="c0", op="is_not_empty", offset=95)
    assert [r["c0"] for r in out["data"]] == list(range(95, 100))


def test_filter_explicit_columns(wide_df):
    out = FilterRowsTool(wide_df).forward(where_col="c0", op="lt", value=3, columns=["c14", "c2"])
    assert out["data"] == [{"c14": i, "c2": i} for i in range(3)]


def test_filter_text_ops_unaffected():
    df = pd.DataFrame({"name": [f"alpha{i}" for i in range(60)] + ["beta", None]})
    tool = FilterRowsTool(df)
    assert tool.forward(where_col="name", op="contains", value="ALPHA")["meta"]["returned"] == 60
    assert tool.forward(where_col="name", op="not_contains", value="alpha")["data"] == [{"name": "beta"}]
    assert tool.forward(where_col="name", op="is_empty")["data"] == [{"name": None}]


def test_filter_invalid_n(wide_df):
    out = FilterRowsTool(wide_df).forward(where_col="c0", op="is_not_empty", n=0)
    assert out["kind"] == "error"
    assert out["code"] == "INVALID_LIMIT"


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


@pytest.fixture
def many_groups_df():
    return pd.DataFrame({"g": [f"g{i:02d}" for i in range(80)], "v": range(80)})


def test_aggregate_returns_all_groups_by_default(many_groups_df):
    out = AggregateTool(many_groups_df).forward(group_by="g", op="count")
    assert len(out["data"]) == 80


def test_aggregate_explicit_n_is_top_n(many_groups_df):
    out = AggregateTool(many_groups_df).forward(group_by="g", op="sum", metric="v", n=3)
    assert [r["g"] for r in out["data"]] == ["g79", "g78", "g77"]
    out = AggregateTool(many_groups_df).forward(group_by="g", op="sum", metric="v", n=500)
    assert len(out["data"]) == 80


def test_aggregate_two_columns_all_groups():
    df = pd.DataFrame({"a": [i // 10 for i in range(100)], "b": [i % 10 for i in range(100)]})
    out = AggregateTool(df).forward(group_by=["a", "b"], op="count")
    assert len(out["data"]) == 100


def test_aggregate_small_sample_note_covers_all_groups(many_groups_df):
    out = AggregateTool(many_groups_df).forward(group_by="g", op="mean", metric="v")
    assert f"80 group(s) are based on fewer than {MIN_RELIABLE_SAMPLE} rows" in out["note"]


@pytest.fixture
def region_df():
    return pd.DataFrame({"region": ["N", "N", "S", "S"], "amount": [10, 20, 30, 40]})


def test_aggregate_valid_pre_filter(region_df):
    out = AggregateTool(region_df).forward(
        group_by="region", op="sum", metric="amount", where_col="amount", op_filter="gt", value=15
    )
    assert {r["region"]: r["sum_amount"] for r in out["data"]} == {"N": 20, "S": 70}


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"where_col": "nope", "value": "N"}, "INVALID_FILTER_COLUMN"),
        ({"where_col": "region", "op_filter": "like", "value": "N"}, "INVALID_FILTER_OP"),
        ({"where_col": "region"}, "MISSING_FILTER_VALUE"),
        ({"value": "N"}, "INVALID_FILTER_COLUMN"),
        ({"where_col": "amount", "op_filter": "gt", "value": "lots"}, "NON_NUMERIC_VALUE"),
        ({"where_col": "region", "value": "N", "where_col2": "nope", "value2": 1}, "INVALID_FILTER_COLUMN"),
        ({"where_col": "region", "value": "N", "where_col2": "amount", "op2_filter": "??", "value2": 1}, "INVALID_FILTER_OP"),
        ({"where_col": "region", "value": "N", "where_col2": "amount"}, "MISSING_FILTER_VALUE"),
    ],
)
def test_aggregate_invalid_pre_filter_fails(region_df, kwargs, code):
    out = AggregateTool(region_df).forward(group_by="region", op="sum", metric="amount", **kwargs)
    assert out["kind"] == "error"
    assert out["code"] == code
    assert "data" not in out


# ---------------------------------------------------------------------------
# unique_values
# ---------------------------------------------------------------------------


def test_unique_values_all_by_default(many_groups_df):
    out = UniqueValuesTool(many_groups_df).forward(column="g")
    assert len(out["data"]) == 80


def test_unique_values_explicit_n():
    df = pd.DataFrame({"v": ["a"] * 3 + ["b"] * 2 + [f"x{i}" for i in range(60)]})
    out = UniqueValuesTool(df).forward(column="v", n=2)
    assert out["data"] == [{"value": "a", "count": 3}, {"value": "b", "count": 2}]


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_df():
    return pd.DataFrame({"d": pd.date_range("2025-01-01", periods=70, freq="D").astype(str)})


def test_trend_full_series_by_default(daily_df):
    out = TrendTool(daily_df).forward(date_col="d", freq="day", op="count")
    assert len(out["data"]) == 70
    assert out["data"][0]["period"] == "2025-01-01"
    assert out["data"][-1]["period"] == "2025-03-11"


def test_trend_explicit_n(daily_df):
    out = TrendTool(daily_df).forward(date_col="d", freq="day", op="count", n=5, ascending=False)
    assert [r["period"] for r in out["data"]][0] == "2025-03-11"
    assert len(out["data"]) == 5


# ---------------------------------------------------------------------------
# describe / missing_values
# ---------------------------------------------------------------------------


@pytest.fixture
def very_wide_df():
    return pd.DataFrame({f"c{i}": [1, None, 3] for i in range(60)})


def test_describe_all_columns_by_default(very_wide_df):
    out = DescribeTool(very_wide_df).forward()
    assert [r["column"] for r in out["data"]] == list(very_wide_df.columns)


def test_describe_explicit_columns(very_wide_df):
    out = DescribeTool(very_wide_df).forward(columns=["c59", "c1"])
    assert [r["column"] for r in out["data"]] == ["c59", "c1"]


def test_missing_values_all_columns_by_default(very_wide_df):
    out = MissingValuesTool(very_wide_df).forward()
    assert len(out["data"]) == 60
    assert all(r["missing"] == 1 for r in out["data"])


def test_missing_values_explicit_columns(very_wide_df):
    out = MissingValuesTool(very_wide_df).forward(columns=["c55"])
    assert [r["column"] for r in out["data"]] == ["c55"]


# ---------------------------------------------------------------------------
# top_rows / sample_rows
# ---------------------------------------------------------------------------


def test_top_rows_keeps_all_columns_and_row_bound(wide_df):
    tool = TopRowsTool(wide_df)
    out = tool.forward(sort_by="c0")
    assert len(out["data"]) == 5
    assert list(out["data"][0].keys()) == WIDE_COLS
    assert out["data"][0]["c0"] == 99
    assert out["meta"]["total_matches"] == 100
    assert len(tool.forward(sort_by="c0", n=500)["data"]) == 20
    assert list(tool.forward(sort_by="c0", columns=["c12"])["data"][0].keys()) == ["c12"]


def test_sample_rows_keeps_all_columns_and_row_bound(wide_df):
    tool = SampleRowsTool(wide_df)
    out = tool.forward()
    assert len(out["data"]) == 5
    assert list(out["data"][0].keys()) == WIDE_COLS
    assert out["meta"] == {"offset": 0, "returned": 5, "total_rows": 100}
    assert len(tool.forward(n=500)["data"]) == 20
    assert tool.forward(n=2, offset=10, columns=["c13"])["data"] == [{"c13": 10}, {"c13": 11}]


# ---------------------------------------------------------------------------
# plot: pie
# ---------------------------------------------------------------------------


def test_pie_slices_keep_full_population():
    totals = pd.Series({"a": 50, "b": 20, "c": 10, "d": 10, "e": 10})
    labels, values = _pie_slices(totals, 2)
    assert labels == ["a", "b", "Other (3 categories)"]
    assert values == [50.0, 20.0, 30.0]
    assert values[0] / sum(values) == 0.5


def test_pie_slices_without_omitted_categories():
    labels, values = _pie_slices(pd.Series({"a": 1, "b": 3}), 5)
    assert labels == ["b", "a"]
    assert values == [3.0, 1.0]


@pytest.mark.parametrize("y", [None, "amount"])
def test_pie_plot_is_bounded_and_uses_full_total(tmp_path, monkeypatch, y):
    df = pd.DataFrame({"cat": [f"k{i:02d}" for i in range(30)] * 2, "amount": [1] * 60})
    captured = {}
    real_pie = plot_tool.plt.pie

    def spy(values, **kwargs):
        captured["values"] = list(values)
        captured["labels"] = kwargs["labels"]
        return real_pie(values, **kwargs)

    monkeypatch.setattr(plot_tool.plt, "pie", spy)
    out = PlotTool(df, str(tmp_path)).forward(kind="pie", x="cat", y=y, agg="sum" if y else None, n=4)
    assert out["kind"] == "image_path"
    assert len(captured["labels"]) == 5
    assert captured["labels"][-1] == "Other (26 categories)"
    assert sum(captured["values"]) == 60
    assert captured["values"][0] / sum(captured["values"]) == pytest.approx(2 / 60)


def test_pie_sum_rejects_negative_category_total_in_other_tail(tmp_path):
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "c", "c", "d", "e"],
            "amount": [50, 40, 5, 1, -3, 2],
        }
    )
    out = PlotTool(df, str(tmp_path)).forward(kind="pie", x="cat", y="amount", agg="sum", n=2)
    assert out["kind"] == "error"
    assert out["code"] == "NEGATIVE_VALUES"


def test_pie_sum_allows_mixed_signs_within_non_negative_category(tmp_path):
    df = pd.DataFrame({"cat": ["a", "a", "b"], "amount": [5, -2, 4]})
    out = PlotTool(df, str(tmp_path)).forward(kind="pie", x="cat", y="amount", agg="sum")
    assert out["kind"] == "image_path"
