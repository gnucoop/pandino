import pandas as pd
import pytest

from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.limits import MIN_RELIABLE_SAMPLE


def _cat(values, categories):
    return pd.Categorical(values, categories=categories)


def _groups(out, cols, value_col):
    return {tuple(r[c] for c in cols): r[value_col] for r in out["data"]}


# ---------------------------------------------------------------------------
# declared-but-unobserved categories are not materialised
# ---------------------------------------------------------------------------


@pytest.fixture
def unobserved_df():
    df = pd.DataFrame(
        {
            "g": _cat(["A", "B", "A", "B"], ["A", "B", "C"]),
            "h": _cat(["x", "x", "y", "y"], ["x", "y", "z"]),
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert isinstance(df["g"].dtype, pd.CategoricalDtype)
    return df


def test_unobserved_one_column_count(unobserved_df):
    out = AggregateTool(unobserved_df).forward(group_by="g", op="count")
    assert _groups(out, ["g"], "count") == {("A",): 2, ("B",): 2}


def test_unobserved_two_columns_count(unobserved_df):
    out = AggregateTool(unobserved_df).forward(group_by=["g", "h"], op="count")
    assert _groups(out, ["g", "h"], "count") == {
        ("A", "x"): 1,
        ("A", "y"): 1,
        ("B", "x"): 1,
        ("B", "y"): 1,
    }


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_unobserved_one_column_non_count(unobserved_df, op):
    out = AggregateTool(unobserved_df).forward(group_by="g", op=op, metric="v")
    assert {r["g"] for r in out["data"]} == {"A", "B"}


def test_unobserved_two_columns_non_count(unobserved_df):
    df = unobserved_df.iloc[:3]  # (B, y) is now unobserved too
    out = AggregateTool(df).forward(group_by=["g", "h"], op="sum", metric="v")
    assert _groups(out, ["g", "h"], "sum_v") == {("A", "x"): 1.0, ("B", "x"): 2.0, ("A", "y"): 3.0}


def test_unobserved_groups_do_not_take_top_n_slots(unobserved_df):
    out = AggregateTool(unobserved_df).forward(group_by="g", op="count", ascending=True, n=1)
    assert len(out["data"]) == 1
    assert out["data"][0]["g"] in {"A", "B"}
    out = AggregateTool(unobserved_df).forward(group_by="g", op="sum", metric="v", ascending=True, n=1)
    assert out["data"] == [{"g": "A", "sum_v": 4.0}]


def test_unobserved_groups_do_not_pollute_small_sample_note(unobserved_df):
    out = AggregateTool(unobserved_df).forward(group_by="g", op="mean", metric="v")
    assert out["note"].startswith("2 group(s)")
    assert "smallest has 2" in out["note"]


def test_no_note_when_only_unobserved_groups_would_be_small():
    df = pd.DataFrame({"g": _cat(["A"] * MIN_RELIABLE_SAMPLE, ["A", "B"]), "v": range(MIN_RELIABLE_SAMPLE)})
    out = AggregateTool(df).forward(group_by="g", op="mean", metric="v")
    assert [r["g"] for r in out["data"]] == ["A"]
    assert "note" not in out


# ---------------------------------------------------------------------------
# rows with a missing Categorical key are kept
# ---------------------------------------------------------------------------


@pytest.fixture
def missing_df():
    df = pd.DataFrame(
        {
            "g": _cat(["A", None, "B", None], ["A", "B"]),
            "h": _cat(["x", "x", None, "y"], ["x", "y"]),
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert isinstance(df["g"].dtype, pd.CategoricalDtype)
    return df


def test_missing_one_column_count(missing_df):
    out = AggregateTool(missing_df).forward(group_by="g", op="count")
    assert _groups(out, ["g"], "count") == {("A",): 1, ("B",): 1, (None,): 2}


def test_missing_in_first_of_two_columns(missing_df):
    out = AggregateTool(missing_df).forward(group_by=["g", "h"], op="count")
    got = _groups(out, ["g", "h"], "count")
    assert got[(None, "x")] == 1
    assert got[(None, "y")] == 1


def test_missing_in_second_of_two_columns(missing_df):
    out = AggregateTool(missing_df).forward(group_by=["g", "h"], op="count")
    assert _groups(out, ["g", "h"], "count")[("B", None)] == 1


@pytest.mark.parametrize(
    "op, expected",
    [
        ("sum", {("A",): 1.0, ("B",): 3.0, (None,): 6.0}),
        ("mean", {("A",): 1.0, ("B",): 3.0, (None,): 3.0}),
        ("min", {("A",): 1.0, ("B",): 3.0, (None,): 2.0}),
        ("max", {("A",): 1.0, ("B",): 3.0, (None,): 4.0}),
    ],
)
def test_missing_non_count(missing_df, op, expected):
    out = AggregateTool(missing_df).forward(group_by="g", op=op, metric="v")
    assert _groups(out, ["g"], f"{op}_v") == expected


@pytest.mark.parametrize("group_by", ["g", "h", ["g", "h"], ["h", "g"]])
def test_missing_count_totals_reconcile(missing_df, group_by):
    out = AggregateTool(missing_df).forward(group_by=group_by, op="count")
    assert sum(r["count"] for r in out["data"]) == len(missing_df)


def test_missing_note_population_matches_result(missing_df):
    out = AggregateTool(missing_df).forward(group_by="g", op="mean", metric="v")
    assert len(out["data"]) == 3
    assert out["note"].startswith("3 group(s)")
    assert "smallest has 1" in out["note"]


# ---------------------------------------------------------------------------
# regression: behaviour outside Categorical keys is unchanged
# ---------------------------------------------------------------------------


def test_object_missing_group_preserved():
    df = pd.DataFrame({"g": ["A", None, "B", float("nan")], "v": [1.0, 2.0, 3.0, 4.0]})
    out = AggregateTool(df).forward(group_by="g", op="count")
    assert _groups(out, ["g"], "count") == {("A",): 1, ("B",): 1, (None,): 2}
    out = AggregateTool(df).forward(group_by="g", op="sum", metric="v")
    assert _groups(out, ["g"], "sum_v") == {("A",): 1.0, ("B",): 3.0, (None,): 6.0}


def test_object_missing_in_two_columns_preserved():
    df = pd.DataFrame({"g": ["A", None, "B"], "h": ["x", "x", None], "v": [1, 2, 3]})
    out = AggregateTool(df).forward(group_by=["g", "h"], op="count")
    assert _groups(out, ["g", "h"], "count") == {("A", "x"): 1, (None, "x"): 1, ("B", None): 1}


def test_categorical_matches_object_for_observed_values():
    values = ["N", "N", "S", "S", "S"]
    obj = pd.DataFrame({"g": values, "v": [1, 2, 3, 4, 5]})
    cat = obj.assign(g=_cat(values, ["N", "S", "W"]))
    for op, metric in (("count", None), ("sum", "v"), ("mean", "v")):
        assert AggregateTool(cat).forward(group_by="g", op=op, metric=metric)["data"] == (
            AggregateTool(obj).forward(group_by="g", op=op, metric=metric)["data"]
        )


def test_numeric_categorical_keys_keep_their_values():
    df = pd.DataFrame({"g": _cat([1, 2, 2], [1, 2, 3]), "v": [1, 2, 3]})
    out = AggregateTool(df).forward(group_by="g", op="count")
    assert out["data"] == [{"g": 2, "count": 2}, {"g": 1, "count": 1}]


def test_bool_number_identity_unchanged():
    # Deliberately not addressed here: pandas grouping still merges True with 1.
    df = pd.DataFrame({"g": pd.Series([True, 1, False, 0], dtype=object)})
    out = AggregateTool(df).forward(group_by="g", op="count")
    assert _groups(out, ["g"], "count") == {(True,): 2, (False,): 2}
