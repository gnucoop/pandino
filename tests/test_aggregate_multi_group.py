"""Tests for two-dimensional group_by on aggregate.

Regression origin: passing a list kept only the first column, so
`group_by=["corso","ruolo"]` returned a one-dimensional table where both courses showed
3.0 — concealing that role X averaged 5.0 and role Y averaged 1.0. The number was wrong,
not merely incomplete, and looked entirely plausible.
"""

import pandas as pd
import pytest

from datachat.tools.aggregate_tool import AggregateTool


@pytest.fixture()
def survey_df():
    return pd.DataFrame(
        {
            "corso": ["A"] * 40 + ["B"] * 40,
            "ruolo": (["X"] * 20 + ["Y"] * 20) * 2,
            "voto": [5.0] * 20 + [1.0] * 20 + [3.0] * 20 + [4.0] * 20,
        }
    )


@pytest.fixture()
def tool(survey_df):
    return AggregateTool(survey_df)


def test_two_columns_keep_both_dimensions(tool):
    result = tool.forward(group_by=["corso", "ruolo"], op="mean", metric="voto")

    assert len(result["data"]) == 4
    for record in result["data"]:
        assert "corso" in record and "ruolo" in record


def test_two_columns_report_the_real_per_cell_means(tool):
    """The exact values the old behaviour hid behind a single 3.0."""
    result = tool.forward(group_by=["corso", "ruolo"], op="mean", metric="voto")
    cells = {(r["corso"], r["ruolo"]): r["mean_voto"] for r in result["data"]}

    assert cells == {("A", "X"): 5.0, ("A", "Y"): 1.0, ("B", "X"): 3.0, ("B", "Y"): 4.0}


def test_single_column_string_still_works(tool):
    result = tool.forward(group_by="corso", op="mean", metric="voto")

    assert len(result["data"]) == 2
    assert all("ruolo" not in r for r in result["data"])


def test_single_element_list_is_treated_as_one_dimension(tool):
    """The original reason the list branch existed: the LLM passing ["col"]."""
    result = tool.forward(group_by=["corso"], op="mean", metric="voto")

    assert len(result["data"]) == 2
    assert all("ruolo" not in r for r in result["data"])


def test_three_columns_are_refused_and_point_at_crosstab(tool):
    result = tool.forward(group_by=["corso", "ruolo", "voto"], op="mean", metric="voto")

    assert result["kind"] == "error"
    assert result["code"] == "TOO_MANY_GROUP_BY"
    assert "crosstab" in result["message"]


def test_two_dimensional_count_works(tool):
    result = tool.forward(group_by=["corso", "ruolo"], op="count")

    assert len(result["data"]) == 4
    assert all(r["count"] == 20 for r in result["data"])


def test_unknown_column_in_the_list_is_named(tool):
    result = tool.forward(group_by=["corso", "nope"], op="count")

    assert result["code"] == "INVALID_GROUP_BY"
    assert "nope" in result["message"]


def test_export_name_covers_both_dimensions(tool):
    result = tool.forward(group_by=["corso", "ruolo"], op="mean", metric="voto")

    assert result["export_name"] == "mean_voto_by_corso_ruolo"


def test_thin_groups_are_flagged_for_averages():
    df = pd.DataFrame({"corso": ["A", "A", "B"], "voto": [5.0, 4.0, 3.0]})
    result = AggregateTool(df).forward(group_by="corso", op="mean", metric="voto")

    assert "Caution" in result["note"]


def test_counts_are_never_flagged_as_unreliable():
    """A count over 3 rows is exact; only averages are fragile."""
    df = pd.DataFrame({"corso": ["A", "A", "B"]})
    result = AggregateTool(df).forward(group_by="corso", op="count")

    assert "note" not in result


def test_empty_group_by_is_rejected(tool):
    assert tool.forward(group_by=[], op="count")["code"] == "MISSING_GROUP_BY"
