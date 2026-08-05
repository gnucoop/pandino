"""Tests for the crosstab tool — two-dimensional breakdowns.

Exists because `aggregate` silently collapsed a two-column request to one dimension,
returning a plausible but wrong number. crosstab is the supported way to ask the question.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.crosstab_tool import CrosstabTool


@pytest.fixture()
def survey_df():
    """2 courses x 2 roles, with a different mean in every cell."""
    return pd.DataFrame(
        {
            "corso": ["A"] * 40 + ["B"] * 40,
            "ruolo": (["X"] * 20 + ["Y"] * 20) * 2,
            "voto": [5.0] * 20 + [1.0] * 20 + [3.0] * 20 + [4.0] * 20,
        }
    )


@pytest.fixture()
def tool(survey_df):
    return CrosstabTool(survey_df)


def test_both_dimensions_survive(tool):
    """The whole point: neither dimension is dropped."""
    result = tool.forward(rows="corso", columns="ruolo")

    assert result["kind"] == "table"
    assert [r["corso"] for r in result["data"]] == ["A", "B"]
    assert set(result["data"][0]) == {"corso", "X", "Y"}


def test_counts_match_pandas(tool, survey_df):
    result = tool.forward(rows="corso", columns="ruolo")
    expected = pd.crosstab(survey_df["corso"], survey_df["ruolo"])

    for record in result["data"]:
        for role in ("X", "Y"):
            assert record[role] == expected.loc[record["corso"], role]


def test_mean_per_cell_reveals_what_one_dimension_hides(tool):
    """Averaged over roles both courses are 3.0; per cell they are anything but."""
    result = tool.forward(rows="corso", columns="ruolo", metric="voto", op="mean")

    by_course = {r["corso"]: r for r in result["data"]}
    assert by_course["A"]["X"] == 5.0
    assert by_course["A"]["Y"] == 1.0
    assert by_course["B"]["X"] == 3.0
    assert by_course["B"]["Y"] == 4.0


def test_row_percentages_sum_to_100(tool):
    result = tool.forward(rows="corso", columns="ruolo", normalize="rows")

    for record in result["data"]:
        total = sum(v for k, v in record.items() if k != "corso")
        assert total == pytest.approx(100.0)


def test_column_percentages_sum_to_100(tool):
    result = tool.forward(rows="corso", columns="ruolo", normalize="columns")

    for role in ("X", "Y"):
        total = sum(r[role] for r in result["data"])
        assert total == pytest.approx(100.0)


def test_blank_values_become_an_explicit_category():
    """Dropping non-responders would misstate every percentage in the table."""
    df = pd.DataFrame(
        {"corso": ["A", "A", "B"], "ruolo": ["X", np.nan, "Y"], "voto": [1.0, 2.0, 3.0]}
    )
    result = CrosstabTool(df).forward(rows="corso", columns="ruolo")

    assert "(vuoto)" in result["data"][0]


def test_thin_cells_are_flagged_for_averages():
    df = pd.DataFrame(
        {"corso": ["A", "A", "B", "B"], "ruolo": ["X", "Y", "X", "Y"], "voto": [1.0, 2.0, 3.0, 4.0]}
    )
    result = CrosstabTool(df).forward(rows="corso", columns="ruolo", metric="voto", op="mean")

    assert "Caution" in result["note"]


def test_counts_are_not_flagged_as_unreliable(tool):
    """A count is exact however few rows it covers."""
    df = pd.DataFrame({"corso": ["A", "B"], "ruolo": ["X", "Y"]})
    result = CrosstabTool(df).forward(rows="corso", columns="ruolo")

    assert "note" not in result


def test_carries_export_name_and_meta(tool):
    result = tool.forward(rows="corso", columns="ruolo", metric="voto", op="mean")

    assert result["export_name"] == "mean_voto_corso_x_ruolo"
    assert result["meta"]["row_values"] == 2
    assert result["meta"]["column_values"] == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_same_column_twice_is_rejected(tool):
    assert tool.forward(rows="corso", columns="corso")["code"] == "SAME_DIMENSION"


def test_unknown_column_is_rejected(tool):
    assert tool.forward(rows="corso", columns="nope")["code"] == "INVALID_COLUMN"


def test_missing_dimension_is_rejected(tool):
    assert tool.forward(rows="corso", columns="")["code"] == "MISSING_DIMENSION"


def test_aggregation_without_metric_is_rejected(tool):
    assert tool.forward(rows="corso", columns="ruolo", op="mean")["code"] == "MISSING_METRIC"


def test_normalize_with_mean_is_rejected(tool):
    """Percentages of a mean are meaningless, so this is an error rather than a guess."""
    result = tool.forward(rows="corso", columns="ruolo", metric="voto", op="mean", normalize="rows")

    assert result["code"] == "INVALID_NORMALIZE_OP"


def test_upstream_data_is_used(tool):
    records = [{"g": "A", "h": "X"}, {"g": "A", "h": "X"}, {"g": "B", "h": "Y"}]
    result = tool.forward(rows="g", columns="h", data=records)

    assert {r["g"] for r in result["data"]} == {"A", "B"}
