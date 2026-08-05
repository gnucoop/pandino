"""Tests for filter_rows null/blank handling and the removal of its silent caps.

Regression origin: asked to "elenca tutte le righe che riportano dei suggerimenti" on a
survey column that was 70% blank, the agent had no operator for "is not empty". It tried
`value=None` and `value=""`, both of which matched nothing because `astype(str)` renders
NaN as the literal "nan", computed 804 - 0 = 804 "rows with suggestions", recognised the
answer was impossible, and looped until it ran out of steps. The real answer was 236.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.filter_rows_tool import FilterRowsTool

COL = "Hai dei suggerimenti per il miglioramento di formazioni come questa?"


@pytest.fixture()
def survey_df():
    """804 rows, 568 blank — NaN, empty string and whitespace-only, as real exports mix."""
    blanks = [np.nan] * 548 + [""] * 10 + ["   "] * 10
    filled = [f"suggerimento {i}" for i in range(236)]
    return pd.DataFrame(
        {
            COL: blanks + filled,
            "corso": ["X"] * 804,
            **{f"extra{i}": range(804) for i in range(11)},
        }
    )


@pytest.fixture()
def tool(survey_df):
    return FilterRowsTool(survey_df)


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


def test_is_not_empty_finds_the_filled_rows(tool):
    result = tool.forward(where_col=COL, op="is_not_empty")

    assert result["meta"]["total_matches"] == 236
    assert len(result["data"]) == 236
    assert all(str(r[COL]).strip() for r in result["data"])


def test_is_empty_finds_the_blank_rows(tool):
    result = tool.forward(where_col=COL, op="is_empty")

    assert result["meta"]["total_matches"] == 568


def test_empty_ops_partition_the_dataset(tool):
    filled = tool.forward(where_col=COL, op="is_not_empty")["meta"]["total_matches"]
    blank = tool.forward(where_col=COL, op="is_empty")["meta"]["total_matches"]

    assert filled + blank == 804


def test_value_none_means_missing_not_the_string_none(tool):
    """The original bug: this compared every cell against the literal "none"."""
    result = tool.forward(where_col=COL, value=None, op="eq")

    assert result["meta"]["total_matches"] == 568


def test_empty_string_value_also_means_missing(tool):
    result = tool.forward(where_col=COL, value="", op="eq")

    assert result["meta"]["total_matches"] == 568


def test_whitespace_only_cells_count_as_empty(tool):
    """A cell of "   " is blank to a user, so it must not appear in is_not_empty."""
    result = tool.forward(where_col=COL, op="is_not_empty")

    assert not any(str(r[COL]).strip() == "" for r in result["data"])


# ---------------------------------------------------------------------------
# No silent caps
# ---------------------------------------------------------------------------


def test_all_matching_rows_are_returned_by_default(tool):
    """Used to cap at 50 with no indication, which truncated the CSV export too."""
    result = tool.forward(where_col=COL, op="is_not_empty")

    assert len(result["data"]) == 236


def test_all_columns_are_returned_by_default(tool, survey_df):
    """Used to keep only the first 10 columns."""
    result = tool.forward(where_col=COL, op="is_not_empty")

    assert len(result["data"][0]) == len(survey_df.columns)


def test_explicit_row_limit_is_honoured_and_disclosed(tool):
    result = tool.forward(where_col=COL, op="is_not_empty", n=10)

    assert len(result["data"]) == 10
    assert result["meta"]["total_matches"] == 236
    assert "236 rows match" in result["note"]


def test_no_note_when_nothing_was_dropped(tool):
    result = tool.forward(where_col=COL, op="is_not_empty")

    assert "note" not in result


def test_explicit_columns_still_narrow_the_result(tool):
    result = tool.forward(where_col=COL, op="is_not_empty", columns=[COL, "corso"])

    assert set(result["data"][0].keys()) == {COL, "corso"}


def test_result_carries_an_export_name(tool):
    result = tool.forward(where_col=COL, op="is_not_empty")

    assert result["export_name"].startswith("filter_")


# ---------------------------------------------------------------------------
# Existing behaviour must survive
# ---------------------------------------------------------------------------


def test_exact_match_still_works(tool):
    result = tool.forward(where_col=COL, value="suggerimento 5", op="eq")

    assert result["meta"]["total_matches"] == 1


def test_numeric_comparison_still_works(tool):
    result = tool.forward(where_col="extra0", value=100, op="lt")

    assert result["meta"]["total_matches"] == 100


def test_numeric_column_empty_ops_use_nan_only():
    df = pd.DataFrame({"n": [1.0, np.nan, 3.0, 0.0]})
    result = FilterRowsTool(df).forward(where_col="n", op="is_empty")

    # 0.0 is a value, not a blank
    assert result["meta"]["total_matches"] == 1


def test_second_condition_can_use_an_empty_op(tool):
    """is_not_empty needs no value, so the AND must activate on the column alone."""
    result = tool.forward(
        where_col="corso", value="X", op="eq", where_col2=COL, op2="is_not_empty"
    )

    assert result["meta"]["total_matches"] == 236


def test_invalid_op_is_rejected(tool):
    result = tool.forward(where_col=COL, value="x", op="regex")

    assert result["kind"] == "error"
    assert result["code"] == "INVALID_FILTER_OP"


def test_unknown_column_is_rejected(tool):
    result = tool.forward(where_col="nope", op="is_not_empty")

    assert result["kind"] == "error"
    assert result["code"] == "INVALID_FILTER_COLUMN"


def test_passed_in_data_is_filtered(tool):
    records = [{"a": "x"}, {"a": None}, {"a": "y"}]
    result = tool.forward(where_col="a", op="is_not_empty", data=records)

    assert result["meta"]["total_matches"] == 2
