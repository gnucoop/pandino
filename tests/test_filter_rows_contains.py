"""Tests for text search on filter_rows.

Before `contains` existed there was no way to search a free-text column at all, on a
dataset whose analytical value is a few hundred open comments.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.filter_rows_tool import FilterRowsTool


@pytest.fixture()
def comments_df():
    return pd.DataFrame(
        {
            "commento": [
                "gli orari sono scomodi",
                "ORARI da rivedere completamente",
                "ottimo corso, docente preparata",
                np.nan,
                "   ",
                "il costo (2.000 euro) è troppo alto",
                "orario ottimo",
            ],
            "corso": ["A", "A", "B", "B", "A", "B", "A"],
        }
    )


@pytest.fixture()
def tool(comments_df):
    return FilterRowsTool(comments_df)


def _texts(result):
    return [r["commento"] for r in result["data"]]


def test_contains_is_case_insensitive(tool):
    """Upper-case 'ORARI' must match a lower-case needle."""
    result = tool.forward(where_col="commento", value="ORARI da rivedere", op="contains")

    assert _texts(result) == ["ORARI da rivedere completamente"]

    lower = tool.forward(where_col="commento", value="orari da rivedere", op="contains")
    assert _texts(lower) == ["ORARI da rivedere completamente"]


def test_contains_matches_substrings_not_whole_words(tool):
    """'orari' also hits 'orario': substring semantics, which is what users expect."""
    result = tool.forward(where_col="commento", value="orari", op="contains")

    assert _texts(result) == [
        "gli orari sono scomodi",
        "ORARI da rivedere completamente",
        "orario ottimo",
    ]


def test_not_contains_excludes_matches_and_blanks(tool):
    result = tool.forward(where_col="commento", value="orari", op="not_contains")
    texts = _texts(result)

    assert "gli orari sono scomodi" not in texts
    assert "ottimo corso, docente preparata" in texts
    # Blanks are not answers that "do not mention" the term.
    assert not any(t is None or str(t).strip() == "" for t in texts)


def test_contains_and_not_contains_partition_the_real_answers(tool, comments_df):
    hits = tool.forward(where_col="commento", value="orari", op="contains")["meta"]["total_matches"]
    misses = tool.forward(where_col="commento", value="orari", op="not_contains")["meta"]["total_matches"]
    non_blank = comments_df["commento"].dropna().astype(str).str.strip().ne("").sum()

    assert hits + misses == non_blank


def test_regex_metacharacters_are_matched_literally(tool):
    """The needle comes from an LLM: '(' must not compile, it must match."""
    result = tool.forward(where_col="commento", value="(2.000", op="contains")

    assert _texts(result) == ["il costo (2.000 euro) è troppo alto"]


def test_wildcard_pattern_matches_nothing(tool):
    """'.*' would match everything as a regex; as a literal it matches nothing here."""
    result = tool.forward(where_col="commento", value=".*", op="contains")

    assert result["data"] == []


def test_unbalanced_bracket_does_not_raise(tool):
    """A regex-mode implementation would throw on this."""
    result = tool.forward(where_col="commento", value="euro)", op="contains")

    assert result["kind"] == "table"
    assert _texts(result) == ["il costo (2.000 euro) è troppo alto"]


def test_empty_needle_matches_nothing(tool):
    result = tool.forward(where_col="commento", value="", op="contains")

    assert result["data"] == []


def test_contains_combines_with_a_second_condition(tool):
    result = tool.forward(
        where_col="commento", value="scomodi", op="contains",
        where_col2="corso", value2="A", op2="eq",
    )

    assert _texts(result) == ["gli orari sono scomodi"]


def test_second_condition_narrows_a_contains_match(tool):
    """Three comments mention 'orari'; all three happen to be course A."""
    both = tool.forward(
        where_col="commento", value="orari", op="contains",
        where_col2="corso", value2="A", op2="eq",
    )

    assert len(both["data"]) == 3
    assert {r["corso"] for r in both["data"]} == {"A"}


def test_contains_works_in_the_second_slot(tool):
    result = tool.forward(
        where_col="corso", value="A", op="eq",
        where_col2="commento", value2="orario", op2="contains",
    )

    assert _texts(result) == ["orario ottimo"]


def test_all_matches_are_returned_and_all_columns_kept(tool):
    result = tool.forward(where_col="commento", value="o", op="contains")

    assert len(result["data"]) == result["meta"]["total_matches"]
    assert set(result["data"][0]) == {"commento", "corso"}
