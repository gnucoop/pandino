"""Tests for Spearman support and matrix mode on the correlation tool.

Pearson assumes evenly spaced values, which a 1-5 rating scale is not; Spearman is the
correct coefficient for ordinal answers. Matrix mode answers "which questions relate to
overall satisfaction?" in one call instead of N.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.correlation_tool import CorrelationTool


@pytest.fixture()
def ratings_df():
    """A monotonic-but-curved relationship plus an unrelated column."""
    x = np.arange(1, 21)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "soddisfazione": x,
            "esponenziale": np.exp(x / 3),
            "rumore": rng.normal(0, 1, 20),
            "costante": [3] * 20,
            "testo": ["a"] * 20,
        }
    )


@pytest.fixture()
def tool(ratings_df):
    return CorrelationTool(ratings_df)


def test_spearman_detects_a_monotonic_curve_pearson_misses(tool):
    """Proves the method is really applied, not just accepted and ignored."""
    pearson = tool.forward(col_x="soddisfazione", col_y="esponenziale", method="pearson")["data"][0]
    spearman = tool.forward(col_x="soddisfazione", col_y="esponenziale", method="spearman")["data"][0]

    assert spearman["correlation"] == pytest.approx(1.0)
    assert pearson["correlation"] < 0.9
    assert spearman["method"] == "spearman"


def test_pearson_remains_the_default(tool):
    result = tool.forward(col_x="soddisfazione", col_y="esponenziale")

    assert result["data"][0]["method"] == "pearson"


def test_strength_label_accompanies_the_coefficient(tool):
    row = tool.forward(col_x="soddisfazione", col_y="esponenziale", method="spearman")["data"][0]

    assert row["strength"] == "very strong"


def test_unsupported_method_is_rejected(tool):
    assert tool.forward(col_x="soddisfazione", col_y="rumore", method="kendall")["code"] == "INVALID_METHOD"


# ---------------------------------------------------------------------------
# Matrix mode
# ---------------------------------------------------------------------------


def test_omitting_col_y_ranks_everything_against_the_anchor(tool):
    result = tool.forward(col_x="soddisfazione", method="spearman")

    assert all(r["col_x"] == "soddisfazione" for r in result["data"])
    assert {r["col_y"] for r in result["data"]} == {"esponenziale", "rumore"}
    assert result["meta"]["anchor"] == "soddisfazione"


def test_matrix_is_sorted_by_absolute_strength(tool):
    rows = tool.forward(col_x="soddisfazione", method="spearman")["data"]
    strengths = [abs(r["correlation"]) for r in rows]

    assert strengths == sorted(strengths, reverse=True)


def test_omitting_both_columns_returns_all_pairs(tool):
    result = tool.forward(method="spearman")

    # 3 usable numeric columns -> 3 unordered pairs
    assert len(result["data"]) == 3
    assert result["export_name"] == "correlation_matrix"


def test_constant_and_text_columns_are_excluded(tool):
    """A column with no variation correlates with nothing; text is not numeric."""
    result = tool.forward(method="spearman")
    involved = {r["col_x"] for r in result["data"]} | {r["col_y"] for r in result["data"]}

    assert "costante" not in involved
    assert "testo" not in involved


def test_anchor_without_variation_is_an_error(tool):
    assert tool.forward(col_x="costante")["code"] == "NO_NUMERIC_DATA"


def test_unknown_anchor_is_an_error(tool):
    assert tool.forward(col_x="nope")["code"] == "INVALID_COLUMN"


def test_matrix_needs_two_numeric_columns():
    df = pd.DataFrame({"solo": [1, 2, 3], "testo": ["a", "b", "c"]})
    result = CorrelationTool(df).forward()

    assert result["code"] == "INSUFFICIENT_DATA"


def test_pairwise_result_carries_export_name(tool):
    result = tool.forward(col_x="soddisfazione", col_y="rumore")

    assert result["export_name"] == "correlation_soddisfazione_rumore"


def test_n_reports_overlapping_pairs():
    df = pd.DataFrame({"a": [1, 2, 3, None, 5], "b": [1, 2, None, 4, 5]})
    row = CorrelationTool(df).forward(col_x="a", col_y="b")["data"][0]

    assert row["n"] == 3
