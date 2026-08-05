"""Tests for compare_groups — telling a real difference from noise.

Exists because ranking group averages is easy and misleading: 800 responses over 80 courses
leaves ~10 each, so the top of any ranking is largely noise. The tool must be willing to say
a gap is not a finding.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.compare_groups_tool import CompareGroupsTool


def _two_groups(mean_a, mean_b, n=60, sd=0.7, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "corso": ["A"] * n + ["B"] * n,
            "voto": list(rng.normal(mean_a, sd, n)) + list(rng.normal(mean_b, sd, n)),
        }
    )


def test_a_real_difference_is_detected():
    result = CompareGroupsTool(_two_groups(4.2, 3.0)).forward(metric="voto", group_col="corso")
    row = result["data"][0]

    assert row["significant"] is True
    assert row["difference"] > 0
    assert result["meta"]["effect"] in {"medium", "large"}
    assert "scores higher" in result["note"]


def test_identical_groups_are_reported_as_indistinguishable():
    """The important half: the tool must refuse to crown a winner."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"corso": ["A"] * 60 + ["B"] * 60, "voto": rng.normal(3.5, 0.8, 120)})

    result = CompareGroupsTool(df).forward(metric="voto", group_col="corso")

    assert result["data"][0]["significant"] is False
    assert "No meaningful difference" in result["note"]
    assert "Do not report one as better" in result["note"]


def test_confidence_interval_brackets_the_true_difference():
    result = CompareGroupsTool(_two_groups(4.2, 3.0)).forward(metric="voto", group_col="corso")
    meta = result["meta"]

    assert meta["ci95_low"] < 1.2 < meta["ci95_high"]
    assert meta["ci95_low"] < result["data"][0]["difference"] < meta["ci95_high"]


def test_the_verdict_survives_the_column_preview():
    """20 fields on one row would have hidden the conclusion: the client keeps only 10."""
    result = CompareGroupsTool(_two_groups(4.2, 3.0)).forward(metric="voto", group_col="corso")

    assert len(result["data"][0]) <= 10
    assert result["note"]
    for key in ("group_a", "group_b", "difference", "p_value", "significant"):
        assert key in result["data"][0]


def test_small_samples_are_flagged():
    result = CompareGroupsTool(_two_groups(4.2, 3.0, n=10)).forward(metric="voto", group_col="corso")

    assert "Caution" in result["note"]
    assert "10" in result["note"]


def test_well_sampled_comparison_carries_no_caution():
    result = CompareGroupsTool(_two_groups(4.2, 3.0, n=60)).forward(metric="voto", group_col="corso")

    # The note still holds the verdict; what must be absent is the small-sample caveat.
    assert "Caution" not in result["note"]


def test_p_value_is_never_reported_as_zero():
    """p=0 is not a measurement; a floor keeps the output honest."""
    result = CompareGroupsTool(_two_groups(1.0, 5.0, n=80, sd=0.3)).forward(
        metric="voto", group_col="corso"
    )

    assert result["meta"]["p_value_exact"] > 0
    assert result["data"][0]["p_value"].startswith("<")
    # And never with two operators glued together.
    assert "p=<" not in result["note"]
    assert "p=0.000" not in result["note"]


def test_ordinal_uses_mann_whitney_by_default():
    result = CompareGroupsTool(_two_groups(4.2, 3.0)).forward(metric="voto", group_col="corso")

    assert result["meta"]["test"] == "mann-whitney"


def test_ordinal_false_uses_welch():
    result = CompareGroupsTool(_two_groups(4.2, 3.0)).forward(
        metric="voto", group_col="corso", ordinal=False
    )

    assert result["meta"]["test"] == "welch t-test"


def test_groups_default_to_the_two_largest():
    df = pd.DataFrame(
        {
            "corso": ["A"] * 30 + ["B"] * 20 + ["C"] * 2,
            "voto": [4.0] * 30 + [3.0] * 20 + [1.0] * 2,
        }
    )
    row = CompareGroupsTool(df).forward(metric="voto", group_col="corso")["data"][0]

    assert {row["group_a"], row["group_b"]} == {"A", "B"}


def test_explicit_groups_are_honoured():
    df = pd.DataFrame(
        {"corso": ["A"] * 20 + ["B"] * 20 + ["C"] * 20, "voto": [4.0] * 20 + [3.0] * 20 + [1.0] * 20}
    )
    row = CompareGroupsTool(df).forward(
        metric="voto", group_col="corso", group_a="A", group_b="C"
    )["data"][0]

    assert {row["group_a"], row["group_b"]} == {"A", "C"}


def test_per_group_statistics_are_reported():
    result = CompareGroupsTool(_two_groups(4.2, 3.0, n=25)).forward(metric="voto", group_col="corso")
    row = result["data"][0]

    assert row["n_a"] == 25
    assert row["n_b"] == 25
    assert isinstance(row["mean_a"], float)
    assert isinstance(row["mean_b"], float)
    # Medians live in meta to keep the visible row within the 10-column preview.
    assert isinstance(result["meta"]["median_a"], float)
    assert isinstance(result["meta"]["median_b"], float)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_unknown_group_is_rejected():
    result = CompareGroupsTool(_two_groups(4.0, 3.0)).forward(
        metric="voto", group_col="corso", group_a="A", group_b="Z"
    )

    assert result["code"] == "INVALID_GROUP"


def test_same_group_twice_is_rejected():
    result = CompareGroupsTool(_two_groups(4.0, 3.0)).forward(
        metric="voto", group_col="corso", group_a="A", group_b="A"
    )

    assert result["code"] == "SAME_GROUP"


def test_group_with_one_value_is_rejected():
    df = pd.DataFrame({"corso": ["A", "B", "B"], "voto": [4.0, 3.0, 3.5]})
    result = CompareGroupsTool(df).forward(metric="voto", group_col="corso")

    assert result["code"] == "GROUP_TOO_SMALL"


def test_non_numeric_metric_is_rejected():
    df = pd.DataFrame({"corso": ["A"] * 5 + ["B"] * 5, "testo": ["x"] * 10})
    result = CompareGroupsTool(df).forward(metric="testo", group_col="corso")

    assert result["code"] == "NO_NUMERIC_DATA"


def test_unknown_column_is_rejected():
    result = CompareGroupsTool(_two_groups(4.0, 3.0)).forward(metric="nope", group_col="corso")

    assert result["code"] == "INVALID_COLUMN"
