"""No DataChat tool may silently truncate its result.

Regression origin: three separate user reports of "table preview shows 20 rows, the
exported CSV has 50". Each time the 50 came from a different tool's hardcoded ceiling
leaking into the export, because the CSV is built from whatever the tool returned. The
caps predate the preview/export layer, which now sizes responses on its own.

The invariant: with no explicit `n`, a tool returns everything. With an explicit `n`, it
returns that many AND says so via `note`.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.limits import resolve_limit, truncation_note
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.unique_values_tool import UniqueValuesTool


@pytest.fixture()
def wide_df():
    """80 distinct groups and 60 columns — past every old ceiling (50 and 20)."""
    n = 800
    return pd.DataFrame(
        {
            "corso": [f"corso {i % 80}" for i in range(n)],
            "voto": [float(i % 5 + 1) for i in range(n)],
            **{f"col{c}": range(n) for c in range(58)},
        }
    )


# ---------------------------------------------------------------------------
# The reported scenario: group-by ranking
# ---------------------------------------------------------------------------


def test_aggregate_ranking_returns_every_group(wide_df):
    """'Group by course, rank by satisfaction' must not stop at 50 groups."""
    result = AggregateTool(wide_df).forward(
        group_by="corso", op="mean", metric="voto"
    )

    assert len(result["data"]) == 80
    # A small-sample caveat is expected here (10 responses per course); what must be
    # absent is a *truncation* note, since nothing was dropped.
    assert "groups available" not in (result.get("note") or "")


def test_aggregate_explicit_limit_is_disclosed(wide_df):
    result = AggregateTool(wide_df).forward(
        group_by="corso", op="mean", metric="voto", n=10
    )

    assert len(result["data"]) == 10
    assert "80 groups available" in result["note"]


def test_aggregate_carries_an_export_name(wide_df):
    result = AggregateTool(wide_df).forward(group_by="corso", op="mean", metric="voto")

    assert result["export_name"] == "mean_voto_by_corso"


# ---------------------------------------------------------------------------
# Every row-returning tool: uncapped by default
# ---------------------------------------------------------------------------


def test_unique_values_returns_every_distinct_value(wide_df):
    result = UniqueValuesTool(wide_df).forward(column="corso")

    assert len(result["data"]) == 80


def test_filter_rows_returns_every_match(wide_df):
    result = FilterRowsTool(wide_df).forward(where_col="corso", op="is_not_empty")

    assert len(result["data"]) == 800


def test_describe_covers_every_column(wide_df):
    result = DescribeTool(wide_df).forward()

    assert len(result["data"]) == len(wide_df.columns)


def test_missing_values_covers_every_column(wide_df):
    result = MissingValuesTool(wide_df).forward()

    assert len(result["data"]) == len(wide_df.columns)


# ---------------------------------------------------------------------------
# Row-wise tools keep every column
# ---------------------------------------------------------------------------


def test_top_rows_keeps_all_columns(wide_df):
    """Used to silently drop everything past the 10th column."""
    result = TopRowsTool(wide_df).forward(sort_by="voto", n=3)

    assert len(result["data"][0]) == len(wide_df.columns)


def test_sample_rows_keeps_all_columns(wide_df):
    result = SampleRowsTool(wide_df).forward(n=3)

    assert len(result["data"][0]) == len(wide_df.columns)


def test_top_rows_honours_a_large_n(wide_df):
    """The old ceiling of 20 silently overrode any larger request."""
    result = TopRowsTool(wide_df).forward(sort_by="voto", n=200)

    assert len(result["data"]) == 200


def test_sample_rows_honours_a_large_n(wide_df):
    result = SampleRowsTool(wide_df).forward(n=200)

    assert len(result["data"]) == 200


def test_top_rows_default_stays_small(wide_df):
    """These tools exist to return a handful; only the hard ceiling was wrong."""
    result = TopRowsTool(wide_df).forward(sort_by="voto")

    assert len(result["data"]) == 5
    assert "800 rows available" in result["note"]


# ---------------------------------------------------------------------------
# The shared helpers
# ---------------------------------------------------------------------------


def test_resolve_limit_treats_absent_as_unlimited():
    assert resolve_limit(None) is None
    assert resolve_limit(0) is None
    assert resolve_limit(25) == 25
    assert resolve_limit(-5) == 1
    assert resolve_limit("bad") is None


def test_resolve_limit_applies_a_default_when_given():
    assert resolve_limit(None, default=5) == 5
    assert resolve_limit(50, default=5) == 50


def test_truncation_note_is_silent_when_nothing_was_dropped():
    assert truncation_note(10, 10) is None
    assert truncation_note(10, 5) is None


def test_truncation_note_names_the_unit():
    note = truncation_note(10, 80, unit="groups")

    assert "80 groups available" in note
    assert "10" in note
