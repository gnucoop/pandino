"""B1: aggregate pre-filter `eq` compares numerically on numeric columns."""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.aggregate_tool import AggregateTool


@pytest.fixture
def df():
    # `score` is float64 because of the NaN, as pandas loads integer CSV columns with gaps.
    return pd.DataFrame(
        {
            "region": ["N", "N", "S", "S", "S", "N"],
            "score": [5.0, np.nan, 5.0, 6.0, 5.0, 4.0],
            "level": [5, 6, 5, 7, 5, 5],
            "code": ["5", "05", "5", "05", "x", "5"],
            "flag": [True, False, True, True, False, False],
            "amount": [10, 20, 30, 40, 50, 60],
        }
    )


def _sums(out):
    assert out["kind"] == "table"
    return {r["region"]: r["sum_amount"] for r in out["data"]}


@pytest.mark.parametrize("value", [5, 5.0, "5", " 5 "])
def test_float_column_with_nan_matches_numerically(df, value):
    out = AggregateTool(df).forward(
        group_by="region", op="sum", metric="amount", where_col="score", value=value
    )
    # Rows with score 5.0: N(10), S(30), S(50); NaN, 6.0 and 4.0 excluded.
    assert _sums(out) == {"N": 10, "S": 80}


def test_int_column_matches_float_value(df):
    out = AggregateTool(df).forward(group_by="region", op="count", where_col="level", value=5.0)
    assert {r["region"]: r["count"] for r in out["data"]} == {"N": 2, "S": 2}


def test_second_filter_uses_numeric_equality(df):
    out = AggregateTool(df).forward(
        group_by="region", op="sum", metric="amount",
        where_col="level", value=5, where_col2="score", value2="5",
    )
    assert _sums(out) == {"N": 10, "S": 80}


def test_object_column_keeps_string_equality(df):
    out = AggregateTool(df).forward(group_by="region", op="sum", metric="amount", where_col="code", value=5)
    # "05" stays distinct from 5.
    assert _sums(out) == {"N": 70, "S": 30}


def test_non_numeric_value_on_numeric_column_is_empty_not_error(df):
    out = AggregateTool(df).forward(group_by="region", op="count", where_col="score", value="abc")
    assert out == {"kind": "table", "data": []}


@pytest.mark.parametrize("value", [True, "true"])
def test_bool_column_behaviour_unchanged(df, value):
    out = AggregateTool(df).forward(group_by="region", op="sum", metric="amount", where_col="flag", value=value)
    assert _sums(out) == {"N": 10, "S": 70}
