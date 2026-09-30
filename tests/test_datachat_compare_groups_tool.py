"""compare_groups: Welch two-sample comparison of a mean between two explicit groups."""

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from datachat.tools.compare_groups_tool import CompareGroupsTool
from tests.test_datachat_result_preview import _ToolEngine, _chat
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)

FIELDS = [
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "mean_a",
    "mean_b",
    "difference",
    "test",
    "statistic",
    "p_value",
    "ci_low",
    "ci_high",
]


def _run(df, **kwargs):
    kwargs.setdefault("metric", "v")
    kwargs.setdefault("group_col", "g")
    return CompareGroupsTool(df).forward(**kwargs)


def _row(out):
    assert out["kind"] == "table", out
    assert len(out["data"]) == 1
    return out["data"][0]


def _frame(a, b, la="A", lb="B"):
    return pd.DataFrame({"g": [la] * len(a) + [lb] * len(b), "v": list(a) + list(b)})


def _normal(mean_a, mean_b, n_a=60, n_b=60, sd_a=0.7, sd_b=0.7, seed=42):
    rng = np.random.default_rng(seed)
    return _frame(rng.normal(mean_a, sd_a, n_a), rng.normal(mean_b, sd_b, n_b))


# --- core -------------------------------------------------------------------


def test_clear_difference():
    row = _row(_run(_normal(4.2, 3.0), group_a="A", group_b="B"))

    assert list(row) == FIELDS
    assert row["test"] == "welch_t"
    assert row["difference"] > 0
    assert row["statistic"] > 0
    assert row["p_value"] < 1e-6
    assert 0 < row["ci_low"] < 1.2 < row["ci_high"]


def test_overlapping_groups_high_p_value():
    rng = np.random.default_rng(7)
    values = rng.normal(3.5, 0.8, 120)
    out = _run(_frame(values[:60], values[60:]), group_a="A", group_b="B")
    row = _row(out)

    assert row["p_value"] > 0.05
    assert row["ci_low"] < 0 < row["ci_high"]
    assert "note" not in out


def test_unequal_sizes_and_variances_match_scipy_welch():
    df = _normal(5.0, 4.0, n_a=12, n_b=80, sd_a=3.0, sd_b=0.5, seed=3)
    row = _row(_run(df, group_a="A", group_b="B"))
    a = df.loc[df.g == "A", "v"].to_numpy()
    b = df.loc[df.g == "B", "v"].to_numpy()
    expected = stats.ttest_ind(a, b, equal_var=False)
    ci = expected.confidence_interval(0.95)

    assert (row["n_a"], row["n_b"]) == (12, 80)
    assert row["statistic"] == pytest.approx(expected.statistic)
    assert row["p_value"] == pytest.approx(expected.pvalue)
    assert (row["ci_low"], row["ci_high"]) == (pytest.approx(ci.low), pytest.approx(ci.high))
    # Welch, not the pooled test: the two disagree with these variances.
    pooled = stats.ttest_ind(a, b, equal_var=True)
    assert row["p_value"] != pytest.approx(pooled.pvalue)


def test_difference_is_mean_a_minus_mean_b_and_ci_follows_it():
    df = _frame([1.0, 2.0, 3.0, 4.0], [10.0, 11.0, 13.0])
    forward = _row(_run(df, group_a="A", group_b="B"))
    backward = _row(_run(df, group_a="B", group_b="A"))

    assert forward["difference"] == forward["mean_a"] - forward["mean_b"] == 2.5 - 34 / 3
    assert forward["ci_low"] < forward["difference"] < forward["ci_high"] < 0
    assert backward["difference"] == pytest.approx(-forward["difference"])
    assert backward["statistic"] == pytest.approx(-forward["statistic"])
    assert backward["p_value"] == pytest.approx(forward["p_value"])
    assert backward["ci_low"] == pytest.approx(-forward["ci_high"])
    assert backward["ci_high"] == pytest.approx(-forward["ci_low"])


def test_numeric_fields_are_numbers():
    row = _row(_run(_normal(4.0, 3.5), group_a="A", group_b="B"))

    assert isinstance(row["n_a"], int) and isinstance(row["n_b"], int)
    for key in ("mean_a", "mean_b", "difference", "statistic", "p_value", "ci_low", "ci_high"):
        assert isinstance(row[key], float), key
        assert math.isfinite(row[key])


def test_p_value_is_not_floored():
    row = _row(_run(_normal(1.0, 5.0, sd_a=0.3, sd_b=0.3, n_a=80, n_b=80), group_a="A", group_b="B"))

    assert isinstance(row["p_value"], float)
    assert row["p_value"] < 1e-6


# --- validation -------------------------------------------------------------


@pytest.fixture
def df():
    return _frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "", "group_a": "A", "group_b": "B"},
        {"group_col": "  ", "group_a": "A", "group_b": "B"},
        {"metric": None, "group_a": "A", "group_b": "B"},
    ],
)
def test_missing_column_parameters(df, kwargs):
    assert _run(df, **kwargs)["code"] == "MISSING_PARAMS"


@pytest.mark.parametrize("missing", [None, "", float("nan")])
def test_missing_group_a(df, missing):
    assert _run(df, group_a=missing, group_b="B")["code"] == "MISSING_PARAMS"


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_group_b(df, missing):
    assert _run(df, group_a="A", group_b=missing)["code"] == "MISSING_PARAMS"


def test_group_arguments_are_required_by_the_signature(df):
    with pytest.raises(TypeError):
        CompareGroupsTool(df).forward(metric="v", group_col="g")


def test_unknown_metric(df):
    assert _run(df, metric="nope", group_a="A", group_b="B")["code"] == "INVALID_COLUMN"


def test_unknown_group_column(df):
    assert _run(df, group_col="nope", group_a="A", group_b="B")["code"] == "INVALID_COLUMN"


def test_metric_and_group_column_must_differ(df):
    assert _run(df, metric="g", group_a="A", group_b="B")["code"] == "SAME_COLUMN"


def test_same_group(df):
    assert _run(df, group_a="A", group_b="A")["code"] == "SAME_GROUP"


def test_absent_group(df):
    out = _run(df, group_a="A", group_b="Z")

    assert out["code"] == "INVALID_GROUP"
    assert "group_b" in out["message"]


def test_group_with_fewer_than_two_usable_values():
    out = _run(_frame([1.0, None, "x"], [4.0, 5.0]), group_a="A", group_b="B")

    assert out["code"] == "GROUP_TOO_SMALL"
    assert "group_a has 1" in out["message"]


def test_no_usable_numeric_data():
    out = _run(_frame(["x", "y"], ["z", None]), group_a="A", group_b="B")

    assert out["code"] == "NO_NUMERIC_DATA"


# --- metric cleaning --------------------------------------------------------


def test_missing_metric_values_are_excluded_and_disclosed():
    out = _run(_frame([1.0, None, 3.0, np.nan], [4.0, 5.0, pd.NA]), group_a="A", group_b="B")
    row = _row(out)

    assert (row["n_a"], row["n_b"]) == (2, 2)
    assert row["mean_a"] == 2.0
    assert out["note"] == "Excluded rows without a usable numeric 'v' value: 2 in group_a, 1 in group_b."


def test_numeric_strings_are_converted():
    out = _run(_frame(["1", " 2 ", "3.5"], [4, "5e0"]), group_a="A", group_b="B")
    row = _row(out)

    assert (row["n_a"], row["n_b"]) == (3, 2)
    assert row["mean_a"] == pytest.approx(6.5 / 3)
    assert "note" not in out


def test_unconvertible_strings_are_excluded_and_disclosed():
    out = _run(_frame(["1", "2", "x", "4,5"], ["5", "6", "n/a"]), group_a="A", group_b="B")
    row = _row(out)

    assert (row["n_a"], row["n_b"]) == (2, 2)
    assert "2 in group_a, 1 in group_b" in out["note"]


def test_infinite_values_are_excluded_and_disclosed():
    out = _run(_frame([1.0, 2.0, np.inf, "inf"], [4.0, -np.inf, 5.0]), group_a="A", group_b="B")
    row = _row(out)

    assert (row["n_a"], row["n_b"]) == (2, 2)
    assert math.isfinite(row["mean_a"]) and math.isfinite(row["difference"])
    assert "2 in group_a, 1 in group_b" in out["note"]


def test_booleans_are_not_treated_as_zero_one():
    out = _run(_frame([True, False, True], [False, False]), group_a="A", group_b="B")

    assert out["code"] == "NO_NUMERIC_DATA"

    mixed = _run(_frame([1.0, 2.0, True], [3.0, 4.0, np.bool_(False)]), group_a="A", group_b="B")
    row = _row(mixed)
    assert (row["n_a"], row["n_b"]) == (2, 2)
    assert "1 in group_a, 1 in group_b" in mixed["note"]


def test_datetimes_are_not_treated_as_nanoseconds():
    dates = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3, "v": pd.date_range("2020-01-01", periods=6)})
    assert _run(dates, group_a="A", group_b="B")["code"] == "NO_NUMERIC_DATA"

    mixed = _frame([1.0, 2.0, dt.date(2020, 1, 1)], [3.0, 4.0, pd.Timestamp("2020-01-01")])
    out = _run(mixed, group_a="A", group_b="B")
    assert (_row(out)["n_a"], _row(out)["n_b"]) == (2, 2)
    assert "1 in group_a, 1 in group_b" in out["note"]


# --- group identity ---------------------------------------------------------


def test_number_one_and_string_one_are_different_groups():
    df = pd.DataFrame({"g": [1, 1, 1, "1", "1", "1", 2, 2], "v": [1, 2, 3, 10, 11, 12, 5, 6]})

    as_string = _row(_run(df, group_a="1", group_b=2))
    as_number = _row(_run(df, group_a=1, group_b=2))
    both = _row(_run(df, group_a=1, group_b="1"))

    assert as_string["group_a"] == "1" and as_string["mean_a"] == 11.0 and as_string["n_a"] == 3
    assert as_number["group_a"] == 1 and as_number["mean_a"] == 2.0 and as_number["n_a"] == 3
    assert (both["mean_a"], both["mean_b"]) == (2.0, 11.0)


def test_boolean_true_is_not_the_number_one():
    df = pd.DataFrame({"g": [True, True, 1, 1, 0, 0], "v": [1, 2, 10, 11, 5, 6]})

    row = _row(_run(df, group_a=True, group_b=1))
    assert (row["group_a"], row["mean_a"], row["mean_b"]) == (True, 1.5, 10.5)
    assert _run(df, group_a="True", group_b=0)["code"] == "INVALID_GROUP"


def test_numeric_text_does_not_select_a_numeric_group():
    df = pd.DataFrame({"g": [1.0, 1.0, 2.0, 2.0], "v": [1, 2, 5, 7]})

    for text in ("1", "01", "1e0", "1.0", " 1"):
        assert _run(df, group_a=text, group_b=2)["code"] == "INVALID_GROUP"
    row = _row(_run(df, group_a=1, group_b=2.0))
    assert (row["group_a"], row["group_b"], row["n_a"]) == (1.0, 2.0, 2)


def test_missing_group_keys_are_not_textual_groups():
    df = pd.DataFrame({"g": [None, None, np.nan, "A", "A", "B", "B"], "v": [1, 2, 3, 4, 5, 6, 7]})

    for literal in ("None", "nan", "(empty)"):
        assert _run(df, group_a=literal, group_b="A")["code"] == "INVALID_GROUP"
    row = _row(_run(df, group_a="A", group_b="B"))
    assert (row["n_a"], row["n_b"]) == (2, 2)


def test_literal_none_string_is_an_ordinary_group():
    df = pd.DataFrame({"g": ["None", "None", None, None, "B", "B"], "v": [1, 2, 30, 40, 5, 6]})

    row = _row(_run(df, group_a="None", group_b="B"))
    assert (row["n_a"], row["mean_a"]) == (2, 1.5)


def test_whitespace_is_part_of_group_identity():
    df = pd.DataFrame({"g": [" A", " A", "A", "A", "B", "B"], "v": [10, 11, 1, 2, 5, 6]})

    padded = _row(_run(df, group_a=" A", group_b="B"))
    plain = _row(_run(df, group_a="A", group_b="B"))
    assert (padded["group_a"], padded["n_a"], padded["mean_a"]) == (" A", 2, 10.5)
    assert (plain["group_a"], plain["n_a"], plain["mean_a"]) == ("A", 2, 1.5)
    assert _run(df, group_a="A ", group_b="B")["code"] == "INVALID_GROUP"


def test_categorical_group_column():
    df = _frame([1.0, 2.0], [3.0, 5.0])
    df["g"] = df["g"].astype("category")

    row = _row(_run(df, group_a="A", group_b="B"))
    assert (row["group_a"], row["n_a"]) == ("A", 2)


# --- degenerate data --------------------------------------------------------


def test_equal_constant_groups_report_no_statistics():
    out = _run(_frame([3.0] * 5, [3.0] * 5), group_a="A", group_b="B")
    row = _row(out)

    assert row["difference"] == 0.0
    assert all(row[k] is None for k in ("statistic", "p_value", "ci_low", "ci_high"))
    assert "zero variance" in out["note"]


def test_different_constant_groups_report_no_statistics():
    out = _run(_frame([3.0] * 5, [4.0] * 5), group_a="A", group_b="B")
    row = _row(out)

    assert row["difference"] == -1.0
    # scipy would give an infinite statistic and p=0 here: not a measurement.
    assert all(row[k] is None for k in ("statistic", "p_value", "ci_low", "ci_high"))
    assert "zero variance" in out["note"]


def test_one_constant_group_is_still_computable():
    row = _row(_run(_frame([3.0] * 5, [1.0, 2.0, 4.0, 5.0]), group_a="A", group_b="B"))

    assert math.isfinite(row["statistic"]) and math.isfinite(row["p_value"])


def test_near_constant_data_discloses_the_precision_warning():
    base = 1e9
    out = _run(
        _frame([base, base, base, base + 1e-6], [base, base, base + 1e-6, base]),
        group_a="A",
        group_b="B",
    )

    assert "precision warning" in out["note"]
    assert "zero variance" not in out["note"]


def test_non_finite_statistical_result_is_not_reported(monkeypatch):
    class _Result:
        statistic = float("nan")
        pvalue = float("nan")

        def confidence_interval(self, confidence_level):
            return stats._stats_py.ConfidenceInterval(float("nan"), float("nan"))

    monkeypatch.setattr(stats, "ttest_ind", lambda *a, **k: _Result())
    out = _run(_frame([1.0, 2.0, 3.0], [4.0, 6.0]), group_a="A", group_b="B")
    row = _row(out)

    assert all(row[k] is None for k in ("statistic", "p_value", "ci_low", "ci_high"))
    assert "did not return finite values" in out["note"]


# --- data= ------------------------------------------------------------------


def test_records_input():
    records = [{"g": "A", "v": 1}, {"g": "A", "v": 2}, {"g": "B", "v": 5}, {"g": "B", "v": 8}]
    session = pd.DataFrame({"g": ["X"], "v": [0]})

    row = _row(CompareGroupsTool(session).forward(metric="v", group_col="g", group_a="A", group_b="B", data=records))
    assert (row["n_a"], row["n_b"], row["difference"]) == (2, 2, -5.0)

    wrapped = CompareGroupsTool(session).forward(
        metric="v", group_col="g", group_a="A", group_b="B", data={"data": records}
    )
    assert _row(wrapped) == row


def test_empty_records():
    out = CompareGroupsTool(pd.DataFrame({"g": ["A"], "v": [1]})).forward(
        metric="v", group_col="g", group_a="A", group_b="B", data=[]
    )
    assert out == {"kind": "table", "data": []}


def test_invalid_records():
    out = CompareGroupsTool(pd.DataFrame()).forward(metric="v", group_col="g", group_a="A", group_b="B", data="x")
    assert out["code"] == "INVALID_DATA"


# --- contract regressions ---------------------------------------------------


def test_no_verdicts_labels_meta_or_export_name():
    for df in (_normal(4.2, 3.0), _normal(3.5, 3.5, seed=7), _normal(4.2, 3.0, n_a=5, n_b=5)):
        out = _run(df, group_a="A", group_b="B")
        assert set(out) <= {"kind", "data", "note"}
        assert "significant" not in out["data"][0]
        text = str(out).lower()
        for phrase in ("meaningful", "indistinguishable", "scores higher", "better", "equivalent", "caution", "effect"):
            assert phrase not in text


def test_no_small_sample_caution_below_fifteen():
    out = _run(_normal(4.2, 3.0, n_a=5, n_b=5), group_a="A", group_b="B")
    assert "note" not in out


def test_no_ordinal_or_test_selection_parameters():
    assert set(CompareGroupsTool.inputs) == {"metric", "group_col", "group_a", "group_b", "data"}


def test_no_mann_whitney_path(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("mannwhitneyu must not be called")

    monkeypatch.setattr(stats, "mannwhitneyu", _forbidden)
    # Skewed data where a rank test and a mean test disagree.
    row = _row(_run(_frame([1] * 29 + [100], [2] * 30), group_a="A", group_b="B"))
    assert row["test"] == "welch_t"
    assert row["p_value"] > 0.05
    assert row["ci_low"] < 0 < row["ci_high"]


def test_no_default_two_largest_groups():
    df = pd.DataFrame({"g": ["A"] * 30 + ["B"] * 20 + ["C"] * 2, "v": [4.0] * 30 + [3.0] * 20 + [1.0, 2.0]})

    assert _run(df, group_a="A", group_b=None)["code"] == "MISSING_PARAMS"
    assert _run(df, group_a=None, group_b=None)["code"] == "MISSING_PARAMS"
    row = _row(_run(df, group_a="C", group_b="B"))
    assert (row["group_a"], row["group_b"]) == ("C", "B")


# --- through POST /datachat ------------------------------------------------------


def test_exclusion_note_reaches_the_client(monkeypatch):
    df = _frame([1.0, 2.0, "x", 4.0], [5.0, 7.0, np.inf])
    body = _chat(monkeypatch, _ToolEngine(lambda: _run(df, group_a="A", group_b="B")))

    assert body["note"] == "Excluded rows without a usable numeric 'v' value: 1 in group_a, 1 in group_b."
    assert body["result_rows"] == 1


def test_clean_comparison_has_no_client_note(monkeypatch):
    df = _frame([1.0, 2.0, 4.0], [5.0, 7.0])
    body = _chat(monkeypatch, _ToolEngine(lambda: _run(df, group_a="A", group_b="B")))

    assert "note" not in body
