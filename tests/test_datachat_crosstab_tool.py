"""crosstab: a descriptive two-dimensional count/distribution matrix."""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.crosstab_tool import CrosstabTool
from infrastructure import datachat_export_store as store
from tests.test_datachat_result_export import _read_csv, _token
from tests.test_datachat_result_preview import _ToolEngine, _chat, _meta
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)


def _run(df, **kwargs):
    return CrosstabTool(df).forward(**kwargs)


def _by_row(out, row_col):
    return {r[row_col]: {k: v for k, v in r.items() if k != row_col} for r in out["data"]}


@pytest.fixture
def survey():
    return pd.DataFrame(
        {
            "course": ["A", "A", "A", "B", "B", "C", "C", "C"],
            "answer": ["yes", "no", "yes", "yes", "yes", "no", "no", "yes"],
        }
    )


# --- counts -----------------------------------------------------------------


def test_absolute_counts_in_wide_form(survey):
    out = _run(survey, rows="course", columns="answer")

    assert out == {
        "kind": "table",
        "data": [
            {"course": "A", "no": 1, "yes": 2},
            {"course": "B", "no": 0, "yes": 2},
            {"course": "C", "no": 2, "yes": 1},
        ],
    }
    assert all(isinstance(v, int) for r in out["data"] for k, v in r.items() if k != "course")


def test_rows_and_columns_are_not_interchangeable(survey):
    out = _run(survey, rows="answer", columns="course")

    assert out["data"] == [
        {"answer": "no", "A": 1, "B": 0, "C": 2},
        {"answer": "yes", "A": 2, "B": 2, "C": 1},
    ]


def test_no_note_meta_or_export_name(survey):
    out = _run(survey, rows="course", columns="answer", normalize="rows")
    assert set(out) == {"kind", "data"}


def test_data_records_are_used_instead_of_the_session_dataset(survey):
    records = [{"g": "x", "h": 1}, {"g": "x", "h": 2}, {"g": "y", "h": 1}]
    out = _run(survey, rows="g", columns="h", data=records)
    assert out["data"] == [{"g": "x", "1": 1, "2": 1}, {"g": "y", "1": 1, "2": 0}]


def test_empty_data_is_an_empty_table(survey):
    assert _run(survey, rows="g", columns="h", data=[]) == {"kind": "table", "data": []}


# --- normalisation ------------------------------------------------------------


def test_normalize_none_is_the_default(survey):
    assert _run(survey, rows="course", columns="answer", normalize="none") == _run(
        survey, rows="course", columns="answer"
    )


def test_normalize_rows(survey):
    got = _by_row(_run(survey, rows="course", columns="answer", normalize="rows"), "course")
    assert got["A"] == pytest.approx({"no": 1 / 3, "yes": 2 / 3})
    assert got["B"] == {"no": 0.0, "yes": 1.0}
    for cells in got.values():
        assert sum(cells.values()) == pytest.approx(1.0)


def test_normalize_columns(survey):
    got = _by_row(_run(survey, rows="course", columns="answer", normalize="columns"), "course")
    assert got["A"] == pytest.approx({"no": 1 / 3, "yes": 2 / 5})
    assert sum(r["yes"] for r in got.values()) == pytest.approx(1.0)
    assert sum(r["no"] for r in got.values()) == pytest.approx(1.0)


def test_normalize_all(survey):
    out = _run(survey, rows="course", columns="answer", normalize="all")
    values = [v for r in out["data"] for k, v in r.items() if k != "course"]
    assert sum(values) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in values)
    assert _by_row(out, "course")["C"]["no"] == pytest.approx(2 / 8)


def test_normalize_is_case_insensitive(survey):
    assert _run(survey, rows="course", columns="answer", normalize=" Rows ") == _run(
        survey, rows="course", columns="answer", normalize="rows"
    )


def test_missing_observations_stay_in_the_denominator():
    df = pd.DataFrame({"g": ["x"] * 10, "answer": ["yes"] * 7 + ["no"] * 2 + [None]})
    out = _run(df, rows="g", columns="answer", normalize="rows")
    assert out["data"] == [{"g": "x", "no": 0.2, "yes": 0.7, "(empty)": 0.1}]


# --- missing values -----------------------------------------------------------


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_real_missing_values_are_one_explicit_category(missing):
    df = pd.DataFrame({"g": ["x", "x", "x", "y"], "v": pd.Series(["a", missing, missing, "a"], dtype=object)})
    out = _run(df, rows="g", columns="v")
    assert out["data"] == [{"g": "x", "a": 1, "(empty)": 2}, {"g": "y", "a": 1, "(empty)": 0}]


def test_nat_is_the_missing_category():
    df = pd.DataFrame({"g": ["x", "x"], "d": pd.to_datetime(["2024-01-01", None])})
    out = _run(df, rows="d", columns="g")
    assert out["data"] == [{"d": "2024-01-01T00:00:00", "x": 1}, {"d": "(empty)", "x": 1}]


def test_null_like_strings_are_ordinary_categories():
    df = pd.DataFrame({"g": ["x"] * 5, "v": ["None", "null", "nan", "(empty)", "NaN"]})
    out = _run(df, rows="g", columns="v")
    assert out["data"] == [{"g": "x", "(empty)": 1, "NaN": 1, "None": 1, "nan": 1, "null": 1}]


# --- observed categories ----------------------------------------------------


def test_unused_categorical_levels_do_not_appear():
    df = pd.DataFrame(
        {
            "g": pd.Categorical(["a", "b", "a"], categories=["a", "b", "never"]),
            "h": pd.Categorical(["p", "p", "q"], categories=["p", "q", "unused"]),
        }
    )
    out = _run(df, rows="g", columns="h")
    assert out["data"] == [{"g": "a", "p": 1, "q": 1}, {"g": "b", "p": 1, "q": 0}]


# --- ordering -----------------------------------------------------------------


def test_numeric_categories_sort_numerically():
    df = pd.DataFrame({"n": [10, 2, 1, 10], "m": [10, 1, 2, 2]})
    out = _run(df, rows="n", columns="m")
    assert [r["n"] for r in out["data"]] == [1, 2, 10]
    assert [k for k in out["data"][0] if k != "n"] == ["1", "2", "10"]


def test_datetime_categories_sort_chronologically():
    df = pd.DataFrame(
        {"d": pd.to_datetime(["2024-10-01", "2024-02-01", "2023-12-31"]), "g": ["x", "x", "x"]}
    )
    out = _run(df, rows="d", columns="g")
    assert [r["d"][:10] for r in out["data"]] == ["2023-12-31", "2024-02-01", "2024-10-01"]


def test_strings_sort_deterministically_and_missing_is_last():
    df = pd.DataFrame({"s": ["b", None, "a", "c"], "g": ["x"] * 4})
    out = _run(df, rows="s", columns="g")
    assert [r["s"] for r in out["data"]] == ["a", "b", "c", "(empty)"]
    shuffled = _run(df.iloc[::-1], rows="s", columns="g")
    assert shuffled == out


def test_booleans_sort_false_first():
    df = pd.DataFrame({"b": [True, False, True], "g": ["x"] * 3})
    out = _run(df, rows="b", columns="g")
    assert out["data"] == [{"b": False, "x": 1}, {"b": True, "x": 2}]


# --- identity preservation -----------------------------------------------------


def test_number_and_string_column_categories_stay_distinct():
    df = pd.DataFrame({"g": ["x", "x", "x"], "v": pd.Series([1, "1", "1"], dtype=object)})
    out = _run(df, rows="g", columns="v")
    assert out["data"] == [{"g": "x", "1 [number]": 1, "1 [string]": 2}]


def test_missing_and_literal_empty_column_categories_stay_distinct():
    df = pd.DataFrame({"g": ["x", "x", "x"], "v": [None, "(empty)", "(empty)"]})
    out = _run(df, rows="g", columns="v")
    assert out["data"] == [{"g": "x", "(empty) [string]": 2, "(empty) [missing]": 1}]


def test_column_category_never_overwrites_the_rows_field():
    df = pd.DataFrame({"status": ["open", "closed"], "v": ["status", "other"]})
    out = _run(df, rows="status", columns="v")
    assert out["data"] == [
        {"status": "closed", "other": 1, "status [string]": 0},
        {"status": "open", "other": 0, "status [string]": 1},
    ]


def test_missing_and_literal_empty_row_categories_stay_distinct():
    df = pd.DataFrame({"v": [None, "(empty)", "(empty)"], "g": ["x", "x", "x"]})
    out = _run(df, rows="v", columns="g")
    assert out["data"] == [{"v": "(empty) [string]", "x": 2}, {"v": "(empty) [missing]", "x": 1}]


def test_number_and_string_row_categories_stay_distinct_others_untouched():
    df = pd.DataFrame({"v": pd.Series([1, "1", 2, "a"], dtype=object), "g": ["x"] * 4})
    out = _run(df, rows="v", columns="g")
    assert [r["v"] for r in out["data"]] == ["1 [number]", 2, "1 [string]", "a"]


def test_boolean_and_number_are_not_merged():
    df = pd.DataFrame({"v": pd.Series([True, 1, 1], dtype=object), "g": ["x"] * 3})
    out = _run(df, rows="g", columns="v")
    assert out["data"] == [{"g": "x", "true": 1, "1": 2}]


def test_qualified_label_does_not_collide_with_a_literal_value():
    df = pd.DataFrame({"g": ["x"] * 3, "v": pd.Series([1, "1", "1 [string]"], dtype=object)})
    out = _run(df, rows="g", columns="v")
    row = out["data"][0]
    assert len(row) == 4
    assert sum(v for k, v in row.items() if k != "g") == 3


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize("rows, columns", [("", "answer"), ("course", ""), (None, "answer"), ("course", None)])
def test_missing_dimension(survey, rows, columns):
    assert _run(survey, rows=rows, columns=columns)["code"] == "MISSING_DIMENSION"


def test_unknown_dimension(survey):
    out = _run(survey, rows="course", columns="nope")
    assert out["kind"] == "error" and out["code"] == "INVALID_COLUMN"
    assert "nope" in out["message"]


def test_same_dimension(survey):
    assert _run(survey, rows="course", columns="course")["code"] == "SAME_DIMENSION"


@pytest.mark.parametrize("normalize", ["index", "percent", "", 1, True])
def test_invalid_normalize(survey, normalize):
    assert _run(survey, rows="course", columns="answer", normalize=normalize)["code"] == "INVALID_NORMALIZE"


def test_no_metric_or_op_parameters():
    assert set(CrosstabTool.inputs) == {"rows", "columns", "normalize", "data"}
    with pytest.raises(TypeError):
        CrosstabTool(pd.DataFrame()).forward(rows="a", columns="b", op="mean")


# --- no hidden cap -------------------------------------------------------------


def test_every_category_is_returned():
    df = pd.DataFrame({"r": [i % 300 for i in range(3000)], "c": [i % 40 for i in range(3000)]})
    out = _run(df, rows="r", columns="c")
    assert len(out["data"]) == 300
    assert all(len(r) == 41 for r in out["data"])
    assert sum(v for r in out["data"] for k, v in r.items() if k != "r") == 3000


# --- through POST /datachat ------------------------------------------------------


def test_wide_crosstab_is_previewed_and_exported_in_full(monkeypatch):
    df = pd.DataFrame({"r": [i % 80 for i in range(800)], "c": [i % 16 for i in range(800)]})
    full = _run(df, rows="r", columns="c")
    body = _chat(monkeypatch, _ToolEngine(lambda: _run(df, rows="r", columns="c")))

    assert _meta(body) == {
        "result_rows": 80,
        "result_columns": 17,
        "preview_rows": 50,
        "preview_columns": 10,
        "truncated": True,
    }
    rows = _read_csv(store.resolve_export(_token(body)).path)
    assert rows[0] == ["r"] + [str(j) for j in range(16)]
    assert len(rows) == 81
    assert rows[1:] == [[str(r[k]) for k in rows[0]] for r in full["data"]]
