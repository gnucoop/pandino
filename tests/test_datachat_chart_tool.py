"""Structured charts (S4 v1): the chart tool, its request-scoped registry and
the additive ``charts`` field of the /datachat response.

Chart data comes only from specs the tool recorded in the same request; the
type/value envelope is unchanged and there is no ``type: "chart"``.
"""

import copy
import dataclasses
import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest
from flask import Flask

from datachat.chart_registry import get_recorded_charts, record_chart
from datachat.smolagents_engine import SmolagentsEngine
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.chart_tool import ChartTool
from datachat.tools.crosstab_tool import CrosstabTool
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    _make_app,
    _patch_success_dependencies,
    _post_chat,
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)

_app = Flask(__name__)


def _run(df, **kwargs):
    """Call the tool inside a request; return its result and the recorded specs."""
    with _app.test_request_context("/datachat"):
        result = ChartTool(df).forward(**kwargs)
        return result, [c.spec for c in get_recorded_charts()]


def _spec(df, **kwargs):
    result, specs = _run(df, **kwargs)
    assert result["kind"] == "text", result
    assert len(specs) == 1
    return specs[0]


def _error(df, **kwargs):
    result, specs = _run(df, **kwargs)
    assert result["kind"] == "error"
    assert specs == []
    return result["code"]


# --- registry -----------------------------------------------------------------


def test_successful_chart_records_one_backend_spec():
    df = pd.DataFrame({"dept": ["a", "b", "a"]})
    spec = _spec(df, kind="bar", x="dept")
    assert spec == {
        "type": "bar",
        "labels": ["a", "b"],
        "datasets": [{"label": "count", "data": [2, 1]}],
        "title": None,
        "x_label": "dept",
        "y_label": "count",
        "stacked": False,
        "horizontal": False,
    }


def test_charts_keep_production_order_without_dedup_and_failures_add_nothing():
    df = pd.DataFrame({"dept": ["a", "b"], "n": [1, 2]})
    with _app.test_request_context("/datachat"):
        tool = ChartTool(df)
        assert tool.forward(kind="bar", x="dept", title="first")["kind"] == "text"
        assert tool.forward(kind="bar", x="dept", title="first")["kind"] == "text"
        assert tool.forward(kind="pie", x="nope")["kind"] == "error"
        titles = [c.spec["title"] for c in get_recorded_charts()]
    assert titles == ["first", "first"]


def test_registry_is_isolated_across_requests():
    with _app.test_request_context("/datachat"):
        record_chart({"type": "bar"})
        assert len(get_recorded_charts()) == 1
    with _app.test_request_context("/datachat"):
        assert get_recorded_charts() == []


def test_registry_is_inert_outside_an_app_context():
    record_chart({"type": "bar"})
    assert get_recorded_charts() == []
    result = ChartTool(pd.DataFrame({"a": [1]})).forward(kind="bar", x="a")
    assert result["kind"] == "text"


def test_engine_holds_no_chart_state():
    names = {f.name for f in dataclasses.fields(SmolagentsEngine)}
    assert not any("chart" in n for n in names)


# --- analytical boundary ---------------------------------------------------------


def test_chart_exposes_no_analytical_parameters():
    params = set(inspect.signature(ChartTool.forward).parameters) - {"self"}
    assert params == {"kind", "x", "y", "series_by", "data", "title", "horizontal"}
    assert set(ChartTool.inputs) == params


def test_hist_kde_box_and_area_are_not_chart_kinds():
    df = pd.DataFrame({"a": [1, 2]})
    for kind in ("hist", "kde", "box", "hexbin", "area"):
        assert _error(df, kind=kind, x="a") == "INVALID_KIND"


def test_metric_without_data_is_refused_rather_than_aggregated():
    df = pd.DataFrame({"dept": ["a", "b"], "score": [1, 2], "g": ["x", "y"]})
    assert _error(df, kind="bar", x="dept", y="score") == "DATA_REQUIRED"
    assert _error(df, kind="bar", x="dept", series_by="g") == "DATA_REQUIRED"


def test_no_source_is_a_structured_invalid_data_error():
    assert _error(None, kind="bar", x="a") == "INVALID_DATA"
    assert _error(pd.DataFrame({"a": [1]}), kind="bar", x="a", data="rows") == "INVALID_DATA"


# --- direct count -------------------------------------------------------------


def test_direct_count_uses_the_complete_distribution_without_cap():
    values = [f"v{i:03d}" for i in range(300)]
    spec = _spec(pd.DataFrame({"c": values}), kind="bar", x="c")
    assert spec["labels"] == values
    assert sum(spec["datasets"][0]["data"]) == 300


@pytest.mark.parametrize("kind", ["pie", "doughnut"])
def test_pie_and_doughnut_count_every_category(kind):
    df = pd.DataFrame({"c": list("aaabbc") + [f"z{i}" for i in range(40)]})
    spec = _spec(df, kind=kind, x="c")
    assert len(spec["labels"]) == 43
    assert "Other" not in spec["labels"]
    assert sum(spec["datasets"][0]["data"]) == 46


def test_real_missing_is_a_category_and_missing_like_strings_are_data():
    df = pd.DataFrame({"c": ["None", "null", "nan", "", None, np.nan, "b"]})
    spec = _spec(df, kind="bar", x="c")
    assert spec["labels"] == ["", "None", "b", "nan", "null", "(empty)"]
    assert spec["datasets"][0]["data"] == [1, 1, 1, 1, 1, 2]


def test_nat_and_pd_na_are_missing():
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-02", None, "2024-01-01"])})
    spec = _spec(df, kind="bar", x="d")
    assert spec["labels"] == ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "(empty)"]
    ints = pd.DataFrame({"n": pd.array([2, None, 10, 2], dtype="Int64")})
    assert _spec(ints, kind="bar", x="n")["labels"] == [2, 10, "(empty)"]


def test_colliding_labels_are_qualified_only_on_collision():
    df = pd.DataFrame({"c": pd.Series(["(empty)", None, 1, "1", "x"], dtype=object)})
    spec = _spec(df, kind="bar", x="c")
    assert spec["labels"] == ["1 [number]", "(empty) [string]", "1 [string]", "x", "(empty) [missing]"]


def test_numeric_categories_are_ordered_numerically():
    spec = _spec(pd.DataFrame({"n": [10, 2, 1, 2]}), kind="bar", x="n")
    assert spec["labels"] == [1, 2, 10]
    assert spec["datasets"][0]["data"] == [1, 2, 1]


def test_direct_count_matches_aggregate_count():
    df = pd.DataFrame({"c": ["a", "b", None, "a", ""]})
    spec = _spec(df, kind="bar", x="c")
    agg = AggregateTool(df).forward(group_by="c", op="count")["data"]
    by_label = {"(empty)" if r["c"] is None else r["c"]: r["count"] for r in agg}
    assert dict(zip(spec["labels"], spec["datasets"][0]["data"])) == by_label


def test_direct_count_keeps_bool_and_number_apart_where_aggregate_merges_them():
    # Lossless category identity, as in crosstab; 1 and 1.0 are one number.
    df = pd.DataFrame({"c": pd.Series([True, 1, 1.0, "1"], dtype=object)})
    spec = _spec(df, kind="bar", x="c")
    assert spec["labels"] == [True, "1 [number]", "1 [string]"]
    assert spec["datasets"][0]["data"] == [1, 2, 1]
    assert sum(spec["datasets"][0]["data"]) == len(df)


def test_categorical_direct_count_is_observed_only_and_keeps_missing():
    df = pd.DataFrame({"c": pd.Categorical(["a", None, "a"], categories=["a", "b"])})
    spec = _spec(df, kind="bar", x="c")
    assert spec["labels"] == ["a", "(empty)"]
    assert spec["datasets"][0]["data"] == [2, 1]


def test_horizontal_is_an_explicit_bar_hint():
    df = pd.DataFrame({"c": ["a"]})
    assert _spec(df, kind="bar", x="c", horizontal=True)["horizontal"] is True
    assert _error(df, kind="line", x="c", horizontal=True) == "INVALID_OPTION"


# --- scatter ------------------------------------------------------------------


def test_scatter_extracts_raw_points_without_aggregation():
    df = pd.DataFrame({"age": [20, 20, 35], "score": [3.5, 4.0, 4.1]})
    result, specs = _run(df, kind="scatter", x="age", y="score")
    assert "skipped" not in result["text"]
    assert specs[0]["labels"] is None
    assert specs[0]["datasets"] == [
        {"label": "age / score", "data": [{"x": 20, "y": 3.5}, {"x": 20, "y": 4.0}, {"x": 35, "y": 4.1}]}
    ]


def test_scatter_skips_unusable_rows_explicitly_and_never_emits_non_finite():
    df = pd.DataFrame(
        {
            "x": pd.Series([1, None, "2.5", "abc", 3, 4, True], dtype=object),
            "y": pd.Series([1.0, 2.0, 3.0, 4.0, math.inf, np.nan, 5.0], dtype=object),
        }
    )
    result, specs = _run(df, kind="scatter", x="x", y="y")
    assert "5 row(s) were skipped" in result["text"]
    assert specs[0]["datasets"][0]["data"] == [{"x": 1, "y": 1.0}, {"x": 2.5, "y": 3.0}]
    json.dumps(specs[0], allow_nan=False)


def test_scatter_needs_y_and_usable_points():
    df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
    assert _error(df, kind="scatter", x="x") == "MISSING_Y"
    assert _error(df, kind="scatter", x="x", y="y") == "NO_POINTS"


def test_scatter_series_keep_missing_series_as_its_own_dataset():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "g": ["b", None, "a"]})
    spec = _spec(df, kind="scatter", x="x", y="y", series_by="g")
    assert [d["label"] for d in spec["datasets"]] == ["a", "b", "(empty)"]
    assert spec["datasets"][2]["data"] == [{"x": 2, "y": 5}]


def test_long_form_scatter_from_records():
    data = [{"x": 1, "y": 2, "g": "a"}, {"x": 3, "y": 4, "g": "b"}]
    spec = _spec(None, kind="scatter", x="x", y="y", series_by="g", data=data)
    assert [d["data"] for d in spec["datasets"]] == [[{"x": 1, "y": 2}], [{"x": 3, "y": 4}]]


# --- precomputed records ------------------------------------------------------


def test_records_render_y_as_supplied_in_upstream_order():
    data = [{"dept": "B", "mean_score": 4.1}, {"dept": "A", "mean_score": 3.7}, {"dept": "C", "mean_score": None}]
    spec = _spec(pd.DataFrame({"other": [1]}), kind="bar", x="dept", y="mean_score", data=data)
    assert spec["labels"] == ["B", "A", "C"]
    assert spec["datasets"] == [{"label": "mean_score", "data": [4.1, 3.7, None]}]
    assert spec["y_label"] == "mean_score"


def test_records_are_never_aggregated():
    data = [{"d": "a", "v": 1}, {"d": "a", "v": 2}]
    assert _error(None, kind="bar", x="d", y="v", data=data) == "DUPLICATE_POINT"


def test_records_refuse_non_numeric_and_non_finite_values():
    assert _error(None, kind="bar", x="d", y="v", data=[{"d": "a", "v": "3"}]) == "NON_NUMERIC_VALUE"
    assert _error(None, kind="bar", x="d", y="v", data=[{"d": "a", "v": math.inf}]) == "NON_FINITE_VALUE"


def test_records_need_y():
    assert _error(None, kind="bar", x="d", data=[{"d": "a", "v": 1}]) == "MISSING_Y"


def test_pie_records_are_rendered_exactly_as_supplied():
    data = [{"c": "a", "count": 5}, {"c": "b", "count": 3}]
    spec = _spec(None, kind="pie", x="c", y="count", data=data)
    assert spec["labels"] == ["a", "b"]
    assert spec["datasets"] == [{"label": "count", "data": [5, 3]}]


def test_wide_tables_and_crosstab_output_are_not_auto_mapped():
    df = pd.DataFrame({"r": ["a", "b", "a"], "c": ["x", "y", "y"]})
    wide = CrosstabTool(df).forward(rows="r", columns="c")["data"]
    # Without y, nothing picks "the other columns" as datasets.
    assert _error(None, kind="bar", x="r", data=wide) == "MISSING_Y"
    spec = _spec(None, kind="bar", x="r", y="x", data=wide)
    assert len(spec["datasets"]) == 1


def test_line_orders_numeric_and_iso_date_axes_but_keeps_other_orders():
    nums = [{"x": 10, "v": 1}, {"x": 2, "v": 2}, {"x": None, "v": 3}, {"x": 1, "v": 4}]
    spec = _spec(None, kind="line", x="x", y="v", data=nums)
    assert spec["labels"] == [1, 2, 10, "(empty)"]
    assert spec["datasets"][0]["data"] == [4, 2, 1, 3]

    dates = [{"d": "2024-03-01", "v": 1}, {"d": "2024-01-01", "v": 2}]
    assert _spec(None, kind="line", x="d", y="v", data=dates)["labels"] == ["2024-01-01", "2024-03-01"]

    words = [{"w": "b", "v": 1}, {"w": "a", "v": 2}]
    assert _spec(None, kind="line", x="w", y="v", data=words)["labels"] == ["b", "a"]
    # Bars keep an upstream ranking even on a numeric axis.
    assert _spec(None, kind="bar", x="x", y="v", data=nums)["labels"] == [10, 2, "(empty)", 1]


# --- multi-series -------------------------------------------------------------


_LONG = [
    {"dept": "A", "sex": "f", "count": 3},
    {"dept": "A", "sex": "m", "count": 2},
    {"dept": "B", "sex": "m", "count": 4},
    {"dept": "C", "sex": None, "count": 1},
]


@pytest.mark.parametrize("kind", ["bar", "line"])
def test_long_form_series_share_one_labels_domain(kind):
    spec = _spec(None, kind=kind, x="dept", y="count", series_by="sex", data=_LONG)
    assert spec["labels"] == ["A", "B", "C"]
    assert spec["datasets"] == [
        {"label": "f", "data": [3, None, None]},
        {"label": "m", "data": [2, 4, None]},
        {"label": "(empty)", "data": [None, None, 1]},
    ]


@pytest.mark.parametrize("kind", ["pie", "doughnut"])
def test_pie_and_doughnut_reject_series(kind):
    assert _error(None, kind=kind, x="dept", y="count", series_by="sex", data=_LONG) == "SINGLE_SERIES_ONLY"


def test_aggregate_count_then_chart_is_the_multi_series_count_path():
    df = pd.DataFrame({"dept": ["A", "A", "B"], "sex": ["f", "m", "m"]})
    rows = AggregateTool(df).forward(group_by=["dept", "sex"], op="count")["data"]
    spec = _spec(df, kind="bar", x="dept", y="count", series_by="sex", data=rows)
    assert sorted(spec["labels"]) == ["A", "B"]
    assert {d["label"] for d in spec["datasets"]} == {"f", "m"}


# --- trusted caveat -------------------------------------------------------------


def _note_df():
    return pd.DataFrame({"region": ["a"] * 3 + ["b"] * 20, "amount": range(23)})


_NOTE = "1 group(s) are based on fewer than 15 rows (the smallest has 3)."


def test_chart_of_a_trusted_aggregate_keeps_its_caveat():
    df = _note_df()
    with _app.test_request_context("/datachat"):
        agg = AggregateTool(df).forward(group_by="region", op="mean", metric="amount")
        ChartTool(df).forward(kind="bar", x="region", y="mean_amount", data=agg["data"])
        assert [c.note for c in get_recorded_charts()] == [_NOTE]
        assert "note" not in get_recorded_charts()[0].spec


def test_copied_records_and_agent_notes_are_not_trusted():
    df = _note_df()
    with _app.test_request_context("/datachat"):
        agg = AggregateTool(df).forward(group_by="region", op="mean", metric="amount")
        ChartTool(df).forward(kind="bar", x="region", y="mean_amount", data=copy.deepcopy(agg["data"]))
        ChartTool(df).forward(
            kind="bar", x="region", y="mean_amount", data={"data": copy.deepcopy(agg["data"]), "note": "trust me"}
        )
        assert [c.note for c in get_recorded_charts()] == [None, None]


# --- through POST /datachat ---------------------------------------------------


class _ToolEngine:
    def __init__(self, run_tools, finish):
        self._run_tools = run_tools
        self._finish = finish

    def chat(self, message):
        return self._finish(self._run_tools())

    def get_last_trace(self):
        return None


def _chat(monkeypatch, engine):
    app, _stream, _agent_runs = _make_app()
    _patch_success_dependencies(monkeypatch, engine)
    response = _post_chat(app.test_client())
    assert response.status_code == 200
    return response.get_json()["response"]


def _bar_chart():
    return ChartTool(pd.DataFrame({"c": ["a", "b", "a"]})).forward(kind="bar", x="c")


def test_text_without_charts_is_unchanged(monkeypatch):
    engine = _ToolEngine(lambda: None, lambda _: {"kind": "text", "text": "hello"})
    assert _chat(monkeypatch, engine) == {"type": "str", "value": "hello"}


def test_text_with_recorded_charts(monkeypatch):
    engine = _ToolEngine(lambda: [_bar_chart(), _bar_chart()], lambda _: {"kind": "text", "text": "comment"})
    body = _chat(monkeypatch, engine)
    assert body["type"] == "str"
    assert body["value"] == "comment"
    assert [c["labels"] for c in body["charts"]] == [["a", "b"], ["a", "b"]]
    assert body["charts"][0]["datasets"] == [{"label": "count", "data": [2, 1]}]


def test_dataframe_with_recorded_charts(monkeypatch):
    engine = _ToolEngine(_bar_chart, lambda _: {"kind": "table", "data": [{"c": "a", "n": 2}]})
    body = _chat(monkeypatch, engine)
    assert body["type"] == "dataframe"
    assert body["value"] == [{"c": "a", "n": 2}]
    assert body["charts"][0]["type"] == "bar"


def test_agent_authored_charts_are_never_forwarded(monkeypatch):
    fake = [{"type": "bar", "labels": ["x"], "datasets": [{"label": "n", "data": [999]}]}]
    engine = _ToolEngine(lambda: None, lambda _: {"kind": "unknown", "charts": fake})
    assert "charts" not in _chat(monkeypatch, engine)

    engine = _ToolEngine(_bar_chart, lambda _: {"kind": "table", "data": {"charts": fake}})
    body = _chat(monkeypatch, engine)
    assert body["charts"][0]["datasets"][0]["data"] == [2, 1]


def test_chart_type_is_not_a_response_type(monkeypatch):
    engine = _ToolEngine(_bar_chart, lambda result: result)
    body = _chat(monkeypatch, engine)
    assert body["type"] == "str"
    assert len(body["charts"]) == 1


def test_trusted_caveat_survives_a_chart_answered_with_text(monkeypatch):
    df = _note_df()

    def tools():
        agg = AggregateTool(df).forward(group_by="region", op="mean", metric="amount")
        ChartTool(df).forward(kind="bar", x="region", y="mean_amount", data=agg["data"])
        return agg

    body = _chat(monkeypatch, _ToolEngine(tools, lambda _: {"kind": "text", "text": "see chart"}))
    assert body["note"] == _NOTE
    assert "note" not in body["charts"][0]

    # Answering with the same trusted table does not repeat the caveat.
    body = _chat(monkeypatch, _ToolEngine(tools, lambda agg: {"kind": "table", "data": agg["data"]}))
    assert body["note"] == _NOTE


def _mean_with_thin_group(size):
    return pd.DataFrame({"region": ["a"] * size + ["b"] * 20, "amount": range(size + 20)})


def test_note_stays_one_trusted_caveat_never_a_composition(monkeypatch):
    thin3, thin5 = _mean_with_thin_group(3), _mean_with_thin_group(5)
    note5 = "1 group(s) are based on fewer than 15 rows (the smallest has 5)."

    def chart_of(df):
        agg = AggregateTool(df).forward(group_by="region", op="mean", metric="amount")
        ChartTool(df).forward(kind="bar", x="region", y="mean_amount", data=agg["data"])
        return agg

    # Two charted tables with distinct caveats and a text answer: the first one only.
    def two_charts():
        chart_of(thin3)
        chart_of(thin5)

    body = _chat(monkeypatch, _ToolEngine(two_charts, lambda _: {"kind": "text", "text": "x"}))
    assert body["note"] == _NOTE
    assert len(body["charts"]) == 2

    # A primary result with its own trusted caveat keeps exactly that caveat.
    def primary_then_other_chart():
        chart_of(thin5)
        return AggregateTool(thin3).forward(group_by="region", op="mean", metric="amount")

    body = _chat(monkeypatch, _ToolEngine(primary_then_other_chart, lambda agg: agg))
    assert body["note"] == _NOTE
    assert note5 not in body["note"]


def test_caveat_of_a_copied_table_does_not_reach_the_response(monkeypatch):
    df = _note_df()

    def tools():
        agg = AggregateTool(df).forward(group_by="region", op="mean", metric="amount")
        ChartTool(df).forward(kind="bar", x="region", y="mean_amount", data=copy.deepcopy(agg["data"]))

    body = _chat(monkeypatch, _ToolEngine(tools, lambda _: {"kind": "text", "text": "x", "note": "agent"}))
    assert "note" not in body
    assert len(body["charts"]) == 1


def test_charts_do_not_leak_into_the_next_request(monkeypatch):
    assert _chat(monkeypatch, _ToolEngine(_bar_chart, lambda _: {"kind": "text", "text": "a"}))["charts"]
    body = _chat(monkeypatch, _ToolEngine(lambda: None, lambda _: {"kind": "text", "text": "b"}))
    assert body == {"type": "str", "value": "b"}


# --- registration and instructions ---------------------------------------------


def _bare_engine(data):
    engine = SmolagentsEngine.__new__(SmolagentsEngine)
    engine.user_name = "tester"
    engine.data = data
    engine._sql_ready = False
    engine._plots_dir = "/tmp/datachat_plots_test"
    return engine


def test_chart_is_registered_and_taught_only_with_a_dataframe(monkeypatch):
    monkeypatch.setattr(
        "datachat.smolagents_engine.load_prompt",
        lambda title, default_text="", **kwargs: default_text,
    )
    with_df = _bare_engine(pd.DataFrame({"a": [1]}))
    assert "chart" in [t.name for t in with_df._data_tools()]
    assert "CHARTS" in with_df._build_instructions(None)

    sql_only = _bare_engine(None)
    assert sql_only._data_tools() == []
    assert "CHARTS" not in sql_only._build_instructions(None)
