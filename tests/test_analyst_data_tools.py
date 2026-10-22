"""Tests for the analyst's DataFrame tools and output tools."""

import json

import numpy as np
import pandas as pd
import pytest

from tests.analyst_helpers import make_state
from analyst.tools.detect_anomalies_tool import DetectAnomaliesTool
from analyst.tools.run_forecast_tool import RunForecastTool
from analyst.tools.run_regression_tool import RunRegressionTool
from analyst.tools.save_chart_tool import SaveChartTool
from analyst.tools.save_sub_task_output_tool import SaveSubTaskOutputTool
from analyst.tools.save_table_tool import SaveTableTool


@pytest.fixture
def datasets():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 10, 200)
    frame = pd.DataFrame({"x": x, "y": 3 * x + 5 + rng.normal(0, 1, 200), "label": ["a"] * 200})
    frame.loc[0, ["x", "y"]] = [1000, -500]
    return {"points": frame}


@pytest.fixture
def state(tmp_path):
    brief = {
        "sub_tasks": [
            {"id": "prep", "name": "Prep", "order": 1, "depends_on": [], "mission": "m",
             "outputs": {"required": ["total", "series"]}},
        ]
    }
    return make_state(tmp_path, brief)


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["isolation_forest", "zscore", "iqr"])
def test_detect_anomalies_finds_the_planted_outlier(datasets, method):
    result = DetectAnomaliesTool(datasets).forward("points", "x, y", method)

    assert result["kind"] == "anomalies"
    assert 0 in result["anomaly_indices"]
    assert result["total_records"] == 200
    assert len(result["sample"]) <= 5
    assert set(result["anomaly_means"]) == {"x", "y"}


def test_detect_anomalies_reports_bad_arguments(datasets):
    tool = DetectAnomaliesTool(datasets)
    assert tool.forward("nope", "x")["code"] == "UNKNOWN_DATASET"
    assert tool.forward("points", "label")["code"] == "INVALID_COLUMN"
    assert tool.forward("points", "x", "magic")["code"] == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# run_regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "ridge", "lasso"])
def test_linear_models_recover_the_slope(datasets, method):
    datasets["points"] = datasets["points"].drop(index=0)
    result = RunRegressionTool(datasets).forward("points", "y", "x", method)

    assert result["r2"] > 0.95
    assert result["coefficients"]["x"] == pytest.approx(3, abs=0.1)
    assert "intercept" in result


def test_random_forest_reports_importances(datasets):
    result = RunRegressionTool(datasets).forward("points", "y", "x", "random_forest")
    assert result["feature_importance"] == {"x": 1.0}


def test_regression_needs_a_feature_other_than_the_target(datasets):
    assert RunRegressionTool(datasets).forward("points", "y", "y")["code"] == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# run_forecast
# ---------------------------------------------------------------------------


def test_forecast_on_text_months_infers_dates():
    frame = pd.DataFrame({"mese": ["202601", "202602", "202603", "202604", "202605"],
                          "viaggi": [100.0, 110.0, 120.0, 130.0, 140.0]})
    result = RunForecastTool({"m": frame}).forward("m", "mese", "viaggi", 2)

    assert result["trend"] == "up"
    assert result["slope_per_period"] == 10.0
    assert result["forecast"] == [{"period": "2026-06-01", "value": 150.0}, {"period": "2026-07-01", "value": 160.0}]
    assert result["mape_percent"] == 0.0
    assert result["first_period"] == "2026-01-01"


def test_forecast_sums_per_period_and_needs_four_periods():
    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
    assert RunForecastTool({"d": frame}).forward("d", "g", "v")["code"] == "INVALID_ARGUMENT"


# ---------------------------------------------------------------------------
# Output tools
# ---------------------------------------------------------------------------


def save_tool(state, max_chars=1000, max_rows=1000):
    return SaveSubTaskOutputTool(state.tasks["prep"], state.datasets, max_chars=max_chars, max_rows=max_rows)


def test_outputs_are_converted_and_report_what_is_missing(state):
    task = state.tasks["prep"]
    tool = save_tool(state)

    result = tool.forward("total", np.float64(12.5))
    assert "still missing: series" in result["text"]
    assert task.outputs["total"] == 12.5

    result = tool.forward("series", pd.Series([1, 2], index=["2026-01", "2026-02"]))
    assert "All required outputs are saved" in result["text"]
    assert task.outputs["series"] == {"2026-01": 1, "2026-02": 2}


@pytest.mark.parametrize("value", [list(range(1000)), "x" * 2000, {"k": "x" * 2000}])
def test_oversized_outputs_that_are_not_tables_are_refused(state, value):
    task = state.tasks["prep"]
    result = save_tool(state).forward("total", value)
    assert result["code"] == "OUTPUT_TOO_LARGE"
    assert "total" not in task.outputs
    assert not state.datasets


def test_invalid_keys_are_refused(state):
    assert save_tool(state).forward("Bad", 1)["code"] == "INVALID_KEY"


def test_a_small_list_of_records_is_saved_as_it_is(state):
    # Not through a DataFrame: no key filled in, no int turned into a float.
    records = [{"client": "a", "trips": 3}, {"client": "b"}]

    save_tool(state).forward("series", records)

    assert state.tasks["prep"].outputs["series"] == records
    assert not state.datasets


def test_a_small_table_is_saved_whole(state):
    save_tool(state).forward("series", pd.DataFrame({"a": [1, 2]}))

    assert state.tasks["prep"].outputs["series"] == [{"a": 1}, {"a": 2}]
    assert not state.datasets


@pytest.mark.parametrize("as_records", [False, True])
def test_an_oversized_table_becomes_a_dataset_with_its_head_saved(state, as_records):
    frame = pd.DataFrame({"client": [f"c{i}" for i in range(300)], "margin": np.arange(300.0)})
    value = frame.to_dict(orient="records") if as_records else frame

    result = save_tool(state).forward("series", value)

    saved = state.tasks["prep"].outputs["series"]
    assert saved["dataset"] == "prep.series"
    assert saved["rows"] == 300
    assert saved["columns"] == ["client", "margin"]
    assert 0 < len(saved["head"]) < 300
    assert saved["head"][0] == {"client": "c0", "margin": 0.0}
    assert len(json.dumps(saved)) <= 1000
    pd.testing.assert_frame_equal(state.datasets["prep.series"], frame)
    assert "datasets['prep.series']" in result["text"]


def test_a_table_above_the_row_limit_is_refused_unconverted(state, monkeypatch):
    converted = []
    monkeypatch.setattr(
        "analyst.tools.save_sub_task_output_tool.to_native", lambda v: converted.append(v) or v
    )
    frame = pd.DataFrame({"v": np.zeros(5000)})

    result = save_tool(state, max_rows=1000).forward("series", frame)

    assert result["code"] == "OUTPUT_TOO_LARGE"
    assert "5000 rows" in result["message"]
    assert converted == [] and not state.datasets


def test_only_as_many_rows_as_could_fit_are_converted(state, monkeypatch):
    from analyst.values import to_native

    sizes = []
    monkeypatch.setattr(
        "analyst.tools.save_sub_task_output_tool.to_native", lambda v: sizes.append(len(v)) or to_native(v)
    )

    save_tool(state, max_rows=100000).forward("series", pd.DataFrame({"v": np.arange(50000)}))

    assert sizes == [500]
    assert state.datasets["prep.series"].shape == (50000, 1)


def chart(points=3):
    return {"type": "bar", "data": {"labels": list(range(points)), "datasets": [{"label": "v", "data": list(range(points))}]}}


def test_charts_are_validated_and_bound_to_the_task(state):
    tool = SaveChartTool(state, "prep", max_charts=2, max_points=10)

    assert tool.forward("volumes", "Volumes", chart())["kind"] == "text"
    assert state.charts["volumes"]["task_id"] == "prep"
    assert tool.forward("big", "Big", chart(11))["code"] == "INVALID_CHART"
    assert tool.forward("x", "X", {"type": "sankey", "data": {}})["code"] == "INVALID_CHART"
    assert tool.forward("x", "X", {"type": "bar", "data": {"datasets": []}})["code"] == "INVALID_CHART"


def test_the_chart_budget_is_enforced_but_replacing_is_allowed(state):
    tool = SaveChartTool(state, "prep", max_charts=1, max_points=10)
    tool.forward("a", "A", chart())
    assert tool.forward("b", "B", chart())["code"] == "CHARTS_FULL"
    assert tool.forward("a", "A again", chart())["kind"] == "text"


def test_tables_accept_dataframes_and_cap_their_size(state):
    tool = SaveTableTool(state, "prep")
    frame = pd.DataFrame({"Cliente": ["CCNO", "METRO"], "Margine": [1104092.0, 652116.0]})

    assert tool.forward("top_clients", "Top clients", frame)["kind"] == "text"
    assert state.tables["top_clients"]["columns"] == ["Cliente", "Margine"]
    assert state.tables["top_clients"]["rows"][0] == ["CCNO", 1104092.0]

    too_long = pd.DataFrame({"v": range(51)})
    assert tool.forward("t", "T", too_long)["code"] == "TABLE_TOO_LARGE"
    assert tool.forward("t", "T", [])["code"] == "INVALID_TABLE"


# ---------------------------------------------------------------------------
# record_insight
# ---------------------------------------------------------------------------


def test_insights_need_evidence_and_are_capped(state):
    from analyst.tools.record_insight_tool import MAX_INSIGHTS_PER_TASK, RecordInsightTool

    seen = []
    tool = RecordInsightTool(state, "prep", max_insights=10, notify=seen.append)

    assert tool.forward("CCNO alone makes half the margin.", {"CCNO": np.float64(0.51)})["kind"] == "text"
    assert state.insights == [
        {"task_id": "prep", "text": "CCNO alone makes half the margin.", "evidence": '{"CCNO": 0.51}'}
    ]
    assert seen == ["CCNO alone makes half the margin."]

    assert tool.forward("A guess.", None)["code"] == "NO_EVIDENCE"
    assert tool.forward("  ", 1)["code"] == "EMPTY_INSIGHT"

    for i in range(MAX_INSIGHTS_PER_TASK - 1):
        tool.forward(f"Fact {i}.", i + 1)
    assert tool.forward("One too many.", 1)["code"] == "INSIGHTS_FULL"


def test_the_run_wide_insight_limit_applies_across_tasks(state):
    from analyst.tools.record_insight_tool import RecordInsightTool

    RecordInsightTool(state, "prep", max_insights=1).forward("First.", 1)
    assert RecordInsightTool(state, "other", max_insights=1).forward("Second.", 2)["code"] == "INSIGHTS_FULL"
