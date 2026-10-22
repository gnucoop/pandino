"""Tests for the analyst's metrics, report rendering and report-agent tools."""

import json

import pytest

from analyst.metrics import compute_delta, fill_metrics, find_previous_run
from analyst.state import COMPLETED, FAILED
from analyst.tools.get_all_saved_charts_tool import GetAllSavedChartsTool
from analyst.tools.get_delta_section_tool import GetDeltaSectionTool
from analyst.tools.get_html_template_tool import GetHtmlTemplateTool
from analyst.tools.get_sub_task_status_tool import GetSubTaskStatusTool
from analyst.tools.read_sub_task_state_tool import ReadSubTaskStateTool
from analyst.tools.save_html_report_tool import SaveHtmlReportTool
from analyst.tools.save_run_metrics_tool import SaveRunMetricsTool
from tests.analyst_helpers import make_state

METRICS_CONFIG = {
    "metrics": {
        "headline_margin": {"label": "Primo margine 2027", "type": "currency", "better": "higher"},
        "concentration_share": {"label": "Concentrazione", "type": "percent", "better": "lower"},
    },
    "categories": {
        "revenue_scenarios": {"label": "Scenari", "icon": "📈", "color": "info"},
    },
}


@pytest.fixture
def state(tmp_path):
    plan = {
        "title": "Forecast 2027",
        "language": "Italian",
        "audience": "Direction",
        "relations": [],
        "tasks": [
            {"id": "prep", "name": "Prep", "mission": "m", "depends_on": [], "outputs": ["revenue_scenarios"]},
            {"id": "final_summary", "name": "Summary", "mission": "m", "depends_on": ["prep"],
             "outputs": ["kpi_summary"]},
        ],
        "charts": [{"id": "c1", "description": "Bars"}],
        "report": {"sections": ["Sintesi"], "tone": "", "length": ""},
        "metrics_config": METRICS_CONFIG,
    }
    state = make_state(tmp_path, {"anything": "goes"}, plan=plan, run_id="run_2", brief_id=7)
    state.tasks["prep"].status = COMPLETED
    state.tasks["prep"].summary = "Scenarios computed."
    state.tasks["prep"].outputs = {
        "revenue_scenarios": [{"g": 0.05, "revenue": 40.6}, {"g": 0.1, "revenue": 42.6}],
        "concentration_share": 0.82,
    }
    state.tasks["final_summary"].status = COMPLETED
    state.tasks["final_summary"].outputs = {"kpi_summary": {"headline_margin": 8300000}, "long": "x" * 5000}
    state.charts["volumes"] = {
        "title": "Viaggi mensili",
        "spec": {"type": "bar", "data": {"labels": ["gen"], "datasets": [{"label": "v", "data": [7300]}]}},
        "task_id": "prep",
    }
    state.tables["top_clients"] = {
        "title": "Top clienti",
        "columns": ["Cliente", "Margine"],
        "rows": [["CCNO <b>", 1104092.0]],
        "task_id": "prep",
    }
    return state


def write_previous_run(state, metrics, *, run_id="run_1", status="completed", finished_at="2026-09-01T10:00:00"):
    folder = state.run_dir.parent / run_id
    folder.mkdir(parents=True)
    (folder / "run_metadata.json").write_text(json.dumps({
        "run_id": run_id, "brief_id": state.brief_id, "status": status,
        "finished_at": finished_at, "metrics": metrics,
    }))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_are_filled_from_outputs_one_level_deep(state):
    filled = fill_metrics(METRICS_CONFIG, state.merged_outputs())

    assert filled["metrics"] == {"headline_margin": 8300000, "concentration_share": 0.82}
    assert len(filled["categories"]["revenue_scenarios"]) == 2
    assert filled["missing"] == []


def test_overrides_fill_what_the_outputs_lack():
    filled = fill_metrics(METRICS_CONFIG, {}, {"headline_margin": {"value": 5}})
    assert filled["metrics"] == {"headline_margin": 5}
    assert filled["missing"] == ["concentration_share", "revenue_scenarios"]


def _reference(rows, head):
    # What save_sub_task_output saves for a table above the output cap.
    return {"dataset": "trips.route_margin", "rows": rows, "columns": ["tratta", "margine"], "head": head}


def test_a_category_saved_as_a_dataset_counts_the_table_rows():
    import pandas as pd

    frame = pd.DataFrame({"tratta": [f"r{i}" for i in range(4067)], "margine": range(4067)})
    outputs = {"revenue_scenarios": _reference(4067, [{"tratta": "r0", "margine": 0}])}

    filled = fill_metrics(METRICS_CONFIG, outputs, datasets={"trips.route_margin": frame})

    items = filled["categories"]["revenue_scenarios"]
    assert len(items) == 4067                     # not 4, the keys of the reference
    assert items[1] == {"tratta": "r1", "margine": 1}


def test_without_its_dataset_a_reference_falls_back_to_its_saved_rows():
    head = [{"tratta": "r0", "margine": 0}, {"tratta": "r1", "margine": 1}]
    filled = fill_metrics(METRICS_CONFIG, {"revenue_scenarios": _reference(4067, head)})
    assert filled["categories"]["revenue_scenarios"] == head


def test_an_ordinary_dict_category_is_unchanged():
    filled = fill_metrics(METRICS_CONFIG, {"revenue_scenarios": {"base": 1, "high": 2}}, datasets={})
    assert filled["categories"]["revenue_scenarios"] == [{"key": "base", "value": 1}, {"key": "high", "value": 2}]


def test_delta_marks_improvements_by_direction():
    current = {"metrics": {"headline_margin": 110, "concentration_share": 0.9},
               "categories": {"revenue_scenarios": [1, 2, 3]}}
    previous = {"metrics": {"headline_margin": 100, "concentration_share": 0.8},
                "categories": {"revenue_scenarios": [1, 2, 4]}}

    delta = compute_delta(METRICS_CONFIG, current, previous)

    assert delta["metrics"]["headline_margin"]["improved"] is True
    assert delta["metrics"]["headline_margin"]["percent"] == 10.0
    assert delta["metrics"]["concentration_share"]["improved"] is False
    assert delta["categories"]["revenue_scenarios"]["added"] == [3]
    assert delta["categories"]["revenue_scenarios"]["removed"] == [4]


def test_the_previous_run_is_the_latest_completed_one_of_the_brief(state):
    write_previous_run(state, {}, run_id="old", finished_at="2026-08-01")
    write_previous_run(state, {}, run_id="new", finished_at="2026-09-01")
    write_previous_run(state, {}, run_id="failed", status="failed", finished_at="2026-10-01")

    previous = find_previous_run(state.run_dir.parent, 7, "run_2")
    assert previous["run_id"] == "new"
    assert find_previous_run(state.run_dir.parent, 8, "run_2") is None


# ---------------------------------------------------------------------------
# Report-agent tools
# ---------------------------------------------------------------------------


def test_read_sub_task_state_previews_long_outputs(state):
    result = ReadSubTaskStateTool(state, preview_chars=200).forward("all")

    long_value = result["tasks"]["final_summary"]["outputs"]["long"]
    assert len(long_value) < 300 and "truncated" in long_value
    assert result["tasks"]["prep"]["summary"] == "Scenarios computed."
    assert ReadSubTaskStateTool(state, preview_chars=200).forward("nope")["code"] == "UNKNOWN_TASK"


def test_sub_task_status_lists_failures(state):
    state.tasks["final_summary"].status = FAILED
    result = GetSubTaskStatusTool(state).forward()
    assert result["completed"] == ["prep"]
    assert result["failed"] == ["final_summary"]


def test_save_run_metrics_then_delta(state):
    result = SaveRunMetricsTool(state).forward()
    assert result["metrics"]["headline_margin"] == 8300000
    assert result["categories"] == {"revenue_scenarios": "2 items"}

    assert "first run" in GetDeltaSectionTool(state).forward()["text"]

    write_previous_run(state, {"metrics": {"headline_margin": 8000000}, "categories": {}})
    delta = GetDeltaSectionTool(state).forward()
    assert delta["metrics"]["headline_margin"]["change"] == 300000
    assert state.delta["previous_run_at"] == "2026-09-01"


def test_saved_charts_are_listed_by_name(state):
    result = GetAllSavedChartsTool(state).forward()
    assert result["charts"] == [{"name": "volumes", "title": "Viaggi mensili", "task_id": "prep", "type": "bar"}]
    assert result["tables"][0]["rows"] == 1


def test_sections_are_validated(state):
    tool = GetHtmlTemplateTool(state)
    assert tool.forward([])["code"] == "INVALID_SECTIONS"
    assert tool.forward([{"title": "A", "content": "x", "charts": ["nope"]}])["code"] == "INVALID_SECTIONS"
    assert tool.forward([{"title": "A", "content": "x", "style": "loud"}])["code"] == "INVALID_SECTIONS"
    assert tool.forward("not json")["code"] == "INVALID_SECTIONS"
    assert state.html is None


def test_the_report_is_rendered_saved_and_safe(state):
    SaveRunMetricsTool(state).forward()
    sections = [
        {"title": "Sintesi", "content": "Il **margine** cresce. <script>alert(1)</script>", "style": "info",
         "charts": ["volumes"]},
    ]
    result = GetHtmlTemplateTool(state, appendix_title="Appendice").forward(json.dumps(sections))
    assert "in the appendix: top_clients" in result["text"]

    saved = SaveHtmlReportTool(state).forward()
    assert saved["kind"] == "text"
    html = (state.run_dir / "report.html").read_text()

    assert "<strong>margine</strong>" in html
    assert "<script>alert(1)</script>" not in html
    assert "CCNO &lt;b&gt;" in html
    assert "Appendice" in html
    assert 'data-chart="volumes"' in html
    assert "chart.js@4.4.1" in html
    assert '"it-IT"' in html
    assert "Primo margine 2027" in html and 'data-num="8300000"' in html
    assert "<h1>Forecast 2027</h1>" in html


def test_save_html_report_needs_a_rendering(state):
    assert SaveHtmlReportTool(state).forward()["code"] == "NOT_RENDERED"


def test_insights_reach_the_report_agent_with_their_task(state):
    state.insights.append({"task_id": "prep", "text": "CCNO makes half the margin.", "evidence": "0.51"})
    state.tasks["prep"].exploratory = True

    tasks = ReadSubTaskStateTool(state, preview_chars=200).forward()["tasks"]

    assert tasks["prep"]["insights"] == [{"text": "CCNO makes half the margin.", "evidence": "0.51"}]
    assert tasks["prep"]["exploratory"] is True
    assert "insights" not in tasks["final_summary"]
