"""Tests for the analyst's run, progress log, state and value helpers."""

import json
import math

import numpy as np
import pandas as pd

from analyst.run import COMPLETED, AnalysisRun, ProgressLog
from tests.analyst_helpers import make_state as state_for
from analyst.values import describe, preview, to_native


def brief(**overrides):
    base = {
        "name": "Test brief",
        "sub_tasks": [
            # Deliberately out of order: the state sorts by `order`.
            {"id": "final_summary", "name": "Summary", "order": 3, "depends_on": ["prep", "trend"],
             "mission": "Sum up.", "outputs": {"required": ["kpi_summary"]}},
            {"id": "prep", "name": "Prep", "order": 1, "depends_on": [],
             "mission": "Prepare.", "outputs": {"required": ["volumes"], "optional": ["notes"]}},
            {"id": "trend", "name": "Trend", "order": 2, "depends_on": ["prep"],
             "mission": "Trend.", "outputs": {"required": ["slope"]}},
        ],
        "metrics_config": None,
    }
    base.update(overrides)
    return base


def make_state(**overrides):
    return state_for(brief(**overrides), brief_id=7)


# ---------------------------------------------------------------------------
# Progress log and run
# ---------------------------------------------------------------------------


def test_progress_events_are_read_from_a_cursor():
    log = ProgressLog()
    log.emit("run_started", "Started")
    log.emit("task_started", "Prep", task_id="prep")

    events, cursor = log.since(0)
    assert [e["type"] for e in events] == ["run_started", "task_started"]
    assert events[1]["task_id"] == "prep"
    assert cursor == 2

    log.emit("task_completed", "Prep done", task_id="prep")
    events, cursor = log.since(cursor)
    assert [e["seq"] for e in events] == [3]
    assert log.since(cursor) == ([], 3)


def test_the_status_payload_carries_the_new_events_only():
    run = AnalysisRun(run_id="r1", brief_id=7, api_key="k", user_email="u@x")
    run.progress.emit("run_started", "Started")
    run.status = COMPLETED
    run.report_ready = True

    payload = run.status_payload(since=1)
    assert payload["events"] == []
    assert payload["next_since"] == 1
    assert payload["status"] == "completed"
    assert payload["report_ready"] is True
    assert run.finished


def test_closing_a_run_cancels_it():
    run = AnalysisRun(run_id="r1", brief_id=7, api_key="k", user_email="u@x")
    assert not run.cancelled
    run.close()
    assert run.cancelled


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_tasks_run_by_order():
    state = make_state()
    assert [t.id for t in state.ordered_tasks()] == ["prep", "trend", "final_summary"]
    assert state.tasks["prep"].required == ["volumes"]
    assert state.tasks["prep"].order == 1


def test_dependency_outputs_hold_only_the_dependencies():
    state = make_state()
    state.tasks["prep"].outputs = {"volumes": [1, 2]}
    state.tasks["trend"].outputs = {"slope": 0.5}

    assert state.dependency_outputs("trend") == {"prep": {"volumes": [1, 2]}}
    assert set(state.dependency_outputs("final_summary")) == {"prep", "trend"}


def test_missing_outputs_lists_unsaved_required_keys():
    task = make_state().tasks["prep"]
    assert task.missing_outputs() == ["volumes"]
    task.outputs["notes"] = "x"
    assert task.missing_outputs() == ["volumes"]
    task.outputs["volumes"] = []
    assert task.missing_outputs() == []


def test_merged_outputs_let_later_tasks_win():
    state = make_state()
    state.tasks["prep"].outputs = {"value": 1, "a": "x"}
    state.tasks["trend"].outputs = {"value": 2}
    assert state.merged_outputs() == {"value": 2, "a": "x"}


def test_the_snapshot_is_the_json_part():
    state = make_state()
    state.datasets["trips"] = pd.DataFrame({"mese": ["2026-01"], "viaggi": [7300]})
    state.tasks["prep"].outputs = {"volumes": [7300]}
    state.charts["c1"] = {"title": "Trips", "spec": {}, "task_id": "prep"}

    saved = json.loads(json.dumps(state.snapshot()))
    assert saved["tasks"]["prep"]["outputs"] == {"volumes": [7300]}
    assert saved["datasets"] == {"trips": ["mese", "viaggi"]}
    assert saved["charts"] == {"c1": "Trips"}


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_to_native_converts_numpy_and_pandas():
    frame = pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]})
    value = {
        "int": np.int64(3),
        "float": np.float64(1.5),
        "nan": float("nan"),
        "bool": np.bool_(True),
        "array": np.array([1, 2]),
        "series": pd.Series([10, 20], index=["x", "y"]),
        "frame": frame,
        "ts": pd.Timestamp("2026-01-01"),
        "nat": pd.NaT,
    }
    native = to_native(value)

    assert native == {
        "int": 3,
        "float": 1.5,
        "nan": None,
        "bool": True,
        "array": [1, 2],
        "series": {"x": 10, "y": 20},
        "frame": [{"a": 1, "b": 0.5}, {"a": 2, "b": None}],
        "ts": "2026-01-01T00:00:00",
        "nat": None,
    }
    json.dumps(native)
    assert not math.isnan(native["float"])


def test_preview_cuts_long_values():
    assert preview([1, 2], 100) == "[1, 2]"
    text = preview("x" * 50, 10)
    assert text.startswith('"xxxxxxxxx') and "truncated" in text


def test_describe_names_the_shape_without_content():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    assert describe(frame) == "DataFrame 3 rows x 1 cols: a"
    assert describe({"k": 1}) == "dict with keys: k"
    assert describe([1, 2]) == "list of 2"


def test_the_state_reads_title_language_and_charts_from_the_plan():
    state = make_state()
    state.plan.update(title="Forecast", language="Italian", charts=[{"id": "c1", "description": "Bars"}])

    assert (state.title, state.language) == ("Forecast", "Italian")
    assert state.charts_requested == [{"id": "c1", "description": "Bars"}]
    assert state.metrics_config is None


def test_insights_are_in_the_snapshot():
    state = make_state()
    state.insights.append({"task_id": "prep", "text": "Fact.", "evidence": "1"})
    assert state.snapshot()["insights"][0]["text"] == "Fact."
