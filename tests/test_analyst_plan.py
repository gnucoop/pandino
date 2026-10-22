"""Tests for the analyst's plan: validation, the code fallback on briefs of any shape, the context text."""

import copy
import json
from pathlib import Path

import pytest

from analyst.plan import (
    brief_context_text,
    brief_task_ids,
    normalize_plan,
    plan_from_brief,
    slug,
    validate_plan,
)
from tests.analyst_helpers import valid_brief

IDENTIFIERS = {"orders": ("country", "amount"), "customers": ("id",)}
EXAMPLE = Path(__file__).resolve().parents[1] / "analyst" / "reports" / "example-brief.json"


def good_plan(**overrides):
    plan = {
        "title": "Revenue",
        "language": "Italian",
        "relations": ["orders"],
        "tasks": [
            {"id": "prep", "name": "Prep", "mission": "Check.", "depends_on": [], "outputs": ["row_count"]},
            {"id": "explore_outliers", "name": "Outliers", "mission": "Look.", "depends_on": ["prep"],
             "outputs": ["outlier_clients"], "exploratory": True},
            {"id": "final_summary", "name": "Summary", "mission": "Sum up.", "depends_on": ["prep"],
             "outputs": ["key_findings"]},
        ],
        "charts": [{"id": "c1", "description": "Bar of revenue by country"}],
        "report": {"sections": ["Summary"]},
        "metrics_config": {"metrics": {"row_count": {"label": "Rows", "type": "number", "better": "higher"}}},
    }
    plan.update(overrides)
    return normalize_plan(plan)


def errors(plan, **kwargs):
    kwargs.setdefault("max_extra", 2)
    return validate_plan(plan, identifiers=IDENTIFIERS, **kwargs)


# ---------------------------------------------------------------------------
# validate_plan
# ---------------------------------------------------------------------------


def test_a_good_plan_passes():
    assert errors(good_plan()) == []


def test_normalize_fills_harmless_defaults():
    plan = normalize_plan({"title": "t", "tasks": [{"id": "a"}]})
    assert plan["relations"] == [] and plan["metrics_config"] is None
    assert plan["report"] == {"sections": [], "tone": "", "length": ""}
    assert plan["tasks"][0]["depends_on"] == [] and plan["tasks"][0]["exploratory"] is False


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda p: p["tasks"][1].update(depends_on=["final_summary"]), "come before it"),
        (lambda p: p["tasks"][0].update(id="Bad Id"), "snake_case"),
        (lambda p: p["tasks"][2].update(outputs=["row_count"]), "already produced"),
        (lambda p: p["tasks"][0].update(outputs=[]), "non-empty list of keys"),
        (lambda p: p.update(charts=[]), "at least one chart"),
        (lambda p: p.update(relations=["ghost"]), "Unknown relations"),
        (lambda p: p.update(language=""), "language"),
        (lambda p: p["metrics_config"]["metrics"].update(nope={"label": "x", "type": "number", "better": "higher"}),
         "output keys"),
        (lambda p: p.update(tasks=[]), "1 to 10 tasks"),
    ],
)
def test_validation_errors_are_worded_for_the_model(change, message):
    plan = good_plan()
    change(plan)
    assert any(message in e for e in errors(plan)), errors(plan)


def test_exploratory_tasks_are_capped():
    assert any("At most 0" in e for e in errors(good_plan(), max_extra=0))


def test_a_brief_task_may_not_depend_on_an_exploratory_one():
    plan = good_plan()
    plan["tasks"][2]["depends_on"] = ["prep", "explore_outliers"]

    assert any(
        "tasks[final_summary] is a brief task and may not depend on exploratory tasks (explore_outliers)" in e
        for e in errors(plan)
    ), errors(plan)


def test_an_exploratory_task_may_depend_on_another_one():
    plan = good_plan()
    plan["tasks"].insert(2, {"id": "explore_more", "name": "More", "mission": "Look further.",
                             "depends_on": ["explore_outliers"], "outputs": ["more"], "exploratory": True})
    assert errors(normalize_plan(plan)) == []


def test_the_brief_tasks_must_be_kept():
    assert any("missing: by_country" in e for e in errors(good_plan(), brief_ids=["prep", "by_country"]))


# ---------------------------------------------------------------------------
# plan_from_brief: any shape in, a valid plan out
# ---------------------------------------------------------------------------


def assert_valid(brief, max_extra=0):
    plan = plan_from_brief(brief, IDENTIFIERS)
    assert validate_plan(plan, identifiers=IDENTIFIERS, brief_ids=brief_task_ids(brief), max_extra=max_extra) == []
    return plan


def test_the_interviewer_brief_keeps_its_tasks_charts_and_metrics():
    plan = assert_valid(valid_brief())

    assert [t["id"] for t in plan["tasks"]] == ["data_preparation", "by_country", "final_summary"]
    assert plan["tasks"][1]["outputs"] == ["total_revenue"]
    assert "Also save, when available: top_country." in plan["tasks"][1]["mission"]
    assert plan["charts"] == [{"id": "c1", "description": "Bar chart"}]
    assert plan["report"]["sections"] == ["Summary"]
    assert plan["language"] == "Italian"
    assert plan["relations"] == ["orders"]
    assert plan["metrics_config"]["metrics"]["total_revenue"]["label"] == "Revenue"


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example brief not present")
def test_the_example_brief_gives_six_tasks_and_six_charts():
    brief = json.loads(EXAMPLE.read_text())
    identifiers = {"Trasporti": (), "Ordini": (), "report_mensile": ()}
    plan = plan_from_brief(brief, identifiers)

    assert validate_plan(plan, identifiers=identifiers, brief_ids=brief_task_ids(brief)) == []
    assert len(plan["tasks"]) == 6 and len(plan["charts"]) == 6
    assert plan["tasks"][-1]["id"] == "final_summary"


def test_renamed_keys_lists_of_outputs_and_no_order():
    brief = {
        "title": "Quarterly check",
        "phases": [
            {"title": "Load the data", "goal": "Count orders", "outputs": ["order_count"]},
            {"title": "Trend", "goal": "Monthly trend", "depends_on": ["load_the_data"]},
        ],
        "graphs": [{"type": "line", "description": "Orders per month"}],
    }
    plan = assert_valid(brief)

    assert [t["id"] for t in plan["tasks"]] == ["load_the_data", "trend"]
    assert plan["tasks"][1]["outputs"] == ["trend_result"]
    assert plan["tasks"][1]["depends_on"] == ["load_the_data"]
    assert plan["charts"][0]["description"] == "Orders per month"
    assert plan["language"] == "English"


def test_tasks_found_under_any_key_and_ordered_by_order():
    brief = {"work": {"steps": [
        {"id": "b", "mission": "second", "order": 2},
        {"id": "a", "mission": "first", "order": 1},
    ]}}
    plan = assert_valid(brief)
    assert [t["id"] for t in plan["tasks"]] == ["a", "b"]
    # No dependencies stated: each task reads the previous one.
    assert plan["tasks"][1]["depends_on"] == ["a"]


@pytest.mark.parametrize("brief", [{}, {"goal": "Find out why margins fell"}, {"tasks": "not a list"}])
def test_a_brief_without_tasks_still_gets_a_plan(brief):
    plan = assert_valid(brief)
    assert [t["id"] for t in plan["tasks"]] == ["analysis", "final_summary"]
    assert len(plan["charts"]) == 1


def test_invalid_metrics_are_dropped_not_fatal():
    brief = copy.deepcopy(valid_brief())
    brief["metrics_config"] = {"metrics": {"unknown_key": {"label": "x"}}}
    assert assert_valid(brief)["metrics_config"] is None


def test_slug():
    assert slug("Baseline 2026!", "x") == "baseline_2026"
    assert slug("2027 forecast", "x") == "t_2027_forecast"
    assert slug("", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Context text
# ---------------------------------------------------------------------------


def test_the_context_carries_everything_but_tasks_report_and_metrics():
    text = brief_context_text(valid_brief())

    assert "Sum amount per country." in text           # mission
    assert "amount is in EUR." in text                 # data_notes
    assert "Only 2026." in text                        # data_scope.filters
    assert "Check amount." not in text                 # a task mission
    assert "Bar chart" not in text                     # the report
    assert "Revenue" in text                           # name


def test_the_context_is_capped():
    text = brief_context_text({"notes": "x" * 500}, max_chars=100)
    assert len(text) < 140 and text.endswith("[brief context truncated]")
