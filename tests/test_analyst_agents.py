"""
Tests for the analyst's planner, task, chart and report agents.

Engines are built through their real __post_init__ with the model, the SQL
datasource and CodeAgent patched, so the SQL gate, the scoped schema and the
tool wiring are exercised. A run's output is scripted on the fake agent and
passed through the engine's own final-answer guard first, as smolagents does.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import datachat.schema_snapshot_loader as loader
from analyst.chart_agent import ChartAgentEngine
from analyst.planner_agent import PlannerAgentEngine
from analyst.prompts import CODE_STEP_RULE
from analyst.report_agent import ReportAgentEngine
from analyst.sandbox import GuardedPythonExecutor
from analyst.state import COMPLETED
from analyst.task_agent import TaskAgentEngine
from tests.analyst_helpers import COLUMNS, Run, make_config, make_state, valid_brief
from tests.fake_sql_datasource import FakeDatasource


@pytest.fixture(autouse=True)
def prompts_from_code():
    default = lambda title, default_text="", **kwargs: default_text  # noqa: E731
    with patch("analyst.task_agent.load_prompt", side_effect=default), \
         patch("analyst.report_agent.load_prompt", side_effect=default), \
         patch("analyst.planner_agent.load_prompt", side_effect=default), \
         patch("analyst.chart_agent.load_prompt", side_effect=default):
        yield


@pytest.fixture(autouse=True)
def reset_snapshot_cache():
    loader.invalidate_snapshot()
    yield
    loader.invalidate_snapshot()


@pytest.fixture
def state():
    return make_state()


def build(engine_cls, module, datasource=None, **kwargs):
    datasource = datasource if datasource is not None else FakeDatasource(
        tables=["orders", "customers"], columns=COLUMNS
    )
    with patch("datachat.smolagents_base.build_litellm_model", return_value=object()), \
         patch("datachat.smolagents_base.sql_datasource.get_datasource", return_value=datasource), \
         patch(f"{module}.CodeAgent") as agent_cls:
        engine = engine_cls(api_key="k", user_name="u@x", config=make_config(), **kwargs)
    if engine._agent is not None:
        engine._agent.state = {}
    return engine, agent_cls


def build_task(state, task_id="by_country", **kwargs):
    return build(TaskAgentEngine, "analyst.task_agent", state=state, task=state.tasks[task_id], **kwargs)


def script(engine, *outputs, side_effect=None):
    """Each run ends with the next output, after the final-answer guard saw it."""
    queue = list(outputs)
    engine.tasks = []
    engine.guard_errors = []

    def run(task, **kwargs):
        engine.tasks.append(task)
        if side_effect:
            side_effect()
        output = queue.pop(0)
        try:
            engine._check_final_answer(output)
        except ValueError as e:
            engine.guard_errors.append(str(e))
        return Run(output, SimpleNamespace(input_tokens=100, output_tokens=20))

    engine._agent.run.side_effect = run


# ---------------------------------------------------------------------------
# Task agent: construction
# ---------------------------------------------------------------------------


def test_the_task_agent_gets_the_tools_the_sandbox_and_the_scoped_schema(state):
    engine, agent_cls = build_task(state)
    kwargs = agent_cls.call_args.kwargs

    names = [t.name for t in kwargs["tools"]]
    assert names == [
        "load_dataset", "sql_stats", "sql_group_stats", "sql_correlation", "sql_outliers",
        "detect_anomalies", "run_regression", "run_forecast",
        "save_sub_task_output", "record_insight", "sql_engine",
    ]
    assert isinstance(kwargs["executor"], GuardedPythonExecutor)
    assert "pandas" not in kwargs["additional_authorized_imports"]

    instructions = kwargs["instructions"]
    assert 'orders' in instructions
    # The relation outside the plan's scope is named, not described.
    assert 'These others also exist' in instructions and '"customers"' in instructions
    assert '"customers"("id"' not in instructions
    assert "Italian" in instructions


def large_orders():
    return FakeDatasource(tables=["orders", "customers"], columns=COLUMNS, rows=[("orders", 15_000_000)])


def test_the_task_agent_is_told_which_tables_are_too_large_to_load(state):
    _, agent_cls = build_task(state, datasource=large_orders())

    instructions = agent_cls.call_args.kwargs["instructions"]
    assert "LARGE TABLES" in instructions and '"orders" ~15.0M' in instructions


def test_small_tables_add_no_large_tables_note(state):
    _, agent_cls = build_task(state)
    assert "LARGE TABLES" not in agent_cls.call_args.kwargs["instructions"]


def test_the_task_agent_needs_the_database(state):
    engine, agent_cls = build_task(state, datasource=FakeDatasource(raises=RuntimeError("down")))
    assert engine._agent is None
    assert engine.run()["code"] == "MISSING_CONFIG"


# ---------------------------------------------------------------------------
# Task agent: running
# ---------------------------------------------------------------------------


def test_the_task_shows_its_mission_inputs_and_datasets_but_not_other_missions(state):
    state.tasks["data_preparation"].outputs = {"row_count": 120}
    state.datasets["orders_2026"] = pd.DataFrame({"country": ["IT"], "amount": [1.0]})
    engine, _ = build_task(state)

    task = engine.render_task()

    assert "By country [by_country], task 2 of 3" in task
    assert "Required outputs: total_revenue" in task
    assert "Never copy numbers from printed results into code or strings." in task
    assert 'inputs["data_preparation"]["row_count"]: int = 120' in task
    assert 'datasets["orders_2026"]: DataFrame 1 rows x 2 cols' in task
    assert "Sum amount per country." in task          # the brief's mission
    assert "Only 2026." in task and "Revenue excludes VAT." in task
    assert "Synthesize." not in task                   # another task's mission
    assert "Bar chart" not in task                     # the report section


def test_inputs_and_datasets_are_placed_in_the_namespace(state):
    state.tasks["data_preparation"].outputs = {"row_count": 120}
    engine, _ = build_task(state)

    def save():
        engine.task.outputs["total_revenue"] = 10

    script(engine, {"kind": "task_done", "summary": "Done."}, side_effect=save)
    engine.run()

    assert engine._agent.state["datasets"] is state.datasets
    assert engine._agent.state["inputs"] == {"data_preparation": {"row_count": 120}}
    # A copy: the agent cannot alter the outputs it was given.
    assert engine._agent.state["inputs"]["data_preparation"] is not state.tasks["data_preparation"].outputs


def test_task_done_is_refused_while_a_required_output_is_missing(state):
    engine, _ = build_task(state)
    script(engine, {"kind": "task_done", "summary": "Done."})

    result = engine.run()

    assert result["code"] == "INVALID_OUTPUT"
    assert "total_revenue" in engine.guard_errors[0]


def test_a_completed_task_reports_its_summary_and_tokens(state):
    engine, _ = build_task(state)

    def save():
        engine.task.outputs["total_revenue"] = 10

    script(engine, '{"kind": "task_done", "summary": "Revenue is 10."}', side_effect=save)

    assert engine.run() == {"kind": "task_done", "summary": "Revenue is 10."}
    assert engine.guard_errors == []
    assert engine.token_usage == {"input": 100, "output": 20}


def test_a_run_out_of_steps_is_an_error_even_with_every_output_saved(state):
    engine, _ = build_task(state)
    # smolagents' answer once the steps are spent: prose, not the contract.
    script(engine, "Revenue totals 10.", side_effect=lambda: engine.task.outputs.update(total_revenue=10))

    assert engine.run()["code"] == "INVALID_OUTPUT"


def test_a_run_out_of_steps_with_an_output_missing_is_an_error(state):
    engine, _ = build_task(state)
    script(engine, "I ran out of steps.")

    assert engine.run()["code"] == "INVALID_OUTPUT"


class _Step:
    def __init__(self, number, observations=None):
        self.step_number, self.observations = number, observations


def test_the_agent_is_told_to_save_three_steps_before_the_cap(state):
    reported = []
    engine, _ = build_task(state, on_step=lambda step, agent=None: reported.append(step.step_number))
    agent = SimpleNamespace(max_steps=20)

    early, warned, error_step = _Step(16, "rows: 5"), _Step(17, "rows: 5"), _Step(17)
    engine._on_step(early, agent=agent)
    engine._on_step(warned, agent=agent)
    engine._on_step(error_step, agent=agent)

    assert early.observations == "rows: 5"
    assert warned.observations.startswith("rows: 5\n\n3 steps left: save every required output")
    assert "(missing: total_revenue)" in warned.observations
    assert error_step.observations.startswith("3 steps left")
    assert reported == [16, 17, 17]          # progress is still reported


def test_no_warning_once_every_output_is_saved(state):
    engine, _ = build_task(state)
    engine.task.outputs["total_revenue"] = 10
    step = _Step(17, "ok")

    engine._on_step(step, agent=SimpleNamespace(max_steps=20))

    assert step.observations == "ok"


def test_the_variables_an_attempt_computed_are_kept_for_the_next(state):
    engine, _ = build_task(state)
    frame = pd.DataFrame({"a": [1]})
    engine._agent.tools = {"sql_engine": object()}
    engine._agent.python_executor = SimpleNamespace(state={
        "__name__": "__main__", "_print_outputs": "x", "datasets": {}, "inputs": {},
        "sql_engine": object(), "json": pd, "helper": lambda: 1,
        "route_margin": frame, "totals": {"IT": 3}, "n": 4, "label": "ok",
    })

    assert engine.variables() == {"route_margin": frame, "totals": {"IT": 3}, "n": 4, "label": "ok"}


def test_a_retry_resumes_from_the_carried_variables(state):
    task = state.tasks["by_country"]
    task.carried = {"route_margin": pd.DataFrame({"a": [1, 2]}), "n": 4}
    engine, _ = build_task(state)

    rendered = engine.render_task()
    assert "Variables left by the previous attempt" in rendered
    assert "- route_margin: DataFrame 2 rows x 1 cols: a" in rendered
    assert "- n: int = 4" in rendered

    script(engine, {"kind": "error", "message": "x"})
    engine.run()
    assert engine._agent.state["n"] == 4 and "route_margin" in engine._agent.state


def test_an_agent_error_is_passed_on(state):
    engine, _ = build_task(state)
    # Even with the outputs saved: the model said the task cannot be done.
    script(engine, {"kind": "error", "message": "No 2026 data."},
           side_effect=lambda: engine.task.outputs.update(total_revenue=10))
    assert engine.run() == {"kind": "error", "message": "No 2026 data.", "code": "AGENT_ERROR"}


def test_a_crashed_run_is_an_error(state):
    engine, _ = build_task(state)
    engine._agent.run.side_effect = RuntimeError("model unavailable")
    result = engine.run()
    assert result["code"] == "RUN_FAILED"
    assert "model unavailable" in result["message"]


# ---------------------------------------------------------------------------
# Report agent
# ---------------------------------------------------------------------------


def test_the_report_agent_gets_the_manager_tools_and_no_sql(state):
    engine, agent_cls = build(ReportAgentEngine, "analyst.report_agent", state=state)
    kwargs = agent_cls.call_args.kwargs

    assert [t.name for t in kwargs["tools"]] == [
        "read_sub_task_state", "get_sub_task_status", "save_run_metrics", "get_delta_section",
        "get_all_saved_charts", "get_html_template", "save_html_report",
    ]
    assert "orders" not in kwargs["instructions"]
    assert engine._configured_model == "report/model"


def test_the_report_task_carries_the_report_shape_and_task_summaries(state):
    state.tasks["by_country"].summary = "Italy leads."
    engine, _ = build(ReportAgentEngine, "analyst.report_agent", state=state)

    task = engine.render_task()

    assert "Report language: Italian" in task
    assert "- Summary" in task and "- c1: Bar chart" in task
    assert "by_country (By country): pending. Italy leads." in task
    assert "Sum amount per country." in task           # the brief, as context
    assert "Check amount." not in task                 # a task's mission


def test_report_saved_is_refused_until_the_report_is_saved(state):
    engine, _ = build(ReportAgentEngine, "analyst.report_agent", state=state)
    script(engine, {"kind": "report_saved"})
    assert engine.run()["code"] == "INVALID_OUTPUT"
    assert "not saved yet" in engine.guard_errors[0]

    state.report_saved = True
    script(engine, {"kind": "report_saved"})
    assert engine.run() == {"kind": "report_saved"}


def test_an_exploratory_task_is_told_so(state):
    state.tasks["by_country"].exploratory = True
    engine, _ = build_task(state)
    assert "exploratory: find what is notable" in engine.render_task()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def build_planner(brief=None, **kwargs):
    return build(PlannerAgentEngine, "analyst.planner_agent", brief=brief or valid_brief(), **kwargs)


def test_every_analyst_agent_cuts_and_retries_hung_model_calls(state):
    state.tasks["by_country"].status = COMPLETED
    engines = [
        (PlannerAgentEngine, "analyst.planner_agent", {"brief": valid_brief()}),
        (TaskAgentEngine, "analyst.task_agent", {"state": state, "task": state.tasks["by_country"]}),
        (ChartAgentEngine, "analyst.chart_agent", {"state": state}),
        (ReportAgentEngine, "analyst.report_agent", {"state": state}),
    ]
    for engine_cls, module, kwargs in engines:
        with patch("datachat.smolagents_base.build_litellm_model", return_value=object()) as build_model, \
             patch("datachat.smolagents_base.sql_datasource.get_datasource",
                   return_value=FakeDatasource(tables=["orders", "customers"], columns=COLUMNS)), \
             patch(f"{module}.CodeAgent"):
            engine_cls(api_key="k", user_name="u@x", config=make_config(), **kwargs)
        assert build_model.call_args.kwargs["timeout"] == 180, engine_cls
        assert build_model.call_args.kwargs["num_retries"] == 2, engine_cls


def test_every_analyst_task_ends_with_the_code_step_rule(state):
    state.tasks["by_country"].status = COMPLETED
    engines = [
        build_planner()[0],
        build_task(state)[0],
        build(ChartAgentEngine, "analyst.chart_agent", state=state)[0],
        build(ReportAgentEngine, "analyst.report_agent", state=state)[0],
    ]
    for engine in engines:
        assert engine.render_task().endswith("\n\n" + CODE_STEP_RULE), type(engine).__name__


def test_the_planner_and_report_agents_run_their_code_through_the_guard(state):
    _, planner_cls = build_planner()
    _, report_cls = build(ReportAgentEngine, "analyst.report_agent", state=state)

    for agent_cls in (planner_cls, report_cls):
        assert type(agent_cls.call_args.kwargs["executor"]) is GuardedPythonExecutor


def test_the_planner_is_told_which_tables_are_too_large_to_load():
    _, agent_cls = build_planner(datasource=large_orders())

    instructions = agent_cls.call_args.kwargs["instructions"]
    assert "LARGE TABLES" in instructions and '"orders" ~15.0M' in instructions


def a_plan(**overrides):
    plan = {
        "title": "Revenue by country", "language": "Italian", "relations": ["orders"],
        "tasks": [
            {"id": "data_preparation", "name": "Prep", "mission": "m", "outputs": ["row_count"]},
            {"id": "explore", "name": "Explore", "mission": "m", "depends_on": ["data_preparation"],
             "outputs": ["outliers"], "exploratory": True},
            {"id": "by_country", "name": "By country", "mission": "m", "depends_on": [], "outputs": ["total_revenue"]},
            {"id": "final_summary", "name": "Summary", "mission": "m", "depends_on": ["by_country"],
             "outputs": ["key_findings"]},
        ],
        "charts": [{"id": "c1", "description": "Bar chart"}],
        "report": {"sections": ["Summary"]},
        "metrics_config": None,
    }
    plan.update(overrides)
    return plan


def test_the_planner_probes_with_statistics_and_sees_the_whole_schema():
    engine, agent_cls = build_planner()
    kwargs = agent_cls.call_args.kwargs

    assert [t.name for t in kwargs["tools"]] == [
        "sql_stats", "sql_group_stats", "sql_correlation", "sql_outliers", "sql_engine",
    ]
    assert '"customers"("id"' in kwargs["instructions"]      # full schema, not scoped
    assert "Add up to 2 exploratory tasks" in kwargs["instructions"]
    assert engine._configured_model == "report/model"


def test_the_planner_reads_the_brief_whatever_its_shape():
    odd = {"Titolo": "Analisi", "fasi": [{"cosa": "contare gli ordini"}]}
    engine, _ = build_planner(odd)
    assert '"fasi":[{"cosa":"contare gli ordini"}]' in engine.render_task()


def test_a_valid_plan_is_returned_normalized():
    engine, _ = build_planner()
    script(engine, {"kind": "plan", "plan": a_plan()})

    result = engine.run()

    assert result["kind"] == "plan"
    assert result["plan"]["tasks"][0]["exploratory"] is False      # default filled in
    assert engine.guard_errors == []


def test_a_plan_dropping_a_brief_task_is_turned_down():
    engine, _ = build_planner()
    plan = a_plan()
    plan["tasks"] = [t for t in plan["tasks"] if t["id"] != "by_country"]
    plan["tasks"][-1]["depends_on"] = []
    script(engine, {"kind": "plan", "plan": plan})

    assert engine.run()["code"] == "INVALID_OUTPUT"
    assert "missing: by_country" in engine.guard_errors[0]


def test_too_many_exploratory_tasks_are_turned_down():
    engine, _ = build_planner()
    plan = a_plan()
    for task in plan["tasks"]:
        task["exploratory"] = True
    script(engine, {"kind": "plan", "plan": plan})
    engine.run()
    assert "At most 2" in engine.guard_errors[0]


# ---------------------------------------------------------------------------
# Chart agent
# ---------------------------------------------------------------------------


def chart_state():
    state = make_state()
    state.plan["charts"] = [{"id": "c1", "description": "Bars"}, {"id": "c2", "description": "Heatmap"}]
    state.tasks["by_country"].status = COMPLETED
    state.tasks["by_country"].outputs = {"total_revenue": {"IT": 10.0, "FR": 5.0}}
    state.tasks["data_preparation"].outputs = {"row_count": 3}      # not completed: not shown
    return state


def a_chart():
    return {"title": "Bars", "spec": {"type": "bar", "data": {"datasets": [{"data": [1]}]}}, "task_id": "charts"}


def test_the_chart_agent_gets_drawing_tools_the_sandbox_and_all_outputs():
    state = chart_state()
    engine, agent_cls = build(ChartAgentEngine, "analyst.chart_agent", state=state)
    kwargs = agent_cls.call_args.kwargs

    assert [t.name for t in kwargs["tools"]] == ["save_chart", "save_table", "load_dataset", "sql_engine"]
    assert isinstance(kwargs["executor"], GuardedPythonExecutor)
    assert "Italian" in kwargs["instructions"]

    task = engine.render_task()
    assert "- c1: Bars" in task and "- c2: Heatmap" in task
    assert 'outputs["by_country"]["total_revenue"]: dict with keys: IT, FR' in task
    assert "row_count" not in task


def test_every_requested_chart_must_be_covered():
    state = chart_state()
    engine, _ = build(ChartAgentEngine, "analyst.chart_agent", state=state)

    def save():
        state.charts["revenue_bars"] = a_chart()

    script(engine, {"kind": "charts_done", "coverage": [{"chart": "c1", "saved": "revenue_bars"}]}, side_effect=save)
    assert engine.run()["code"] == "INVALID_OUTPUT"
    assert "chart c2 is not in the coverage" in engine.guard_errors[0]

    script(engine, {"kind": "charts_done", "coverage": [
        {"chart": "c1", "saved": "ghost"}, {"chart": "c2", "saved": None, "reason": ""}]})
    engine.run()
    assert "nothing was saved as 'ghost'" in engine.guard_errors[0]
    assert "no reason given" in engine.guard_errors[0]


def test_covered_charts_are_reported_with_the_skipped_ones():
    state = chart_state()
    engine, _ = build(ChartAgentEngine, "analyst.chart_agent", state=state)

    def save():
        state.charts["revenue_bars"] = a_chart()

    script(engine, {"kind": "charts_done", "coverage": [
        {"chart": "c1", "saved": "revenue_bars"},
        {"chart": "c2", "saved": None, "reason": "No matrix in the data."}]}, side_effect=save)

    result = engine.run()

    assert result == {"kind": "charts_done", "saved": 1, "skipped": [{"chart": "c2", "reason": "No matrix in the data."}]}
    assert engine._agent.state["outputs"] == {"by_country": {"total_revenue": {"IT": 10.0, "FR": 5.0}}}
    assert engine._agent.state["datasets"] is state.datasets
