"""
Tests for the analyst orchestrator: brief loading, planning and its fallback,
task order, retries, skipped dependents, the chart phase, the report phase and
its fallback, cancellation, metadata and progress events.

Engines are scripted fakes: the planner returns a plan (by default the one the
code fallback derives), each task engine runs a function of the task id that
saves outputs and returns a result, the chart engine saves a chart: exactly
what real agents would leave behind.
"""

import json

import pytest

import analyst.orchestrator as orchestrator
import datachat.schema_snapshot_loader as loader
from analyst.orchestrator import Orchestrator
from analyst.plan import plan_from_brief
from analyst.run import AnalysisRun
from tests.analyst_helpers import COLUMNS, make_config, valid_brief
from tests.fake_sql_datasource import FakeDatasource

DONE = {"kind": "task_done", "summary": "ok"}


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    loader.invalidate_snapshot()
    env = {"brief": valid_brief(), "consumed": []}
    monkeypatch.setattr(
        orchestrator, "get_brief",
        lambda brief_id, user_email=None: {"brief": env["brief"]} if env["brief"] else None,
    )
    monkeypatch.setattr(
        orchestrator.sql_datasource, "get_datasource",
        lambda: FakeDatasource(tables=["orders"], columns=COLUMNS),
    )
    monkeypatch.setattr(orchestrator, "mark_brief_consumed", lambda brief_id: env["consumed"].append(brief_id))
    yield env
    loader.invalidate_snapshot()


def make_run():
    return AnalysisRun(run_id="r1", brief_id=3, api_key="k", user_email="u@x")


OUTPUTS = {
    "data_preparation": {"row_count": 10},
    "by_country": {"total_revenue": 99.5},
    "final_summary": {"key_findings": ["Italy leads"]},
}


def planner_engine(behaviour=None):
    """A PlannerAgentEngine stand-in; behaviour(brief) -> result, default the code plan."""

    class FakePlanner:
        def __init__(self, *, brief, **kwargs):
            self.brief = brief
            self.token_usage = {"input": 30, "output": 3}

        def run(self):
            if behaviour is not None:
                return behaviour(self.brief)
            return {"kind": "plan", "plan": plan_from_brief(self.brief, {"orders": ()})}

    return FakePlanner


def task_engine(behaviour):
    """A TaskAgentEngine stand-in; behaviour(task, attempt) -> result."""
    calls = []

    class FakeTaskEngine:
        def __init__(self, *, state, task, on_step, on_insight, **kwargs):
            self.state, self.task, self.on_step, self.on_insight = state, task, on_step, on_insight
            self.token_usage = {"input": 10, "output": 1}

        def run(self):
            calls.append(self.task.id)
            return behaviour(self.task, calls.count(self.task.id))

        def variables(self):
            return {}

    FakeTaskEngine.calls = calls
    return FakeTaskEngine


def succeed(task, attempt):
    task.outputs.update(OUTPUTS.get(task.id) or {key: 1 for key in task.required})
    return DONE


class SavingChartEngine:
    def __init__(self, *, state, **kwargs):
        self.state = state
        self.token_usage = {"input": 20, "output": 2}

    def run(self):
        self.state.charts["revenue"] = {
            "title": "Revenue", "task_id": "charts",
            "spec": {"type": "bar", "data": {"labels": ["IT"], "datasets": [{"data": [99.5]}]}},
        }
        return {"kind": "charts_done", "saved": 1, "skipped": []}


class FailingChartEngine:
    def __init__(self, **kwargs):
        self.token_usage = {}

    def run(self):
        return {"kind": "error", "message": "model down", "code": "RUN_FAILED"}


class SavingReportEngine:
    def __init__(self, *, state, **kwargs):
        self.state = state
        self.token_usage = {"input": 50, "output": 5}

    def run(self):
        path = self.state.run_dir / "report.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html>report</html>")
        self.state.report_path = path
        return {"kind": "report_saved"}


class FailingReportEngine:
    def __init__(self, **kwargs):
        self.token_usage = {}

    def run(self):
        return {"kind": "error", "message": "model down", "code": "RUN_FAILED"}


def execute(
    tmp_path, task_behaviour=succeed, report=SavingReportEngine, retries=1, run=None,
    planner=None, charts=SavingChartEngine,
):
    run = run or make_run()
    engine = task_engine(task_behaviour)
    config = make_config(reports_dir=str(tmp_path), task_retries=retries)
    Orchestrator(
        run, config,
        planner_engine=planner or planner_engine(),
        task_engine=engine,
        chart_engine=charts,
        report_engine=report,
    ).execute()
    return run, engine.calls


def events(run):
    return [e["type"] for e in run.progress.since(0)[0]]


def metadata(tmp_path):
    return json.loads((tmp_path / "r1" / "run_metadata.json").read_text())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_run_executes_every_task_in_order_then_writes_the_report(tmp_path, environment):
    run, calls = execute(tmp_path)

    assert calls == ["data_preparation", "by_country", "final_summary"]
    assert run.status == "completed" and run.report_ready
    assert (run.tasks_done, run.tasks_total) == (3, 3)
    assert events(run) == [
        "run_started", "plan_started", "plan_ready",
        "task_started", "task_completed",
        "task_started", "task_completed",
        "task_started", "task_completed",
        "charts_started", "charts_saved",
        "report_started", "report_saved", "run_completed",
    ]
    assert environment["consumed"] == [3]

    meta = metadata(tmp_path)
    assert meta["status"] == "completed"
    assert meta["metrics"]["metrics"] == {"total_revenue": 99.5}
    assert meta["tokens"]["by_country"] == {"input": 10, "output": 1}
    assert meta["tokens"]["report"] == {"input": 50, "output": 5}
    assert meta["tokens"]["plan"] == {"input": 30, "output": 3}
    assert meta["tokens"]["charts"] == {"input": 20, "output": 2}
    assert meta["report"] == "report.html"
    assert meta["chart_agent"] == {"done": True, "error": None, "skipped": [], "step_errors": []}
    assert meta["report_agent"] == {"fallback": False, "error": None, "step_errors": []}
    assert (tmp_path / "r1" / "sub_task_state.json").exists()
    assert json.loads((tmp_path / "r1" / "plan.json").read_text())["title"] == "Revenue by country"


# ---------------------------------------------------------------------------
# Failures and retries
# ---------------------------------------------------------------------------


def test_a_failed_attempt_is_retried(tmp_path):
    def flaky(task, attempt):
        if task.id == "by_country" and attempt == 1:
            return {"kind": "error", "message": "missing outputs", "code": "INVALID_OUTPUT"}
        return succeed(task, attempt)

    run, calls = execute(tmp_path, flaky)

    assert calls.count("by_country") == 2
    assert "task_retry" in events(run)
    assert run.status == "completed"


def test_a_failed_exploratory_task_is_not_retried(tmp_path):
    def planned(brief):
        plan = plan_from_brief(brief, {"orders": ()})
        plan["tasks"].insert(2, {"id": "explore", "name": "Explore", "mission": "Look around.",
                                 "depends_on": [], "outputs": ["oddities"], "exploratory": True})
        return {"kind": "plan", "plan": plan}

    def flaky(task, attempt):
        if attempt == 1 and task.id in ("by_country", "explore"):
            return {"kind": "error", "message": "missing outputs", "code": "INVALID_OUTPUT"}
        return succeed(task, attempt)

    run, calls = execute(tmp_path, flaky, retries=3, planner=planner_engine(planned))

    assert calls.count("explore") == 1
    assert calls.count("by_country") == 2
    assert metadata(tmp_path)["tasks"]["explore"] == "failed"
    assert run.status == "completed"


def test_a_retry_receives_what_the_failed_attempt_computed(tmp_path):
    seen = []

    def flaky(task, attempt):
        if task.id == "by_country":
            seen.append(dict(task.carried))
            if attempt == 1:
                return {"kind": "error", "message": "out of steps", "code": "INVALID_OUTPUT"}
        return succeed(task, attempt)

    engine = task_engine(flaky)
    engine.variables = lambda self: {"partial": 42}
    run = make_run()
    Orchestrator(
        run, make_config(reports_dir=str(tmp_path), task_retries=1),
        planner_engine=planner_engine(), task_engine=engine,
        chart_engine=SavingChartEngine, report_engine=SavingReportEngine,
    ).execute()

    assert seen == [{}, {"partial": 42}]
    assert metadata(tmp_path)["tasks"]["by_country"] == "completed"


def test_an_impossible_task_is_not_retried_and_its_dependents_are_skipped(tmp_path):
    def impossible(task, attempt):
        if task.id == "by_country":
            return {"kind": "error", "message": "No data.", "code": "AGENT_ERROR"}
        return succeed(task, attempt)

    run, calls = execute(tmp_path, impossible)

    assert calls == ["data_preparation", "by_country"]
    meta = metadata(tmp_path)
    assert meta["tasks"] == {"data_preparation": "completed", "by_country": "failed", "final_summary": "failed"}
    assert run.status == "completed"


def test_a_run_with_nothing_completed_fails_without_a_report(tmp_path, environment):
    def broken(task, attempt):
        return {"kind": "error", "message": "boom", "code": "RUN_FAILED"}

    run, calls = execute(tmp_path, broken, retries=0)

    assert run.status == "failed"
    assert "nothing to report" in run.error
    assert events(run)[-1] == "run_failed"
    assert environment["consumed"] == []


def test_a_missing_brief_fails_the_run(tmp_path, environment):
    environment["brief"] = None
    run, calls = execute(tmp_path)
    assert run.status == "failed" and "not found" in run.error
    assert calls == []


# ---------------------------------------------------------------------------
# Planning and charts
# ---------------------------------------------------------------------------


def test_the_planner_plan_drives_the_run(tmp_path):
    def planned(brief):
        plan = plan_from_brief(brief, {"orders": ()})
        plan["tasks"].insert(2, {"id": "explore", "name": "Explore", "mission": "Look around.",
                                 "depends_on": [], "outputs": ["oddities"], "exploratory": True})
        return {"kind": "plan", "plan": plan}

    run, calls = execute(tmp_path, planner=planner_engine(planned))

    assert calls == ["data_preparation", "by_country", "explore", "final_summary"]
    ready = [e for e in run.progress.since(0)[0] if e["type"] == "plan_ready"][0]
    assert ready["data"] == {"tasks": 4, "exploratory": 1, "charts": 1, "planner": True, "planner_error": None}
    assert metadata(tmp_path)["planner"] == {"used": True, "error": None, "step_errors": []}


def test_a_failed_planner_falls_back_to_the_code_plan(tmp_path):
    message = "The planner failed: litellm.Timeout: Request timed out"
    failing = planner_engine(lambda brief: {"kind": "error", "message": message, "code": "RUN_FAILED"})
    run, calls = execute(tmp_path, planner=failing)

    assert run.status == "completed"
    assert calls == ["data_preparation", "by_country", "final_summary"]
    ready = [e for e in run.progress.since(0)[0] if e["type"] == "plan_ready"][0]
    assert ready["data"]["planner"] is False
    # Why it failed is kept, not only that it did.
    assert ready["data"]["planner_error"] == message
    assert metadata(tmp_path)["planner"] == {"used": False, "error": message, "step_errors": []}


def test_the_errors_of_a_failed_task_steps_are_kept_in_the_state(tmp_path):
    def failing(task, attempt):
        if task.id == "data_preparation":
            for step in range(1, 5):
                task_step = _Step(step, "x = 1", error=RuntimeError(f"QUERY_TIMEOUT on step {step}\nselect ..."))
                failing.callbacks[task.id](task_step, agent=_Agent())
            return {"kind": "error", "message": "out of steps", "code": "INVALID_OUTPUT"}
        return succeed(task, attempt)

    engine = task_engine(failing)
    failing.callbacks = {}
    original_init = engine.__init__

    def capture(self, *, task, on_step, **kwargs):
        original_init(self, task=task, on_step=on_step, **kwargs)
        failing.callbacks[task.id] = on_step

    engine.__init__ = capture
    run = make_run()
    Orchestrator(
        run, make_config(reports_dir=str(tmp_path), task_retries=1),
        planner_engine=planner_engine(), task_engine=engine,
        chart_engine=SavingChartEngine, report_engine=SavingReportEngine,
    ).execute()

    state = json.loads((tmp_path / "r1" / "sub_task_state.json").read_text())
    errors = state["tasks"]["data_preparation"]["step_errors"]
    # Two attempts of four errors each: the last five are kept, first line only.
    assert len(errors) == 5
    assert errors[0] == "attempt 1, step 4: RuntimeError: QUERY_TIMEOUT on step 4"
    assert errors[-1] == "attempt 2, step 4: RuntimeError: QUERY_TIMEOUT on step 4"
    assert all("select" not in e for e in errors)


@pytest.mark.parametrize(
    "brief",
    [{"goal": "Why did margins fall?"}, {"fasi": [{"titolo": "Conta", "obiettivo": "Contare gli ordini"}]}, "just text"],
)
def test_a_brief_of_any_shape_gets_analysed(tmp_path, environment, brief):
    environment["brief"] = brief
    failing = planner_engine(lambda b: {"kind": "error", "message": "x", "code": "RUN_FAILED"})

    run, calls = execute(tmp_path, planner=failing)

    assert run.status == "completed", run.error
    assert calls


def test_a_missing_planner_configuration_fails_the_run(tmp_path):
    missing = planner_engine(lambda b: {"kind": "error", "message": "x", "code": "MISSING_CONFIG"})
    run, calls = execute(tmp_path, planner=missing)
    assert run.status == "failed" and "not configured" in run.error
    assert calls == []


def test_failed_charts_do_not_stop_the_report(tmp_path):
    run, _ = execute(tmp_path, charts=FailingChartEngine)

    assert run.status == "completed"
    assert "charts_failed" in events(run)
    assert metadata(tmp_path)["chart_agent"] == {
        "done": False, "error": "model down", "skipped": [], "step_errors": [],
    }


def test_the_reasons_for_skipped_charts_are_kept(tmp_path):
    class SkippingChartEngine(SavingChartEngine):
        def run(self):
            return {"kind": "charts_done", "saved": 0,
                    "skipped": [{"chart": "revenue", "reason": "no output holds revenue by country"}]}

    execute(tmp_path, charts=SkippingChartEngine)

    assert metadata(tmp_path)["chart_agent"]["skipped"] == [
        {"chart": "revenue", "reason": "no output holds revenue by country"}
    ]


def test_insights_are_reported_as_progress(tmp_path):
    class InsightTaskEngine(task_engine(succeed)):
        def run(self):
            if self.task.id == "by_country":
                self.on_insight("Italy is 80% of revenue.")
            return super().run()

    run = make_run()
    config = make_config(reports_dir=str(tmp_path))
    Orchestrator(run, config, planner_engine=planner_engine(), task_engine=InsightTaskEngine,
                 chart_engine=SavingChartEngine, report_engine=SavingReportEngine).execute()

    insight = [e for e in run.progress.since(0)[0] if e["type"] == "insight"][0]
    assert insight["message"] == "Italy is 80% of revenue." and insight["task_id"] == "by_country"


def test_a_failed_report_agent_falls_back_to_a_plain_report(tmp_path):
    run, _ = execute(tmp_path, report=FailingReportEngine)

    assert run.status == "completed"
    html = (tmp_path / "r1" / "report.html").read_text()
    assert "Revenue by country" in html and "<h2>By country</h2>" in html
    assert metadata(tmp_path)["report_agent"] == {"fallback": True, "error": "model down", "step_errors": []}


def test_an_internal_error_fails_the_run_cleanly(tmp_path):
    def crash(task, attempt):
        raise RuntimeError("bug")

    run, _ = execute(tmp_path, crash)
    assert run.status == "failed"
    assert run.error == "The analysis failed because of an internal error."
    assert metadata(tmp_path)["status"] == "failed"


# ---------------------------------------------------------------------------
# Cancellation and progress
# ---------------------------------------------------------------------------


def test_a_cancelled_run_stops_between_tasks(tmp_path, environment):
    run = make_run()

    def cancel_after_first(task, attempt):
        run.cancel()
        return succeed(task, attempt)

    run, calls = execute(tmp_path, cancel_after_first, run=run)

    assert calls == ["data_preparation"]
    assert run.status == "cancelled"
    assert events(run)[-1] == "run_cancelled"
    assert environment["consumed"] == []


class _Step:
    def __init__(self, number, code, error=None):
        self.step_number, self.code_action, self.error = number, code, error


class _Agent:
    tools = {"load_dataset": None, "sql_stats": None, "final_answer": None}

    def __init__(self):
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True


def test_the_step_callback_names_the_tools_called_and_never_the_code():
    run = make_run()
    callback = orchestrator.step_callback(run, "prep")

    callback(_Step(2, 'df = load_dataset(name="x", query="SELECT secret FROM t")\nfinal_answer({})'), agent=_Agent())

    event = run.progress.since(0)[0][0]
    assert event["type"] == "step" and event["task_id"] == "prep"
    assert event["data"] == {"step": 2, "tools": ["load_dataset"], "error": False}
    assert "secret" not in json.dumps(event)


def test_a_step_error_is_collected_but_stays_out_of_the_event():
    run = make_run()
    errors = []
    callback = orchestrator.step_callback(run, "prep", errors, prefix="attempt 1, ")

    callback(_Step(3, "x = 1", error=ValueError("column \"secret\" does not exist")), agent=_Agent())

    assert errors == ['attempt 1, step 3: ValueError: column "secret" does not exist']
    event = run.progress.since(0)[0][0]
    assert event["data"]["error"] is True
    assert "secret" not in json.dumps(event)


def test_the_step_callback_interrupts_a_cancelled_run():
    run = make_run()
    run.cancel()
    agent = _Agent()
    orchestrator.step_callback(run, None)(_Step(1, ""), agent=agent)
    assert agent.interrupted
    assert run.progress.since(0)[0] == []
