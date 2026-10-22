"""
orchestrator.py
---------------

run_analysis(): one analysis of one brief, from the stored brief to the report.

1. load the brief (owned by the user), whatever its shape;
2. plan: the PlannerAgentEngine turns it into a plan (analyst/plan.py), probing
   the data and adding exploration tasks. Should it fail, plan_from_brief()
   derives one by code, so no brief stops a run. From here on only the plan is
   read; the brief is context text;
3. run the tasks in order, one fresh TaskAgentEngine each. A task is retried
   (ANALYST_TASK_RETRIES) when its agent failed or left a required output
   unsaved; a task whose dependency failed is skipped, not run blind;
4. draw the charts: the ChartAgentEngine, from the results. Should it fail, the
   report has no charts but is still written;
5. run the ReportAgentEngine on the results. Should it fail, a plain report is
   rendered from the task summaries and insights, so a run that analysed the
   data always ends with a report;
6. write run_metadata.json (status, metrics for the next run's comparison,
   token usage) and mark the brief consumed.

Everything a client sees goes through run.progress; cancellation is checked
between tasks and, through the step callback, between agent steps.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import datachat.schema_snapshot_loader as schema_snapshot_loader
import datachat.sql_datasource as sql_datasource
from analyst.chart_agent import ChartAgentEngine
from analyst.metrics import METADATA_FILE, fill_metrics
from analyst.plan import plan_from_brief
from analyst.planner_agent import PlannerAgentEngine
from analyst.report_agent import ReportAgentEngine
from analyst.report_template import render_report
from analyst.run import (
    CANCELLED,
    CHARTING,
    COMPLETED,
    FAILED,
    PLANNING,
    RUNNING,
    WRITING_REPORT,
    AnalysisRun,
)
from analyst.state import RunState, TaskRecord, write_json
from analyst.state import COMPLETED as TASK_COMPLETED
from analyst.state import FAILED as TASK_FAILED
from analyst.state import RUNNING as TASK_RUNNING
from analyst.task_agent import TaskAgentEngine
from analyst.tools.save_html_report_tool import REPORT_FILE
from config import AnalystConfig
from services.analysis_brief_service import get_brief, mark_brief_consumed
from utils.logging_config import reset_request_context, set_request_context

logger = logging.getLogger(__name__)

# Errors that retrying cannot fix: the agent judged the task impossible, or
# nothing could be built at all.
_NO_RETRY_CODES = ("AGENT_ERROR", "MISSING_CONFIG")

# Step errors kept per task (and for the planner), each cut to its first line.
_MAX_STEP_ERRORS = 5
_STEP_ERROR_CHARS = 300

PLAN_FILE = "plan.json"


class AnalysisAborted(Exception):
    """The run cannot go on; the message is shown to the user."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def step_callback(
    run: AnalysisRun,
    task_id: Optional[str],
    errors: Optional[list[str]] = None,
    prefix: str = "",
) -> Callable[..., None]:
    """
    Report each agent step as a progress event, and stop the agent once the run
    is cancelled. The event names the tools the step's code called, never its
    data or SQL.

    The error of a failed step goes to `errors` instead (the last
    _MAX_STEP_ERRORS, persisted with the run state): it may quote SQL or data,
    so it stays out of the events, but without it a failed run cannot be
    explained.
    """

    def on_step(step: Any, agent: Any = None) -> None:
        if run.cancelled and agent is not None:
            agent.interrupt()
            return
        code = getattr(step, "code_action", None) or ""
        tools = sorted(
            name
            for name in getattr(agent, "tools", {}) or {}
            if name != "final_answer" and re.search(rf"\b{re.escape(name)}\s*\(", code)
        )
        number = getattr(step, "step_number", None)
        error = getattr(step, "error", None)
        if error is not None and errors is not None:
            lines = str(error).strip().splitlines() or [""]
            errors.append(f"{prefix}step {number}: {type(error).__name__}: {lines[0][:_STEP_ERROR_CHARS]}")
            del errors[:-_MAX_STEP_ERRORS]
        run.progress.emit(
            "step",
            f"Step {number}: {', '.join(tools) if tools else 'computing'}",
            task_id=task_id,
            data={"step": number, "tools": tools, "error": error is not None},
        )

    return on_step


class Orchestrator:
    """Runs one AnalysisRun. Engine factories are injectable for tests."""

    def __init__(
        self,
        run: AnalysisRun,
        config: AnalystConfig,
        *,
        planner_engine: Callable[..., Any] = PlannerAgentEngine,
        task_engine: Callable[..., Any] = TaskAgentEngine,
        chart_engine: Callable[..., Any] = ChartAgentEngine,
        report_engine: Callable[..., Any] = ReportAgentEngine,
    ) -> None:
        self.run = run
        self.config = config
        self.planner_engine = planner_engine
        self.task_engine = task_engine
        self.chart_engine = chart_engine
        self.report_engine = report_engine
        self.state: Optional[RunState] = None
        self.tokens: dict[str, dict[str, int]] = {}
        # What the planner, chart and report agents did, for the metadata.
        self.planner: Optional[dict[str, Any]] = None
        self.chart_agent: Optional[dict[str, Any]] = None
        self.report_agent: Optional[dict[str, Any]] = None
        self.started_at = _now()

    # --- entry point ---

    def execute(self) -> None:
        run = self.run
        context = set_request_context(request_id=run.run_id)
        try:
            run.status = PLANNING
            run.progress.emit("run_started", "Analysis started")
            logger.info("event=analyst_run_start run_id=%s brief_id=%s", run.run_id, run.brief_id)

            self.state = self._plan(self._load_brief())
            run.tasks_total = len(self.state.tasks)
            run.status = RUNNING
            self._write_metadata(RUNNING)

            for task in self.state.ordered_tasks():
                self._check_cancelled()
                self._run_task(task)
                run.tasks_done += 1
                self.state.persist()

            completed = [t for t in self.state.ordered_tasks() if t.status == TASK_COMPLETED]
            if not completed:
                raise AnalysisAborted("No task could be completed, so there is nothing to report.")

            self._check_cancelled()
            self._draw_charts()
            self._check_cancelled()
            self._write_report()

            self._write_metadata(COMPLETED)
            self._mark_consumed()
            run.report_ready = True
            run.current_task = None
            run.status = COMPLETED
            run.progress.emit("run_completed", "The report is ready")
            logger.info("event=analyst_run_end run_id=%s status=completed", run.run_id)

        except AnalysisAborted as e:
            self._finish(CANCELLED if run.cancelled else FAILED, str(e))
        except Exception:
            logger.exception("event=analyst_run_failed run_id=%s", run.run_id)
            self._finish(FAILED, "The analysis failed because of an internal error.")
        finally:
            reset_request_context(context)

    # --- phases ---

    def _load_brief(self) -> dict[str, Any]:
        run = self.run
        row = get_brief(run.brief_id, run.user_email)
        if row is None:
            raise AnalysisAborted(f"Analysis brief {run.brief_id} was not found.")
        brief = row["brief"]
        # Any shape is accepted; only something that is not a JSON object is wrapped.
        return brief if isinstance(brief, dict) else {"brief": brief}

    def _plan(self, brief: dict[str, Any]) -> RunState:
        """The plan, by the planner or, failing that, by code; then the run state built on it."""
        run = self.run
        datasource = sql_datasource.get_datasource()
        snapshot = schema_snapshot_loader.get_schema_snapshot(datasource) if datasource else None
        if snapshot is None:
            raise AnalysisAborted("The SQL database is not available.")

        run.progress.emit("plan_started", "Planning the analysis")
        step_errors: list[str] = []
        engine = self.planner_engine(
            api_key=run.api_key,
            user_name=run.user_email,
            config=self.config,
            brief=brief,
            on_step=step_callback(run, None, step_errors),
        )
        result = engine.run()
        self._count_tokens("plan", engine)
        self._check_cancelled()
        planner_error = None if result["kind"] == "plan" else result.get("message")
        self.planner = {"used": result["kind"] == "plan", "error": planner_error, "step_errors": step_errors}

        if result["kind"] == "plan":
            plan = result["plan"]
        else:
            if result.get("code") == "MISSING_CONFIG":
                raise AnalysisAborted("The analyst is not configured: check ANALYST_* and the SQL datasource.")
            logger.warning("event=analyst_plan_fallback run_id=%s code=%s", run.run_id, result.get("code"))
            plan = plan_from_brief(brief, schema_snapshot_loader.build_identifier_index(snapshot))

        state = RunState.from_plan(
            run_id=run.run_id,
            brief_id=run.brief_id,
            brief=brief,
            plan=plan,
            run_dir=Path(self.config.reports_dir) / run.run_id,
        )
        write_json(state.run_dir / PLAN_FILE, plan)
        exploratory = sum(1 for t in state.ordered_tasks() if t.exploratory)
        run.progress.emit(
            "plan_ready",
            f"{len(state.tasks)} tasks ({exploratory} exploratory), {len(state.charts_requested)} charts",
            data={
                "tasks": len(state.tasks),
                "exploratory": exploratory,
                "charts": len(state.charts_requested),
                "planner": result["kind"] == "plan",
                "planner_error": planner_error,
            },
        )
        return state

    def _run_task(self, task: TaskRecord) -> None:
        run = self.run
        assert self.state is not None
        run.current_task = task.id

        failed_dependencies = [
            dep for dep in task.depends_on if self.state.tasks[dep].status == TASK_FAILED
        ]
        if failed_dependencies:
            task.status = TASK_FAILED
            task.error = f"Skipped: it depends on {', '.join(failed_dependencies)}, which failed."
            run.progress.emit("task_failed", task.error, task_id=task.id)
            return

        # Exploratory tasks are the planner's extras: a failed one costs only its
        # own insights, not worth a whole new attempt.
        retries = 0 if task.exploratory else self.config.task_retries
        for attempt in range(1 + retries):
            self._check_cancelled()
            task.attempts = attempt + 1
            task.status = TASK_RUNNING
            if attempt == 0:
                run.progress.emit("task_started", task.name, task_id=task.id)
            else:
                run.progress.emit("task_retry", f"{task.name}: attempt {attempt + 1}", task_id=task.id)

            engine = self.task_engine(
                api_key=run.api_key,
                user_name=run.user_email,
                config=self.config,
                state=self.state,
                task=task,
                on_step=step_callback(run, task.id, task.step_errors, prefix=f"attempt {attempt + 1}, "),
                on_insight=lambda text, task_id=task.id: run.progress.emit("insight", text, task_id=task_id),
            )
            result = engine.run()
            self._count_tokens(task.id, engine)
            self._check_cancelled()

            if result["kind"] == "task_done":
                task.status = TASK_COMPLETED
                task.summary = result["summary"]
                task.error = None
                task.carried = {}
                run.progress.emit("task_completed", task.summary, task_id=task.id)
                logger.info("event=analyst_task_end run_id=%s task=%s status=completed attempts=%s",
                            run.run_id, task.id, task.attempts)
                return

            task.error = result.get("message") or "The task failed."
            # The next attempt resumes from what this one computed.
            task.carried = engine.variables()
            if result.get("code") == "MISSING_CONFIG":
                raise AnalysisAborted("The analyst is not configured: check ANALYST_* and the SQL datasource.")
            if result.get("code") in _NO_RETRY_CODES:
                break

        task.status = TASK_FAILED
        task.carried = {}
        run.progress.emit("task_failed", task.error or "The task failed.", task_id=task.id)
        logger.info("event=analyst_task_end run_id=%s task=%s status=failed attempts=%s",
                    self.run.run_id, task.id, task.attempts)

    def _draw_charts(self) -> None:
        """Draw the planned charts. A failure leaves the report without charts, not the run failed."""
        run = self.run
        assert self.state is not None
        run.status = CHARTING
        run.current_task = None
        run.progress.emit("charts_started", f"Drawing {len(self.state.charts_requested)} charts")

        step_errors: list[str] = []
        engine = self.chart_engine(
            api_key=run.api_key,
            user_name=run.user_email,
            config=self.config,
            state=self.state,
            on_step=step_callback(run, None, step_errors),
        )
        result = engine.run()
        self._count_tokens("charts", engine)
        self._check_cancelled()
        done = result["kind"] == "charts_done"
        self.chart_agent = {
            "done": done,
            "error": None if done else result.get("message"),
            # Why the planned charts that are missing were not drawn.
            "skipped": (result.get("skipped") or []) if done else [],
            "step_errors": step_errors,
        }

        if result["kind"] == "charts_done":
            run.progress.emit(
                "charts_saved",
                f"{len(self.state.charts)} charts and {len(self.state.tables)} tables saved",
                data={"charts": len(self.state.charts), "tables": len(self.state.tables),
                      "skipped": result.get("skipped") or []},
            )
        else:
            logger.warning("event=analyst_charts_failed run_id=%s code=%s", run.run_id, result.get("code"))
            run.progress.emit("charts_failed", result.get("message") or "The charts could not be drawn.")
        self.state.persist()

    def _write_report(self) -> None:
        run = self.run
        assert self.state is not None
        run.status = WRITING_REPORT
        run.current_task = None
        run.progress.emit("report_started", "Writing the report")

        step_errors: list[str] = []
        engine = self.report_engine(
            api_key=run.api_key,
            user_name=run.user_email,
            config=self.config,
            state=self.state,
            on_step=step_callback(run, None, step_errors),
        )
        result = engine.run()
        self._count_tokens("report", engine)
        self._check_cancelled()

        fallback = result["kind"] != "report_saved" or self.state.report_path is None
        self.report_agent = {
            "fallback": fallback,
            "error": (result.get("message") or "No report was saved.") if fallback else None,
            "step_errors": step_errors,
        }
        if fallback:
            logger.warning("event=analyst_report_fallback run_id=%s code=%s", run.run_id, result.get("code"))
            self._write_fallback_report()

        run.progress.emit("report_saved", "Report saved")

    def _write_fallback_report(self) -> None:
        """A plain report from the task summaries, insights, charts and tables: no model involved."""
        state = self.state
        assert state is not None
        if state.metrics is None:
            state.metrics = self._metrics()
        sections = [
            {
                "title": task.name,
                "content": task.summary or task.error or "",
                "style": "danger" if task.status == TASK_FAILED else "",
                "charts": [n for n, c in state.charts.items() if c["task_id"] == task.id],
                "tables": [n for n, t in state.tables.items() if t["task_id"] == task.id],
            }
            for task in state.ordered_tasks()
        ]
        if state.insights:
            sections.append({
                "title": "Additional findings",
                "content": "\n".join(f"- {i['text']}" for i in state.insights),
                "style": "info",
                "charts": [],
                "tables": [],
            })
        state.html = render_report(
            state,
            title=None,
            subtitle=None,
            sections=sections,
            footer=f"Maui · analysis brief #{state.brief_id} · run {state.run_id}",
        )
        state.report_path = state.run_dir / REPORT_FILE
        state.report_path.parent.mkdir(parents=True, exist_ok=True)
        state.report_path.write_text(state.html, encoding="utf-8")

    # --- helpers ---

    def _check_cancelled(self) -> None:
        if self.run.cancelled:
            raise AnalysisAborted("The analysis was cancelled.")

    def _count_tokens(self, label: str, engine: Any) -> None:
        usage = getattr(engine, "token_usage", None) or {}
        total = self.tokens.setdefault(label, {"input": 0, "output": 0})
        total["input"] += int(usage.get("input", 0))
        total["output"] += int(usage.get("output", 0))

    def _metrics(self) -> dict[str, Any]:
        assert self.state is not None
        if self.state.metrics is not None:
            return self.state.metrics
        filled = fill_metrics(
            self.state.metrics_config, self.state.merged_outputs(), datasets=self.state.datasets
        )
        filled.pop("missing")
        return filled

    def _write_metadata(self, status: str, error: Optional[str] = None) -> None:
        state = self.state
        if state is None:
            return
        finished = status in (COMPLETED, FAILED, CANCELLED)
        write_json(state.run_dir / METADATA_FILE, {
            "run_id": state.run_id,
            "brief_id": state.brief_id,
            "title": state.title,
            "user_email": self.run.user_email,
            "status": status,
            "error": error,
            "started_at": self.started_at,
            "finished_at": _now() if finished else None,
            "metrics": self._metrics() if status == COMPLETED else None,
            "tasks": {t.id: t.status for t in state.ordered_tasks()},
            "planner": self.planner,
            "chart_agent": self.chart_agent,
            "report_agent": self.report_agent,
            "tokens": self.tokens,
            "models": {
                "plan": self.config.report_model,
                "tasks": self.config.model,
                "charts": self.config.model,
                "report": self.config.report_model,
            },
            "report": REPORT_FILE if state.report_path else None,
        })

    def _mark_consumed(self) -> None:
        try:
            mark_brief_consumed(self.run.brief_id)
        except RuntimeError:
            # The report exists: a stale brief status is not worth failing the run.
            logger.exception("event=analyst_brief_status_failed brief_id=%s", self.run.brief_id)

    def _finish(self, status: str, message: str) -> None:
        run = self.run
        run.status = status
        run.error = message
        run.current_task = None
        try:
            self._write_metadata(status, message)
        except OSError:
            logger.exception("event=analyst_metadata_failed run_id=%s", run.run_id)
        run.progress.emit("run_cancelled" if status == CANCELLED else "run_failed", message)
        logger.info("event=analyst_run_end run_id=%s status=%s", run.run_id, status)


def run_analysis(run: AnalysisRun, config: AnalystConfig) -> None:
    """Run the analysis to its end. Never raises: the outcome is on `run`."""
    Orchestrator(run, config).execute()
