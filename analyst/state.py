"""
state.py
--------

RunState: everything an analysis run produces, shared by its agents.

The state lives in memory, owned by the orchestrator, and is built from the
plan (analyst/plan.py), never from the brief's keys: the brief is kept only as
context text. The agents reach the state through their tools:

- task agents save outputs and insights for *their* task (the task id is bound
  into the tools, never guessed from the prompt), and load DataFrames into
  `datasets`, shared by the whole run so a later task can reuse them;
- the chart agent saves charts and tables;
- the report agent reads all of it back, in bounded previews, and turns it
  into the report.

snapshot() is the JSON-compatible part, which the orchestrator stores with the
run (analysis_reports.state), for inspection.

Tasks run in plan order: validate_plan() guarantees that every dependency
comes before the task depending on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import pandas as pd

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


@dataclass
class TaskRecord:
    id: str
    name: str
    order: int
    mission: str
    depends_on: list[str]
    required: list[str]
    # Added by the planner to explore the data beyond what the brief asked.
    exploratory: bool = False

    status: str = PENDING
    outputs: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: Optional[str] = None
    attempts: int = 0
    # The last errors of the agent's steps, across attempts: what explains a failure.
    step_errors: list[str] = field(default_factory=list)
    # The variables a failed attempt computed, handed to the next one. Not persisted.
    carried: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_plan(cls, task: Mapping[str, Any], order: int) -> "TaskRecord":
        return cls(
            id=task["id"],
            name=str(task.get("name") or task["id"]),
            order=order,
            mission=str(task.get("mission") or ""),
            depends_on=list(task.get("depends_on") or []),
            required=list(task.get("outputs") or []),
            exploratory=bool(task.get("exploratory")),
        )

    def missing_outputs(self) -> list[str]:
        return [key for key in self.required if key not in self.outputs]


@dataclass
class RunState:
    run_id: str
    brief_id: int
    brief: dict[str, Any]
    plan: dict[str, Any]

    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    datasets: dict[str, pd.DataFrame] = field(default_factory=dict)
    # name -> {"title", "spec", "task_id"}; spec is a Chart.js configuration.
    charts: dict[str, dict[str, Any]] = field(default_factory=dict)
    # name -> {"title", "columns", "rows", "task_id"}
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    # [{"task_id", "text", "evidence"}], from record_insight.
    insights: list[dict[str, Any]] = field(default_factory=list)
    # {"metrics": {key: number}, "categories": {key: [items]}}, from save_run_metrics.
    metrics: Optional[dict[str, Any]] = None
    # Comparison with the previous completed run of the same brief, from get_delta_section.
    delta: Optional[dict[str, Any]] = None
    # The rendered report, from get_html_template; final once save_html_report
    # marks it saved, and stored with the run by the orchestrator.
    html: Optional[str] = None
    report_saved: bool = False

    @classmethod
    def from_plan(
        cls, *, run_id: str, brief_id: int, brief: dict[str, Any], plan: dict[str, Any]
    ) -> "RunState":
        state = cls(run_id=run_id, brief_id=brief_id, brief=brief, plan=plan)
        state.tasks = {
            t["id"]: TaskRecord.from_plan(t, order) for order, t in enumerate(plan["tasks"], start=1)
        }
        return state

    # --- reading ---

    def ordered_tasks(self) -> list[TaskRecord]:
        return list(self.tasks.values())

    def dependency_outputs(self, task_id: str) -> dict[str, dict[str, Any]]:
        """Outputs of the tasks `task_id` depends on, keyed by task id."""
        return {
            dep: dict(self.tasks[dep].outputs)
            for dep in self.tasks[task_id].depends_on
            if dep in self.tasks
        }

    def merged_outputs(self) -> dict[str, Any]:
        """Every output by key, later tasks winning a key clash."""
        merged: dict[str, Any] = {}
        for task in self.ordered_tasks():
            merged.update(task.outputs)
        return merged

    @property
    def title(self) -> str:
        return str(self.plan.get("title") or "Analysis")

    @property
    def language(self) -> str:
        return str(self.plan.get("language") or "English")

    @property
    def audience(self) -> str:
        return str(self.plan.get("audience") or "")

    @property
    def charts_requested(self) -> list[dict[str, Any]]:
        return list(self.plan.get("charts") or [])

    @property
    def metrics_config(self) -> Optional[dict[str, Any]]:
        config = self.plan.get("metrics_config")
        return config if isinstance(config, dict) else None

    # --- persistence ---

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "brief_id": self.brief_id,
            "tasks": {
                t.id: {
                    "status": t.status,
                    "exploratory": t.exploratory,
                    "attempts": t.attempts,
                    "summary": t.summary,
                    "error": t.error,
                    "step_errors": t.step_errors,
                    "outputs": t.outputs,
                }
                for t in self.ordered_tasks()
            },
            "datasets": {name: list(map(str, df.columns)) for name, df in self.datasets.items()},
            "charts": {name: c["title"] for name, c in self.charts.items()},
            "tables": {name: t["title"] for name, t in self.tables.items()},
            "insights": self.insights,
            "metrics": self.metrics,
        }
