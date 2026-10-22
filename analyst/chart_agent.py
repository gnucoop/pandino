"""
chart_agent.py
--------------

ChartAgentEngine: one agent run that draws the report's charts.

It runs after the tasks, so every result exists: its namespace holds
`outputs` (every task's outputs, at full precision) and `datasets`, and it
builds each Chart.js configuration in code from them, so no number goes
through the model. The charts to draw come from the plan: the brief's own
requests, or the ones the planner designed when the brief named none.

Its final answer is guarded: every planned chart must be covered, either by a
chart or table it saved, or skipped with a reason.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from smolagents import CodeAgent, LiteLLMModel

from analyst.model_options import AnalystModelOptions
from analyst.prompts import (
    CODE_STEP_RULE,
    ANALYST_CHART_SYSTEM_DEFAULT,
    ANALYST_CHART_SYSTEM_TITLE,
    ANALYST_CHART_TURN_DEFAULT,
    ANALYST_CHART_TURN_TITLE,
    ANALYST_TASK_SQL_ADDENDUM_DEFAULT,
    ANALYST_TASK_SQL_ADDENDUM_TITLE,
)
from analyst.sandbox import AUTHORIZED_IMPORTS, GuardedPythonExecutor
from analyst.sql_prompt import build_scoped_sql_addendum, scope_relations
from analyst.state import COMPLETED, RunState
from analyst.tools.load_dataset_tool import LoadDatasetTool
from analyst.tools.save_chart_tool import SaveChartTool
from analyst.tools.save_table_tool import SaveTableTool
from analyst.tools.sql_engine_tool import AnalystSqlEngineTool
from analyst.values import describe, preview
from config import AnalystConfig
from datachat.final_answer_contract import coerce_final_payload, validate_text_or_message
from datachat.smolagents_base import SmolagentsEngineBase
from datachat.sql_datasource import SqlDatasource
from infrastructure.prompt_utils import load_prompt, render_prompt

# The owner recorded on the charts and tables this agent saves.
CHART_TASK_ID = "charts"

# One output as the task describes it: enough to know its shape.
_OUTPUT_PREVIEW_CHARS = 200


def _validate_charts_done(payload: dict[str, Any]) -> Optional[str]:
    return None if isinstance(payload.get("coverage"), list) else "MISSING_COVERAGE"


_CHART_KIND_VALIDATORS = {
    "charts_done": _validate_charts_done,
    "error": validate_text_or_message,
}


def _contract_failure_message(reason: str) -> str:
    return (
        f"The final answer does not satisfy the output contract ({reason}). Call final_answer "
        'with exactly {"kind":"charts_done","coverage":[{"chart":"<request id>",'
        '"saved":"<name>"|null,"reason":"..."}]}, with no wrapper and no surrounding prose.'
    )


@dataclass
class ChartAgentEngine(AnalystModelOptions, SmolagentsEngineBase):
    config: AnalystConfig
    state: RunState
    on_step: Optional[Callable[..., None]] = None

    token_usage: dict[str, int] = field(default_factory=dict, init=False)

    # --- hooks ---

    def _init_config(self) -> None:
        self._provider = self.config.provider
        self._configured_model = self.config.model
        self._max_steps = self.config.chart_max_steps

    def _build_instructions(self, datasource: Optional[SqlDatasource] = None) -> str:
        template = load_prompt(ANALYST_CHART_SYSTEM_TITLE, default_text=ANALYST_CHART_SYSTEM_DEFAULT)
        instructions = render_prompt(
            template, language=self.state.language, max_charts=self.config.max_charts
        )
        if datasource is None:
            return instructions
        result = build_scoped_sql_addendum(
            datasource,
            template=load_prompt(
                ANALYST_TASK_SQL_ADDENDUM_TITLE, default_text=ANALYST_TASK_SQL_ADDENDUM_DEFAULT
            ),
            relations=scope_relations(self.state.plan),
            max_rows=self.config.dataset_max_rows,
        )
        if result is None:
            return instructions
        self._sql_ready = True
        self._sql_identifiers = result.identifiers
        return instructions + "\n\n" + result.text

    def _extra_tools(self) -> list[Any]:
        cfg = self.config
        tools: list[Any] = [
            SaveChartTool(self.state, CHART_TASK_ID, max_charts=cfg.max_charts, max_points=cfg.chart_max_points),
            SaveTableTool(self.state, CHART_TASK_ID),
        ]
        if self._sql_datasource is not None and self._sql_ready:
            tools.append(
                LoadDatasetTool(
                    self._sql_datasource,
                    self.state.datasets,
                    max_rows=cfg.dataset_max_rows,
                    identifiers=self._sql_identifiers,
                )
            )
        return tools

    def _sql_tools(self, datasource: Optional[SqlDatasource] = None) -> list[Any]:
        if datasource is None or not self._sql_ready:
            return []
        return [AnalystSqlEngineTool(datasource, max_rows=self.config.sql_max_rows, identifiers=self._sql_identifiers)]

    def _build_agent(self, model: LiteLLMModel, instructions: str) -> Optional[CodeAgent]:
        self._final_answer_checks_supported = True
        return CodeAgent(
            tools=[*self._extra_tools(), *self._sql_tools(self._sql_datasource)],
            model=model,
            instructions=instructions,
            max_steps=self._max_steps,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            executor=GuardedPythonExecutor(),
            step_callbacks=[self.on_step] if self.on_step else None,
            final_answer_checks=[self._check_final_answer],
        )

    # --- final answer ---

    def _coverage_problem(self, coverage: list[Any]) -> Optional[str]:
        """Why the coverage does not account for every planned chart, or None."""
        listed: dict[str, Any] = {}
        for item in coverage:
            if isinstance(item, dict) and item.get("chart") is not None:
                listed[str(item["chart"])] = item

        problems = []
        for chart in self.state.charts_requested:
            item = listed.get(str(chart["id"]))
            if item is None:
                problems.append(f"chart {chart['id']} is not in the coverage")
                continue
            saved = item.get("saved")
            if saved:
                if saved not in self.state.charts and saved not in self.state.tables:
                    problems.append(f"chart {chart['id']}: nothing was saved as {saved!r}")
            elif not str(item.get("reason") or "").strip():
                problems.append(f"chart {chart['id']}: not saved and no reason given")
        if not problems:
            return None
        return (
            "The coverage is not complete: " + "; ".join(problems) + ". Draw the missing charts "
            "(or skip them with a reason) and answer again."
        )

    def _evaluate(self, output: Any) -> Tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
        payload, passed, kind, reason = coerce_final_payload(output, _CHART_KIND_VALIDATORS)
        if not passed or payload is None:
            return payload, kind, _contract_failure_message(reason)
        if kind == "charts_done":
            return payload, kind, self._coverage_problem(payload["coverage"])
        return payload, kind, None

    def _check_final_answer(self, *args: Any, **kwargs: Any) -> bool:
        candidate = args[0] if args else (
            kwargs.get("final_answer") or kwargs.get("answer") or kwargs.get("output")
        )
        _, _, problem = self._evaluate(candidate)
        if problem is not None:
            raise ValueError(problem)
        return True

    # --- run ---

    def _outputs(self) -> dict[str, dict[str, Any]]:
        return {t.id: t.outputs for t in self.state.ordered_tasks() if t.status == COMPLETED}

    def render_task(self) -> str:
        charts = [f"- {c['id']}: {c['description']}" for c in self.state.charts_requested]
        outputs = [
            f'- outputs["{task_id}"]["{key}"]: {describe(value)} = {preview(value, _OUTPUT_PREVIEW_CHARS)}'
            for task_id, task_outputs in self._outputs().items()
            for key, value in task_outputs.items()
        ]
        datasets = [f'- datasets["{name}"]: {describe(frame)}' for name, frame in self.state.datasets.items()]
        template = load_prompt(ANALYST_CHART_TURN_TITLE, default_text=ANALYST_CHART_TURN_DEFAULT)
        return render_prompt(
            template,
            charts="\n".join(charts) or "- (none)",
            outputs="\n".join(outputs) or "- (none)",
            datasets="\n".join(datasets) or "- (none)",
        ) + "\n\n" + CODE_STEP_RULE

    def run(self) -> dict[str, Any]:
        """
        Draw the charts: {"kind": "charts_done", "saved": n, "skipped": [{"chart", "reason"}]}
        or {"kind": "error", "message", "code"}.
        """
        if self._agent is None:
            return {"kind": "error", "message": "The chart agent is not configured.", "code": "MISSING_CONFIG"}
        if not self.state.charts_requested:
            return {"kind": "charts_done", "saved": 0, "skipped": []}

        self._agent.state.update({
            "outputs": copy.deepcopy(self._outputs()),
            "datasets": self.state.datasets,
        })

        run_result, run_error = self._run_agent(self.render_task())
        if run_result is None:
            return {"kind": "error", "message": f"The chart agent failed: {run_error}", "code": "RUN_FAILED"}

        usage = getattr(run_result, "token_usage", None)
        if usage is not None:
            self.token_usage = {"input": usage.input_tokens, "output": usage.output_tokens}

        payload, kind, problem = self._evaluate(getattr(run_result, "output", None))
        if problem is not None or payload is None:
            return {"kind": "error", "message": problem or "No final answer.", "code": "INVALID_OUTPUT"}
        if kind == "error":
            return {"kind": "error", "message": str(payload.get("message")), "code": "AGENT_ERROR"}
        skipped = [
            {"chart": str(item.get("chart")), "reason": str(item.get("reason") or "")}
            for item in payload["coverage"]
            if isinstance(item, dict) and not item.get("saved")
        ]
        return {"kind": "charts_done", "saved": len(self.state.charts) + len(self.state.tables), "skipped": skipped}
