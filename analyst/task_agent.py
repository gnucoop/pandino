"""
task_agent.py
-------------

TaskAgentEngine: one agent run executing one task of the plan.

A smolagents CodeAgent built on SmolagentsEngineBase, like the interviewer, and
built fresh for every task: its context holds that task alone, never the
history of the previous ones. What it gets:

- instructions: the analyst persona, plus the schema of the plan's relations
  (analyst.sql_prompt) instead of the whole database;
- the task: the brief as context text (plan.brief_context_text, whatever its
  shape), its own mission and output keys, and a description - not the content
  - of its inputs and of the datasets loaded so far;
- its namespace: `datasets` (shared by the run) and `inputs` (the outputs of
  the tasks it depends on), placed in agent.state rather than passed as
  run(additional_args=...), which would print them into the task;
- its tools: the SQL and analysis tools, save_sub_task_output and
  record_insight, bound to its task. Charts are the chart agent's job.

Its code runs in GuardedPythonExecutor. Its final answer is guarded: task_done
is turned down, and the reason replayed to the model, while a required output
is unsaved.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from smolagents import CodeAgent, LiteLLMModel

from analyst.model_options import AnalystModelOptions
from analyst.prompts import (
    CODE_STEP_RULE,
    ANALYST_TASK_SQL_ADDENDUM_DEFAULT,
    ANALYST_TASK_SQL_ADDENDUM_TITLE,
    ANALYST_TASK_SYSTEM_DEFAULT,
    ANALYST_TASK_SYSTEM_TITLE,
    ANALYST_TASK_TURN_DEFAULT,
    ANALYST_TASK_TURN_TITLE,
)
from analyst.sandbox import AUTHORIZED_IMPORTS, GuardedPythonExecutor
from analyst.plan import brief_context_text
from analyst.sql_prompt import build_scoped_sql_addendum, scope_relations
from analyst.state import RunState, TaskRecord
from analyst.tools.detect_anomalies_tool import DetectAnomaliesTool
from analyst.tools.load_dataset_tool import LoadDatasetTool
from analyst.tools.run_forecast_tool import RunForecastTool
from analyst.tools.run_regression_tool import RunRegressionTool
from analyst.tools.record_insight_tool import RecordInsightTool
from analyst.tools.save_sub_task_output_tool import SaveSubTaskOutputTool
from analyst.tools.sql_correlation_tool import SqlCorrelationTool
from analyst.tools.sql_engine_tool import AnalystSqlEngineTool
from analyst.tools.sql_group_stats_tool import SqlGroupStatsTool
from analyst.tools.sql_outliers_tool import SqlOutliersTool
from analyst.tools.sql_stats_tool import SqlStatsTool
from analyst.values import describe, preview
from config import AnalystConfig
from datachat.final_answer_contract import coerce_final_payload, validate_text_or_message
from datachat.smolagents_base import SmolagentsEngineBase
from datachat.sql_datasource import SqlDatasource
from infrastructure.prompt_utils import load_prompt, render_prompt

# One input value as the task describes it: enough to know what it holds.
_INPUT_PREVIEW_CHARS = 300

# In the task text, built in code so that a prompt stored in the database
# cannot drop it: models otherwise retype printed numbers into long literals,
# which are both a transcription risk and the code they most often mangle.
_OUTPUTS_FROM_VARIABLES = (
    "Variables persist between steps: build each output from the variables you computed "
    '(e.g. quality_report = {"rows": n_rows, ...}). Never copy numbers from printed '
    "results into code or strings."
)

# How many steps before the cap the agent is told to save what it computed.
_BUDGET_WARNING_STEPS = 3

# What a failed attempt hands to the next one: the data it computed, never
# tools, modules or functions, and at most this many names.
_CARRIED_TYPES = (pd.DataFrame, pd.Series, dict, list, tuple, str, int, float, np.generic)
_NOT_CARRIED = frozenset({"datasets", "inputs"})
_MAX_CARRIED = 30
_CARRIED_PREVIEW_CHARS = 80

StepCallback = Callable[..., None]


def _validate_task_done(payload: dict[str, Any]) -> Optional[str]:
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return "MISSING_SUMMARY"
    return None


_TASK_KIND_VALIDATORS = {
    "task_done": _validate_task_done,
    "error": validate_text_or_message,
}


def _contract_failure_message(reason: str) -> str:
    return (
        f"The final answer does not satisfy the output contract ({reason}). Call "
        'final_answer with exactly {"kind":"task_done","summary":"<2-3 sentences>"} '
        '(or {"kind":"error","message":"..."} if the sub-task is impossible), with no '
        "wrapper and no surrounding prose."
    )


@dataclass
class TaskAgentEngine(AnalystModelOptions, SmolagentsEngineBase):
    config: AnalystConfig
    state: RunState
    task: TaskRecord
    # Called as on_step(step, agent=agent) after every agent step (progress, cancel).
    on_step: Optional[StepCallback] = None
    # Called with the text of every insight recorded (progress).
    on_insight: Optional[Callable[[str], None]] = None

    # Token usage of the last run: {"input": n, "output": n}.
    token_usage: dict[str, int] = field(default_factory=dict, init=False)

    # --- hooks ---

    def _init_config(self) -> None:
        self._provider = self.config.provider
        self._configured_model = self.config.model
        self._max_steps = self.config.task_max_steps

    def _build_instructions(self, datasource: Optional[SqlDatasource] = None) -> str:
        template = load_prompt(ANALYST_TASK_SYSTEM_TITLE, default_text=ANALYST_TASK_SYSTEM_DEFAULT)
        instructions = render_prompt(template, language=self.state.language)

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

    def _requirements_met(self) -> bool:
        # The sub-tasks analyse the database: nothing to do without it.
        return self._sql_ready

    def _extra_tools(self) -> list[Any]:
        ds = self._sql_datasource
        assert ds is not None
        ids = self._sql_identifiers
        datasets = self.state.datasets
        cfg = self.config
        return [
            LoadDatasetTool(ds, datasets, max_rows=cfg.dataset_max_rows, identifiers=ids),
            SqlStatsTool(ds, ids),
            SqlGroupStatsTool(ds, max_rows=cfg.sql_max_rows, identifiers=ids),
            SqlCorrelationTool(ds, ids),
            SqlOutliersTool(ds, max_rows=cfg.sql_max_rows, identifiers=ids),
            DetectAnomaliesTool(datasets),
            RunRegressionTool(datasets),
            RunForecastTool(datasets),
            SaveSubTaskOutputTool(
                self.task, datasets, max_chars=cfg.output_max_chars, max_rows=cfg.dataset_max_rows
            ),
            RecordInsightTool(self.state, self.task.id, max_insights=cfg.max_insights, notify=self.on_insight),
        ]

    def _sql_tools(self, datasource: Optional[SqlDatasource] = None) -> list[Any]:
        if datasource is None or not self._sql_ready:
            return []
        return [
            AnalystSqlEngineTool(
                datasource, max_rows=self.config.sql_max_rows, identifiers=self._sql_identifiers
            )
        ]

    def _build_agent(self, model: LiteLLMModel, instructions: str) -> Optional[CodeAgent]:
        tools = [*self._extra_tools(), *self._sql_tools(self._sql_datasource)]
        self._final_answer_checks_supported = True
        return CodeAgent(
            tools=tools,
            model=model,
            instructions=instructions,
            max_steps=self._max_steps,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            executor=GuardedPythonExecutor(),
            step_callbacks=[self._on_step],
            final_answer_checks=[self._check_final_answer],
        )

    def _on_step(self, step: Any, agent: Any = None) -> None:
        """
        A few steps before the cap, tell the agent to save what it computed:
        otherwise its work stays in variables and a run that reaches the cap
        leaves nothing. The note is appended to the step's observation, which
        the model reads at its next step. Then report progress as usual.
        """
        number = getattr(step, "step_number", None)
        max_steps = getattr(agent, "max_steps", None)
        if isinstance(number, int) and isinstance(max_steps, int) and max_steps - number == _BUDGET_WARNING_STEPS:
            missing = self.task.missing_outputs()
            if missing:
                note = (
                    f"{_BUDGET_WARNING_STEPS} steps left: save every required output you have "
                    f"computed now with save_sub_task_output (missing: {', '.join(missing)}), "
                    "then call final_answer."
                )
                step.observations = f"{step.observations}\n\n{note}" if step.observations else note
        if self.on_step is not None:
            self.on_step(step, agent=agent)

    def variables(self) -> dict[str, Any]:
        """
        The data the agent computed in its namespace, for a retry to resume from:
        a new attempt starts with a new executor, and would otherwise recompute it.
        """
        state = getattr(getattr(self._agent, "python_executor", None), "state", None) or {}
        tools = set(getattr(self._agent, "tools", None) or {})
        kept: dict[str, Any] = {}
        for name, value in state.items():
            if len(kept) >= _MAX_CARRIED:
                break
            if name.startswith("_") or name in _NOT_CARRIED or name in tools:
                continue
            if isinstance(value, _CARRIED_TYPES):
                kept[name] = value
        return kept

    # --- final answer ---

    def _evaluate(self, output: Any) -> Tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
        """(payload, kind, problem); problem is None when the answer is acceptable."""
        payload, passed, kind, reason = coerce_final_payload(output, _TASK_KIND_VALIDATORS)
        if not passed or payload is None:
            return payload, kind, _contract_failure_message(reason)
        if kind == "task_done":
            missing = self.task.missing_outputs()
            if missing:
                return payload, kind, (
                    f"Required outputs not saved yet: {', '.join(missing)}. Compute them, "
                    "save each with save_sub_task_output, then answer again."
                )
        return payload, kind, None

    def _check_final_answer(self, *args: Any, **kwargs: Any) -> bool:
        """Raising is the only way to tell the model why its answer was turned down."""
        candidate = args[0] if args else (
            kwargs.get("final_answer") or kwargs.get("answer") or kwargs.get("output")
        )
        _, _, problem = self._evaluate(candidate)
        if problem is not None:
            raise ValueError(problem)
        return True

    # --- run ---

    def render_task(self) -> str:
        task = self.task
        position = [t.id for t in self.state.ordered_tasks()].index(task.id) + 1
        lines = [
            f"{task.name} [{task.id}], task {position} of {len(self.state.tasks)}"
            + (" (exploratory: find what is notable, record it with record_insight)" if task.exploratory else ""),
            task.mission,
            f"Required outputs: {', '.join(task.required)}",
            _OUTPUTS_FROM_VARIABLES,
        ]
        # A retry: say what the failed attempt left, so that it is not redone.
        if task.error:
            lines.append(f"A previous attempt failed: {task.error}")
        if task.outputs:
            lines.append(f"Already saved (keep unless wrong): {', '.join(task.outputs)}")
        if task.carried:
            lines.append("Variables left by the previous attempt (still defined; save them if they are right):")
            lines.extend(
                f"- {name}: {describe(value)}"
                + ("" if isinstance(value, pd.DataFrame) else f" = {preview(value, _CARRIED_PREVIEW_CHARS)}")
                for name, value in task.carried.items()
            )
        sub_task = "\n".join(lines)

        inputs = [
            f'- inputs["{dep}"]["{key}"]: {describe(value)} = {preview(value, _INPUT_PREVIEW_CHARS)}'
            for dep, outputs in self.state.dependency_outputs(task.id).items()
            for key, value in outputs.items()
        ]
        datasets = [f'- datasets["{name}"]: {describe(frame)}' for name, frame in self.state.datasets.items()]

        template = load_prompt(ANALYST_TASK_TURN_TITLE, default_text=ANALYST_TASK_TURN_DEFAULT)
        return render_prompt(
            template,
            brief_context=brief_context_text(self.state.brief),
            sub_task=sub_task,
            inputs="\n".join(inputs) or "- (none)",
            datasets="\n".join(datasets) or "- (none)",
        ) + "\n\n" + CODE_STEP_RULE

    def run(self) -> dict[str, Any]:
        """
        Execute the sub-task: {"kind": "task_done", "summary"} or
        {"kind": "error", "message", "code"}. Outputs land in the task record.
        """
        if self._agent is None:
            return {"kind": "error", "message": "The task agent is not configured.", "code": "MISSING_CONFIG"}

        # A copy: the agent may reshape what it is given without touching the
        # outputs of the tasks it read them from.
        self._agent.state.update({
            # What a failed attempt computed (see variables()); none on a first attempt.
            **self.task.carried,
            "datasets": self.state.datasets,
            "inputs": copy.deepcopy(self.state.dependency_outputs(self.task.id)),
        })

        run_result, run_error = self._run_agent(self.render_task())
        if run_result is None:
            return {"kind": "error", "message": f"The task agent failed: {run_error}", "code": "RUN_FAILED"}

        usage = getattr(run_result, "token_usage", None)
        if usage is not None:
            self.token_usage = {"input": usage.input_tokens, "output": usage.output_tokens}

        # Re-checked here: when the agent runs out of steps smolagents produces
        # a final answer the guard never saw.
        payload, kind, problem = self._evaluate(getattr(run_result, "output", None))
        if problem is not None or payload is None:
            return {"kind": "error", "message": problem or "No final answer.", "code": "INVALID_OUTPUT"}
        if kind == "error":
            message = payload.get("message") or payload.get("text")
            return {"kind": "error", "message": str(message), "code": "AGENT_ERROR"}
        return {"kind": "task_done", "summary": str(payload["summary"]).strip()}
