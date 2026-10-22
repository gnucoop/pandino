"""
planner_agent.py
----------------

PlannerAgentEngine: one agent run that turns a brief, whatever its shape, into a plan.

The brief is written by the interviewer's model, so its keys are not
guaranteed; the plan (analyst/plan.py) is. The planner reads the brief as it
is, probes the data with a few read-only statistics calls, and answers with a
plan: the brief's tasks kept and made self-contained, up to
ANALYST_MAX_EXTRA_TASKS exploration tasks of its own, the charts to draw (the
brief's, or designed), the report shape, the language and the tracked metrics.

Its final answer is guarded: validate_plan() turns it down, and the reason is
replayed to the model, until the plan is consistent and keeps every task the
brief defined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from smolagents import CodeAgent, LiteLLMModel

from analyst.plan import brief_task_ids, normalize_plan, validate_plan
from analyst.model_options import AnalystModelOptions
from analyst.prompts import (
    CODE_STEP_RULE,
    ANALYST_PLAN_SYSTEM_DEFAULT,
    ANALYST_PLAN_SYSTEM_TITLE,
    ANALYST_PLAN_TURN_DEFAULT,
    ANALYST_PLAN_TURN_TITLE,
    ANALYST_TASK_SQL_ADDENDUM_DEFAULT,
    ANALYST_TASK_SQL_ADDENDUM_TITLE,
)
from analyst.sandbox import GuardedPythonExecutor
from analyst.sql_prompt import large_relations_note
from analyst.tools.sql_correlation_tool import SqlCorrelationTool
from analyst.tools.sql_engine_tool import AnalystSqlEngineTool
from analyst.tools.sql_group_stats_tool import SqlGroupStatsTool
from analyst.tools.sql_outliers_tool import SqlOutliersTool
from analyst.tools.sql_stats_tool import SqlStatsTool
from config import AnalystConfig
from datachat.final_answer_contract import coerce_final_payload, validate_text_or_message
from datachat.smolagents_base import SmolagentsEngineBase
from datachat.sql_datasource import SqlDatasource
from infrastructure.prompt_utils import load_prompt, render_prompt

# The brief as the planner reads it, compact JSON; a longer one is cut.
_MAX_BRIEF_CHARS = 40000


def _validate_plan_payload(payload: dict[str, Any]) -> Optional[str]:
    return None if isinstance(payload.get("plan"), dict) else "MISSING_PLAN"


_PLAN_KIND_VALIDATORS = {
    "plan": _validate_plan_payload,
    "error": validate_text_or_message,
}


def _contract_failure_message(reason: str) -> str:
    return (
        f"The final answer does not satisfy the output contract ({reason}). Call "
        'final_answer with exactly {"kind":"plan","plan":{...}}, with no wrapper and '
        "no surrounding prose."
    )


def _plan_errors_message(errors: list[str]) -> str:
    return "The plan is not valid yet. Fix every point below and answer again:\n" + "\n".join(
        f"- {error}" for error in errors
    )


@dataclass
class PlannerAgentEngine(AnalystModelOptions, SmolagentsEngineBase):
    config: AnalystConfig
    brief: dict[str, Any]
    on_step: Optional[Callable[..., None]] = None

    token_usage: dict[str, int] = field(default_factory=dict, init=False)

    # --- hooks ---

    def _init_config(self) -> None:
        self._provider = self.config.report_provider
        self._configured_model = self.config.report_model
        self._max_steps = self.config.plan_max_steps

    def _build_instructions(self, datasource: Optional[SqlDatasource] = None) -> str:
        template = load_prompt(ANALYST_PLAN_SYSTEM_TITLE, default_text=ANALYST_PLAN_SYSTEM_DEFAULT)
        instructions = render_prompt(template, max_extra=self.config.max_extra_tasks)
        # The whole schema: the planner chooses the relations the analysis reads.
        addendum = self._sql_addendum(
            datasource,
            lambda: load_prompt(
                ANALYST_TASK_SQL_ADDENDUM_TITLE, default_text=ANALYST_TASK_SQL_ADDENDUM_DEFAULT
            ),
        )
        if addendum is None:
            return instructions
        # Large tables shape the plan: missions should aggregate or sample them.
        note = large_relations_note(datasource, self._sql_identifiers, max_rows=self.config.dataset_max_rows)
        return instructions + "\n\n" + addendum + ("\n\n" + note if note else "")

    def _requirements_met(self) -> bool:
        return self._sql_ready

    def _extra_tools(self) -> list[Any]:
        ds = self._sql_datasource
        assert ds is not None
        ids = self._sql_identifiers
        rows = self.config.sql_max_rows
        return [
            SqlStatsTool(ds, ids),
            SqlGroupStatsTool(ds, max_rows=rows, identifiers=ids),
            SqlCorrelationTool(ds, ids),
            SqlOutliersTool(ds, max_rows=rows, identifiers=ids),
        ]

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
            additional_authorized_imports=["json"],
            executor=GuardedPythonExecutor(),
            step_callbacks=[self.on_step] if self.on_step else None,
            final_answer_checks=[self._check_final_answer],
        )

    # --- final answer ---

    def _evaluate(self, output: Any) -> Tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
        """(plan, kind, problem); problem is None when the plan is acceptable."""
        payload, passed, kind, reason = coerce_final_payload(output, _PLAN_KIND_VALIDATORS)
        if not passed or payload is None:
            return None, kind, _contract_failure_message(reason)
        if kind == "error":
            return payload, kind, None

        plan = normalize_plan(payload["plan"])
        errors = validate_plan(
            plan,
            identifiers=self._sql_identifiers,
            brief_ids=brief_task_ids(self.brief),
            max_extra=self.config.max_extra_tasks,
        )
        if errors:
            return plan, kind, _plan_errors_message(errors)
        return plan, kind, None

    def _check_final_answer(self, *args: Any, **kwargs: Any) -> bool:
        candidate = args[0] if args else (
            kwargs.get("final_answer") or kwargs.get("answer") or kwargs.get("output")
        )
        _, _, problem = self._evaluate(candidate)
        if problem is not None:
            raise ValueError(problem)
        return True

    # --- run ---

    def render_task(self) -> str:
        brief = json.dumps(self.brief, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(brief) > _MAX_BRIEF_CHARS:
            brief = brief[:_MAX_BRIEF_CHARS] + "… [brief truncated]"
        template = load_prompt(ANALYST_PLAN_TURN_TITLE, default_text=ANALYST_PLAN_TURN_DEFAULT)
        return render_prompt(template, brief=brief) + "\n\n" + CODE_STEP_RULE

    def run(self) -> dict[str, Any]:
        """{"kind": "plan", "plan": {...}} (normalized, valid) or {"kind": "error", "message", "code"}."""
        if self._agent is None:
            return {"kind": "error", "message": "The planner is not configured.", "code": "MISSING_CONFIG"}

        run_result, run_error = self._run_agent(self.render_task())
        if run_result is None:
            return {"kind": "error", "message": f"The planner failed: {run_error}", "code": "RUN_FAILED"}

        usage = getattr(run_result, "token_usage", None)
        if usage is not None:
            self.token_usage = {"input": usage.input_tokens, "output": usage.output_tokens}

        plan, kind, problem = self._evaluate(getattr(run_result, "output", None))
        if problem is not None or plan is None:
            return {"kind": "error", "message": problem or "No final answer.", "code": "INVALID_OUTPUT"}
        if kind == "error":
            return {"kind": "error", "message": str(plan.get("message")), "code": "AGENT_ERROR"}
        return {"kind": "plan", "plan": plan}
