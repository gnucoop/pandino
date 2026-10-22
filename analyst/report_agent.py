"""
report_agent.py
---------------

ReportAgentEngine: the agent run that turns the results into the report.

The tasks and the chart agent have already run (the orchestrator ran them), so
it only reads the results, insights, charts and tables, and writes the report.
What it knows of the brief comes from the plan, never from the brief's keys.
It never queries the database and never handles HTML: it
writes markdown sections and places charts and tables by name, and
get_html_template renders the page.

Its final answer is guarded: report_saved is turned down until the report
file has been written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from smolagents import CodeAgent, LiteLLMModel

from analyst.model_options import AnalystModelOptions
from analyst.prompts import (
    CODE_STEP_RULE,
    ANALYST_REPORT_SYSTEM_DEFAULT,
    ANALYST_REPORT_SYSTEM_TITLE,
    ANALYST_REPORT_TURN_DEFAULT,
    ANALYST_REPORT_TURN_TITLE,
    APPENDIX_TITLES,
)
from analyst.plan import brief_context_text
from analyst.sandbox import GuardedPythonExecutor
from analyst.state import RunState
from analyst.tools.get_all_saved_charts_tool import GetAllSavedChartsTool
from analyst.tools.get_delta_section_tool import GetDeltaSectionTool
from analyst.tools.get_html_template_tool import GetHtmlTemplateTool
from analyst.tools.get_sub_task_status_tool import GetSubTaskStatusTool
from analyst.tools.read_sub_task_state_tool import ReadSubTaskStateTool
from analyst.tools.save_html_report_tool import SaveHtmlReportTool
from analyst.tools.save_run_metrics_tool import SaveRunMetricsTool
from config import AnalystConfig
from datachat.final_answer_contract import coerce_final_payload, validate_text_or_message
from datachat.smolagents_base import SmolagentsEngineBase
from datachat.sql_datasource import SqlDatasource
from infrastructure.prompt_utils import load_prompt, render_prompt

_REPORT_KIND_VALIDATORS = {
    "report_saved": lambda payload: None,
    "error": validate_text_or_message,
}


def _contract_failure_message(reason: str) -> str:
    return (
        f"The final answer does not satisfy the output contract ({reason}). After "
        'save_html_report succeeded, call final_answer with exactly {"kind":"report_saved"}.'
    )


def _bullets(items: Any) -> str:
    return "\n".join(f"- {item}" for item in items or []) or "- (none)"


# The brief as the report agent reads it: enough for the business context and
# the assumptions to disclose; the results come from the tools.
_BRIEF_EXCERPT_CHARS = 4000


def render_brief_report(state: RunState) -> str:
    """What the report agent needs: the subject, the reader, the report shape, from the plan."""
    report = state.plan.get("report") or {}
    return "\n".join([
        f"Title: {state.title}",
        f"Audience: {state.audience or '(not stated)'}",
        f"Report language: {state.language}",
        f"Tone: {report.get('tone') or '(not stated)'}",
        f"Length: {report.get('length') or '(not stated)'}",
        "Sections, in this order:",
        _bullets(report.get("sections")) if report.get("sections") else "- (not stated: design them)",
        "Charts requested:",
        _bullets(f"{c['id']}: {c['description']}" for c in state.charts_requested),
        f"Insights recorded by the analysts: {len(state.insights)}",
        "",
        "The brief (excerpt):",
        brief_context_text(state.brief, max_chars=_BRIEF_EXCERPT_CHARS),
    ])


@dataclass
class ReportAgentEngine(AnalystModelOptions, SmolagentsEngineBase):
    config: AnalystConfig
    state: RunState
    on_step: Optional[Callable[..., None]] = None

    token_usage: dict[str, int] = field(default_factory=dict, init=False)

    # --- hooks ---

    def _init_config(self) -> None:
        self._provider = self.config.report_provider
        self._configured_model = self.config.report_model
        self._max_steps = self.config.report_max_steps

    def _build_instructions(self, datasource: Optional[SqlDatasource] = None) -> str:
        template = load_prompt(ANALYST_REPORT_SYSTEM_TITLE, default_text=ANALYST_REPORT_SYSTEM_DEFAULT)
        return render_prompt(template, language=self.state.language)

    def _extra_tools(self) -> list[Any]:
        language = self.state.language.strip().lower()
        return [
            ReadSubTaskStateTool(self.state, preview_chars=self.config.preview_max_chars),
            GetSubTaskStatusTool(self.state),
            SaveRunMetricsTool(self.state),
            GetDeltaSectionTool(self.state),
            GetAllSavedChartsTool(self.state),
            GetHtmlTemplateTool(
                self.state,
                appendix_title=APPENDIX_TITLES.get(language, "Appendix"),
                footer=f"Maui · analysis brief #{self.state.brief_id} · run {self.state.run_id}",
            ),
            SaveHtmlReportTool(self.state),
        ]

    def _sql_tools(self, datasource: Optional[SqlDatasource] = None) -> list[Any]:
        return []

    def _build_agent(self, model: LiteLLMModel, instructions: str) -> Optional[CodeAgent]:
        self._final_answer_checks_supported = True
        return CodeAgent(
            tools=self._extra_tools(),
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
        payload, passed, kind, reason = coerce_final_payload(output, _REPORT_KIND_VALIDATORS)
        if not passed or payload is None:
            return payload, kind, _contract_failure_message(reason)
        if kind == "report_saved" and self.state.report_path is None:
            return payload, kind, (
                "The report is not saved yet: render it with get_html_template, write it "
                "with save_html_report, then answer again."
            )
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

    def render_task(self) -> str:
        tasks = "\n".join(
            f"- {t.id} ({t.name}): {t.status}. {t.summary or t.error or ''}".rstrip()
            for t in self.state.ordered_tasks()
        )
        template = load_prompt(ANALYST_REPORT_TURN_TITLE, default_text=ANALYST_REPORT_TURN_DEFAULT)
        return (
            render_prompt(template, brief_report=render_brief_report(self.state), tasks=tasks)
            + "\n\n" + CODE_STEP_RULE
        )

    def run(self) -> dict[str, Any]:
        """Write the report: {"kind": "report_saved"} or {"kind": "error", "message", "code"}."""
        if self._agent is None:
            return {"kind": "error", "message": "The report agent is not configured.", "code": "MISSING_CONFIG"}

        run_result, run_error = self._run_agent(self.render_task())
        if run_result is None:
            return {"kind": "error", "message": f"The report agent failed: {run_error}", "code": "RUN_FAILED"}

        usage = getattr(run_result, "token_usage", None)
        if usage is not None:
            self.token_usage = {"input": usage.input_tokens, "output": usage.output_tokens}

        payload, kind, problem = self._evaluate(getattr(run_result, "output", None))
        if problem is not None or payload is None:
            return {"kind": "error", "message": problem or "No final answer.", "code": "INVALID_OUTPUT"}
        if kind == "error":
            return {"kind": "error", "message": str(payload.get("message")), "code": "AGENT_ERROR"}
        return {"kind": "report_saved"}
