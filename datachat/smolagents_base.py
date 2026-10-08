"""
smolagents_base.py
------------------

What every smolagents-backed engine shares, whatever it is for.

DataChat answers questions about data; the interviewer questions the user about
the analysis they want. Both are a CodeAgent on a LiteLLM model, both may read
the same read-only SQL datasource through the same sql_engine tool, and both
guard their final answer with a contract the model is told about when it
breaks it. This base owns that plumbing:

- model construction from a provider/model pair;
- the single SQL gate: datasource resolved once, schema rendered into the
  instructions, sql_engine handed the identifiers from that same snapshot;
- agent construction with final_answer_checks (and the fallback for a
  smolagents that does not support them);
- one agent run with timing, failure capture and the trace the routes log.

Subclasses decide the rest through hooks: where their config comes from
(_init_config), what their instructions are (_build_instructions), which tools
they add besides sql_engine (_extra_tools), what a valid final answer is
(_check_final_answer) and how a turn is shaped (chat).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional

from smolagents import CodeAgent, LiteLLMModel

import datachat.sql_datasource as sql_datasource
from datachat.sql_datasource import SqlDatasource
from datachat.sql_prompt import build_sql_addendum
from datachat.tools.sql_engine_tool import SqlEngineTool
from llm.litellm_factory import build_litellm_model


@dataclass
class SmolagentsEngineBase:
    api_key: str
    user_name: str

    # Label used in every runtime log line, and the logger those lines go to.
    ENGINE_NAME: ClassVar[str] = "smolagents"
    RUNTIME_LOGGER_NAME: ClassVar[str] = "datachat.runtime"

    _agent: Optional[CodeAgent] = field(default=None, init=False, repr=False)
    _model: Optional[LiteLLMModel] = field(default=None, init=False, repr=False)

    _last_run_result: Any = field(default=None, init=False, repr=False)
    _last_run_duration_ms: Optional[float] = field(default=None, init=False, repr=False)

    _provider: str = field(default="", init=False, repr=False)
    _configured_model: str = field(default="", init=False, repr=False)
    _max_steps: int = field(default=12, init=False, repr=False)
    _instructions: str = field(default="", init=False, repr=False)

    _last_final_answer_check_passed: Optional[bool] = field(default=None, init=False, repr=False)
    _last_final_kind: Optional[str] = field(default=None, init=False, repr=False)
    _active_request_id: Optional[str] = field(default=None, init=False, repr=False)

    _final_answer_checks_supported: bool = field(default=False, init=False, repr=False)

    # The SQL datasource is resolved exactly once, in __post_init__, and passed
    # down to _build_instructions() and _sql_tools(). `None` means the
    # datasource is disabled or unconfigured, and is the single gate for
    # everything SQL: no reflection, no schema in the prompt, no sql_engine tool.
    _sql_datasource: Optional[SqlDatasource] = field(default=None, init=False, repr=False)
    _sql_ready: bool = field(default=False, init=False, repr=False)

    # Real relation -> real columns, from the very snapshot rendered into the
    # prompt. Derived there rather than fetched by the tool so that a TTL expiry
    # mid-session can never have the tool quoting names the agent was not shown.
    _sql_identifiers: dict[str, tuple[str, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._init_paths()
        self._init_config()

        self._log.info(
            "engine_init engine=%s user=%s provider=%s model=%s max_steps=%s",
            self.ENGINE_NAME,
            self.user_name,
            self._provider,
            self._configured_model or "missing",
            self._max_steps,
        )

        if not self._configured_model:
            self._set_missing_config("MISSING_CONFIG")
            return

        self._model = self._build_model()
        if self._model is None:
            self._set_missing_config("MISSING_CONFIG")
            return

        # Single gate: everything SQL hangs off this one resolution.
        self._sql_datasource = sql_datasource.get_datasource()

        self._instructions = self._build_instructions(self._sql_datasource)
        if not self._requirements_met():
            self._set_missing_config("SQL_UNAVAILABLE")
            return

        self._agent = self._build_agent(self._model, self._instructions)

        self._log.info(
            "engine_init_result engine=%s user=%s status=%s sql=%s",
            self.ENGINE_NAME,
            self.user_name,
            "ok" if self._agent is not None else "error",
            "ready" if self._sql_ready else ("error" if self._sql_datasource is not None else "off"),
        )

    @property
    def _log(self) -> logging.Logger:
        return logging.getLogger(self.RUNTIME_LOGGER_NAME)

    # --- hooks ---

    def _init_paths(self) -> None:
        """Per-session filesystem setup; none by default."""

    def _init_config(self) -> None:
        """Set _provider, _configured_model and _max_steps."""
        raise NotImplementedError

    def _build_instructions(self, datasource: Optional[SqlDatasource] = None) -> str:
        """The agent's system instructions, SQL addendum included when ready."""
        raise NotImplementedError

    def _extra_tools(self) -> list[Any]:
        """Tools the engine adds besides sql_engine; they come first."""
        return []

    def _requirements_met(self) -> bool:
        """Whether the engine can run once its instructions are built."""
        return True

    def _check_final_answer(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError

    # --- init helpers ---

    def _set_missing_config(self, code: str) -> None:
        self._model = None
        self._agent = None
        self._log.info(
            "engine_init_result engine=%s user=%s status=error error_code=%s",
            self.ENGINE_NAME,
            self.user_name,
            code,
        )

    def _build_model(self) -> Optional[LiteLLMModel]:
        try:
            return build_litellm_model(
                provider=self._provider,
                configured_model=self._configured_model,
                temperature=0.0,
            )
        except Exception:
            return None

    def _sql_addendum(
        self,
        datasource: Optional[SqlDatasource],
        load_template: Callable[[], str],
    ) -> Optional[str]:
        """
        The rendered SQL addendum, or None when SQL must stay out of the agent.

        Sets _sql_ready and _sql_identifiers on success, which is why
        instructions must be built before the tools.

        :param load_template: resolves the addendum prompt; only called when
            there is a datasource, so a disabled one costs no prompt lookup.
        """
        if datasource is None:
            return None

        result = build_sql_addendum(
            datasource,
            template=load_template(),
            engine_name=self.ENGINE_NAME,
            user_name=self.user_name,
        )
        if result is None:
            return None

        self._sql_ready = True
        self._sql_identifiers = result.identifiers
        return result.text

    def _sql_tools(self, datasource: Optional[SqlDatasource] = None) -> list[Any]:
        """
        SQL tools, or an empty list when the datasource is disabled or its
        schema could not be reflected.

        Discovery is not a tool: the schema is reflected once and rendered into
        the instructions by _build_instructions(), so sql_engine is all the
        agent needs.
        """
        if datasource is None or not self._sql_ready:
            return []
        return [SqlEngineTool(datasource, identifiers=self._sql_identifiers)]

    def _build_agent(self, model: LiteLLMModel, instructions: str) -> Optional[CodeAgent]:
        tools = []

        extra_tools = self._extra_tools()
        tools.extend(extra_tools)

        sql_tools = self._sql_tools(self._sql_datasource)
        tools.extend(sql_tools)

        self._log.info(
            "engine_init_tools engine=%s user=%s tool_count=%s sql_tools=%s",
            self.ENGINE_NAME,
            self.user_name,
            len(tools),
            len(sql_tools),
        )

        base_kwargs: dict[str, Any] = {
            "tools": tools,
            "model": model,
            "instructions": instructions,
            "max_steps": self._max_steps,
            "additional_authorized_imports": ["json"],
        }

        try:
            agent = CodeAgent(**{**base_kwargs, "final_answer_checks": [self._check_final_answer]})
            self._final_answer_checks_supported = True
            self._log.info(
                "engine_init_guardrail engine=%s user=%s final_answer_checks_supported=%s",
                self.ENGINE_NAME,
                self.user_name,
                True,
            )
            return agent
        except TypeError:
            self._final_answer_checks_supported = False
            self._log.info(
                "engine_init_guardrail engine=%s user=%s final_answer_checks_supported=%s",
                self.ENGINE_NAME,
                self.user_name,
                False,
            )
            return CodeAgent(**base_kwargs)

    # --- run ---

    def _run_agent(self, task: str) -> tuple[Any, Optional[str]]:
        """
        One agent run on a fresh memory.

        Memory is reset on every run: whatever the agent must know across turns
        is put into the task or the instructions by the subclass.

        :return: (run_result, None) on success, (None, error message) on failure.
        """
        assert self._agent is not None
        try:
            started = time.time()
            run_result = self._agent.run(str(task), reset=True, return_full_result=True)
            self._last_run_result = run_result
            self._last_run_duration_ms = round((time.time() - started) * 1000, 2)
            return run_result, None
        except Exception as e:
            self._last_run_result = None
            self._last_run_duration_ms = None
            self._log.info(
                "chat_error request_id=%s engine=%s user=%s error_code=RUN_FAILED error_message_short=%s",
                self._active_request_id,
                self.ENGINE_NAME,
                self.user_name,
                str(e)[:160],
            )
            return None, str(e)

    def get_last_trace(self) -> Optional[dict[str, Any]]:
        if self._last_run_result is None:
            return None
        return {"run_result": self._last_run_result, "duration_ms": self._last_run_duration_ms}

    def close(self) -> None:
        """Release per-session resources; none by default."""
