"""
smolagents_base.py
------------------

What a smolagents-backed engine needs, whatever it is for.

An engine here is a CodeAgent on a LiteLLM model that may read the read-only
SQL datasource through the guarded sql_engine tool, and that guards its final
answer with a contract the model is told about when it breaks it. This base
owns that plumbing:

- model construction from a provider/model pair;
- the single SQL gate: datasource resolved once, schema rendered into the
  instructions, sql_engine handed the identifiers from that same snapshot;
- agent construction with final_answer_checks (and the fallback for a
  smolagents that does not support them);
- one agent run on a fresh memory, with failure capture.

The analysis interviewer is built on it. DataChat still carries its own copy of
this plumbing in SmolagentsEngine.

Subclasses decide the rest through hooks: where their config comes from
(_init_config), what their instructions are (_build_instructions), which tools
they add besides sql_engine (_extra_tools), what a valid final answer is
(_check_final_answer) and how a turn is shaped (chat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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

    _agent: Optional[CodeAgent] = field(default=None, init=False, repr=False)
    _model: Optional[LiteLLMModel] = field(default=None, init=False, repr=False)

    _provider: str = field(default="", init=False, repr=False)
    _configured_model: str = field(default="", init=False, repr=False)
    _max_steps: int = field(default=12, init=False, repr=False)
    _instructions: str = field(default="", init=False, repr=False)

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

        if not self._configured_model:
            self._set_missing_config()
            return

        self._model = self._build_model()
        if self._model is None:
            self._set_missing_config()
            return

        # Single gate: everything SQL hangs off this one resolution.
        self._sql_datasource = sql_datasource.get_datasource()

        self._instructions = self._build_instructions(self._sql_datasource)
        if not self._requirements_met():
            self._set_missing_config()
            return

        self._agent = self._build_agent(self._model, self._instructions)

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

    def _set_missing_config(self) -> None:
        self._model = None
        self._agent = None

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

        result = build_sql_addendum(datasource, template=load_template())
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
        tools.extend(self._extra_tools())
        tools.extend(self._sql_tools(self._sql_datasource))

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
            return agent
        except TypeError:
            self._final_answer_checks_supported = False
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
            run_result = self._agent.run(str(task), reset=True, return_full_result=True)
            return run_result, None
        except Exception as e:
            return None, str(e)

    def close(self) -> None:
        """Release per-session resources; none by default."""
