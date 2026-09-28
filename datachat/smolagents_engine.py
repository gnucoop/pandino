import json
import logging
import os
import re
import shutil
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import pandas as pd
from smolagents import CodeAgent, LiteLLMModel

import datachat.schema_snapshot_loader as schema_snapshot_loader
import datachat.sql_datasource as sql_datasource
from datachat.bootstrap_static import get_static_bootstrap_html
from datachat.engine_interface import DataChatEngine, EngineBootstrapResult
from datachat.sql_datasource import SqlDatasource
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.correlation_tool import CorrelationTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.plot_tool import PlotTool
from datachat.tools.row_count_tool import RowCountTool
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.sql_engine_tool import SqlEngineTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.trend_tool import TrendTool
from datachat.tools.unique_values_tool import UniqueValuesTool
from llm.litellm_factory import build_litellm_model
from infrastructure.prompt_utils import load_prompt, render_prompt
from utils.logging_config import get_request_id

runtime_logger = logging.getLogger("datachat.runtime")

_ALLOWED_FINAL_KINDS = {"text", "table", "image_path", "error"}

# How many times a run may have an empty table turned down as its final answer.
# One is deliberate: it buys exactly one verification round. An empty result is a
# legitimate answer to "which clients lost money?", so the guard must force the
# agent to check its filters, not forbid it from ever reporting nothing.
_MAX_EMPTY_FINAL_REJECTIONS = 1

_EMPTY_TABLE_GUIDANCE = (
    "The query matched no rows, so this answer would show the user an empty table. "
    "Zero rows is far more often a filter written with the wrong value than data that "
    "is genuinely absent: the value's format, its spelling, or the column it lives in "
    "may differ from the way the question phrased it. Verify before answering — re-run "
    "the query without its narrowest filter, or SELECT DISTINCT the columns you "
    "filtered on, and compare the real values against the ones you used. "
    "If that turns up the data, answer with it. If you confirm that nothing matches, "
    'do not return an empty table: return {"kind":"text","text":"..."} stating what '
    "you searched for and that no matching rows exist."
)

# Appended to the system instructions only when the SQL datasource is enabled.
# {sql_schema} is replaced with the rendered schema snapshot; every other brace
# is literal and is passed through untouched by render_prompt().
_SQL_ADDENDUM_DEFAULT = textwrap.dedent(
    """\
    SQL DATABASE
    - Besides the optional (might be missing) uploaded dataset, a read-only SQL database is available.
    - Use the dataset tools for questions about the uploaded data; use sql_engine
      only when the user asks for data that is not in the uploaded columns.

    {sql_schema}

    SQL RULES
    - The schema above is already complete. Do not try to discover it: go straight
      to sql_engine and use exactly the names listed, spelled exactly as shown.
    - Identifiers are case-sensitive and every name in the schema is shown
      double-quoted: write it with those quotes. PostgreSQL folds an unquoted name
      to lowercase, so SELECT "Idcliente" FROM "Clienti" works where
      SELECT Idcliente FROM Clienti fails with 'relation "clienti" does not exist'.
      Double quotes are for names; single quotes are for string values.
    - If the data the user asks for is not in the schema above, say so plainly
      instead of guessing a table or column name.
    - Views and materialized views are read exactly like tables in a SELECT. A view
      that already joins and filters the base tables is usually the better source.
    - Prefer a materialized view when one answers the question: it is precomputed and
      much faster, but its data is only as fresh as its last refresh.
    - Only a single SELECT (or WITH ... SELECT) is accepted. INSERT, UPDATE,
      DELETE and DDL are rejected before reaching the database.
    - Add an explicit LIMIT while exploring. Results are capped: if meta.truncated
      is true the answer is incomplete, so narrow the query and run it again.
    - If sql_engine returns kind="error", read its "code" field, fix the query and
      retry at most twice before explaining the limitation to the user.
    - The values shown after "e.g." in the schema are samples, not the full set a
      column holds. They tell you how a column is written — its format, whether it
      holds a code or a label — not which values exist.
    - Zero rows back usually means a filter value is wrong, not that the data is
      missing. Before you accept an empty result, re-run without the narrowest
      filter, or SELECT DISTINCT the columns you filtered on, and compare the real
      values against the ones you used.
    - Never answer with an empty table. If nothing matches after you have checked,
      answer kind="text" saying what you searched for and that no rows match.

    The final-answer rules stated above are unchanged: sql_engine already returns
    a valid table payload and can be passed to final_answer as it is.
    """
)


# ----------------------------
# Final answer contract utils
# ----------------------------

def _extract_json_object(s: str) -> Optional[dict[str, Any]]:
    """
    Best-effort extraction of a JSON object from a string.
    - strict json.loads
    - if that fails, try substring between first '{' and last '}'.
    """
    s = (s or "").strip()
    if not s:
        return None

    # A) strict JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        # Sometimes LLM returns a JSON string containing JSON.
        if isinstance(obj, str):
            obj2 = json.loads(obj)
            if isinstance(obj2, dict):
                return obj2
    except Exception:
        pass

    # B) conservative extraction
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def _unwrap_nested_table(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Fix common LLM mistake:
      {"kind":"table","data":{"kind":"table","data":[...]}}
    -> {"kind":"table","data":[...]}
    """
    try:
        kind = str(payload.get("kind") or "").strip().lower()
        if kind != "table":
            return payload

        data = payload.get("data")
        if isinstance(data, list):
            return payload

        if isinstance(data, dict):
            nested_data = data.get("data")
            if isinstance(nested_data, list):
                payload["data"] = nested_data
        return payload
    except Exception:
        return payload


def _validate_contract_payload(payload: dict[str, Any]) -> Tuple[bool, Optional[str], str]:
    """
    Returns:
      (passed, final_kind, reason)
    """
    raw_kind = payload.get("kind")
    kind = str(raw_kind or "").strip().lower()

    if kind not in _ALLOWED_FINAL_KINDS:
        return False, (kind or None), "INVALID_KIND"

    if kind in {"text", "error"}:
        text_val = payload.get("text")
        msg_val = payload.get("message")
        if text_val is None and msg_val is None:
            return False, kind, "MISSING_TEXT_OR_MESSAGE"
        if text_val is not None and not str(text_val).strip() and msg_val is None:
            return False, kind, "EMPTY_TEXT"
        if msg_val is not None and not str(msg_val).strip() and text_val is None:
            return False, kind, "EMPTY_MESSAGE"
        return True, kind, "OK"

    if kind == "table":
        if "data" not in payload:
            return False, kind, "MISSING_DATA"
        return True, kind, "OK"

    if kind == "image_path":
        path = payload.get("path")
        if not isinstance(path, str) or not path.strip():
            return False, kind, "MISSING_PATH"
        return True, kind, "OK"

    return False, (kind or None), "INVALID_KIND"


def _is_empty_table(payload: Optional[dict[str, Any]]) -> bool:
    """Whether a validated payload is a table that carries no rows."""
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return isinstance(data, list) and not data


def _contract_failure_message(reason: str) -> str:
    """
    Explain a rejected final answer to the model.

    smolagents interpolates whatever this check raises into the step error and
    replays it to the model, so the reason has to travel inside the exception:
    returning False raises a bare AssertionError, which reaches the model as
    "failed with error:" and nothing more.
    """
    return (
        f"The final answer does not satisfy the output contract ({reason}). "
        'Return exactly one JSON object carrying a "kind" field, with no wrapper, '
        "no extra nesting and no surrounding prose. Valid shapes: "
        '{"kind":"text","text":"..."}, {"kind":"table","data":[...]}, '
        '{"kind":"image_path","path":"..."}, {"kind":"error","message":"..."}.'
    )


def _coerce_final_payload(output: Any) -> Tuple[Optional[dict[str, Any]], bool, Optional[str], str]:
    """
    Parse + unwrap + validate.
    Returns:
      (payload_or_none, passed, final_kind, reason)
    """
    if isinstance(output, dict):
        candidate = output
    elif isinstance(output, str):
        candidate = _extract_json_object(output)
        if candidate is None:
            return None, False, None, "NON_JSON_OR_NO_OBJECT"
    else:
        return None, False, None, "NON_JSON_OR_UNSUPPORTED_TYPE"

    if "kind" not in candidate:
        return None, False, None, "NO_KIND"

    candidate = _unwrap_nested_table(candidate)
    passed, final_kind, reason = _validate_contract_payload(candidate)
    return candidate, passed, final_kind, reason


# ----------------------------
# Engine
# ----------------------------

@dataclass
class SmolagentsEngine(DataChatEngine):
    api_key: str
    user_name: str
    llm: Any  # kept for interface compatibility; not used here
    data: pd.DataFrame|None

    _agent: Optional[CodeAgent] = field(default=None, init=False, repr=False)
    _model: Optional[LiteLLMModel] = field(default=None, init=False, repr=False)

    _plots_dir: Optional[str] = field(default=None, init=False, repr=False)
    _user_plots_dir: Optional[str] = field(default=None, init=False, repr=False)

    _last_run_result: Any = field(default=None, init=False, repr=False)
    _last_run_duration_ms: Optional[float] = field(default=None, init=False, repr=False)

    _provider: str = field(default="", init=False, repr=False)
    _configured_model: str = field(default="", init=False, repr=False)
    _max_steps: int = field(default=12, init=False, repr=False)
    _instructions: str = field(default="", init=False, repr=False)

    # Per-run budget for the empty-table guard, reset at the top of chat().
    # The only per-run state kept on the instance: the smolagents final-answer
    # check is invoked by the agent mid-run, so the budget cannot live as a
    # chat() local. Request identity is ambient (get_request_id()), never stored.
    _empty_final_rejections: int = field(default=0, init=False, repr=False)

    _final_answer_checks_supported: bool = field(default=False, init=False, repr=False)

    # The SQL datasource is resolved exactly once, here, and passed down to
    # _build_instructions() and _sql_tools(). `None` means the datasource is
    # disabled or unconfigured, and is the single gate for everything SQL: no
    # reflection, no schema in the prompt, no sql_engine tool.
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

        runtime_logger.info(
            "engine_init request_id=%s engine=smolagents user=%s provider=%s model=%s max_steps=%s",
            get_request_id(),
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
        self._agent = self._build_agent(self._model, self._instructions)

        runtime_logger.info(
            "engine_init_result request_id=%s engine=smolagents user=%s status=%s sql=%s",
            get_request_id(),
            self.user_name,
            "ok" if self._agent is not None else "error",
            "ready" if self._sql_ready else ("error" if self._sql_datasource is not None else "off"),
        )

    # --- init helpers ---

    def _init_paths(self) -> None:
        safe_user = re.sub(r"[^A-Za-z0-9._-]+", "_", str(self.user_name or "user")).strip("_")
        session_id = uuid.uuid4().hex
        base_dir = os.getenv("DATACHAT_PLOTS_DIR", "/tmp/datachat_plots")
        self._user_plots_dir = os.path.join(base_dir, safe_user)
        self._plots_dir = os.path.join(self._user_plots_dir, session_id)

    def _init_config(self) -> None:
        self._provider = os.getenv("DATACHAT_PROVIDER", "Deepinfra").strip()
        self._configured_model = os.getenv("DATACHAT_MODEL", "").strip()
        try:
            self._max_steps = max(1, int(os.getenv("DATACHAT_MAX_STEPS", "12")))
        except ValueError:
            self._max_steps = 12

    def _set_missing_config(self, code: str) -> None:
        self._model = None
        self._agent = None
        runtime_logger.info(
            "engine_init_result request_id=%s engine=smolagents user=%s status=error error_code=%s",
            get_request_id(),
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

    def _build_instructions(self, datasource: Optional[SqlDatasource] = None) -> str:
        cols = list(self.data.columns) if self.data is not None else []
        default_context = textwrap.dedent(
            """\
            You are DataChat, a cognitive assistant that helps users explore a tabular dataset.

            DATASET
            - The dataset has the following columns: {columns}.
            - If {columns} is empty, no dataset is loaded.

            PURPOSE
            - Understand the user’s intent.
            - If the user requests a concrete data operation, translate it into explicit tool calls.
            - Do NOT invent columns, rows, or values.

            REQUEST TYPES

            1) Concrete data operations
            Examples: counts, filtering, summaries, correlations, charts, trends.
            -> You MUST call the appropriate tool.

            2) High-level dataset questions
            Examples: "What is this dataset about?", "What information does it contain?"
            -> Return a short natural-language summary (kind="text"), based only on the column names
               (and optionally a small sample via sample_rows if needed).
            -> Do NOT return raw describe output unless the user explicitly asks for statistics.

            3) Meta-system questions
            Examples: "What can you do?", "What analyses are possible?"
            -> Return a short explanation (kind="text") describing the available analyses supported by the tools.

            RULES
            - Use plain text only (avoid markdown formatting).
            - If a request cannot be expressed with the available tools, explain the limitation briefly.

            OUTPUT
            - The final result must be exactly one JSON object.
            - The JSON must contain a "kind" field.
            - Valid kinds are: text, table, image_path, error.
            - Do not add wrappers, metadata, or extra nesting.
            - Never use a "value" field in the final answer.

            Final-answer schemas (strict):
            - kind="text"       -> {"kind":"text","text":"..."}
            - kind="table"      -> {"kind":"table","data":[...]}
            - kind="image_path" -> {"kind":"image_path","path":"..."}
            - kind="error"      -> {"kind":"error","message":"..."}

            When a tool already returns a valid final contract object:
            - pass it directly to final_answer(...) without re-wrapping.
            """
        )

        template = load_prompt("data_chat_system", default_text=default_context)
        instructions = render_prompt(template, columns=cols)

        # Gate: with no datasource nothing SQL is appended and no reflection is
        # attempted, so the agent is exactly what it is with SQL disabled.
        if datasource is None:
            return instructions

        snapshot = schema_snapshot_loader.get_schema_snapshot(datasource)
        if snapshot is None:
            # Reflection failed. An agent that can write SQL but was never told
            # the schema is worse than one without SQL, so _sql_tools() drops
            # sql_engine too and the instructions stay silent about the database.
            runtime_logger.info(
                "engine_init_sql engine=smolagents user=%s status=error reason=SCHEMA_UNAVAILABLE",
                self.user_name,
            )
            return instructions

        rendered = schema_snapshot_loader.render_schema(
            snapshot, datasource.schema_max_chars
        )

        # Appended as a separate prompt so that a DB-stored override of
        # data_chat_system keeps working and can predate the SQL support.
        addendum = load_prompt(
            "data_chat_sql_addendum", default_text=_SQL_ADDENDUM_DEFAULT
        )
        if "{sql_schema}" in addendum:
            addendum = render_prompt(addendum, sql_schema=rendered)
        else:
            # An override stored before the schema placeholder existed: append
            # the schema rather than silently dropping it.
            addendum = addendum + "\n\n" + rendered

        self._sql_ready = True
        self._sql_identifiers = schema_snapshot_loader.build_identifier_index(snapshot)
        runtime_logger.info(
            "engine_init_sql engine=smolagents user=%s status=ready relations=%s schema_chars=%s",
            self.user_name,
            len(snapshot.relations),
            len(rendered),
        )
        return instructions + "\n\n" + addendum

    def _data_tools(self) -> list[Any]:
        """DATA tools, or an empty list when data (csv datasource) is None."""
        datasource = self.data
        if datasource is None:
            return []
        return [
            DescribeTool(datasource),
            MissingValuesTool(datasource),
            UniqueValuesTool(datasource),
            CorrelationTool(datasource),
            SampleRowsTool(datasource),
            TopRowsTool(datasource),
            FilterRowsTool(datasource),
            RowCountTool(datasource),
            AggregateTool(datasource),
            PlotTool(datasource, output_dir=self._plots_dir or os.getenv("DATACHAT_PLOTS_DIR", "/tmp/datachat_plots")),
            TrendTool(datasource),
        ]

    def _sql_tools(self, datasource: Optional[SqlDatasource] = None) -> list[Any]:
        """
        SQL tools, or an empty list when the datasource is disabled or its
        schema could not be reflected.

        Discovery is no longer a tool: the schema is reflected once and rendered
        into the instructions by _build_instructions(), so sql_engine is all the
        agent needs. _sql_ready is set there, which is why instructions must be
        built first.
        """
        if datasource is None or not self._sql_ready:
            return []
        return [SqlEngineTool(datasource, identifiers=self._sql_identifiers)]

    def _build_agent(self, model: LiteLLMModel, instructions: str) -> Optional[CodeAgent]:
        tools = []

        data_tools = self._data_tools()
        tools.extend(data_tools)

        sql_tools = self._sql_tools(self._sql_datasource)
        tools.extend(sql_tools)

        runtime_logger.info(
            "engine_init_tools request_id=%s engine=smolagents user=%s tool_count=%s sql_tools=%s",
            get_request_id(),
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
            runtime_logger.info(
                "engine_init_guardrail request_id=%s engine=smolagents user=%s final_answer_checks_supported=%s",
                get_request_id(),
                self.user_name,
                True,
            )
            return agent
        except TypeError:
            self._final_answer_checks_supported = False
            runtime_logger.info(
                "engine_init_guardrail request_id=%s engine=smolagents user=%s final_answer_checks_supported=%s",
                get_request_id(),
                self.user_name,
                False,
            )
            return CodeAgent(**base_kwargs)

    def _check_final_answer(self, *args: Any, **kwargs: Any) -> bool:
        """
        Guard the final answer before smolagents hands it back to the caller.

        smolagents calls this as check(final_answer, memory, agent=agent) and
        asserts the result, wrapping anything raised into the step error it
        replays to the model. Raising is therefore the only way to tell the model
        *why* its answer was turned down: returning False reaches it as
        "failed with error:" and nothing else, which it cannot act on.
        """
        candidate = args[0] if args else (
            kwargs.get("final_answer") or kwargs.get("answer") or kwargs.get("output")
        )
        payload, passed, final_kind, reason = _coerce_final_payload(candidate)

        runtime_logger.info(
            "final_answer_check request_id=%s engine=smolagents user=%s passed=%s final_kind=%s reason=%s",
            get_request_id(),
            self.user_name,
            passed,
            final_kind or "none",
            reason,
        )

        if not passed:
            raise ValueError(_contract_failure_message(reason))

        # The contract is satisfied but the answer is blank. Spend the
        # verification budget before letting it reach the user.
        if final_kind == "table" and _is_empty_table(payload):
            if self._empty_final_rejections < _MAX_EMPTY_FINAL_REJECTIONS:
                self._empty_final_rejections += 1
                runtime_logger.info(
                    "final_answer_empty_rejected request_id=%s engine=smolagents user=%s attempt=%s",
                    get_request_id(),
                    self.user_name,
                    self._empty_final_rejections,
                )
                raise ValueError(_EMPTY_TABLE_GUIDANCE)

            # Budget spent: the agent has had its verification round and still
            # reports nothing, so an empty result is taken at face value.
            runtime_logger.info(
                "final_answer_empty_accepted request_id=%s engine=smolagents user=%s rejections=%s",
                get_request_id(),
                self.user_name,
                self._empty_final_rejections,
            )

        return True

    # --- public API ---

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        html = get_static_bootstrap_html(lang)
        return EngineBootstrapResult(suggested_questions_html=html)

    def chat(self, message: str) -> Any:
        # Reset the per-run empty-table budget. Request identity is ambient.
        self._empty_final_rejections = 0

        runtime_logger.info(
            "chat_start request_id=%s engine=smolagents user=%s message_len=%s",
            get_request_id(),
            self.user_name,
            len(str(message or "")),
        )

        if self._agent is None:
            runtime_logger.info(
                "chat_error request_id=%s engine=smolagents user=%s error_code=MISSING_CONFIG",
                get_request_id(),
                self.user_name,
            )
            return {
                "kind": "error",
                "message": (
                    "SmolagentsEngine non è configurato correttamente: "
                    "verifica DATACHAT_PROVIDER/DATACHAT_MODEL e la relativa API key."
                ),
                "code": "MISSING_CONFIG",
            }

        try:
            started = time.time()
            run_result = self._agent.run(str(message), reset=True, return_full_result=True)
            self._last_run_result = run_result
            self._last_run_duration_ms = round((time.time() - started) * 1000, 2)
        except Exception as e:
            self._last_run_result = None
            self._last_run_duration_ms = None
            runtime_logger.info(
                "chat_error request_id=%s engine=smolagents user=%s error_code=RUN_FAILED error_message_short=%s",
                get_request_id(),
                self.user_name,
                str(e)[:160],
            )
            return {"kind": "error", "message": f"SmolagentsEngine failed to run: {e}", "code": "RUN_FAILED"}

        out = getattr(run_result, "output", None)

        payload, passed, final_kind, reason = _coerce_final_payload(out)

        if passed and payload is not None:
            payload["kind"] = final_kind  # normalize casing
            result_payload = payload
        else:
            # Safe fallback: keep it user-friendly, and avoid claiming tool results.
            safe_text = out.strip() if isinstance(out, str) else ""
            result_payload = {
                "kind": "text",
                "text": safe_text or f"Nessun output finale valido prodotto dall'agente ({reason}).",
                "format": "plain",
            }
            final_kind = None

        # An empty table is returned as an empty table. The guard in
        # _check_final_answer has already bought the agent a verification round,
        # which is where a wrong filter value gets caught; past that point the
        # emptiness is the answer, and rewriting it into a sentence would change
        # the response kind out from under callers that asked for a table.
        # Presenting an empty result is the client's display decision.
        if _is_empty_table(result_payload):
            runtime_logger.info(
                "chat_empty_table request_id=%s engine=smolagents user=%s",
                get_request_id(),
                self.user_name,
            )

        runtime_logger.info(
            "chat_end request_id=%s engine=smolagents user=%s duration_ms=%s response_kind=%s final_answer_check_passed=%s final_kind=%s",
            get_request_id(),
            self.user_name,
            self._last_run_duration_ms,
            result_payload.get("kind"),
            bool(passed),
            final_kind or "none",
        )
        return result_payload

    def get_last_trace(self) -> Optional[dict[str, Any]]:
        if self._last_run_result is None:
            return None
        return {"run_result": self._last_run_result, "duration_ms": self._last_run_duration_ms}

    def close(self) -> None:
        plots_dir_removed = False
        user_dir_removed = False
        cleanup_error = ""

        try:
            if self._plots_dir and os.path.exists(self._plots_dir):
                shutil.rmtree(self._plots_dir, ignore_errors=True)
                plots_dir_removed = not os.path.exists(self._plots_dir)

            if self._user_plots_dir and os.path.isdir(self._user_plots_dir):
                try:
                    if not os.listdir(self._user_plots_dir):
                        os.rmdir(self._user_plots_dir)
                        user_dir_removed = not os.path.exists(self._user_plots_dir)
                except Exception as e:
                    cleanup_error = str(e)[:160]
        except Exception as e:
            cleanup_error = str(e)[:160]

        runtime_logger.info(
            "cleanup_result request_id=%s engine=smolagents user=%s plots_dir_removed=%s user_dir_removed=%s cleanup_error=%s",
            get_request_id(),
            self.user_name,
            plots_dir_removed,
            user_dir_removed,
            cleanup_error or "none",
        )
        return
