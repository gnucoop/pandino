import os
import re
import shutil
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Tuple

import pandas as pd

from datachat.bootstrap_static import get_static_bootstrap_html
from datachat.engine_interface import DataChatEngine, EngineBootstrapResult
from datachat.final_answer_contract import (
    coerce_final_payload,
    extract_json_object,
    is_empty_table,
    unwrap_nested_table,
    validate_contract_payload,
    validate_table,
    validate_text_or_message,
)
from datachat.smolagents_base import SmolagentsEngineBase
from datachat.sql_datasource import SqlDatasource
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.correlation_tool import CorrelationTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.plot_tool import PlotTool
from datachat.tools.row_count_tool import RowCountTool
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.trend_tool import TrendTool
from datachat.tools.unique_values_tool import UniqueValuesTool
from infrastructure.prompt_utils import load_prompt, render_prompt

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

def _validate_image_path(payload: dict[str, Any]) -> Optional[str]:
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        return "MISSING_PATH"
    return None


_DATACHAT_KIND_VALIDATORS = {
    "text": validate_text_or_message,
    "error": validate_text_or_message,
    "table": validate_table,
    "image_path": _validate_image_path,
}
assert set(_DATACHAT_KIND_VALIDATORS) == _ALLOWED_FINAL_KINDS

# Kept under their historical names: they are DataChat's view of the shared
# contract helpers in datachat.final_answer_contract.
_extract_json_object = extract_json_object
_unwrap_nested_table = unwrap_nested_table
_is_empty_table = is_empty_table


def _validate_contract_payload(payload: dict[str, Any]) -> Tuple[bool, Optional[str], str]:
    """
    Returns:
      (passed, final_kind, reason)
    """
    return validate_contract_payload(payload, _DATACHAT_KIND_VALIDATORS)


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
    return coerce_final_payload(
        output, _DATACHAT_KIND_VALIDATORS, repair=_unwrap_nested_table
    )


# ----------------------------
# Engine
# ----------------------------

@dataclass
class SmolagentsEngine(SmolagentsEngineBase, DataChatEngine):
    llm: Any  # kept for interface compatibility; not used here
    data: pd.DataFrame|None

    ENGINE_NAME: ClassVar[str] = "smolagents"
    RUNTIME_LOGGER_NAME: ClassVar[str] = "datachat.runtime"

    _plots_dir: Optional[str] = field(default=None, init=False, repr=False)
    _user_plots_dir: Optional[str] = field(default=None, init=False, repr=False)

    # Per-run budget for the empty-table guard, reset at the top of chat().
    _empty_final_rejections: int = field(default=0, init=False, repr=False)

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

        # Appended as a separate prompt so that a DB-stored override of
        # data_chat_system keeps working and can predate the SQL support.
        # With no datasource, or when reflection fails, nothing SQL is appended:
        # an agent that can write SQL but was never told the schema is worse
        # than one without SQL, so _sql_tools() drops sql_engine too.
        addendum = self._sql_addendum(
            datasource,
            lambda: load_prompt("data_chat_sql_addendum", default_text=_SQL_ADDENDUM_DEFAULT),
        )
        if addendum is None:
            return instructions
        return instructions + "\n\n" + addendum

    def _extra_tools(self) -> list[Any]:
        return self._data_tools()

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

        self._last_final_answer_check_passed = passed
        self._last_final_kind = final_kind

        self._log.info(
            "final_answer_check request_id=%s engine=smolagents user=%s passed=%s final_kind=%s reason=%s",
            self._active_request_id or "n/a",
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
                self._last_final_answer_check_passed = False
                self._log.info(
                    "final_answer_empty_rejected request_id=%s engine=smolagents user=%s attempt=%s",
                    self._active_request_id or "n/a",
                    self.user_name,
                    self._empty_final_rejections,
                )
                raise ValueError(_EMPTY_TABLE_GUIDANCE)

            # Budget spent: the agent has had its verification round and still
            # reports nothing, so an empty result is taken at face value.
            self._log.info(
                "final_answer_empty_accepted request_id=%s engine=smolagents user=%s rejections=%s",
                self._active_request_id or "n/a",
                self.user_name,
                self._empty_final_rejections,
            )

        return True

    # --- public API ---

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        html = get_static_bootstrap_html(lang)
        return EngineBootstrapResult(suggested_questions_html=html)

    def chat(self, message: str, request_id: Optional[str] = None) -> Any:
        self._active_request_id = request_id or "n/a"
        self._last_final_answer_check_passed = None
        self._last_final_kind = None
        self._empty_final_rejections = 0

        self._log.info(
            "chat_start request_id=%s engine=smolagents user=%s message_len=%s",
            self._active_request_id,
            self.user_name,
            len(str(message or "")),
        )

        if self._agent is None:
            self._log.info(
                "chat_error request_id=%s engine=smolagents user=%s error_code=MISSING_CONFIG",
                self._active_request_id,
                self.user_name,
            )
            self._active_request_id = None
            return {
                "kind": "error",
                "message": (
                    "SmolagentsEngine non è configurato correttamente: "
                    "verifica DATACHAT_PROVIDER/DATACHAT_MODEL e la relativa API key."
                ),
                "code": "MISSING_CONFIG",
            }

        run_result, run_error = self._run_agent(str(message))
        if run_result is None:
            self._active_request_id = None
            return {"kind": "error", "message": f"SmolagentsEngine failed to run: {run_error}", "code": "RUN_FAILED"}

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
            self._last_final_answer_check_passed = False
            self._last_final_kind = None

        # An empty table is returned as an empty table. The guard in
        # _check_final_answer has already bought the agent a verification round,
        # which is where a wrong filter value gets caught; past that point the
        # emptiness is the answer, and rewriting it into a sentence would change
        # the response kind out from under callers that asked for a table.
        # Presenting an empty result is the client's display decision.
        if _is_empty_table(result_payload):
            self._log.info(
                "chat_empty_table request_id=%s engine=smolagents user=%s",
                self._active_request_id,
                self.user_name,
            )

        self._log.info(
            "chat_end request_id=%s engine=smolagents user=%s duration_ms=%s response_kind=%s final_answer_check_passed=%s final_kind=%s",
            self._active_request_id,
            self.user_name,
            self._last_run_duration_ms,
            result_payload.get("kind"),
            bool(self._last_final_answer_check_passed),
            self._last_final_kind or "none",
        )
        self._active_request_id = None
        return result_payload

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

        self._log.info(
            "cleanup_result engine=smolagents user=%s plots_dir_removed=%s user_dir_removed=%s cleanup_error=%s",
            self.user_name,
            plots_dir_removed,
            user_dir_removed,
            cleanup_error or "none",
        )
        return
