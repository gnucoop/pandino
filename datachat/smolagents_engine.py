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

from datachat.bootstrap_static import get_static_bootstrap_html
from datachat.engine_interface import DataChatEngine, EngineBootstrapResult
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.classify_tool import ClassifyTool
from datachat.tools.compare_groups_tool import CompareGroupsTool
from datachat.tools.correlation_tool import CorrelationTool
from datachat.tools.crosstab_tool import CrosstabTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.export_csv_tool import ExportCsvTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.keywords_tool import KeywordsTool
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.plot_tool import PlotTool
from datachat.tools.row_count_tool import RowCountTool
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.sentiment_tool import SentimentAnalysisTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.trend_tool import TrendTool
from datachat.tools.unique_values_tool import UniqueValuesTool
from llm.litellm_factory import build_litellm_model
from infrastructure.prompt_utils import load_prompt, render_prompt

runtime_logger = logging.getLogger("datachat.runtime")

_ALLOWED_FINAL_KINDS = {"text", "table", "image_path", "error"}

# Cap on full-result CSVs kept per session; older ones are deleted on overflow.
_MAX_SESSION_EXPORTS = 20

# Suggested download names come from column names, which can be full sentences.
_MAX_EXPORT_NAME_LEN = 60


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
    data: pd.DataFrame

    _agent: Optional[CodeAgent] = field(default=None, init=False, repr=False)
    _model: Optional[LiteLLMModel] = field(default=None, init=False, repr=False)

    _plots_dir: Optional[str] = field(default=None, init=False, repr=False)
    _user_plots_dir: Optional[str] = field(default=None, init=False, repr=False)

    # download token -> (absolute CSV path, suggested download name); see register_export
    _exports: dict[str, Tuple[str, str]] = field(default_factory=dict, init=False, repr=False)

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

    def __post_init__(self) -> None:
        self._init_paths()
        self._init_config()

        runtime_logger.info(
            "engine_init engine=smolagents user=%s provider=%s model=%s max_steps=%s",
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

        self._instructions = self._build_instructions()
        self._agent = self._build_agent(self._model, self._instructions)

        runtime_logger.info(
            "engine_init_result engine=smolagents user=%s status=%s",
            self.user_name,
            "ok" if self._agent is not None else "error",
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
            "engine_init_result engine=smolagents user=%s status=error error_code=%s",
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

    def _build_instructions(self) -> str:
        cols = list(self.data.columns)
        default_context = textwrap.dedent(
            """\
            You are DataChat, a cognitive assistant that helps users explore a tabular dataset.

            DATASET
            - The dataset has the following columns: {columns}.

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

            CHOOSING COLUMNS
            - When the user names a concept rather than an exact column, pick the SINGLE
              column whose name best matches it. Do not combine several columns into one
              answer unless the user asked for a combination.
            - Always state which column you used, quoting its name, so the user can correct
              you. Example: 'Nella colonna "...qualunque commento" ci sono 212 righe compilate.'
            - If two or more columns match the request equally well, say so and ask which
              one the user means instead of guessing or merging them.

            PICKING THE RIGHT TOOL
            - Two dimensions ("X per A e per B", "distribuzione delle risposte per gruppo")
              -> crosstab. Never issue two separate aggregate calls and stitch the numbers
              together yourself.
            - Free text mentioning a word ("commenti che parlano di orario")
              -> filter_rows with op="contains".
            - What open answers are about -> keywords first (it is free and instant), then
              classify or sentiment_analysis only if the user wants grouping or tone.
            - Before saying one group scores better than another -> compare_groups. An
              average from aggregate cannot distinguish a real gap from noise.
            - Relationships between rating questions -> correlation with method="spearman",
              and omit col_y to rank every question against one of them at once.

            REPORTING RESULTS HONESTLY
            - When a tool returns a "note", repeat its substance in your answer. It reports
              things like rows that could not be analyzed or groups too small to trust.
            - Never present an average over a handful of answers as a firm result.

            COUNTING FILLED OR EMPTY ROWS
            - To count rows that have a value in a column, call
              filter_rows(where_col=..., op="is_not_empty") and then row_count on the result.
            - To count rows that are blank, use op="is_empty" the same way.
            - Do NOT compute these by subtracting one count from another: it is easy to get
              wrong and it hides which column you actually used.

            TABLE RESULTS ARE PREVIEWS
            - Tables you return are shown to the user as a preview of the first 20 rows only.
            - Whenever the result is larger, the system automatically attaches a download link
              to the complete CSV. You do not need to do anything for this to happen.
            - Therefore: never state or imply that a table contains all the matching rows,
              and never quote the number of rows in a table as if it were the total.
            - Never try to paginate, split a result into batches, or re-run a query to
              "show the remaining rows". Return the full result once and let the system
              handle the preview.
            - If the user needs every row, tell them to use the download link.

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
        rendered = render_prompt(template, columns=cols)

        # render_prompt uses str.format, which raises on the literal JSON braces in the
        # contract examples above and then returns the template untouched -- leaving
        # "{columns}" in place, so the model never saw the column names. Substitute it
        # directly; this also covers a DB-stored prompt with the same placeholder.
        if "{columns}" in rendered:
            rendered = rendered.replace("{columns}", ", ".join(str(c) for c in cols))
        return rendered

    def _build_agent(self, model: LiteLLMModel, instructions: str) -> Optional[CodeAgent]:
        tools = [
            DescribeTool(self.data),
            MissingValuesTool(self.data),
            UniqueValuesTool(self.data),
            CorrelationTool(self.data),
            SampleRowsTool(self.data),
            TopRowsTool(self.data),
            FilterRowsTool(self.data),
            RowCountTool(self.data),
            AggregateTool(self.data),
            CrosstabTool(self.data),
            CompareGroupsTool(self.data),
            KeywordsTool(self.data),
            PlotTool(self.data, output_dir=self._plots_dir or os.getenv("DATACHAT_PLOTS_DIR", "/tmp/datachat_plots")),
            TrendTool(self.data),
            SentimentAnalysisTool(self.data, model=self._model),
            ClassifyTool(self.data, model=self._model),
            ExportCsvTool(self.data, exporter=self),
        ]

        base_kwargs: dict[str, Any] = {
            "tools": tools,
            "model": model,
            "instructions": instructions,
            "max_steps": self._max_steps,
            "additional_authorized_imports": ["json"],
        }

        def _final_answer_contract_check(*args: Any, **kwargs: Any) -> bool:
            candidate = args[0] if args else (
                kwargs.get("final_answer") or kwargs.get("answer") or kwargs.get("output")
            )
            payload, passed, final_kind, reason = _coerce_final_payload(candidate)

            self._last_final_answer_check_passed = passed
            self._last_final_kind = final_kind

            runtime_logger.info(
                "final_answer_check request_id=%s engine=smolagents user=%s passed=%s final_kind=%s reason=%s",
                self._active_request_id or "n/a",
                self.user_name,
                passed,
                final_kind or "none",
                reason,
            )
            return passed

        try:
            agent = CodeAgent(**{**base_kwargs, "final_answer_checks": [_final_answer_contract_check]})
            self._final_answer_checks_supported = True
            runtime_logger.info(
                "engine_init_guardrail engine=smolagents user=%s final_answer_checks_supported=%s",
                self.user_name,
                True,
            )
            return agent
        except TypeError:
            self._final_answer_checks_supported = False
            runtime_logger.info(
                "engine_init_guardrail engine=smolagents user=%s final_answer_checks_supported=%s",
                self.user_name,
                False,
            )
            return CodeAgent(**base_kwargs)

    # --- public API ---

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        html = get_static_bootstrap_html(lang)
        return EngineBootstrapResult(suggested_questions_html=html)

    def chat(self, message: str, request_id: Optional[str] = None) -> Any:
        self._active_request_id = request_id or "n/a"
        self._last_final_answer_check_passed = None
        self._last_final_kind = None

        runtime_logger.info(
            "chat_start request_id=%s engine=smolagents user=%s message_len=%s",
            self._active_request_id,
            self.user_name,
            len(str(message or "")),
        )

        if self._agent is None:
            runtime_logger.info(
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
                self._active_request_id,
                self.user_name,
                str(e)[:160],
            )
            self._active_request_id = None
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
            self._last_final_answer_check_passed = False
            self._last_final_kind = None

        runtime_logger.info(
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

    def get_last_trace(self) -> Optional[dict[str, Any]]:
        if self._last_run_result is None:
            return None
        return {"run_result": self._last_run_result, "duration_ms": self._last_run_duration_ms}

    # --- full-result CSV exports ---

    def register_export(self, records: list[dict[str, Any]], hint: str = "export") -> Tuple[str, str]:
        """
        Write `records` as a complete CSV and return (token, download_filename).

        Table responses are truncated to a preview before reaching the client, so the
        full result is written here and handed back as an opaque token.

        The on-disk name is the token itself: the caller-supplied `hint` only ever
        becomes the *suggested download* name, never a path component, so no caller
        (tool, LLM or client) can steer where we write.
        """
        if not self._plots_dir:
            raise RuntimeError("export directory not initialised")

        os.makedirs(self._plots_dir, exist_ok=True)

        safe_hint = re.sub(r"[^A-Za-z0-9._-]+", "_", str(hint or "export")).strip("_") or "export"
        if safe_hint.lower().endswith(".csv"):
            safe_hint = safe_hint[:-4]
        # Column names can be whole sentences (survey headers); keep the filename usable.
        safe_hint = safe_hint[:_MAX_EXPORT_NAME_LEN].strip("_") or "export"
        download_filename = f"{safe_hint}.csv"

        token = uuid.uuid4().hex
        path = os.path.join(self._plots_dir, f"{token}.csv")
        pd.DataFrame(records).to_csv(path, index=False)

        self._exports[token] = (path, download_filename)
        self._evict_old_exports()

        runtime_logger.info(
            "export_registered engine=smolagents user=%s token=%s rows=%s cols=%s",
            self.user_name,
            token,
            len(records),
            len(records[0]) if records else 0,
        )
        return token, download_filename

    def resolve_export(self, token: str) -> Optional[Tuple[str, str]]:
        """Return (path, download_filename) for a token issued by this session, or None."""
        entry = self._exports.get(str(token or ""))
        if entry and os.path.isfile(entry[0]):
            return entry
        return None

    def _evict_old_exports(self) -> None:
        """Keep only the most recent exports so a long session cannot fill the disk."""
        while len(self._exports) > _MAX_SESSION_EXPORTS:
            # dicts preserve insertion order, so the first key is the oldest export
            old_token = next(iter(self._exports))
            old_path, _ = self._exports.pop(old_token)
            try:
                os.unlink(old_path)
            except OSError:
                pass

    def close(self) -> None:
        plots_dir_removed = False
        user_dir_removed = False
        cleanup_error = ""

        try:
            # export tokens die with the session: the CSVs live under _plots_dir
            self._exports.clear()

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
            "cleanup_result engine=smolagents user=%s plots_dir_removed=%s user_dir_removed=%s cleanup_error=%s",
            self.user_name,
            plots_dir_removed,
            user_dir_removed,
            cleanup_error or "none",
        )
        return
