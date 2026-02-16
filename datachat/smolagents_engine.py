import os
import json
import textwrap
import re
import shutil
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from smolagents import CodeAgent, LiteLLMModel

from datachat.bootstrap_static import get_static_bootstrap_html
from datachat.engine_interface import DataChatEngine, EngineBootstrapResult
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.row_count_tool import RowCountTool
from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.unique_values_tool import UniqueValuesTool
from datachat.tools.correlation_tool import CorrelationTool
from datachat.tools.plot_tool import PlotTool
from datachat.tools.trend_tool import TrendTool
from llm.litellm_factory import build_litellm_model
from prompt_utils import load_prompt, render_prompt


plots_dir = os.getenv("DATACHAT_PLOTS_DIR", "/tmp/datachat_plots")
runtime_logger = logging.getLogger("datachat.runtime")
_ALLOWED_FINAL_KINDS = {"text", "table", "image_path", "error"}


def _parse_kind_payload_obj(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict) and "kind" in obj:
        return obj
    return None


def _parse_kind_payload_str(s: str) -> dict[str, Any] | None:
    s = (s or "").strip()
    if not s:
        return None

    # A) strict JSON
    try:
        obj = json.loads(s)
        parsed = _parse_kind_payload_obj(obj)
        if parsed:
            return parsed

        if isinstance(obj, str):
            obj2 = json.loads(obj)
            parsed2 = _parse_kind_payload_obj(obj2)
            if parsed2:
                return parsed2
    except Exception:
        pass

    # B) conservative extraction: take substring between first { and last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1].strip()
        try:
            obj = json.loads(candidate)
            parsed = _parse_kind_payload_obj(obj)
            if parsed:
                return parsed
        except Exception:
            pass

    return None


def _unwrap_nested_table(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Fix common LLM mistake:
    {"kind":"table","data": {"kind":"table","data":[...]}}
    -> {"kind":"table","data":[...]}
    """
    try:
        kind = str(obj.get("kind") or "").strip().lower()
        if kind != "table":
            return obj

        data = obj.get("data")

        if isinstance(data, list):
            return obj

        if isinstance(data, dict):
            nested_kind = str(data.get("kind") or "").strip().lower()
            nested_data = data.get("data")

            if nested_kind == "table" and isinstance(nested_data, list):
                obj["data"] = nested_data
                return obj

            if isinstance(nested_data, list):
                obj["data"] = nested_data
                return obj

        return obj
    except Exception:
        return obj


def _validate_contract_payload(payload: dict[str, Any]) -> tuple[bool, str | None, str]:
    raw_kind = payload.get("kind")
    kind = str(raw_kind or "").strip().lower()

    if kind not in _ALLOWED_FINAL_KINDS:
        return False, kind or None, "INVALID_KIND"

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

    return False, kind or None, "INVALID_KIND"


@dataclass
class SmolagentsEngine(DataChatEngine):
    """
    Smolagents engine.

    Enabled via DATACHAT_ENGINE=smolagents.
    """

    api_key: str
    user_name: str
    llm: Any  # kept for interface compatibility; not used here yet
    data: pd.DataFrame

    _agent: CodeAgent | None = field(default=None, init=False, repr=False)
    _model: LiteLLMModel | None = field(default=None, init=False, repr=False)
    _plots_dir: str | None = field(default=None, init=False, repr=False)
    _user_plots_dir: str | None = field(default=None, init=False, repr=False)
    _last_run_result: Any | None = field(default=None, init=False, repr=False)
    _last_run_duration_ms: float | None = field(default=None, init=False, repr=False)
    _configured_model: str = field(default="", init=False, repr=False)
    _provider: str = field(default="", init=False, repr=False)
    _max_steps: int = field(default=12, init=False, repr=False)
    _instructions: str = field(default="", init=False, repr=False)
    _last_final_answer_check_passed: bool | None = field(default=None, init=False, repr=False)
    _last_final_kind: str | None = field(default=None, init=False, repr=False)
    _active_request_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Create a per-user/per-session plots directory for safe cleanup.
        safe_user = re.sub(r"[^A-Za-z0-9._-]+", "_", str(self.user_name or "user")).strip("_")
        session_id = uuid.uuid4().hex
        base_dir = os.getenv("DATACHAT_PLOTS_DIR", "/tmp/datachat_plots")
        self._user_plots_dir = os.path.join(base_dir, safe_user)
        self._plots_dir = os.path.join(self._user_plots_dir, session_id)

        provider = os.getenv("DATACHAT_PROVIDER", "Deepinfra").strip()
        configured_model = os.getenv("DATACHAT_MODEL", "").strip()
        self._provider = provider
        self._configured_model = configured_model
        try:
            self._max_steps = max(1, int(os.getenv("DATACHAT_MAX_STEPS", "12")))
        except ValueError:
            self._max_steps = 12

        runtime_logger.info(
            "engine_init engine=smolagents user=%s provider=%s model=%s max_steps=%s",
            self.user_name,
            provider,
            configured_model or "missing",
            self._max_steps,
        )

        if not configured_model:
            self._model = None
            self._agent = None
            runtime_logger.info(
                "engine_init_result engine=smolagents user=%s status=error error_code=MISSING_CONFIG",
                self.user_name,
            )
            return

        try:
            model = build_litellm_model(
                provider=provider,
                configured_model=configured_model,
                temperature=0.0,
            )
        except Exception:
            self._model = None
            self._agent = None
            runtime_logger.info(
                "engine_init_result engine=smolagents user=%s status=error error_code=MISSING_CONFIG",
                self.user_name,
            )
            return

        self._model = model

        # Build custom instructions once per session and inject them into CodeAgent.
        cols = list(self.data.columns)
        default_context = textwrap.dedent(
            '''\
            You are DataChat, a cognitive assistant that helps users explore a tabular dataset.

            DATASET
            - The dataset has the following columns: {columns}.

            PURPOSE
            - Understand the user’s intent.
            - If the user requests a concrete data operation, translate it into explicit tool calls.
            - Do NOT invent columns, rows, or values.
            - 

            REQUEST TYPES

            1) Concrete data operations  
            Examples: counts, filtering, summaries, correlations, charts, trends.
            → You MUST call the appropriate tool.

            2) High-level dataset questions  
            Examples: “What is this dataset about?”, “What information does it contain?”
            → Return a short natural-language summary (kind=""text""), based only on the column names (and optionally a small sample via sample_rows if needed).  
            → Do NOT return raw describe output unless the user explicitly asks for statistics.

            3) Meta-system questions  
            Examples: “What can you do?”, “What analyses are possible?”
            → Return a short explanation (kind=""text"") describing the available analyses supported by the tools.

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
            - kind="text"       -> {{"kind":"text","text":"..."}}
            - kind="table"      -> {{"kind":"table","data":[...]}}
            - kind="image_path" -> {{"kind":"image_path","path":"..."}}
            - kind="error"      -> {{"kind":"error","message":"..."}}

            When a tool already returns a valid final contract object:
            - pass it directly to final_answer(...) without re-wrapping.

            Examples:
            - Wrong: {{"kind":"image_path","value":{{"kind":"image_path","path":"/tmp/plot.png"}}}}
            - Correct: {{"kind":"image_path","path":"/tmp/plot.png"}}
        '''
        )
        context_template = load_prompt(
            "data_chat_system",
            default_text=default_context,
        )
        self._instructions = render_prompt(context_template, columns=cols)

        def _final_answer_contract_check(*args: Any, **kwargs: Any) -> bool:
            candidate = args[0] if args else (
                kwargs.get("final_answer")
                or kwargs.get("answer")
                or kwargs.get("output")
            )

            parsed = _parse_kind_payload_obj(candidate)
            if parsed is None and isinstance(candidate, str):
                parsed = _parse_kind_payload_str(candidate)

            if parsed is None:
                self._last_final_answer_check_passed = False
                self._last_final_kind = None
                runtime_logger.info(
                    "final_answer_check request_id=%s engine=smolagents user=%s passed=%s final_kind=%s reason=%s",
                    self._active_request_id or "n/a",
                    self.user_name,
                    False,
                    "none",
                    "NON_JSON_OR_NO_KIND",
                )
                return False

            parsed = _unwrap_nested_table(parsed)
            passed, final_kind, reason = _validate_contract_payload(parsed)
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

        agent_kwargs: dict[str, Any] = {
            "tools": [
                DescribeTool(self.data),
                MissingValuesTool(self.data),
                UniqueValuesTool(self.data),
                CorrelationTool(self.data),
                SampleRowsTool(self.data),
                TopRowsTool(self.data),
                FilterRowsTool(self.data),
                RowCountTool(self.data),
                AggregateTool(self.data),
                PlotTool(self.data, output_dir=self._plots_dir or plots_dir),
                TrendTool(self.data),
            ],
            "model": model,
            "instructions": self._instructions,
            "max_steps": self._max_steps,
            "additional_authorized_imports": ["json"],
        }
        agent_kwargs_with_checks = dict(agent_kwargs)
        agent_kwargs_with_checks["final_answer_checks"] = [_final_answer_contract_check]

        try:
            self._agent = CodeAgent(**agent_kwargs_with_checks)
            runtime_logger.info(
                "engine_init_guardrail engine=smolagents user=%s final_answer_checks_supported=%s",
                self.user_name,
                True,
            )
        except TypeError:
            self._agent = CodeAgent(**agent_kwargs)
            runtime_logger.info(
                "engine_init_guardrail engine=smolagents user=%s final_answer_checks_supported=%s",
                self.user_name,
                False,
            )

        runtime_logger.info(
            "engine_init_result engine=smolagents user=%s status=ok",
            self.user_name,
        )

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        """
        Bootstrap statico locale:
        - usa il parametro lang ricevuto dal frontend
        - restituisce HTML statico localizzato con fallback a inglese
        - non chiama il model (nessuna generazione LLM)
        """
        html = get_static_bootstrap_html(lang)
        return EngineBootstrapResult(suggested_questions_html=html)

    def chat(self, message: str, request_id: str | None = None) -> Any:
        self._active_request_id = request_id or "n/a"
        self._last_final_answer_check_passed = None
        self._last_final_kind = None

        runtime_logger.info(
            "chat_start request_id=%s engine=smolagents user=%s message_len=%s",
            request_id or "n/a",
            self.user_name,
            len(str(message or "")),
        )

        if self._agent is None:
            self._last_run_result = None
            self._last_run_duration_ms = None
            runtime_logger.info(
                "chat_error request_id=%s engine=smolagents user=%s error_code=MISSING_CONFIG",
                request_id or "n/a",
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
            run_result = self._agent.run(
                str(message),
                reset=True,
                return_full_result=True,
            )
            self._last_run_result = run_result
            self._last_run_duration_ms = round((time.time() - started) * 1000, 2)
        except Exception as e:
            self._last_run_result = None
            self._last_run_duration_ms = None
            runtime_logger.info(
                "chat_error request_id=%s engine=smolagents user=%s error_code=RUN_FAILED error_message_short=%s",
                request_id or "n/a",
                self.user_name,
                str(e)[:160],
            )
            self._active_request_id = None
            return {
                "kind": "error",
                "message": f"SmolagentsEngine failed to run: {e}",
                "code": "RUN_FAILED",
            }

        out = getattr(run_result, "output", None)

        parsed = _parse_kind_payload_obj(out)
        if parsed is None and isinstance(out, str):
            parsed = _parse_kind_payload_str(out)

        result_payload: dict[str, Any]
        if parsed is not None:
            parsed = _unwrap_nested_table(parsed)
            passed, final_kind, reason = _validate_contract_payload(parsed)
            self._last_final_answer_check_passed = passed
            self._last_final_kind = final_kind

            if passed:
                parsed["kind"] = final_kind
                result_payload = parsed
            else:
                result_payload = {
                    "kind": "text",
                    "text": f"Nessun output finale valido prodotto dall'agente ({reason}).",
                    "format": "plain",
                }
        else:
            self._last_final_answer_check_passed = False
            self._last_final_kind = None
            safe_text = out.strip() if isinstance(out, str) else ""
            result_payload = {
                "kind": "text",
                "text": safe_text or "Nessun output valido prodotto dall'agente.",
                "format": "plain",
            }

        runtime_logger.info(
            "chat_end request_id=%s engine=smolagents user=%s duration_ms=%s response_kind=%s final_answer_check_passed=%s final_kind=%s",
            request_id or "n/a",
            self.user_name,
            self._last_run_duration_ms,
            result_payload.get("kind"),
            bool(self._last_final_answer_check_passed),
            self._last_final_kind or "none",
        )
        self._active_request_id = None
        return result_payload


    def get_last_trace(self) -> dict[str, Any] | None:
        if self._last_run_result is None:
            return None
        return {
            "run_result": self._last_run_result,
            "duration_ms": self._last_run_duration_ms,
        }


    def close(self) -> None:
        # Remove only the session plots directory; keep user dir unless empty.
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
            "cleanup_result engine=smolagents user=%s plots_dir_removed=%s user_dir_removed=%s cleanup_error=%s",
            self.user_name,
            plots_dir_removed,
            user_dir_removed,
            cleanup_error or "none",
        )
        return
