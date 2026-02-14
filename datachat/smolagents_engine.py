import os
import json
import textwrap
import re
import shutil
import uuid
import time
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

@dataclass
class SmolagentsEngine(DataChatEngine):
    """
    Smolagents engine (initial wiring).

    NOTE: enabled via DATACHAT_ENGINE=smolagents. Tools and real behavior will be added step-by-step.
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

    def __post_init__(self) -> None:
        # Create a per-user/per-session plots directory for safe cleanup.
        safe_user = re.sub(r"[^A-Za-z0-9._-]+", "_", str(self.user_name or "user")).strip("_")
        session_id = uuid.uuid4().hex
        base_dir = os.getenv("DATACHAT_PLOTS_DIR", "/tmp/datachat_plots")
        self._user_plots_dir = os.path.join(base_dir, safe_user)
        self._plots_dir = os.path.join(self._user_plots_dir, session_id)
        
        provider = os.getenv("DATACHAT_PROVIDER", "Deepinfra").strip()
        configured_model = os.getenv("DATACHAT_MODEL", "").strip()

        if not configured_model:
            self._model = None
            self._agent = None
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
            return

        self._model = model
        
        self._agent = CodeAgent(
            tools=[
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
                TrendTool(self.data)
            ],
            model=model,
            max_steps=5,
            additional_authorized_imports=["json"],
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

    def chat(self, message: str) -> Any:
        if self._agent is None:
            self._last_run_result = None
            self._last_run_duration_ms = None
            return {
                "kind": "error",
                "message": (
                    "SmolagentsEngine non è configurato correttamente: "
                    "verifica DATACHAT_PROVIDER/DATACHAT_MODEL e la relativa API key."
                ),
                "code": "MISSING_CONFIG",
            }

        # Contesto minimo (non serializziamo tutto il dataframe: troppo grande)
        cols = list(self.data.columns)
        default_context = textwrap.dedent('''\
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
            → You MUST call the appropriate tool.
            → If aggregation depends on filtered data, prefer using the aggregate tool with filtering parameters instead of combining filter_rows and Python post-processing.
            → If the request is clear, do not call sample_rows or unique_values just to "check"; proceed directly with the necessary tool calls.

            2) High-level dataset questions  
            Examples: “What is this dataset about?”, “What information does it contain?”
            → Return a short natural-language summary (kind="text"), based only on the column names (and optionally a small sample via sample_rows if needed).  
            → Do NOT return raw describe output unless the user explicitly asks for statistics.

            3) Meta-system questions  
            Examples: “What can you do?”, “What analyses are possible?”
            → Return a short explanation (kind="text") describing the available analyses supported by the tools.

            TOOLS
            Use the appropriate tool when needed:
            • total record count → row_count
            • summaries or group statistics → aggregate
            • filtering conditions → filter_rows
            • previews or examples → sample_rows or top_rows
            • missing values → missing_values
            • unique values or categories → unique_values
            • correlations → correlation
            • charts or visualizations → plot
            • trends over time → trend
            • dataset overview statistics → describe

            RULES
            - Always answer in the same language as the user.
            - Use plain text only (no markdown formatting).
            - filter_rows returns a limited number of rows (pagination). For complete statistics, retrieve all pages using offset.
            - If a tool returns an error, return that error as-is.
            - If a request cannot be expressed with the available tools, explain the limitation briefly.

            OUTPUT
            - The final result must be a JSON object with a "kind" field.
            - Valid kinds are: text, table, image_path, error.
            - Do not add extra wrapping structures.
            - Be concise and concrete.
        ''')
        context_template = load_prompt(
            "data_chat_system",
            default_text=default_context,
        )
        context = render_prompt(context_template, columns=cols)

        try:
            started = time.time()
            
            run_result = self._agent.run(
                context + "\n\nUser question: " + str(message),
                reset=True,
                return_full_result=True,
            )
            
            self._last_run_result = run_result
            self._last_run_duration_ms = round((time.time() - started) * 1000, 2)
        
        except Exception as e:
            self._last_run_result = None
            self._last_run_duration_ms = None
            return {
                "kind": "error",
                "message": f"SmolagentsEngine failed to run: {e}",
                "code": "RUN_FAILED",
            }

        out = getattr(run_result, "output", None)


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

                # Case A: data is already correct
                if isinstance(data, list):
                    return obj

                # Case B: data is a nested payload dict
                if isinstance(data, dict):
                    nested_kind = str(data.get("kind") or "").strip().lower()
                    nested_data = data.get("data")

                    # {"data": {"kind":"table","data":[...]}}
                    if nested_kind == "table" and isinstance(nested_data, list):
                        obj["data"] = nested_data
                        return obj

                    # {"data": {"data":[...]}}  (no nested kind)
                    if isinstance(nested_data, list):
                        obj["data"] = nested_data
                        return obj

                return obj
            except Exception:
                return obj


        def _parse_kind_payload_obj(obj: Any) -> dict[str, Any] | None:
            if isinstance(obj, dict) and "kind" in obj:
                return obj
            return None

        def _parse_kind_payload_str(s: str) -> dict[str, Any] | None:
            s = s.strip()
            if not s:
                return None

            # A) strict JSON
            try:
                obj = json.loads(s)
                parsed = _parse_kind_payload_obj(obj)
                if parsed:
                    return parsed

                # JSON string inside JSON
                if isinstance(obj, str):
                    obj2 = json.loads(obj)
                    parsed2 = _parse_kind_payload_obj(obj2)
                    if parsed2:
                        return parsed2
            except Exception:
                pass

            # B) extract {...}
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

        # 1) Prefer structured output (dict)
        parsed = _parse_kind_payload_obj(out)
        if parsed is not None:
            return _unwrap_nested_table(parsed)

        # 2) If string, parse it
        if isinstance(out, str):
            parsed = _parse_kind_payload_str(out)
            if parsed is not None:
                return _unwrap_nested_table(parsed)

        # 3) Last fallback: plain text (avoid leaking RunResult(...))
        safe_text = out.strip() if isinstance(out, str) else ""
        return {
            "kind": "text",
            "text": safe_text or "Nessun output valido prodotto dall'agente.",
            "format": "plain",
        }

    def get_last_trace(self) -> dict[str, Any] | None:
        if self._last_run_result is None:
            return None
        return {
            "run_result": self._last_run_result,
            "duration_ms": self._last_run_duration_ms,
        }


    def close(self) -> None:
        # Remove only the session plots directory; keep user dir unless empty.
        if self._plots_dir and os.path.exists(self._plots_dir):
            shutil.rmtree(self._plots_dir, ignore_errors=True)

        if self._user_plots_dir and os.path.isdir(self._user_plots_dir):
            try:
                if not os.listdir(self._user_plots_dir):
                    os.rmdir(self._user_plots_dir)
            except Exception:
                pass
        return
