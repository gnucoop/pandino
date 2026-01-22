import os
import json
from dataclasses import dataclass, field
from typing import Any
import logging

import pandas as pd
from smolagents import CodeAgent, LiteLLMModel
from datachat.bootstrap import build_bootstrap_question
from datachat.engine_interface import DataChatEngine, EngineBootstrapResult
from datachat.tools.sample_rows_tool import SampleRowsTool
from llm.litellm_factory import build_litellm_model


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

    def __post_init__(self) -> None:
        
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
            # Conservative: do not crash engine creation. We'll return a user-facing error in chat().
            self._model = None
            self._agent = None
            return

        self._model = model
        
        self._agent = CodeAgent(
            tools=[SampleRowsTool(self.data)],
            model=model,
            max_steps=2,
            additional_authorized_imports=["json"],
        )


    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        """
        1:1 con pandasai bootstrap:
        - usa build_bootstrap_question(data, lang)
        - chiama il model e restituisce HTML (o comunque testo) come suggested_questions_html
        """
        question = build_bootstrap_question(self.data, lang)

        if self._model is None:
            logging.error("[datachat][smolagents][bootstrap] model is not configured")
            return EngineBootstrapResult(suggested_questions_html=None)

        try:
            msg = [{"role": "user", "content": question}]
            out = self._model(msg)

            if isinstance(out, str):
                content = out
            else:
                content = getattr(out, "content", None)
                if content is None:
                    content = str(out)

            html = str(content).strip()
            if not html:
                logging.error("[datachat][smolagents][bootstrap] empty content")
                return EngineBootstrapResult(suggested_questions_html=None)

            return EngineBootstrapResult(suggested_questions_html=html)

        except Exception as e:
            logging.exception(f"[datachat][smolagents][bootstrap] model call failed: {e}")
            return EngineBootstrapResult(suggested_questions_html=None)

    def chat(self, message: str) -> Any:
        if self._agent is None:
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
        context = (
            "You are a DataChat assistant. You help users understand a tabular dataset.\n"
            f"Dataset columns: {cols}\n\n"
            "RULES:\n"
            "- Answer in the same language used by the user.\n"
            "- Use Python code ONLY if strictly necessary for data analysis "
            "(e.g., to compute statistics or describe subsets).\n"
            "- ALWAYS wrap any Python code in <code>...</code>.\n"
            "- If no analysis is needed, respond directly by executing "
            "<code>final_answer('your concise answer')</code>.\n"
            "- If the user asks for a plot or a table, you MAY return a structured table "
            "as described below, or describe what would be shown.\n"
            "- Keep answers concise and concrete.\n\n"
            "TOOLS:\n"
            "- You have access to a tool named 'sample_rows'.\n"
            "- To call it, use: sample_rows(n=<int>, columns=[\"col1\",\"col2\",...]).\n"
            "- If the user asks to see example rows / a preview / 'show me N rows' / a small table preview, "
            "you MUST call 'sample_rows' and use its output as the table data.\n"
            "- Do NOT invent rows. Do NOT fabricate values.\n"
            "- If the user asks for a table that would require filtering/aggregation beyond a preview, "
            "do NOT fabricate data; answer in text and explain that only sample previews are available for tables right now.\n"
            "- After calling the tool (if needed), you MUST return the final answer as JSON using final_answer(...).\n\n"
            "- When using the output of a tool, you MUST serialize it using json.dumps(...) before passing it to final_answer.\n\n"
            "- NEVER use str(...) to serialize tool outputs.\n\n"
            "OUTPUT FORMAT (MANDATORY):\n"
            "- You MUST respond by calling <code>final_answer(...)</code>.\n"
            "- The argument of final_answer MUST be a JSON string.\n"
            "- The JSON string MUST start with '{' and end with '}'.\n"
            "- Return ONLY this code block, nothing else.\n\n"
            "- All string values MUST be valid JSON strings (escape inner quotes using \\\" ).\n\n"
            "Valid JSON schemas:\n"
            "1) Text:\n"
            "<code>final_answer('{\"kind\":\"text\",\"text\":\"...\",\"format\":\"plain\"}')</code>\n"
            "2) Table (small result only):\n"
            "<code>final_answer('{\"kind\":\"table\",\"data\":[{\"col1\":\"value1\",\"col2\":\"value2\"}]}')</code>\n\n"
            "TABLE RULES (MANDATORY):\n"
            "- 'data' MUST be a JSON array of objects (list of dicts).\n"
            "- Use at most 5 rows and at most 10 columns.\n"
            "- All cell values MUST be JSON scalars only: string, number, boolean, or null.\n"
            "- NEVER include '{' or '}' characters inside any string cell value.\n"
            "- If a value is structured (object/array/JSON), convert it to a flat string like "
            "\"key1=value1; key2=value2\" (no braces, no inner quotes).\n"
            "- If a cell value would be complex (object/array) or would require nested quotes, "
            "replace it with a short string summary.\n\n"
            "IMPORTANT:\n"
            "- The JSON must be valid.\n"
            "- Never include Python code inside the JSON.\n"
        )

        try:
            out = self._agent.run(context + "\n\nUser question: " + str(message), reset=False)
        except Exception as e:
            return {
                "kind": "error",
                "message": f"SmolagentsEngine failed to run: {e}",
                "code": "RUN_FAILED",
            }

        raw = str(out).strip()

        def _parse_kind_payload(s: str) -> dict[str, Any] | None:
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "kind" in obj:
                    return obj
            except Exception:
                return None
            return None

        # 1) Tentativo diretto
        parsed = _parse_kind_payload(raw)
        if parsed is not None:
            return parsed

        # 2) Se è una stringa JSON "annidata" (es. "\"{...}\""), un json.loads in più la sblocca
        try:
            unwrapped = json.loads(raw)
            if isinstance(unwrapped, str):
                parsed = _parse_kind_payload(unwrapped)
                if parsed is not None:
                    return parsed
        except Exception:
            pass

        # 3) Estrazione conservativa tra prima { e ultima } (per output tipo final_answer('...'))
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            parsed = _parse_kind_payload(candidate)
            if parsed is not None:
                return parsed

        # 4) Ultimo fallback: testo
        return {"kind": "text", "text": raw, "format": "plain"}


    def close(self) -> None:
        return
