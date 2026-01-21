from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from smolagents import CodeAgent

from datachat.engine_interface import DataChatEngine, EngineBootstrapResult
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

    def __post_init__(self) -> None:
        
        provider = os.getenv("DATACHAT_PROVIDER", "Deepinfra").strip()
        configured_model = os.getenv("DATACHAT_MODEL", "").strip()

        if not configured_model:
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
            self._agent = None
            return

        self._agent = CodeAgent(
            tools=[],
            model=model,
            max_steps=2,
            additional_authorized_imports=[],
        )

        

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        # For now: no real bootstrap. We keep it explicit and safe.
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
            f"Dataset columns: {cols}\n"
            "Rules:\n"
            "- Answer in the same language used by the user.\n"
            "- Use Python code ONLY if strictly necessary for data analysis (e.g., to compute statistics or describe subsets). ALWAYS wrap any code in <code>...</code>.\n"
            "- If no analysis is needed, respond directly by executing <code>final_answer('your concise answer')</code>.\n"
            "- If the user asks for a plot/table, describe what you would show (e.g., 'I would show a bar chart with...'), but do not generate code yet.\n"
            "- Keep answers concise and concrete.\n"
        )

        try:
            out = self._agent.run(context + "\n\nUser question: " + str(message), reset=False)
        except Exception as e:
            return {
                "kind": "error",
                "message": f"SmolagentsEngine failed to run: {e}",
                "code": "RUN_FAILED",
            }

        return {"kind": "text", "text": str(out), "format": "plain"}


    def close(self) -> None:
        return
