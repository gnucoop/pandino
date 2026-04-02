import os
import shutil
from dataclasses import dataclass
from typing import Any

import pandas as pd

# from pandasai import Agent
try:
    from pandasai import Agent
except ImportError:
    Agent = None


from datachat.bootstrap import build_bootstrap_question
from datachat.engine_interface import DataChatEngine, EngineBootstrapResult


@dataclass
class PandasAIEngine(DataChatEngine):
    api_key: str
    user_name: str
    llm: Any
    data: pd.DataFrame
    open_charts: bool = False

    def __post_init__(self) -> None:
        if Agent is None:
            raise ImportError("pandasai is not installed")
        agent_config = {
            "llm": self.llm,
            "open_charts": self.open_charts,
            "save_charts": True,
            "save_charts_path": f"exports/charts/{self.user_name}",
            "custom_whitelisted_dependencies": ["tabulate"],
        }
        self._agent = Agent(self.data, config=agent_config)

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        # PandasAI must keep legacy bootstrap prompt to avoid regressions
        question = build_bootstrap_question(self.data, lang, prompt_version=1)
        resp = self.llm.invoke(question)
        html = getattr(resp, "content", None) if resp else None
        return EngineBootstrapResult(suggested_questions_html=html)

    def chat(self, message: str, request_id: str | None = None) -> Any:
        return self._agent.chat(message)

    def close(self) -> None:
        folder_path = f"exports/charts/{self.user_name}"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
