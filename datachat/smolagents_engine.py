from dataclasses import dataclass
from typing import Any

import pandas as pd

from datachat.engine_interface import DataChatEngine, EngineBootstrapResult


@dataclass
class SmolagentsEngine(DataChatEngine):
    """
    Skeleton Smolagents engine.

    IMPORTANT: This class is not wired into the factory yet.
    It exists only to establish the integration surface and imports.
    """
    api_key: str
    user_name: str
    llm: Any
    data: pd.DataFrame

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        # For now: no real bootstrap. We keep it explicit and safe.
        return EngineBootstrapResult(suggested_questions_html=None)

    def chat(self, message: str) -> Any:
        # For now: return a contract-mode error output (dict-based)
        return {
            "kind": "error",
            "message": "SmolagentsEngine is not enabled yet.",
            "code": "NOT_ENABLED",
        }

    def close(self) -> None:
        # Nothing to clean up yet.
        return
