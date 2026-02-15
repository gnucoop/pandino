
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class EngineBootstrapResult:
    """
    Result of the DataChat bootstrap phase.
    For AS-IS compatibility we keep it minimal and aligned with Dino expectations.
    """
    suggested_questions_html: str | None


class DataChatEngine(Protocol):
    """
    Minimal engine interface for DataChat.
    - No HTTP knowledge
    - No token management
    - No output normalization
    """

    def bootstrap(self, lang: str) -> EngineBootstrapResult: ...
    def chat(self, message: str, request_id: str | None = None) -> Any: ...
    def close(self) -> None: ...
