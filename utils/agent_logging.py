# utils/agent_logging.py
"""
Module for structured logging of agent runs.
Configures a dedicated logger and records results in JSON format per line.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = ["setup_agent_logger", "log_runresult"]

_agent_logger = logging.getLogger("agent_runs")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRunRecord:
    """Data structure for an agent run log record."""
    timestamp: str
    user: str
    namespace: str
    language: str
    question: str
    request_id: str = "-"
    state: Optional[Any] = None
    steps_count: int = 0
    duration_ms: Optional[float] = None
    token_usage: Optional[Dict[str, Optional[int]]] = None
    tool_calls: List[Dict[str, Any]] = None  # type: ignore
    vectors_count: int = 0
    answer_excerpt: str = ""
    # Extra field for extensibility (Open-Closed)
    extra: Optional[Dict[str, Any]] = None


def setup_agent_logger(
    path: str = "logs/agent_runs.log",
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
) -> None:
    """
    Configure the agent run logger if not already set up.

    :param path: Log file path.
    :param level: Logging level (default: INFO).
    :param formatter: Custom formatter (default: JSON message only).
    """
    # avoid duplicates
    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename.endswith("agent_runs.log")
        for h in _agent_logger.handlers
    ):
        return

    _agent_logger.setLevel(level)
    _agent_logger.propagate = False

    fh = logging.FileHandler(path)
    fh.setLevel(level)
    if formatter is None:
        formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    _agent_logger.addHandler(fh)


def log_runresult(
    result: Any,
    *,
    user: str,
    namespace: str,
    language: str,
    question: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a RunResult in structured JSON format.

    :param result: RunResult object from smolagents.
    :param user: Username of the user.
    :param namespace: Retrieval namespace.
    :param language: Query language.
    :param question: User question (truncated).
    :param extra: Additional fields for extensibility.
    """
    # Function-local import, not hoistable to module level: utils/logging_config.py
    # imports setup_agent_logger from this module at its own module level, so when
    # this module is loaded as part of that import chain, utils.logging_config is
    # already present in sys.modules but get_request_id is not yet defined on it
    # (this import runs before that name is bound). By the time log_runresult is
    # actually called, both modules are fully loaded and the import resolves.
    from utils.logging_config import get_request_id

    # Resolved outside the try block below: a failure here must not be
    # miscategorised as one of the RunResult-shape errors that block catches
    # and silently downgrades to a WARNING with no record written at all.
    request_id = get_request_id()

    try:
        steps: List[Dict[str, Any]] = getattr(result, "steps", []) or []
        timing = getattr(result, "timing", None)
        token_usage = getattr(result, "token_usage", None)

        
        vectors_count = sum(
            len(obs.get("vectors", []))
            for s in steps
            if isinstance(obs := s.get("observations"), dict) and "vectors" in obs
        )

        
        tool_calls = [
            {"tool": fn.get("name"), "args": fn.get("arguments")}
            for s in steps
            for tc in (s.get("tool_calls", []) or [])
            if (fn := (tc.get("function") or {}))
        ]

        
        answer_excerpt = ""
        if isinstance(out := getattr(result, "output", None), dict):
            answer_excerpt = str(out.get("answer", "")).strip()[:180]

        
        truncated_question = question[:200]

        
        record = AgentRunRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user=user,
            namespace=namespace,
            language=language,
            question=truncated_question,
            request_id=request_id,
            state=getattr(result, "state", None),
            steps_count=len(steps),
            duration_ms=round(float(getattr(timing, "duration", 0.0)) * 1000, 2) if timing else None,
            token_usage={
                "input": getattr(token_usage, "input_tokens", None),
                "output": getattr(token_usage, "output_tokens", None),
                "total": getattr(token_usage, "total_tokens", None),
            } if token_usage else None,
            tool_calls=tool_calls,
            vectors_count=vectors_count,
            answer_excerpt=answer_excerpt,
            extra=extra,
        )

        _agent_logger.info(json.dumps(asdict(record), ensure_ascii=False))

    except (AttributeError, TypeError, ValueError) as e:
        # Specific errors for failed getattr or wrong types
        logger.warning("event=agent_run_log_failed error=%s", e)

