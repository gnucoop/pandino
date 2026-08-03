# utils/logging_config.py
"""
Centralised logging bootstrap.

This module is imported before everything else in ``main.py``, so it is
deliberately kept free of application imports: stdlib only, plus ``dotenv``
and the two existing logging helpers (``utils.agent_logging``,
``utils.runtime_logging``). Importing it must not pull in config, routes,
services, infrastructure, datachat, llm, flask, pandas, matplotlib,
smolagents, litellm or langchain.
"""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from utils.agent_logging import setup_agent_logger
from utils.runtime_logging import setup_datachat_runtime_logger

__all__ = ["bootstrap_logging"]

DEFAULT_LOG_LEVEL = "WARNING"
DEFAULT_AGENT_RUNS_LOG_PATH = "logs/agent_runs.log"

#: Marker attribute used to make :func:`bootstrap_logging` idempotent,
#: mirroring the pattern in ``utils/runtime_logging.py``.
_HANDLER_MARKER = "_maui_bootstrap"

#: Log levels accepted from the environment. Deliberately a whitelist rather
#: than a ``logging.getLevelName`` lookup: that would accept ``NOTSET``,
#: which on the ROOT logger means no threshold at all and would let every
#: DEBUG record from every third-party library through, silently defeating
#: the WARNING default.
_ALLOWED_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "request_id=%(request_id)s app_id=%(app_id)s %(message)s"
)


class UtcIsoFormatter(logging.Formatter):
    """Formatter emitting UTC ISO-8601 timestamps.

    The offset is rendered as ``+00:00`` with microseconds, matching
    ``datetime.now(timezone.utc).isoformat()`` as used by the agent_runs
    JSONL channel, so the two channels stay correlatable.
    """

    def formatTime(self, record, datefmt=None):  # noqa: N802 (stdlib API)
        return datetime.fromtimestamp(record.created, timezone.utc).isoformat()


class ContextDefaultsFilter(logging.Filter):
    """Ensure every record carries ``request_id`` and ``app_id``.

    Attached to the handler, never to a logger: records propagated up from
    child loggers bypass ancestor logger filters but always traverse
    ancestor handlers. Handler placement is what makes the defaults
    universal, including for third-party libraries.
    """

    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "app_id"):
            record.app_id = "-"
        return True


def _resolve_level(raw: str) -> "tuple[int, str | None]":
    """Resolve a textual log level, never raising.

    :return: ``(level, warning_message_or_None)``.
    """
    if not raw or not raw.strip():
        return logging.WARNING, (
            "LOG_LEVEL is set to an empty or whitespace-only value; "
            "falling back to %s." % DEFAULT_LOG_LEVEL
        )

    candidate = raw.strip().upper()
    if candidate in _ALLOWED_LEVELS:
        return _ALLOWED_LEVELS[candidate], None

    return logging.WARNING, (
        "LOG_LEVEL=%r is not one of %s; falling back to %s."
        % (raw, sorted(_ALLOWED_LEVELS), DEFAULT_LOG_LEVEL)
    )


def bootstrap_logging() -> logging.Logger:
    """Configure the operational, audit and datachat logging channels.

    Idempotent: repeated calls never add a second root handler.

    :return: the configured ``datachat.runtime`` logger, so callers can
        publish it into ``app.config`` once the Flask app exists.
    """
    load_dotenv()

    level, level_warning = _resolve_level(os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL))
    agent_runs_path = (
        os.getenv("AGENT_RUNS_LOG_PATH") or DEFAULT_AGENT_RUNS_LOG_PATH
    )

    # --- Operational channel: root logger, stream only. Rotation, collection
    # and retention belong to the runtime environment, not to the process.
    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(h, _HANDLER_MARKER, False) for h in root.handlers):
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        handler.setLevel(level)
        handler.setFormatter(UtcIsoFormatter(LOG_FORMAT))
        handler.addFilter(ContextDefaultsFilter())
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            if getattr(handler, _HANDLER_MARKER, False):
                handler.setLevel(level)

    logger = logging.getLogger(__name__)

    if level_warning:
        logger.warning(level_warning)

    # --- Audit channel: FileHandler opens eagerly, so the parent directory
    # must exist first. It is gitignored and created by nothing in the image.
    try:
        parent = os.path.dirname(agent_runs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        setup_agent_logger(path=agent_runs_path)
    except Exception as exc:  # noqa: BLE001 - startup must never be aborted here
        logger.warning(
            "Agent run logging disabled: cannot open %s (%s: %s). "
            "The application will start without its audit channel.",
            agent_runs_path,
            type(exc).__name__,
            exc,
        )

    # --- DataChat runtime channel: declared exception, isolated from root.
    return setup_datachat_runtime_logger()
