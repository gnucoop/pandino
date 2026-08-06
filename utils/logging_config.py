# utils/logging_config.py
"""
Centralised logging bootstrap and request logging context.

Two responsibilities live here: configuring the operational, audit and
datachat channels (:func:`bootstrap_logging`), and owning the per-request
logging context - the ContextVars, their accessors, the filter that reads
them and the Flask hooks that bind them
(:func:`register_request_context_hooks`).

This module is imported before everything else in ``main.py``, so it is
deliberately kept free of application imports: stdlib only, plus ``dotenv``
and the two existing logging helpers (``utils.agent_logging``,
``utils.runtime_logging``). Importing it must not pull in config, routes,
services, infrastructure, datachat, llm, flask, pandas, matplotlib,
smolagents, litellm or langchain. The single Flask import is therefore
function-local inside :func:`register_request_context_hooks`, which runs
only after ``main.py`` has created the app.
"""

import logging
import os
import secrets
from contextvars import ContextVar
from datetime import datetime, timezone

from dotenv import load_dotenv

from utils.agent_logging import setup_agent_logger
from utils.runtime_logging import setup_datachat_runtime_logger

__all__ = [
    "bootstrap_logging",
    "get_request_id",
    "register_request_context_hooks",
    "reset_request_context",
    "set_request_context",
]

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

#: Value rendered when no request context is bound.
CONTEXT_UNSET = "-"

#: Attribute under which the request-context tokens are parked on ``flask.g``.
_G_TOKENS_ATTR = "_maui_log_context_tokens"

#: Marker attribute recording that the hooks are already registered on an app.
_HOOKS_MARKER = "_maui_request_context_hooks"

# Created once per process, at module level. Creating a ContextVar inside a
# function is a documented anti-pattern: the objects are never garbage
# collected, so a per-call ContextVar leaks. The "-" default is what keeps
# every record emitted outside an HTTP request - bootstrap, CLI, background
# code - renderable without special casing.
_request_id_var: ContextVar[str] = ContextVar("maui_request_id", default=CONTEXT_UNSET)
_app_id_var: ContextVar[str] = ContextVar("maui_app_id", default=CONTEXT_UNSET)

logger = logging.getLogger(__name__)


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

    The values are read from the ambient ContextVars, which under gevent
    resolve per greenlet, so concurrent requests cannot borrow each other's
    id. Outside a request the vars hold ``"-"``.

    Attached to the handler, never to a logger: records propagated up from
    child loggers bypass ancestor logger filters but always traverse
    ancestor handlers. Handler placement is what makes the defaults
    universal, including for third-party libraries.
    """

    def filter(self, record):
        # The hasattr guard comes first so an explicit
        # extra={"request_id": ...} from a caller still wins over ambient.
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_var.get() or CONTEXT_UNSET
        if not hasattr(record, "app_id"):
            record.app_id = _app_id_var.get() or CONTEXT_UNSET
        return True


def get_request_id() -> str:
    """Return the ambient request id, or ``"-"`` outside a request."""
    return _request_id_var.get() or CONTEXT_UNSET


def set_request_context(request_id=None, app_id=None) -> tuple:
    """Bind the given context values, leaving the others untouched.

    :return: the ``(request_id_token, app_id_token)`` pair to hand back to
        :func:`reset_request_context`. Either element is ``None`` when the
        corresponding value was not supplied.
    """
    request_token = _request_id_var.set(request_id) if request_id is not None else None
    app_token = _app_id_var.set(app_id) if app_id is not None else None
    return request_token, app_token


def reset_request_context(tokens) -> None:
    """Restore the values captured by :func:`set_request_context`.

    Accepts ``None`` and is a no-op in that case, so a teardown that runs
    without a matching bind cannot raise.
    """
    if not tokens:
        return

    request_token, app_token = tokens
    if request_token is not None:
        _request_id_var.reset(request_token)
    if app_token is not None:
        _app_id_var.reset(app_token)


def register_request_context_hooks(app) -> None:
    """Bind a fresh request id for the lifetime of each HTTP request.

    Must be called before the app serves its first request: since Flask 2.3,
    registering a ``before_request`` handler on an app that has already
    handled one raises a setup error.

    Idempotent: a marker on the app makes a second call a no-op, so an
    accidental double registration cannot bind two ids per request or leave
    a token pair orphaned on ``flask.g``.

    ``app_id`` is deliberately left at ``"-"``: the system has no reliable
    tenant concept yet (``X-CLIENT`` is read only in ``routes/auth.py``,
    defaulted to "dino", and discarded), so populating it is a Phase 3
    decision. :func:`set_request_context` already accepts it.
    """
    # Function-local by design: this module is imported as the first
    # statement of main.py, before Flask exists. By the time this function
    # is called, main.py has already created the Flask app, so flask is
    # loaded and this import is free.
    from flask import g  # noqa: PLC0415

    if getattr(app, _HOOKS_MARKER, False):
        return
    setattr(app, _HOOKS_MARKER, True)

    @app.before_request
    def _bind_request_context():
        setattr(g, _G_TOKENS_ATTR, set_request_context(request_id=secrets.token_hex(8)))

    @app.after_request
    def _emit_request_id_header(response):
        request_id = get_request_id()
        if request_id != CONTEXT_UNSET:
            response.headers["X-Request-ID"] = request_id
        return response

    @app.teardown_request
    def _unbind_request_context(exc=None):
        # Runs even when the view raised. abort() (routes/utils.py) yields
        # an HTTPException that Flask finalises into a response, so
        # after_request still runs there - but an unhandled exception skips
        # it entirely, and only teardown is guaranteed. Without the reset a
        # keep-alive greenlet, which serves several sequential requests,
        # would attribute the next request's records to this id.
        reset_request_context(getattr(g, _G_TOKENS_ATTR, None))


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

    if level_warning:
        logger.warning("event=log_level_resolution_fallback detail=%s", level_warning)

    # --- Audit channel: FileHandler opens eagerly, so the parent directory
    # must exist first. It is gitignored and created by nothing in the image.
    try:
        parent = os.path.dirname(agent_runs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        setup_agent_logger(path=agent_runs_path)
    except Exception as exc:  # noqa: BLE001 - startup must never be aborted here
        logger.warning(
            "event=agent_run_log_channel_unavailable path=%s error_type=%s error=%s",
            agent_runs_path,
            type(exc).__name__,
            exc,
        )

    # --- DataChat runtime channel: declared exception, isolated from root.
    return setup_datachat_runtime_logger()
