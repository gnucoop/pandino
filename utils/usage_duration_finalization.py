# utils/usage_duration_finalization.py
"""Best-effort Usage duration persistence orchestration.

Owns exactly one responsibility: at the ``after_request`` boundary, combine
the authoritative request duration (``utils.request_duration``) with the
request-local Usage row identity (``utils.usage_request_state``) to
finalize ``logs.duration_ms`` via ``update_usage_duration()``
(``infrastructure/database_pg.py``).

This module does not measure duration, does not know how a Usage row's
``log_id`` was captured, and does not implement any DB primitive - it only
composes the settled public APIs of those three modules. It never creates
a Usage row, never retries, and never lets a persistence failure alter an
otherwise valid HTTP response: a missing row or a database exception is
observed and logged, not propagated.
"""

import logging

from infrastructure.database_pg import update_usage_duration
from utils.request_duration import get_request_duration_ms
from utils.usage_request_state import get_usage_log_id

logger = logging.getLogger(__name__)

__all__ = [
    "register_usage_duration_finalization_hooks",
]

#: Marker attribute recording that the hooks are already registered on an
#: app, distinct from logging_config's and request_duration's own hooks
#: markers.
_HOOKS_MARKER = "_maui_usage_duration_finalization_hooks"


def register_usage_duration_finalization_hooks(app) -> None:
    """Bind best-effort Usage duration finalization for each HTTP request.

    Must be called before the app serves its first request, per the same
    Flask 2.3+ constraint documented on
    :func:`utils.logging_config.register_request_context_hooks`.

    Idempotent: a marker on the app makes a second call a no-op, mirroring
    the pattern used by :func:`utils.request_duration.register_request_duration_hooks`
    and :func:`utils.logging_config.register_request_context_hooks`, so this
    module owns its own marker and does not share or couple with either.

    Registration order relative to
    :func:`utils.request_duration.register_request_duration_hooks` is a
    correctness invariant, not style: Flask runs ``after_request`` hooks in
    reverse registration order (LIFO), so this must be registered *before*
    ``register_request_duration_hooks`` for this hook to observe the
    finalized duration on the same request.
    """
    if getattr(app, _HOOKS_MARKER, False):
        return
    setattr(app, _HOOKS_MARKER, True)

    @app.after_request
    def _finalize_usage_duration(response):
        duration_ms = get_request_duration_ms()
        if duration_ms is None:
            return response

        log_id = get_usage_log_id()
        if log_id is None:
            return response

        try:
            updated = update_usage_duration(log_id, duration_ms)
        except Exception as exc:
            logger.exception(
                "event=usage_duration_update_failed "
                "log_id=%s duration_ms=%s error_type=%s error=%s",
                log_id,
                duration_ms,
                type(exc).__name__,
                exc,
            )
            return response

        if not updated:
            logger.warning(
                "event=usage_duration_update_not_found log_id=%s duration_ms=%s",
                log_id,
                duration_ms,
            )

        return response
