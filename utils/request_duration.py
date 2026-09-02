# utils/request_duration.py
"""Authoritative HTTP request-duration timer lifecycle.

Owns exactly one responsibility: measure, once per request, the elapsed
server-side time between the application-wide ``before_request`` and
``after_request`` boundaries, using a monotonic clock
(:func:`time.perf_counter`). The finalized value is kept as request-local
state on ``flask.g`` and exposed through :func:`get_request_duration_ms`.

This module does not persist anything, does not know about Usage
``log_id``, and does not call ``update_usage_duration()``
(``infrastructure/database_pg.py``). ``utils.usage_duration_finalization``
reads this timer's output and owns that persistence; wiring it here would
pull Usage-persistence concerns into a module deliberately kept to
lifecycle timing only.
"""

import time

__all__ = [
    "get_request_duration_ms",
    "register_request_duration_hooks",
]

#: Attribute under which the raw perf_counter() start value is parked on
#: ``flask.g``. Private to this module, namespaced like
#: ``utils.logging_config``'s own ``_maui_*`` g attributes.
_G_START_ATTR = "_maui_request_duration_start"

#: Attribute under which the finalized, rounded duration_ms is parked on
#: ``flask.g``, once ``after_request`` has run.
_G_DURATION_ATTR = "_maui_request_duration_ms"

#: Marker attribute recording that the hooks are already registered on an
#: app, distinct from logging_config's own hooks marker.
_HOOKS_MARKER = "_maui_request_duration_hooks"


def get_request_duration_ms() -> "int | None":
    """Return the finalized duration for the current request, in whole ms.

    :return: ``None`` before finalization (or when no request context is
        active); the rounded ``int`` once ``after_request`` has run.
    """
    from flask import g  # noqa: PLC0415

    return getattr(g, _G_DURATION_ATTR, None)


def register_request_duration_hooks(app) -> None:
    """Bind an authoritative request-duration timer for each HTTP request.

    Must be called before the app serves its first request, per the same
    Flask 2.3+ constraint documented on
    :func:`utils.logging_config.register_request_context_hooks`.

    Idempotent: a marker on the app makes a second call a no-op, mirroring
    the pattern used there, so this module owns its own marker and does not
    share or couple with logging_config's hooks.
    """
    from flask import g  # noqa: PLC0415

    if getattr(app, _HOOKS_MARKER, False):
        return
    setattr(app, _HOOKS_MARKER, True)

    @app.before_request
    def _start_request_timer():
        setattr(g, _G_START_ATTR, time.perf_counter())

    @app.after_request
    def _finalize_request_duration(response):
        start = getattr(g, _G_START_ATTR, None)
        if start is not None:
            elapsed_ms = round((time.perf_counter() - start) * 1000)
            setattr(g, _G_DURATION_ATTR, elapsed_ms)
        return response
