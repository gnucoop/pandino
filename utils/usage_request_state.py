# utils/usage_request_state.py
"""Request-local Usage row identity.

Owns exactly one responsibility: hold the ``log_id`` of the Usage row
created (if any) by the current HTTP request, so a later lifecycle
orchestration step can read it. The value is kept as request-local state
on ``flask.g`` and exposed through :func:`get_usage_log_id`.

This module does not persist anything, does not measure time, does not
register Flask hooks, and does not call ``update_usage_duration()``
(``infrastructure/database_pg.py``). Those are the concern of a future
slice that orchestrates this identity together with
``utils.request_duration``'s duration measurement; wiring them here would
pull that orchestration into a module deliberately kept to identity
storage only.
"""

__all__ = [
    "get_usage_log_id",
    "set_usage_log_id",
]

#: Attribute under which the successfully-created Usage row id is parked
#: on ``flask.g``. Private to this module, namespaced like
#: ``utils.logging_config``'s and ``utils.request_duration``'s own
#: ``_maui_*`` g attributes.
_G_LOG_ID_ATTR = "_maui_usage_log_id"


def set_usage_log_id(log_id: int) -> None:
    """Register the current request's successfully-created Usage row id.

    Callers must invoke this only after a successful Usage INSERT
    (``log_token_usage()`` returning normally). Not idempotent-guarded
    beyond a single request: at most one Usage row is created per request
    today, so a single slot is sufficient.
    """
    from flask import g  # noqa: PLC0415

    setattr(g, _G_LOG_ID_ATTR, log_id)


def get_usage_log_id() -> "int | None":
    """Return the current request's registered Usage row id, if any.

    :return: ``None`` when no Usage row has been registered for this
        request (no request context, Usage write skipped, or Usage write
        failed); the registered ``int`` otherwise.
    """
    from flask import g  # noqa: PLC0415

    return getattr(g, _G_LOG_ID_ATTR, None)
