# usage/request_state.py
"""Request-local Usage row identity.

Owns exactly one responsibility: hold the ``log_id`` of every Usage row
created (if any) by the current HTTP request, so the lifecycle
orchestration that finalizes those rows can read them. The state is kept
as request-local state on ``flask.g`` and exposed through two readers:

* :func:`get_usage_log_id` - the most recently *set* single id, the
  original accessor, retained for its existing readers;
* :func:`get_usage_log_ids` - every id registered during the request, in
  first-seen order and de-duplicated.

A request may create several Usage rows, so the single slot is no longer
the whole picture. :func:`set_usage_log_id` keeps its exact previous
meaning *and* registers into the ordered collection, which is why its
callers need no change; :func:`register_usage_log_id` registers only,
leaving the single slot alone.

This module does not persist anything, does not measure time, does not
register Flask hooks, and does not call ``update_usage_duration()``
(``infrastructure/database_pg.py``). That orchestration lives in
``usage.duration_finalization``, which combines this identity state
with ``utils.request_duration``'s duration measurement; wiring it here
would pull that orchestration into a module deliberately kept to identity
storage only.
"""

__all__ = [
    "get_usage_log_id",
    "get_usage_log_ids",
    "register_usage_log_id",
    "set_usage_log_id",
]

#: Attribute under which the most recently set Usage row id is parked on
#: ``flask.g``. Private to this module, namespaced like
#: ``utils.logging_config``'s and ``utils.request_duration``'s own
#: ``_maui_*`` g attributes.
_G_LOG_ID_ATTR = "_maui_usage_log_id"

#: Attribute under which the ordered list of every Usage row id
#: registered during the request is parked on ``flask.g``. Private to
#: this module; read through :func:`get_usage_log_ids`.
_G_LOG_IDS_ATTR = "_maui_usage_log_ids"


def _append_log_id(log_id: int) -> None:
    """Append ``log_id`` to the request-local ordered list.

    Order is preserved as appended; de-duplication happens on read, so a
    repeated registration is harmless here.
    """
    from flask import g  # noqa: PLC0415

    if log_id is None:
        return

    log_ids = getattr(g, _G_LOG_IDS_ATTR, None)
    if log_ids is None:
        log_ids = []
        setattr(g, _G_LOG_IDS_ATTR, log_ids)
    log_ids.append(log_id)


def set_usage_log_id(log_id: int) -> None:
    """Register the current request's most recent Usage row id.

    Callers must invoke this only after a successful Usage INSERT
    (``log_token_usage()`` returning normally). The id both replaces the
    single most-recent slot read by :func:`get_usage_log_id` and is
    registered into the ordered collection read by
    :func:`get_usage_log_ids`, so callers that create exactly one Usage
    row per request need to do nothing else.
    """
    from flask import g  # noqa: PLC0415

    setattr(g, _G_LOG_ID_ATTR, log_id)
    _append_log_id(log_id)


def register_usage_log_id(log_id: int) -> None:
    """Register an additional Usage row id for the current request.

    Unlike :func:`set_usage_log_id` this does not touch the single
    most-recent slot: :func:`get_usage_log_id` keeps returning whatever
    was last *set*, while the id becomes visible to
    :func:`get_usage_log_ids`. Intended for producers that create Usage
    rows alongside a request's primary row.
    """
    _append_log_id(log_id)


def get_usage_log_id() -> "int | None":
    """Return the most recently set Usage row id for this request, if any.

    :return: ``None`` when no Usage row has been set for this request (no
        request context, Usage write skipped, or Usage write failed); the
        registered ``int`` otherwise. Ids registered only through
        :func:`register_usage_log_id` are not reported here - see
        :func:`get_usage_log_ids`.
    """
    from flask import g  # noqa: PLC0415

    return getattr(g, _G_LOG_ID_ATTR, None)


def get_usage_log_ids() -> "tuple[int, ...]":
    """Return every Usage row id registered during this request.

    :return: an immutable tuple in first-seen registration order, with
        duplicates removed; empty when nothing has been registered (no
        request context included). Not sorted - the order reflects when
        each row was registered, never the numeric value.
    """
    from flask import g  # noqa: PLC0415

    log_ids = getattr(g, _G_LOG_IDS_ATTR, None)
    if not log_ids:
        return ()

    return tuple(dict.fromkeys(log_ids))
