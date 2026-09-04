# usage/attribution_state.py
"""Request-local Usage attribution metadata.

Owns exactly one responsibility: hold the *who and where* of the current
HTTP request - the resolved Maui user id, the service the request hit and
the client source it came from - so a later accounting step can attribute
the rows it writes without re-deriving identity.

Deliberately a sibling of ``usage.request_state`` rather than part
of it: that module stores the ids of Usage rows a request created, this
one stores the metadata a Usage row is written *with*. Keeping them apart
keeps each a single storage concern.

Storage only. This module does not resolve users, does not read the
request headers or body, does not touch authentication, does not persist
anything, does not register Flask hooks and does not log. Whoever binds
the attribution is responsible for resolving an honest user id first; a
missing or unresolvable identity is represented by leaving the state
unbound, never by a fabricated value.
"""

from dataclasses import dataclass

__all__ = [
    "UsageAttribution",
    "bind_usage_attribution",
    "get_usage_attribution",
]

#: Attribute under which the current request's attribution is parked on
#: ``flask.g``. Private to this module, namespaced like
#: ``usage.request_state``'s and ``utils.request_duration``'s own
#: ``_maui_*`` g attributes.
_G_ATTRIBUTION_ATTR = "_maui_usage_attribution"


@dataclass(frozen=True, slots=True)
class UsageAttribution:
    """The attribution metadata of one HTTP request.

    Narrow by construction: only what a Usage row needs and cannot
    reconstruct on its own. Everything else a row records - provider,
    model, quantities, duration, request id - is owned by other request
    state and is deliberately absent here.

    ``user_id`` is a resolved Maui user id; there is no sentinel for
    "unknown", because an unknown user means no attribution is bound at
    all. ``source`` is nullable: a request may legitimately carry no
    client source.
    """

    user_id: int
    service: str
    source: "str | None"


def bind_usage_attribution(user_id: int, service: str, source: "str | None") -> None:
    """Bind the current request's attribution metadata.

    Callers must invoke this only once identity has been honestly
    resolved. Binding is last-write-wins, mirroring
    ``usage.request_state.set_usage_log_id``: a second call replaces
    the first rather than raising or being ignored, so a re-bind is a
    correction, not an error.
    """
    from flask import g  # noqa: PLC0415

    setattr(
        g,
        _G_ATTRIBUTION_ATTR,
        UsageAttribution(user_id=user_id, service=service, source=source),
    )


def get_usage_attribution() -> "UsageAttribution | None":
    """Return the current request's attribution metadata, if any.

    :return: ``None`` when nothing has been bound for this request - no
        attribution resolved, or resolution failed; the bound immutable
        :class:`UsageAttribution` otherwise. Never a reconstructed or
        defaulted value.
    """
    from flask import g  # noqa: PLC0415

    return getattr(g, _G_ATTRIBUTION_ATTR, None)
