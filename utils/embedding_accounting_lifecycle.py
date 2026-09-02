# utils/embedding_accounting_lifecycle.py
"""Request-lifecycle binding of the embedding-accounting sink.

Owns exactly one responsibility: for the lifetime of each HTTP request,
publish a sink over that request's accumulator, and unpublish it on
teardown. Nothing else. It reads no contributions, writes no Usage row,
touches no schema and knows no endpoint: it only closes the last gap
between the capture layer (which knows *what* it observed) and the
request-scoped accumulator (which knows *where* it belongs).

Without this hook the sink stays at its ``no_op_sink`` default:
provider capture would normalize a contribution correctly and then discard
it, because nothing in production ever entered
:func:`utils.embedding_usage_state.bind_embedding_accounting`. That
function is a context manager, which cannot span Flask's
``before_request``/``teardown_request`` boundary, so this module drives the
same two primitives it composes -- creating the accumulator and setting the
sink -- in hook form instead, keeping a single mechanism rather than a
second one.

Deliberately a sibling of ``utils.request_duration`` and
``utils.logging_config``'s hooks, not an extension of either: it owns its
own app marker and its own token attribute on ``flask.g``, so a double
registration cannot bind two sinks or orphan a token.
"""

from utils.embedding_accounting_sink import (
    reset_embedding_accounting_sink,
    set_embedding_accounting_sink,
)
from utils.embedding_usage_state import get_embedding_accumulator

__all__ = ["register_embedding_accounting_hooks"]

#: Attribute under which this request's sink token is parked on ``flask.g``,
#: namespaced like the other ``_maui_*`` request-local attributes.
_G_TOKEN_ATTR = "_maui_embedding_accounting_token"

#: Marker recording that the hooks are already registered on an app.
_HOOKS_MARKER = "_maui_embedding_accounting_hooks"


def register_embedding_accounting_hooks(app) -> None:
    """Bind a fresh accumulator behind the sink for each HTTP request.

    Must be called before the app serves its first request, per the same
    Flask 2.3+ constraint documented on
    :func:`utils.logging_config.register_request_context_hooks`.

    Idempotent via an app marker, mirroring that function.

    The sink closes over the accumulator *instance*, never over ``flask.g``:
    provider capture may deliver from ``PGVectorStore``'s background loop
    thread, which has no Flask context, and must still reach the right
    request's accumulator. ``flask.g`` is therefore touched only here,
    in the request greenlet.
    """
    from flask import g  # noqa: PLC0415

    if getattr(app, _HOOKS_MARKER, False):
        return
    setattr(app, _HOOKS_MARKER, True)

    @app.before_request
    def _bind_embedding_accounting_sink():
        accumulator = get_embedding_accumulator(create=True)
        setattr(g, _G_TOKEN_ATTR, set_embedding_accounting_sink(accumulator.add))

    @app.teardown_request
    def _unbind_embedding_accounting_sink(exc=None):
        # Teardown, not after_request: an unhandled view exception skips
        # after_request entirely, and only teardown is guaranteed. Without
        # the reset a keep-alive greenlet would carry this request's
        # accumulator into the next request's embedding work.
        reset_embedding_accounting_sink(getattr(g, _G_TOKEN_ATTR, None))
        setattr(g, _G_TOKEN_ATTR, None)
