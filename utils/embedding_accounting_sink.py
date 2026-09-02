# utils/embedding_accounting_sink.py
"""Context-propagated delivery seam for accounting contributions.

Owns exactly one responsibility: publish an opaque, immutable *sink
callable* in its own ContextVar so Flask-blind capture code can deliver a
normalized contribution without knowing what receives it:

    get_embedding_accounting_sink()(contribution)

The sink travels *downward*, which is the direction and mutability
verified as safe across ``PGVectorStore``'s background-loop hop and
``langchain_core``'s ``run_in_executor`` fallback. Mutation happens on the
far side of the call, inside the Flask-aware owner of the accumulator
(``utils.embedding_usage_state``), so context propagation and
accumulation stay separate primitives.

The default is a no-op, so capture code never branches on "is there a
request", never raises outside request scope, and stays honestly
Flask-blind — this module imports nothing from Flask.

Deliberately not an event bus, a publish/subscribe framework or an observer
registry: one ContextVar holding one callable is the whole mechanism.
"""

from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "no_op_sink",
    "get_embedding_accounting_sink",
    "set_embedding_accounting_sink",
    "reset_embedding_accounting_sink",
    "embedding_accounting_sink",
]


def no_op_sink(contribution) -> None:
    """Discard a contribution. The default when nothing is bound.

    Silent by construction: the absence of a bound sink is the normal case
    outside an HTTP request (direct tests, the namespace probe, reusable
    Flask-independent services), not an anomaly worth logging — and the
    contribution must never be logged in any case.
    """
    return None


# Created once per process, at module level, for the same reason recorded in
# utils/logging_config.py: a ContextVar created per call leaks.
_sink_var: ContextVar = ContextVar("maui_embedding_accounting_sink", default=no_op_sink)


def get_embedding_accounting_sink():
    """Return the bound sink, or :func:`no_op_sink` when none is bound."""
    return _sink_var.get() or no_op_sink


def set_embedding_accounting_sink(sink):
    """Bind ``sink``, returning the token needed to unbind it.

    :raises TypeError: if ``sink`` is not callable — an unusable sink would
        otherwise fail far away, inside capture code, on the far side of a
        thread hop.
    """
    if not callable(sink):
        raise TypeError("embedding accounting sink must be callable")
    return _sink_var.set(sink)


def reset_embedding_accounting_sink(token) -> None:
    """Restore the value captured by :func:`set_embedding_accounting_sink`.

    Tolerates ``None`` so an unwind without a matching bind cannot raise.
    """
    if token is None:
        return
    _sink_var.reset(token)


@contextmanager
def embedding_accounting_sink(sink):
    """Bind ``sink`` for the duration of the block, restoring on exit.

    Restoration runs in a ``finally``: without it a keep-alive greenlet
    would carry a stale sink — one closing over the *previous* request's
    accumulator — into the next request, which is a cross-request
    correctness bug rather than a local one.
    """
    token = set_embedding_accounting_sink(sink)
    try:
        yield sink
    finally:
        reset_embedding_accounting_sink(token)
