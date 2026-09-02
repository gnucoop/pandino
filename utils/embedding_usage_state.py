# utils/embedding_usage_state.py
"""Request-scoped accumulation of embedding-accounting contributions.

Owns exactly one responsibility: hold the ordered 0..N contributions
produced by the current HTTP request, as request-local state on
``flask.g``, and validate the request-level provider/model invariant
across them.

Sibling to — not an extension of — ``utils.usage_request_state``, which is
documented as a single slot because at most one Usage row exists per
request. Embedding work is 0..N per request and its question ("what was
consumed") is a different one from that module's ("which row id").

This module does not persist anything, creates no Usage row, registers no
Flask hook, and knows nothing about provider response shapes. It is the one
Flask-aware piece of the foundation: everything above it (contract,
operation context, sink) is Flask-blind, and reaches this state only
through the sink callable bound by :func:`bind_embedding_accounting`.
"""

import logging
from contextlib import contextmanager

from utils.embedding_accounting_sink import embedding_accounting_sink

__all__ = [
    "EmbeddingAccountingAccumulator",
    "get_embedding_accumulator",
    "get_embedding_contributions",
    "bind_embedding_accounting",
]

#: Attribute under which the accumulator is parked on ``flask.g``. Private
#: to this module, namespaced like ``utils.request_duration``'s and
#: ``utils.usage_request_state``'s own ``_maui_*`` g attributes.
_G_ACCUMULATOR_ATTR = "_maui_embedding_accumulator"

logger = logging.getLogger(__name__)


class EmbeddingAccountingAccumulator:
    """Ordered, mutable collection of contributions for one request.

    Plain object with no Flask knowledge, so it is directly constructible in
    tests and usable outside a request context. The request-scoped lifetime
    comes from where it is stored, not from what it is.
    """

    __slots__ = ("_contributions", "_invariant_violations")

    def __init__(self) -> None:
        self._contributions = []
        self._invariant_violations = 0

    def add(self, contribution) -> None:
        """Append one contribution, validating the provider/model invariant.

        The invariant "one provider/model per request" is real but
        **configuration-derived and unenforced**: it holds because the call
        sites read the same two config keys, not because any code asserts it.

        This foundation records and warns, and still keeps the
        contribution. Raising here would run inside the capture path, on the
        far side of a thread hop, and would convert an accounting anomaly
        into a user-visible request failure — a poor trade for an
        observation-only subsystem whose contract is that
        consumption is independent of HTTP outcome. Dropping the
        contribution instead would lose real consumption, which is the one
        thing this module exists to keep. The violation is counted so a
        later phase can see the anomaly rather than infer it.

        The warning names provider and model only: both are configuration
        identities, never user content.
        """
        first = self._contributions[0] if self._contributions else None
        if first is not None and (
            first.provider != contribution.provider or first.model != contribution.model
        ):
            self._invariant_violations += 1
            logger.warning(
                "embedding accounting provider/model invariant violated within one "
                "request: expected %s/%s, received %s/%s",
                first.provider,
                first.model,
                contribution.provider,
                contribution.model,
            )
        self._contributions.append(contribution)

    @property
    def contributions(self) -> tuple:
        """The appended contributions, in order, as an immutable snapshot."""
        return tuple(self._contributions)

    @property
    def invariant_violations(self) -> int:
        """How many appends violated the provider/model invariant."""
        return self._invariant_violations


def get_embedding_accumulator(create: bool = False):
    """Return this request's accumulator.

    :param create: when ``True``, lazily create and park one if absent.
        Readers pass ``False``, so observing the accumulator can never
        manufacture request state; only the request-lifecycle owner creates.
    :return: ``None`` when absent and ``create`` is ``False``.
    """
    from flask import g  # noqa: PLC0415

    accumulator = getattr(g, _G_ACCUMULATOR_ATTR, None)
    if accumulator is None and create:
        accumulator = EmbeddingAccountingAccumulator()
        setattr(g, _G_ACCUMULATOR_ATTR, accumulator)
    return accumulator


def get_embedding_contributions() -> tuple:
    """Return this request's contributions, or ``()`` when none were made."""
    accumulator = get_embedding_accumulator()
    return accumulator.contributions if accumulator is not None else ()


@contextmanager
def bind_embedding_accounting():
    """Bind this request's accumulator behind a sink, for the block.

    Creates (or reuses) the request's accumulator, publishes a sink closing
    over that **instance**, and unbinds on exit including on exception.

    Closing over the instance is what keeps ``flask.g`` out of the far side:
    the background loop thread appends to a plain object it already holds,
    and ``flask.g`` is touched only here, in the request greenlet.

    Requires an active Flask context, like any other ``flask.g`` access.
    The production request path does not go through this context manager: the
    hooks registered by :mod:`utils.embedding_accounting_lifecycle` drive the
    same two primitives across ``before_request``/``teardown_request``. This
    is the scoped form, for callers that own their own block.
    """
    accumulator = get_embedding_accumulator(create=True)
    with embedding_accounting_sink(accumulator.add):
        yield accumulator
