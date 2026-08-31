# utils/embedding_operation_context.py
"""Scoped ambient context naming the current logical embedding operation.

Owns exactly one responsibility (DC5): carry an *immutable* value — one of
``query``, ``document`` or ``probe`` (DC4) — downward from the narrowest
Maui-owned layer that knows the semantics into the provider-adjacent
capture layer, with token/reset semantics and exception-safe restoration.

Deliberately separate from ``utils.logging_config``'s request-id/app-id
ContextVars: those are owned by logging, carry different values and are
unwound by a logging-owned teardown hook. The discipline is mirrored, the
state is not shared.

This module is Flask-blind: it imports nothing from Flask, needs no request
context, and is safe to use from ``services/``, ``infrastructure/``, a
background event-loop thread, or no request at all. It performs no logging,
no I/O and no accumulation — accumulation is a separate primitive (DC6).
"""

from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "OPERATION_QUERY",
    "OPERATION_DOCUMENT",
    "OPERATION_PROBE",
    "OPERATION_KINDS",
    "get_embedding_operation",
    "set_embedding_operation",
    "reset_embedding_operation",
    "embedding_operation",
]

#: Retrieval embedding. Shared by /completion.json and /agentchat: DC4
#: records that they are the same normalized work and differ only in HTTP
#: flow, which is a request-scope property.
OPERATION_QUERY = "query"

#: Document/chunk embedding during ingestion.
OPERATION_DOCUMENT = "document"

#: Infrastructure dimensionality probe (``embed_query("test")``).
OPERATION_PROBE = "probe"

#: The complete CURRENT taxonomy (DC4). No fourth kind is invented here;
#: this is also the vocabulary the contribution contract validates against.
OPERATION_KINDS = frozenset({OPERATION_QUERY, OPERATION_DOCUMENT, OPERATION_PROBE})

# Created once per process, at module level: creating a ContextVar inside a
# function is a documented anti-pattern (the objects are never garbage
# collected), the same reasoning recorded in utils/logging_config.py.
# ``None`` — not a sentinel string — is the default, so "no embedding
# operation is in progress" stays distinguishable from every valid kind.
_operation_var: "ContextVar[str | None]" = ContextVar(
    "maui_embedding_operation", default=None
)


def get_embedding_operation() -> "str | None":
    """Return the operation kind in scope, or ``None`` outside any scope."""
    return _operation_var.get()


def set_embedding_operation(kind: str):
    """Enter an operation scope, returning the token needed to leave it.

    The low-level primitive: the caller keeps the token and must hand it to
    :func:`reset_embedding_operation`, including on the exception path.
    Prefer :func:`embedding_operation`, which owns that discipline.

    :raises ValueError: if ``kind`` is not in :data:`OPERATION_KINDS`. A bad
        kind is a programmer error at an authored call site, not a runtime
        input, so it fails loudly rather than recording an unusable scope.
    """
    if kind not in OPERATION_KINDS:
        raise ValueError(
            f"unknown embedding operation kind: {kind!r}; "
            f"expected one of {sorted(OPERATION_KINDS)}"
        )
    return _operation_var.set(kind)


def reset_embedding_operation(token) -> None:
    """Restore the value captured by :func:`set_embedding_operation`.

    Accepts ``None`` and is a no-op in that case, so a teardown that runs
    without a matching bind cannot raise — the same tolerance
    :func:`utils.logging_config.reset_request_context` offers.
    """
    if token is None:
        return
    _operation_var.reset(token)


@contextmanager
def embedding_operation(kind: str):
    """Scope an operation kind, restoring the previous value on exit.

    Restoration runs in a ``finally``, so an exception raised inside the
    scope still unwinds it. Without that, a keep-alive greenlet serving
    sequential requests would attribute the next request's embedding work to
    this scope — the concrete hazard ``logging_config`` documents for the
    request-id vars.
    """
    token = set_embedding_operation(kind)
    try:
        yield kind
    finally:
        reset_embedding_operation(token)
