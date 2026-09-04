# utils/embedding_usage_persistence.py
"""Request-lifecycle persistence of embedding consumption as Usage rows.

Owns exactly one responsibility: at the end of each HTTP request, turn the
0..N embedding-accounting contributions that request accumulated into 0..N
persisted Usage rows, and register the resulting row ids so the existing
duration finalizer treats them like any other Usage row of the request.

The pipeline is a composition of settled modules, and nothing here
duplicates what any of them owns::

    accumulator (utils.embedding_usage_state)
      -> pure aggregation (utils.embedding_usage_aggregation)
      -> attribution (utils.usage_attribution_state)
      -> provenance vocabulary (utils.usage_provenance)
      -> batch writer (infrastructure.database_pg)
      -> row-id registration (utils.usage_request_state)

Deliberately *not* an extension of ``utils.embedding_accounting_lifecycle``,
whose single responsibility is binding and unbinding the capture sink: that
module makes collection possible, this one decides what the collection
means. It knows no route, no service, no provider client and no request
payload; identity arrives only as already-resolved attribution, and a
request that bound none writes no row rather than inferring one.

Accounting is observation. Every failure in this module produces one log
line and an unchanged HTTP response, and a request attempts persistence at
most once - a failed write is never retried, because a failure between the
INSERT and the commit acknowledgement cannot be distinguished from a
success and a retry would risk double-counting real money.
"""

import logging

from infrastructure.database_pg import (
    ResolvedCostUsageEntry,
    log_resolved_cost_usage_batch,
)
from utils.embedding_accounting import COST_NO_PROVIDER_BILLING
from utils.embedding_usage_aggregation import aggregate_embedding_contributions
from utils.embedding_usage_state import get_embedding_contributions
from utils.logging_config import get_request_id
from utils.usage_attribution_state import get_usage_attribution
from utils.usage_provenance import cost_origin_from_cost_state
from utils.usage_request_state import register_usage_log_id

logger = logging.getLogger(__name__)

__all__ = ["register_embedding_usage_persistence_hooks"]

#: Attribute recording that this request has already attempted embedding
#: Usage persistence. Shared by both seams, namespaced like the other
#: ``_maui_*`` request-local attributes.
_G_PERSISTED_ATTR = "_maui_embedding_usage_persisted"

#: Marker attribute recording that the hooks are already registered on an
#: app, distinct from every other lifecycle module's own marker.
_HOOKS_MARKER = "_maui_embedding_usage_persistence_hooks"


def _claim_persistence_attempt() -> bool:
    """Claim this request's single persistence attempt.

    :return: ``True`` for the first caller, ``False`` for every later one.

    The claim is taken *before* the write, not after it, so whichever seam
    runs first is the only seam that ever writes.
    """
    from flask import g  # noqa: PLC0415

    if getattr(g, _G_PERSISTED_ATTR, False):
        return False
    setattr(g, _G_PERSISTED_ATTR, True)
    return True


def _resolve_cost(aggregate) -> "tuple[float, str] | None":
    """Resolve one aggregate's persisted cost and cost origin.

    :return: the ``(cost, cost_origin)`` pair to persist, or ``None`` when
        the partition has no honest monetary value and must be skipped.

    A provider-authoritative aggregate carries the provider's own summed
    amount. An aggregate under no provider billing persists ``0.0``, which
    its ``cost_origin`` marks as an absence of billing rather than a priced
    zero. A partition that would need Maui-side pricing but has no resolved
    amount is skipped: writing ``0.0`` would claim free consumption and
    writing an estimate would fabricate money.
    """
    if aggregate.cost_state == COST_NO_PROVIDER_BILLING:
        return 0.0, cost_origin_from_cost_state(COST_NO_PROVIDER_BILLING)

    if aggregate.resolved_cost is None:
        return None

    return float(aggregate.resolved_cost), cost_origin_from_cost_state(
        aggregate.cost_state
    )


def _build_entries(aggregates, attribution, request_id) -> list:
    """Map aggregates onto persistence-ready Usage entries, in order.

    Partitions with no honest cost are dropped and logged individually, so
    one unpriceable partition never costs the request its other rows.

    ``token_output`` is ``0`` by the established non-token Usage
    convention, and no duration is supplied: rows are inserted with
    ``duration_ms`` NULL and finalized later by the duration hook.
    """
    entries = []
    for aggregate in aggregates:
        resolved = _resolve_cost(aggregate)
        if resolved is None:
            logger.warning(
                "event=embedding_usage_cost_unresolved request_id=%s service=%s "
                "provider=%s model=%s operation_kind=%s cost_state=%s",
                request_id,
                attribution.service,
                aggregate.provider,
                aggregate.model,
                aggregate.operation_kind,
                aggregate.cost_state,
            )
            continue

        cost, cost_origin = resolved
        entries.append(
            ResolvedCostUsageEntry(
                user_id=attribution.user_id,
                cost=cost,
                model=aggregate.model,
                provider=aggregate.provider,
                service=attribution.service,
                request_id=request_id,
                source=attribution.source,
                token_input=aggregate.input_quantity,
                token_output=0,
                embedding_operation_kind=aggregate.operation_kind,
                quantity_origin=aggregate.quantity_origin,
                cost_origin=cost_origin,
            )
        )
    return entries


def _persist_embedding_usage() -> None:
    """Persist this request's embedding consumption, at most once.

    Returns silently when there is nothing to persist, so a request that
    performed no embedding work opens no database connection. Never
    raises: every failure below is an accounting failure and the caller is
    an HTTP lifecycle hook, so the whole body is guarded and reduces to a
    single log line.
    """
    attribution = None
    contributions = ()
    request_id = None
    try:
        contributions = get_embedding_contributions()
        if not contributions:
            return

        # Claimed only once there is something to persist, so a request
        # with no embedding work leaves both seams free and a request with
        # embedding work is decided exactly once.
        if not _claim_persistence_attempt():
            return

        request_id = get_request_id()

        attribution = get_usage_attribution()
        if attribution is None:
            # Consumption that cannot be honestly attributed is observed
            # and not persisted; a row's user_id is never inferred here.
            logger.warning(
                "event=embedding_usage_persistence_skipped reason=no_attribution "
                "request_id=%s contributions=%s",
                request_id,
                len(contributions),
            )
            return

        aggregates = aggregate_embedding_contributions(contributions)
        entries = _build_entries(aggregates, attribution, request_id)

        if not entries:
            logger.warning(
                "event=embedding_usage_persistence_skipped "
                "reason=no_persistable_partitions request_id=%s service=%s "
                "partitions=%s",
                request_id,
                attribution.service,
                len(aggregates),
            )
            return

        log_ids = log_resolved_cost_usage_batch(entries)

        for log_id in log_ids:
            # register_, never set_: these ids must join the request's
            # duration lifecycle without displacing the single most-recent
            # id that existing readers still consume.
            register_usage_log_id(log_id)
    except Exception as exc:
        logger.exception(
            "event=embedding_usage_persistence_failed request_id=%s service=%s "
            "contributions=%s error_type=%s error=%s",
            request_id,
            attribution.service if attribution is not None else None,
            len(contributions),
            type(exc).__name__,
            exc,
        )


def register_embedding_usage_persistence_hooks(app) -> None:
    """Bind end-of-request embedding Usage persistence for each request.

    Must be called before the app serves its first request, per the same
    Flask 2.3+ constraint documented on
    :func:`utils.logging_config.register_request_context_hooks`.

    Idempotent: a marker on the app makes a second call a no-op, mirroring
    every other lifecycle module, so this one owns its own marker and
    couples to none of them.

    Registration order is a correctness invariant, not style. Flask runs
    ``after_request`` hooks in reverse registration order, so this must be
    registered *after* the Usage duration finalizer (whose updates must see
    these row ids) and *before* the request-duration timer (whose finalized
    value those updates need). Teardown is likewise LIFO, so this must be
    registered after the request-context hooks, whose ``request_id`` the
    fallback still needs. The capture sink may already be unbound by then,
    which costs the fallback nothing: contributions live on the
    ``flask.g`` accumulator, not behind the sink, so everything captured
    during the request stays readable here.

    ``after_request`` is the primary seam; ``teardown_request`` is a
    fallback for requests where an exception propagated and
    ``after_request`` never ran. Both share one request-local guard, so a
    request persists at most once.
    """
    if getattr(app, _HOOKS_MARKER, False):
        return
    setattr(app, _HOOKS_MARKER, True)

    @app.after_request
    def _persist_embedding_usage_after_request(response):
        _persist_embedding_usage()
        return response

    @app.teardown_request
    def _persist_embedding_usage_on_teardown(exc=None):
        # The teardown exception is not business input: it says only that
        # after_request may have been skipped. Consumption that really
        # happened is persisted either way, and the guard makes a
        # normal-path second attempt impossible.
        _persist_embedding_usage()
