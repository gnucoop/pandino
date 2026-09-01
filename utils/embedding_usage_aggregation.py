# utils/embedding_usage_aggregation.py
"""Pure aggregation of embedding-accounting contributions.

Collapses the 0..N contributions a single request produced into the
minimum set of records a consumer can persist, one per distinct
consumption invariant.

Pure by construction: no Flask, no database, no request context, no
provider I/O, no logging. The input contributions are frozen and are never
mutated, and this module re-validates nothing the contribution contract
already enforces.
"""

from dataclasses import dataclass

from utils.embedding_accounting import COST_PROVIDER_AUTHORITATIVE

__all__ = [
    "AggregatedEmbeddingUsage",
    "aggregate_embedding_contributions",
]


@dataclass(frozen=True, slots=True)
class AggregatedEmbeddingUsage:
    """Summed consumption for one distinct set of embedding invariants.

    Deliberately not an ``EmbeddingAccountingContribution``: a contribution
    is one successful provider accounting response, while this is a fact
    about a different cardinality, and the per-call provider identifiers a
    contribution carries are meaningless once summed.

    ``resolved_cost`` is the summed provider cost when the partition is
    provider-authoritative, and ``None`` otherwise — never ``0.0``, which
    would be an invented price. Resolving the remaining states belongs to
    the persistence layer.

    ``contribution_count`` is diagnostic only: it says how many responses
    were folded together and is never part of what distinguishes one
    aggregate from another.
    """

    provider: str
    model: str
    operation_kind: str
    input_quantity: int
    quantity_origin: str
    cost_state: str
    resolved_cost: "float | None"
    contribution_count: int


def aggregate_embedding_contributions(contributions) -> tuple:
    """Aggregate contributions into ``AggregatedEmbeddingUsage`` records.

    Contributions are grouped by ``(provider, model, operation_kind,
    quantity_unit, quantity_origin, cost_state)``. Nothing is summed across
    a differing key, so a provider/model anomaly within one request yields
    two records rather than one silently wrong total.

    ``quantity_unit`` partitions but is not carried on the result: the
    supported quantity is input tokens, and mixing units into one sum would
    be meaningless.

    Within a group, ``input_quantity`` is summed, and ``provider_cost`` is
    summed into ``resolved_cost`` only for a provider-authoritative group.
    Records are emitted in order of each group's first appearance in the
    input; an empty input yields an empty tuple.

    :raises ValueError: if a provider-authoritative group contains a
        contribution with no cost, which can only mean an upstream
        contract regression.
    """
    groups = {}

    for contribution in contributions:
        key = (
            contribution.provider,
            contribution.model,
            contribution.operation_kind,
            contribution.quantity_unit,
            contribution.quantity_origin,
            contribution.cost_state,
        )
        # dicts preserve insertion order, which is exactly the required
        # "first appearance" emission order — no separate ordering pass.
        group = groups.get(key)
        if group is None:
            groups[key] = group = {
                "quantity": 0,
                "cost": None,
                "count": 0,
                "missing_cost": False,
            }
        group["quantity"] += contribution.input_quantity
        group["count"] += 1
        if contribution.cost_state == COST_PROVIDER_AUTHORITATIVE:
            if contribution.provider_cost is None:
                # The contribution contract forbids this; never sum around it.
                group["missing_cost"] = True
            else:
                group["cost"] = (group["cost"] or 0.0) + contribution.provider_cost

    aggregates = []
    for key, group in groups.items():
        provider, model, operation_kind, _unit, quantity_origin, cost_state = key
        resolved_cost = group["cost"]
        if cost_state == COST_PROVIDER_AUTHORITATIVE and (
            group["missing_cost"] or resolved_cost is None
        ):
            raise ValueError(
                "provider_authoritative aggregate has a contribution without a "
                "provider_cost: "
                f"{provider}/{model} ({operation_kind})"
            )
        aggregates.append(
            AggregatedEmbeddingUsage(
                provider=provider,
                model=model,
                operation_kind=operation_kind,
                input_quantity=group["quantity"],
                quantity_origin=quantity_origin,
                cost_state=cost_state,
                resolved_cost=resolved_cost,
                contribution_count=group["count"],
            )
        )
    return tuple(aggregates)
