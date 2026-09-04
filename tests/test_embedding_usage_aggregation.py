"""Tests for usage.embedding_aggregation.

Scope: the pure contribution -> aggregate transformation — partition key,
sum rules, emission order, empty input, and input immutability.
"""

import dataclasses

import pytest

from usage.embedding_accounting import (
    COST_NO_PROVIDER_BILLING,
    COST_PROVIDER_ABSENT_RESOLVABLE,
    COST_PROVIDER_AUTHORITATIVE,
    ORIGIN_MAUI_DERIVED,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_UNIT_INPUT_TOKENS,
    EmbeddingAccountingContribution,
)
from usage.embedding_aggregation import (
    AggregatedEmbeddingUsage,
    aggregate_embedding_contributions,
)


def _contribution(**overrides):
    fields = dict(
        provider="deepinfra",
        model="BAAI/bge-m3",
        input_quantity=10,
        quantity_unit=QUANTITY_UNIT_INPUT_TOKENS,
        quantity_origin=ORIGIN_PROVIDER_REPORTED,
        cost_state=COST_PROVIDER_AUTHORITATIVE,
        operation_kind="query",
        provider_cost=1.0e-07,
    )
    fields.update(overrides)
    return EmbeddingAccountingContribution(**fields)


def _unbilled(**overrides):
    fields = dict(cost_state=COST_NO_PROVIDER_BILLING, provider_cost=None)
    fields.update(overrides)
    return _contribution(**fields)


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


def test_empty_input_returns_an_empty_tuple():
    assert aggregate_embedding_contributions([]) == ()


def test_single_contribution_preserves_every_semantic_field():
    (aggregate,) = aggregate_embedding_contributions([_contribution()])
    assert isinstance(aggregate, AggregatedEmbeddingUsage)
    assert aggregate.provider == "deepinfra"
    assert aggregate.model == "BAAI/bge-m3"
    assert aggregate.operation_kind == "query"
    assert aggregate.input_quantity == 10
    assert aggregate.quantity_origin == ORIGIN_PROVIDER_REPORTED
    assert aggregate.cost_state == COST_PROVIDER_AUTHORITATIVE
    assert aggregate.resolved_cost == pytest.approx(1.0e-07)
    assert aggregate.contribution_count == 1


def test_aggregate_is_frozen():
    (aggregate,) = aggregate_embedding_contributions([_contribution()])
    with pytest.raises(dataclasses.FrozenInstanceError):
        aggregate.input_quantity = 99


def test_aggregate_does_not_carry_per_call_provider_identifiers():
    fields = {f.name for f in dataclasses.fields(AggregatedEmbeddingUsage)}
    assert "provider_request_id" not in fields
    assert "provider_runtime_ms" not in fields
    assert "quantity_unit" not in fields


# --------------------------------------------------------------------------
# Sum rules
# --------------------------------------------------------------------------


def test_matching_contributions_collapse_into_one_summed_aggregate():
    (aggregate,) = aggregate_embedding_contributions(
        [
            _contribution(input_quantity=10, provider_cost=1.0e-07),
            _contribution(input_quantity=4, provider_cost=4.0e-08),
            _contribution(input_quantity=1, provider_cost=1.0e-08),
        ]
    )
    assert aggregate.input_quantity == 15
    assert aggregate.resolved_cost == pytest.approx(1.5e-07)
    assert aggregate.contribution_count == 3


@pytest.mark.parametrize(
    "cost_state",
    [COST_PROVIDER_ABSENT_RESOLVABLE, COST_NO_PROVIDER_BILLING],
)
def test_non_authoritative_states_keep_resolved_cost_none(cost_state):
    (aggregate,) = aggregate_embedding_contributions(
        [
            _contribution(cost_state=cost_state, provider_cost=None, input_quantity=3),
            _contribution(cost_state=cost_state, provider_cost=None, input_quantity=7),
        ]
    )
    assert aggregate.input_quantity == 10
    assert aggregate.resolved_cost is None
    assert aggregate.contribution_count == 2


def test_zero_quantities_still_produce_an_aggregate():
    (aggregate,) = aggregate_embedding_contributions(
        [_contribution(input_quantity=0, provider_cost=0.0)]
    )
    assert aggregate.input_quantity == 0
    assert aggregate.resolved_cost == 0.0


# --------------------------------------------------------------------------
# Partition boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "openai"},
        {"model": "text-embedding-3-small"},
        {"operation_kind": "document"},
        {"quantity_origin": ORIGIN_MAUI_DERIVED},
        {"cost_state": COST_NO_PROVIDER_BILLING, "provider_cost": None},
    ],
    ids=["provider", "model", "operation_kind", "quantity_origin", "cost_state"],
)
def test_each_partition_dimension_splits_the_aggregate(overrides):
    aggregates = aggregate_embedding_contributions(
        [_contribution(), _contribution(**overrides)]
    )
    assert len(aggregates) == 2
    assert all(a.contribution_count == 1 for a in aggregates)


def test_quantity_unit_splits_the_aggregate():
    # The contribution contract admits exactly one unit today, so the unit
    # dimension is exercised with a minimal stand-in: the aggregator reads
    # attributes and re-validates nothing.
    only_unit = _contribution()
    other_unit = dataclasses.replace(only_unit)
    object.__setattr__(other_unit, "quantity_unit", "characters")

    aggregates = aggregate_embedding_contributions([only_unit, other_unit])
    assert len(aggregates) == 2
    assert all(a.contribution_count == 1 for a in aggregates)


def test_non_adjacent_matching_contributions_still_collapse():
    aggregates = aggregate_embedding_contributions(
        [
            _contribution(input_quantity=1),
            _contribution(provider="openai", input_quantity=2),
            _contribution(input_quantity=4),
        ]
    )
    assert len(aggregates) == 2
    assert aggregates[0].input_quantity == 5
    assert aggregates[0].contribution_count == 2
    assert aggregates[1].input_quantity == 2


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


def test_output_follows_first_appearance_of_each_partition():
    aggregates = aggregate_embedding_contributions(
        [
            _contribution(provider="zeta"),
            _contribution(provider="beta"),
            _contribution(provider="zeta"),
            _contribution(provider="alpha"),
        ]
    )
    assert [a.provider for a in aggregates] == ["zeta", "beta", "alpha"]


# --------------------------------------------------------------------------
# Immutability and input handling
# --------------------------------------------------------------------------


def test_inputs_are_not_mutated():
    contributions = [_contribution(input_quantity=3), _contribution(input_quantity=5)]
    before = [dataclasses.astuple(c) for c in contributions]

    aggregate_embedding_contributions(contributions)

    assert [dataclasses.astuple(c) for c in contributions] == before
    assert len(contributions) == 2


def test_an_arbitrary_iterable_is_accepted():
    aggregates = aggregate_embedding_contributions(iter([_contribution()]))
    assert len(aggregates) == 1


# --------------------------------------------------------------------------
# Upstream contract regression
# --------------------------------------------------------------------------


def test_authoritative_partition_without_a_cost_raises():
    broken = _contribution()
    object.__setattr__(broken, "provider_cost", None)
    with pytest.raises(ValueError):
        aggregate_embedding_contributions([broken])


def test_authoritative_partition_with_one_missing_cost_raises():
    broken = _contribution()
    object.__setattr__(broken, "provider_cost", None)
    with pytest.raises(ValueError):
        aggregate_embedding_contributions([_contribution(), broken])
