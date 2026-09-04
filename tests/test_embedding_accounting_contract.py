"""Tests for usage.embedding_accounting (design §21.1).

Scope: the normalized contribution contract only — required vs optional
fields, quantity origin, the three cost states, absence != zero, and the
operation taxonomy.
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


def _contribution(**overrides):
    fields = dict(
        provider="deepinfra",
        model="BAAI/bge-m3",
        input_quantity=14,
        quantity_unit=QUANTITY_UNIT_INPUT_TOKENS,
        quantity_origin=ORIGIN_PROVIDER_REPORTED,
        cost_state=COST_PROVIDER_AUTHORITATIVE,
        operation_kind="query",
        provider_cost=1.4e-07,
    )
    fields.update(overrides)
    return EmbeddingAccountingContribution(**fields)


# --------------------------------------------------------------------------
# Required fields and immutability
# --------------------------------------------------------------------------


def test_required_fields_are_carried():
    c = _contribution()
    assert (c.provider, c.model, c.input_quantity) == ("deepinfra", "BAAI/bge-m3", 14)
    assert c.quantity_unit == QUANTITY_UNIT_INPUT_TOKENS
    assert c.quantity_origin == ORIGIN_PROVIDER_REPORTED
    assert c.cost_state == COST_PROVIDER_AUTHORITATIVE
    assert c.operation_kind == "query"


def test_contribution_is_frozen():
    c = _contribution()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.input_quantity = 99


def test_optional_provider_fields_default_to_absent():
    c = _contribution(cost_state=COST_NO_PROVIDER_BILLING, provider_cost=None)
    assert c.provider_cost is None
    assert c.provider_request_id is None
    assert c.provider_runtime_ms is None


def test_optional_provider_fields_are_carried_when_supplied():
    c = _contribution(provider_request_id="abc-123", provider_runtime_ms=106)
    assert c.provider_request_id == "abc-123"
    assert c.provider_runtime_ms == 106


def test_no_free_form_details_field_exists():
    names = {f.name for f in dataclasses.fields(EmbeddingAccountingContribution)}
    assert names == {
        "provider",
        "model",
        "input_quantity",
        "quantity_unit",
        "quantity_origin",
        "cost_state",
        "operation_kind",
        "provider_cost",
        "provider_request_id",
        "provider_runtime_ms",
    }


def test_no_request_id_endpoint_or_payload_fields():
    names = {f.name for f in dataclasses.fields(EmbeddingAccountingContribution)}
    for forbidden in ("request_id", "service", "path", "endpoint", "success", "text",
                      "vectors", "response", "details"):
        assert forbidden not in names


# --------------------------------------------------------------------------
# Identity and quantity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["provider", "model"])
def test_identity_must_be_a_non_empty_string(field):
    with pytest.raises(ValueError):
        _contribution(**{field: ""})


def test_unknown_quantity_unit_is_rejected():
    with pytest.raises(ValueError):
        _contribution(quantity_unit="characters")


def test_unknown_quantity_origin_is_rejected():
    with pytest.raises(ValueError):
        _contribution(quantity_origin="guessed")


def test_maui_derived_origin_is_representable():
    c = _contribution(
        quantity_origin=ORIGIN_MAUI_DERIVED,
        cost_state=COST_PROVIDER_ABSENT_RESOLVABLE,
        provider_cost=None,
    )
    assert c.quantity_origin == ORIGIN_MAUI_DERIVED


def test_negative_or_non_integer_quantity_is_rejected():
    with pytest.raises(ValueError):
        _contribution(input_quantity=-1)
    with pytest.raises(ValueError):
        _contribution(input_quantity="14")
    with pytest.raises(ValueError):
        _contribution(input_quantity=True)


# --------------------------------------------------------------------------
# Cost semantics (DC7): absence is never zero
# --------------------------------------------------------------------------


def test_authoritative_cost_requires_a_value():
    with pytest.raises(ValueError):
        _contribution(cost_state=COST_PROVIDER_AUTHORITATIVE, provider_cost=None)


def test_a_zero_cost_is_valid_and_distinct_from_absence():
    priced = _contribution(provider_cost=0.0)
    unpriced = _contribution(
        cost_state=COST_PROVIDER_ABSENT_RESOLVABLE, provider_cost=None
    )
    assert priced.provider_cost == 0.0
    assert unpriced.provider_cost is None
    assert priced.cost_state != unpriced.cost_state


def test_the_three_cost_states_are_distinguishable():
    resolvable = _contribution(
        cost_state=COST_PROVIDER_ABSENT_RESOLVABLE, provider_cost=None
    )
    no_billing = _contribution(cost_state=COST_NO_PROVIDER_BILLING, provider_cost=None)
    authoritative = _contribution()
    states = {resolvable.cost_state, no_billing.cost_state, authoritative.cost_state}
    assert len(states) == 3


def test_non_authoritative_state_must_not_carry_a_cost():
    with pytest.raises(ValueError):
        _contribution(cost_state=COST_NO_PROVIDER_BILLING, provider_cost=0.5)


def test_unknown_cost_state_is_rejected():
    with pytest.raises(ValueError):
        _contribution(cost_state="free")


# --------------------------------------------------------------------------
# Operation taxonomy (DC4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["query", "document", "probe"])
def test_taxonomy_accepts_exactly_the_three_kinds(kind):
    assert _contribution(operation_kind=kind).operation_kind == kind


def test_taxonomy_rejects_anything_else():
    with pytest.raises(ValueError):
        _contribution(operation_kind="agentchat")


def test_provider_runtime_must_be_a_non_negative_int_when_present():
    with pytest.raises(ValueError):
        _contribution(provider_runtime_ms=-1)
    with pytest.raises(ValueError):
        _contribution(provider_runtime_ms=1.5)
