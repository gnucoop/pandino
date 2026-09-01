"""Tests for utils.usage_provenance.

Scope: the persisted cost-origin vocabulary and the pure mapping from the
capture-time cost state onto it.
"""

import pytest

from utils.embedding_accounting import (
    COST_NO_PROVIDER_BILLING,
    COST_PROVIDER_ABSENT_RESOLVABLE,
    COST_PROVIDER_AUTHORITATIVE,
    ORIGIN_MAUI_DERIVED,
    ORIGIN_PROVIDER_REPORTED,
)
from utils.usage_provenance import (
    COST_ORIGIN_MAUI_RESOLVED,
    COST_ORIGIN_NO_PROVIDER_BILLING,
    COST_ORIGIN_PROVIDER_AUTHORITATIVE,
    COST_ORIGINS,
    cost_origin_from_cost_state,
)
import utils.usage_provenance as provenance


# --------------------------------------------------------------------------
# Exported vocabulary
# --------------------------------------------------------------------------


def test_cost_origin_constants_carry_the_accepted_strings():
    assert COST_ORIGIN_PROVIDER_AUTHORITATIVE == "provider_authoritative"
    assert COST_ORIGIN_MAUI_RESOLVED == "maui_resolved"
    assert COST_ORIGIN_NO_PROVIDER_BILLING == "no_provider_billing"


def test_cost_origins_is_exactly_the_three_values():
    assert COST_ORIGINS == {
        "provider_authoritative",
        "maui_resolved",
        "no_provider_billing",
    }


def test_reused_vocabularies_are_the_capture_ones():
    assert provenance.ORIGIN_PROVIDER_REPORTED is ORIGIN_PROVIDER_REPORTED
    assert provenance.ORIGIN_MAUI_DERIVED is ORIGIN_MAUI_DERIVED
    assert provenance.OPERATION_KINDS == {"query", "document", "probe"}


# --------------------------------------------------------------------------
# cost_state -> cost_origin
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cost_state,expected",
    [
        (COST_PROVIDER_AUTHORITATIVE, "provider_authoritative"),
        (COST_PROVIDER_ABSENT_RESOLVABLE, "maui_resolved"),
        (COST_NO_PROVIDER_BILLING, "no_provider_billing"),
    ],
)
def test_mapping_is_the_accepted_one(cost_state, expected):
    assert cost_origin_from_cost_state(cost_state) == expected


@pytest.mark.parametrize("unknown", ["free", "maui_resolved", "", None])
def test_unknown_cost_state_raises(unknown):
    with pytest.raises(ValueError):
        cost_origin_from_cost_state(unknown)
