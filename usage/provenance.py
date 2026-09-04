# usage/provenance.py
"""Usage-facing provenance vocabulary for persisted consumption rows.

Owns the words a persisted Usage row uses to say *where its numbers came
from*, so persistence depends on one Usage-owned vocabulary rather than on
capture-layer internals, and a future non-embedding producer can adopt the
same words without importing anything embedding-shaped.

Two of the three vocabularies are identical to the capture ones and are
re-exported verbatim rather than respelled: the operation taxonomy and the
quantity origin. The third, ``cost_origin``, is a genuine mapping:
``cost_state`` describes what the *provider* supplied when the call
happened, while ``cost_origin`` describes where the *persisted number*
came from. A provider that supplied no cost but can be priced locally is
``provider_absent_resolvable`` at capture time and ``maui_resolved`` once a
row exists.

Pure: no Flask, no database, no I/O, no logging.
"""

from usage.embedding_accounting import (
    COST_NO_PROVIDER_BILLING,
    COST_PROVIDER_ABSENT_RESOLVABLE,
    COST_PROVIDER_AUTHORITATIVE,
    ORIGIN_MAUI_DERIVED,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_ORIGINS,
)
from usage.embedding_operation_context import (
    OPERATION_DOCUMENT,
    OPERATION_KINDS,
    OPERATION_PROBE,
    OPERATION_QUERY,
)

__all__ = [
    "OPERATION_QUERY",
    "OPERATION_DOCUMENT",
    "OPERATION_PROBE",
    "OPERATION_KINDS",
    "ORIGIN_PROVIDER_REPORTED",
    "ORIGIN_MAUI_DERIVED",
    "QUANTITY_ORIGINS",
    "COST_ORIGIN_PROVIDER_AUTHORITATIVE",
    "COST_ORIGIN_MAUI_RESOLVED",
    "COST_ORIGIN_NO_PROVIDER_BILLING",
    "COST_ORIGINS",
    "cost_origin_from_cost_state",
]

#: The provider billed the amount and reported it; the stored number is the
#: provider's own.
COST_ORIGIN_PROVIDER_AUTHORITATIVE = "provider_authoritative"

#: The provider billed but supplied no amount; Maui priced the quantity, so
#: the stored number is Maui's.
COST_ORIGIN_MAUI_RESOLVED = "maui_resolved"

#: No monetary provider billing exists at all. Says that a stored zero is an
#: absence of billing, not a priced zero.
COST_ORIGIN_NO_PROVIDER_BILLING = "no_provider_billing"

COST_ORIGINS = frozenset(
    {
        COST_ORIGIN_PROVIDER_AUTHORITATIVE,
        COST_ORIGIN_MAUI_RESOLVED,
        COST_ORIGIN_NO_PROVIDER_BILLING,
    }
)

_COST_ORIGIN_BY_COST_STATE = {
    COST_PROVIDER_AUTHORITATIVE: COST_ORIGIN_PROVIDER_AUTHORITATIVE,
    COST_PROVIDER_ABSENT_RESOLVABLE: COST_ORIGIN_MAUI_RESOLVED,
    COST_NO_PROVIDER_BILLING: COST_ORIGIN_NO_PROVIDER_BILLING,
}


def cost_origin_from_cost_state(cost_state: str) -> str:
    """Map a capture-time cost state onto the persisted cost origin.

    Total over the supported capture vocabulary and defined nowhere else:
    an unknown state raises rather than falling back or echoing its input,
    because a row whose provenance word is a guess is worse than no row.

    :raises ValueError: if ``cost_state`` is not a supported capture state.
    """
    try:
        return _COST_ORIGIN_BY_COST_STATE[cost_state]
    except KeyError:
        raise ValueError(
            f"unknown cost_state: {cost_state!r}; expected one of "
            f"{sorted(_COST_ORIGIN_BY_COST_STATE)}"
        ) from None
