# utils/embedding_accounting.py
"""Normalized embedding-accounting contribution contract (§9).

One :class:`EmbeddingAccountingContribution` represents exactly one
successful provider accounting response (DC3). It is provider-agnostic: no
native provider field name survives into it, and no raw provider payload,
query text, document chunk or embedding vector may ever reach it (§17).

The type is frozen: a contribution is a fact about a call that already
happened. Aggregation (§9.6) is the concern of whatever consumes the
accumulator, not of this module — this module owns the vocabulary and the
per-item invariants only. It writes nothing, logs nothing, and knows
nothing about Flask, HTTP endpoints or the Maui request id (§9.3).
"""

from dataclasses import dataclass

from utils.embedding_operation_context import OPERATION_KINDS

__all__ = [
    "QUANTITY_UNIT_INPUT_TOKENS",
    "QUANTITY_UNITS",
    "ORIGIN_PROVIDER_REPORTED",
    "ORIGIN_MAUI_DERIVED",
    "QUANTITY_ORIGINS",
    "COST_PROVIDER_AUTHORITATIVE",
    "COST_PROVIDER_ABSENT_RESOLVABLE",
    "COST_NO_PROVIDER_BILLING",
    "COST_STATES",
    "EmbeddingAccountingContribution",
]

#: The only unit the CURRENT provider set reports. Named rather than
#: implied, because DC7 forbids a bare total: a Maui-derived quantity is not
#: the same unit as a provider's input-token count.
QUANTITY_UNIT_INPUT_TOKENS = "input_tokens"

QUANTITY_UNITS = frozenset({QUANTITY_UNIT_INPUT_TOKENS})

#: Read from the provider's own accounting response.
ORIGIN_PROVIDER_REPORTED = "provider_reported"

#: Computed locally by Maui. No CURRENT capture path produces this; the
#: vocabulary exists so a future one cannot be silently indistinguishable
#: from an authoritative count (DC7).
ORIGIN_MAUI_DERIVED = "maui_derived"

QUANTITY_ORIGINS = frozenset({ORIGIN_PROVIDER_REPORTED, ORIGIN_MAUI_DERIVED})

#: The provider billed an amount and reported it (DeepInfra).
COST_PROVIDER_AUTHORITATIVE = "provider_authoritative"

#: The provider bills, but supplies no cost at this seam; a Maui-side price
#: could resolve it later from the quantity (OpenAI, Mistral).
COST_PROVIDER_ABSENT_RESOLVABLE = "provider_absent_resolvable"

#: No monetary provider billing exists at all (Ollama). Distinct from the
#: state above: "resolvable later" and "meaningless" must not collapse.
COST_NO_PROVIDER_BILLING = "no_provider_billing"

COST_STATES = frozenset(
    {
        COST_PROVIDER_AUTHORITATIVE,
        COST_PROVIDER_ABSENT_RESOLVABLE,
        COST_NO_PROVIDER_BILLING,
    }
)


@dataclass(frozen=True, slots=True)
class EmbeddingAccountingContribution:
    """One normalized accounting fact per successful provider response.

    Required (§9.1): ``provider``, ``model``, ``input_quantity``,
    ``quantity_unit``, ``quantity_origin``, ``cost_state``,
    ``operation_kind``.

    Optional (§9.2), ``None`` when the provider does not supply them:
    ``provider_cost``, ``provider_request_id``, ``provider_runtime_ms``.

    ``provider_cost is None`` means *absent*, never zero: ``0.0`` is a valid
    cost, a distinction ``infrastructure/asr_accounting.py`` already draws.
    ``cost_state`` is what says *why* a cost is absent, so absence is never
    silently read as free.
    """

    provider: str
    model: str
    input_quantity: int
    quantity_unit: str
    quantity_origin: str
    cost_state: str
    operation_kind: str
    provider_cost: "float | None" = None
    provider_request_id: "str | None" = None
    provider_runtime_ms: "int | None" = None

    def __post_init__(self) -> None:
        _require_identity("provider", self.provider)
        _require_identity("model", self.model)
        _require_member("quantity_unit", self.quantity_unit, QUANTITY_UNITS)
        _require_member("quantity_origin", self.quantity_origin, QUANTITY_ORIGINS)
        _require_member("cost_state", self.cost_state, COST_STATES)
        _require_member("operation_kind", self.operation_kind, OPERATION_KINDS)

        # bool is an int subclass; a flag is never a quantity.
        if isinstance(self.input_quantity, bool) or not isinstance(
            self.input_quantity, int
        ):
            raise ValueError("input_quantity must be an int")
        if self.input_quantity < 0:
            raise ValueError("input_quantity must not be negative")

        if self.provider_cost is not None:
            if isinstance(self.provider_cost, bool) or not isinstance(
                self.provider_cost, (int, float)
            ):
                raise ValueError("provider_cost must be a number or None")
            if self.provider_cost < 0:
                raise ValueError("provider_cost must not be negative")

        # The cost state and the cost value must agree, so a reader never has
        # to guess which one is authoritative.
        if self.cost_state == COST_PROVIDER_AUTHORITATIVE:
            if self.provider_cost is None:
                raise ValueError(
                    "cost_state 'provider_authoritative' requires a provider_cost"
                )
        elif self.provider_cost is not None:
            raise ValueError(
                f"cost_state {self.cost_state!r} must not carry a provider_cost"
            )

        if self.provider_request_id is not None:
            _require_identity("provider_request_id", self.provider_request_id)

        if self.provider_runtime_ms is not None:
            if isinstance(self.provider_runtime_ms, bool) or not isinstance(
                self.provider_runtime_ms, int
            ):
                raise ValueError("provider_runtime_ms must be an int or None")
            if self.provider_runtime_ms < 0:
                raise ValueError("provider_runtime_ms must not be negative")


def _require_identity(field: str, value) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_member(field: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"unknown {field}: {value!r}; expected one of {sorted(allowed)}")
