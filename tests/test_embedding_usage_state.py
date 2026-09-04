"""Tests for usage.embedding_state (design §21.3).

Scope: the request-scoped accumulator and the DC8 request binding — 0/1/N
contributions, order, the provider/model invariant reaction, request
isolation, and exception-safe unbinding.
"""

import logging

import pytest

from usage.embedding_accounting import (
    COST_PROVIDER_AUTHORITATIVE,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_UNIT_INPUT_TOKENS,
    EmbeddingAccountingContribution,
)
from usage.embedding_accounting_sink import (
    get_embedding_accounting_sink,
    no_op_sink,
)
from usage.embedding_state import (
    EmbeddingAccountingAccumulator,
    bind_embedding_accounting,
    get_embedding_accumulator,
    get_embedding_contributions,
)


def _make_app():
    from flask import Flask

    return Flask(__name__)


def _contribution(provider="deepinfra", model="BAAI/bge-m3", quantity=14):
    return EmbeddingAccountingContribution(
        provider=provider,
        model=model,
        input_quantity=quantity,
        quantity_unit=QUANTITY_UNIT_INPUT_TOKENS,
        quantity_origin=ORIGIN_PROVIDER_REPORTED,
        cost_state=COST_PROVIDER_AUTHORITATIVE,
        operation_kind="query",
        provider_cost=1.4e-07,
    )


# --------------------------------------------------------------------------
# Accumulator, standalone (no Flask required)
# --------------------------------------------------------------------------


def test_accumulator_starts_empty():
    assert EmbeddingAccountingAccumulator().contributions == ()


def test_accumulator_preserves_order():
    acc = EmbeddingAccountingAccumulator()
    for i in range(3):
        acc.add(_contribution(quantity=i))
    assert [c.input_quantity for c in acc.contributions] == [0, 1, 2]


def test_contributions_snapshot_is_immutable():
    acc = EmbeddingAccountingAccumulator()
    acc.add(_contribution())
    snapshot = acc.contributions
    acc.add(_contribution())
    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 1


def test_matching_provider_model_is_not_a_violation():
    acc = EmbeddingAccountingAccumulator()
    acc.add(_contribution())
    acc.add(_contribution())
    assert acc.invariant_violations == 0


def test_invariant_violation_is_recorded_and_warned_but_never_raises(caplog):
    acc = EmbeddingAccountingAccumulator()
    acc.add(_contribution())
    with caplog.at_level(logging.WARNING, logger="usage.embedding_state"):
        acc.add(_contribution(provider="openai", model="text-embedding-3-small"))

    assert acc.invariant_violations == 1
    # The contribution is kept: real consumption is never discarded.
    assert len(acc.contributions) == 2
    assert "invariant violated" in caplog.text
    # Only configuration identities are named, never content.
    assert "openai" in caplog.text


# --------------------------------------------------------------------------
# Request scoping
# --------------------------------------------------------------------------


def test_accumulator_absent_until_created():
    with _make_app().test_request_context("/"):
        assert get_embedding_accumulator() is None
        assert get_embedding_contributions() == ()


def test_lazy_creation_is_stable_within_one_request():
    with _make_app().test_request_context("/"):
        first = get_embedding_accumulator(create=True)
        assert get_embedding_accumulator() is first


def test_requests_are_isolated():
    app = _make_app()
    with app.test_request_context("/"):
        with bind_embedding_accounting():
            get_embedding_accounting_sink()(_contribution())
        assert len(get_embedding_contributions()) == 1
    with app.test_request_context("/"):
        assert get_embedding_contributions() == ()


# --------------------------------------------------------------------------
# DC8 binding
# --------------------------------------------------------------------------


def test_binding_delivers_zero_contributions():
    with _make_app().test_request_context("/"):
        with bind_embedding_accounting():
            pass
        assert get_embedding_contributions() == ()


def test_binding_delivers_n_contributions_in_order():
    with _make_app().test_request_context("/"):
        with bind_embedding_accounting():
            for i in range(4):
                get_embedding_accounting_sink()(_contribution(quantity=i))
        assert [c.input_quantity for c in get_embedding_contributions()] == [0, 1, 2, 3]


def test_binding_unbinds_the_sink_on_exit():
    with _make_app().test_request_context("/"):
        with bind_embedding_accounting():
            assert get_embedding_accounting_sink() is not no_op_sink
        assert get_embedding_accounting_sink() is no_op_sink


def test_binding_unbinds_the_sink_on_exception_and_keeps_delivered_work():
    with _make_app().test_request_context("/"):
        with pytest.raises(RuntimeError):
            with bind_embedding_accounting():
                get_embedding_accounting_sink()(_contribution())
                raise RuntimeError("boom")
        assert get_embedding_accounting_sink() is no_op_sink
        assert len(get_embedding_contributions()) == 1
