"""Tests for usage.embedding_accounting_sink (DC8).

Scope: the sink ContextVar only — no-op default, delivery of 0..N
contributions, token/reset, exception-safe restoration, Flask independence.
"""

import pytest

from usage.embedding_accounting_sink import (
    embedding_accounting_sink,
    get_embedding_accounting_sink,
    no_op_sink,
    reset_embedding_accounting_sink,
    set_embedding_accounting_sink,
)


def test_default_sink_is_the_no_op():
    assert get_embedding_accounting_sink() is no_op_sink


def test_no_sink_bound_is_a_safe_no_op():
    assert get_embedding_accounting_sink()(object()) is None


def test_bound_sink_receives_one_contribution():
    received = []
    with embedding_accounting_sink(received.append):
        get_embedding_accounting_sink()("c1")
    assert received == ["c1"]


def test_bound_sink_receives_n_contributions_in_order():
    received = []
    with embedding_accounting_sink(received.append):
        for i in range(5):
            get_embedding_accounting_sink()(i)
    assert received == [0, 1, 2, 3, 4]


def test_reset_restores_the_no_op_default():
    token = set_embedding_accounting_sink(lambda c: None)
    assert get_embedding_accounting_sink() is not no_op_sink
    reset_embedding_accounting_sink(token)
    assert get_embedding_accounting_sink() is no_op_sink


def test_nested_binding_restores_the_outer_sink():
    outer, inner = [], []
    with embedding_accounting_sink(outer.append):
        with embedding_accounting_sink(inner.append):
            get_embedding_accounting_sink()("inner")
        get_embedding_accounting_sink()("outer")
    assert outer == ["outer"]
    assert inner == ["inner"]
    assert get_embedding_accounting_sink() is no_op_sink


def test_exception_inside_scope_still_restores():
    with pytest.raises(RuntimeError):
        with embedding_accounting_sink(lambda c: None):
            raise RuntimeError("boom")
    assert get_embedding_accounting_sink() is no_op_sink


def test_reset_tolerates_none_token():
    reset_embedding_accounting_sink(None)
    assert get_embedding_accounting_sink() is no_op_sink


def test_non_callable_sink_is_rejected():
    with pytest.raises(TypeError):
        set_embedding_accounting_sink("not callable")
    assert get_embedding_accounting_sink() is no_op_sink


def test_module_does_not_import_flask():
    import usage.embedding_accounting_sink as module

    source = open(module.__file__).read()
    assert "import flask" not in source
    assert "from flask" not in source
