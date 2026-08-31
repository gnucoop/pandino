"""Tests for utils.embedding_operation_context (design §21.2).

Scope: the scoped operation ContextVar only — taxonomy, token/reset,
exception-safe restoration, sequential scopes, and separation from the
logging request-id context.
"""

import pytest

from utils.embedding_operation_context import (
    OPERATION_DOCUMENT,
    OPERATION_KINDS,
    OPERATION_PROBE,
    OPERATION_QUERY,
    embedding_operation,
    get_embedding_operation,
    reset_embedding_operation,
    set_embedding_operation,
)


def test_taxonomy_is_exactly_query_document_probe():
    assert OPERATION_KINDS == {"query", "document", "probe"}


def test_no_active_operation_by_default():
    assert get_embedding_operation() is None


@pytest.mark.parametrize("kind", [OPERATION_QUERY, OPERATION_DOCUMENT, OPERATION_PROBE])
def test_each_kind_is_observable_inside_its_scope(kind):
    with embedding_operation(kind):
        assert get_embedding_operation() == kind
    assert get_embedding_operation() is None


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        set_embedding_operation("retrieval")
    assert get_embedding_operation() is None


def test_token_reset_restores_previous_value():
    token = set_embedding_operation(OPERATION_QUERY)
    assert get_embedding_operation() == OPERATION_QUERY
    reset_embedding_operation(token)
    assert get_embedding_operation() is None


def test_reset_tolerates_none_token():
    reset_embedding_operation(None)
    assert get_embedding_operation() is None


def test_nested_scopes_restore_the_outer_value():
    with embedding_operation(OPERATION_PROBE):
        with embedding_operation(OPERATION_DOCUMENT):
            assert get_embedding_operation() == OPERATION_DOCUMENT
        assert get_embedding_operation() == OPERATION_PROBE
    assert get_embedding_operation() is None


def test_exception_inside_scope_still_restores():
    with pytest.raises(RuntimeError):
        with embedding_operation(OPERATION_DOCUMENT):
            raise RuntimeError("boom")
    assert get_embedding_operation() is None


def test_sequential_scopes_do_not_bleed():
    observed = []
    for kind in (OPERATION_PROBE, OPERATION_DOCUMENT, OPERATION_QUERY):
        with embedding_operation(kind):
            observed.append(get_embedding_operation())
        assert get_embedding_operation() is None
    assert observed == [OPERATION_PROBE, OPERATION_DOCUMENT, OPERATION_QUERY]


def test_context_is_separate_from_the_logging_request_id_vars():
    from utils import logging_config

    with embedding_operation(OPERATION_QUERY):
        # The logging context is untouched by an embedding scope.
        assert logging_config.get_request_id() == logging_config.CONTEXT_UNSET

    tokens = logging_config.set_request_context(request_id="req-1")
    try:
        assert get_embedding_operation() is None
    finally:
        logging_config.reset_request_context(tokens)


def test_module_does_not_import_flask():
    import utils.embedding_operation_context as module

    source = open(module.__file__).read()
    assert "import flask" not in source
    assert "from flask" not in source
