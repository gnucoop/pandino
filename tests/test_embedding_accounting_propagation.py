"""Mandatory runtime/thread propagation regression test (design §21.5).

Proves that BOTH the operation ContextVar and the sink ContextVar survive
the execution path ``PGVectorStore`` uses, without any network or database
call.

The path is reproduced from verified upstream source rather than mocked
away:

* ``PGEngine._run_as_sync`` is ``asyncio.run_coroutine_threadsafe(coro,
  self._loop).result()`` against a background event-loop thread created at
  engine construction. The hop is reproduced here with a real background
  loop thread and the same submission call.
* ``langchain_core.runnables.config.run_in_executor`` is the async fallback
  DeepInfra's embeddings route through; it is imported and exercised for
  real, not re-implemented.

`[V]` Both propagations rest on third-party implementation choices Maui
neither owns nor pins, which is why this is a regression test rather than
an assumption — the same standard
``docs/logging/contextvars_gevent_verification.md`` set for litellm's
background thread.
"""

import asyncio
import threading

import pytest
from langchain_core.runnables.config import run_in_executor

from usage.embedding_accounting import (
    COST_NO_PROVIDER_BILLING,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_UNIT_INPUT_TOKENS,
    EmbeddingAccountingContribution,
)
from usage.embedding_accounting_sink import get_embedding_accounting_sink
from usage.embedding_operation_context import (
    OPERATION_DOCUMENT,
    OPERATION_PROBE,
    OPERATION_QUERY,
    embedding_operation,
    get_embedding_operation,
)
from usage.embedding_state import (
    bind_embedding_accounting,
    get_embedding_contributions,
)


@pytest.fixture
def background_loop():
    """A background event-loop thread, as PGEngine creates at construction."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


@pytest.fixture
def request_ctx():
    """An active Flask request context: `bind_embedding_accounting` is the one
    Flask-aware piece of the foundation and needs `flask.g` on the caller side."""
    from flask import Flask

    with Flask(__name__).test_request_context("/"):
        yield


def _run_as_sync(loop, coro):
    """Mirror of ``PGEngine._run_as_sync``."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=5)


def _contribution(kind, quantity=1):
    """A contribution the far side builds from what it can see ambiently."""
    return EmbeddingAccountingContribution(
        provider="ollama",
        model="bge-m3",
        input_quantity=quantity,
        quantity_unit=QUANTITY_UNIT_INPUT_TOKENS,
        quantity_origin=ORIGIN_PROVIDER_REPORTED,
        cost_state=COST_NO_PROVIDER_BILLING,
        operation_kind=kind,
    )


def test_both_contextvars_are_visible_on_the_background_loop_thread(background_loop, request_ctx):
    seen = {}

    async def _capture():
        seen["thread"] = threading.current_thread().name
        seen["operation"] = get_embedding_operation()
        seen["sink"] = get_embedding_accounting_sink()

    with embedding_operation(OPERATION_QUERY):
        with bind_embedding_accounting() as accumulator:
            caller_thread = threading.current_thread().name
            caller_sink = get_embedding_accounting_sink()
            _run_as_sync(background_loop, _capture())

    assert seen["thread"] != caller_thread  # the hop really happened
    assert seen["operation"] == OPERATION_QUERY
    # A bound method is a fresh object per attribute access, so identity is
    # asserted on what it closes over: the caller's accumulator instance.
    assert seen["sink"] == caller_sink
    assert seen["sink"].__self__ is accumulator


def test_contribution_delivered_from_the_background_thread_reaches_the_request(
    background_loop,
):
    from flask import Flask

    app = Flask(__name__)

    async def _capture_and_deliver():
        # Exactly what capture code will do: read the ambient kind, build the
        # contribution, hand it to the ambient sink. No Flask, no g.
        kind = get_embedding_operation()
        get_embedding_accounting_sink()(_contribution(kind))

    with app.test_request_context("/"):
        with bind_embedding_accounting():
            with embedding_operation(OPERATION_DOCUMENT):
                _run_as_sync(background_loop, _capture_and_deliver())

        contributions = get_embedding_contributions()

    assert len(contributions) == 1
    assert contributions[0].operation_kind == OPERATION_DOCUMENT


def test_both_contextvars_survive_langchain_run_in_executor_fallback(background_loop, request_ctx):
    """DeepInfra's async path routes back into sync code through this."""
    seen = {}

    def _sync_capture():
        seen["thread"] = threading.current_thread().name
        seen["operation"] = get_embedding_operation()
        get_embedding_accounting_sink()(_contribution(get_embedding_operation()))

    async def _async_entry():
        # executor_or_config=None is the fallback branch langchain_core takes,
        # which wraps the call in partial(copy_context().run, wrapper).
        await run_in_executor(None, _sync_capture)

    with embedding_operation(OPERATION_PROBE):
        with bind_embedding_accounting() as accumulator:
            caller_thread = threading.current_thread().name
            _run_as_sync(background_loop, _async_entry())

    assert seen["thread"] != caller_thread
    assert seen["operation"] == OPERATION_PROBE
    assert [c.operation_kind for c in accumulator.contributions] == [OPERATION_PROBE]


def test_sequential_operations_across_the_hop_do_not_bleed(background_loop, request_ctx):
    async def _deliver():
        get_embedding_accounting_sink()(_contribution(get_embedding_operation()))

    with bind_embedding_accounting() as accumulator:
        for kind in (OPERATION_PROBE, OPERATION_DOCUMENT, OPERATION_DOCUMENT):
            with embedding_operation(kind):
                _run_as_sync(background_loop, _deliver())
        # The scope is closed again on the caller side after every hop.
        assert get_embedding_operation() is None

    assert [c.operation_kind for c in accumulator.contributions] == [
        OPERATION_PROBE,
        OPERATION_DOCUMENT,
        OPERATION_DOCUMENT,
    ]


def test_far_side_set_does_not_leak_back_to_the_caller(background_loop):
    """One-way propagation: the far side cannot mutate the caller's scope."""
    from usage.embedding_operation_context import set_embedding_operation

    async def _rebind():
        set_embedding_operation(OPERATION_DOCUMENT)  # token deliberately dropped

    with embedding_operation(OPERATION_QUERY):
        _run_as_sync(background_loop, _rebind())
        assert get_embedding_operation() == OPERATION_QUERY


def test_exception_on_the_far_side_still_restores_caller_scopes(background_loop, request_ctx):
    async def _boom():
        raise RuntimeError("provider exploded")

    with pytest.raises(RuntimeError):
        with embedding_operation(OPERATION_QUERY):
            with bind_embedding_accounting():
                _run_as_sync(background_loop, _boom())

    from usage.embedding_accounting_sink import no_op_sink

    assert get_embedding_operation() is None
    assert get_embedding_accounting_sink() is no_op_sink
