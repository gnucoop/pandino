"""Tests for utils.embedding_usage_persistence.

Focused on the module's own concern only: composing the request-scoped
accumulator, the pure aggregator, the request attribution, the provenance
mapping, the batch writer and Usage row-id registration at the end of one
HTTP request. The settled modules it composes are not re-tested here -
they are either used for real (aggregation, provenance, request state,
attribution state, accumulator) or monkeypatched exactly as this module
consumes them (the batch writer, which is the only DB boundary).
"""

import ast
import logging
import os
from unittest.mock import patch

import pytest
from flask import Flask

from utils.embedding_accounting import (
    COST_NO_PROVIDER_BILLING,
    COST_PROVIDER_ABSENT_RESOLVABLE,
    COST_PROVIDER_AUTHORITATIVE,
    EmbeddingAccountingContribution,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_UNIT_INPUT_TOKENS,
)
from utils.embedding_operation_context import OPERATION_DOCUMENT, OPERATION_QUERY
from utils.embedding_usage_persistence import (
    _G_PERSISTED_ATTR,
    _HOOKS_MARKER,
    register_embedding_usage_persistence_hooks,
)
from utils.embedding_usage_state import get_embedding_accumulator
from utils.usage_attribution_state import bind_usage_attribution
from utils.usage_request_state import get_usage_log_id, get_usage_log_ids

MODULE = "utils.embedding_usage_persistence"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _contribution(
    provider="deepinfra",
    model="BAAI/bge-m3",
    input_quantity=10,
    quantity_origin=ORIGIN_PROVIDER_REPORTED,
    cost_state=COST_PROVIDER_AUTHORITATIVE,
    operation_kind=OPERATION_QUERY,
    provider_cost=0.25,
):
    """One valid contribution, with every discriminating field overridable."""
    return EmbeddingAccountingContribution(
        provider=provider,
        model=model,
        input_quantity=input_quantity,
        quantity_unit=QUANTITY_UNIT_INPUT_TOKENS,
        quantity_origin=quantity_origin,
        cost_state=cost_state,
        operation_kind=operation_kind,
        provider_cost=provider_cost,
    )


def _make_app(contributions=(), attribution=(7, "/completion.json", "web")):
    """Throwaway app carrying only this slice's hooks and a probe route.

    The accumulator and the attribution are seeded inside the request, the
    way the sink hook and a route would seed them in production, so the
    lifecycle module is exercised through its real request-local seams
    rather than through patched readers.

    :param attribution: the ``(user_id, service, source)`` to bind, or
        ``None`` to leave the request unattributed.
    """
    app = Flask(__name__)
    register_embedding_usage_persistence_hooks(app)

    @app.before_request
    def _seed():
        accumulator = get_embedding_accumulator(create=True)
        for contribution in contributions:
            accumulator.add(contribution)
        if attribution is not None:
            user_id, service, source = attribution
            bind_usage_attribution(user_id, service, source)

    @app.route("/ping")
    def ping():
        return "pong"

    @app.route("/boom")
    def boom():
        raise RuntimeError("view exploded")

    return app


class _RecordingWriter:
    """Stand-in for log_resolved_cost_usage_batch, recording every call."""

    def __init__(self, ids=None, exc=None):
        self.calls = []
        self._ids = ids
        self._exc = exc

    def __call__(self, entries):
        entries = list(entries)
        self.calls.append(entries)
        if self._exc is not None:
            raise self._exc
        return list(self._ids) if self._ids is not None else list(
            range(100, 100 + len(entries))
        )


def _run(app, writer, path="/ping"):
    """Drive one request with the batch writer replaced.

    :return: the ``(response, registered_ids, single_slot_id)`` triple. The
        ids are read inside a teardown handler registered last - and so run
        first, before the persistence fallback - which on the normal path
        observes exactly what the duration finalizer already saw.
    """
    holder = {}

    @app.teardown_request
    def _read(exc=None):
        holder["ids"] = get_usage_log_ids()
        holder["single"] = get_usage_log_id()

    with patch(f"{MODULE}.log_resolved_cost_usage_batch", writer):
        response = app.test_client().get(path)

    return response, holder.get("ids", ()), holder.get("single")


# --------------------------------------------------------------------------
# Normal persistence path
# --------------------------------------------------------------------------


def test_single_contribution_persists_one_row_with_mapped_fields():
    writer = _RecordingWriter(ids=[41])
    app = _make_app([_contribution(input_quantity=12, provider_cost=0.5)])

    response, ids, single = _run(app, writer)

    assert response.status_code == 200
    assert len(writer.calls) == 1
    (entry,) = writer.calls[0]
    assert entry.user_id == 7
    assert entry.service == "/completion.json"
    assert entry.source == "web"
    assert entry.provider == "deepinfra"
    assert entry.model == "BAAI/bge-m3"
    assert entry.token_input == 12
    assert entry.token_output == 0
    assert entry.cost == pytest.approx(0.5)
    assert entry.embedding_operation_kind == OPERATION_QUERY
    assert entry.quantity_origin == ORIGIN_PROVIDER_REPORTED
    assert entry.cost_origin == "provider_authoritative"
    assert entry.request_id
    # Duration is never supplied at insert; the finalizer owns it later.
    assert not hasattr(entry, "duration_ms")
    assert ids == (41,)


def test_matching_contributions_collapse_into_one_entry():
    writer = _RecordingWriter(ids=[1])
    app = _make_app(
        [
            _contribution(input_quantity=10, provider_cost=0.1),
            _contribution(input_quantity=5, provider_cost=0.4),
        ]
    )

    _run(app, writer)

    (entry,) = writer.calls[0]
    assert entry.token_input == 15
    assert entry.cost == pytest.approx(0.5)


def test_multiple_partitions_are_one_batch_in_first_appearance_order():
    writer = _RecordingWriter(ids=[11, 12])
    app = _make_app(
        [
            _contribution(operation_kind=OPERATION_DOCUMENT, input_quantity=3),
            _contribution(operation_kind=OPERATION_QUERY, input_quantity=4),
        ]
    )

    _, ids, _ = _run(app, writer)

    assert len(writer.calls) == 1
    first, second = writer.calls[0]
    assert first.embedding_operation_kind == OPERATION_DOCUMENT
    assert second.embedding_operation_kind == OPERATION_QUERY
    assert ids == (11, 12)


def test_ids_are_registered_not_set_into_the_legacy_single_slot():
    writer = _RecordingWriter(ids=[55])
    app = _make_app([_contribution()])

    _, ids, single = _run(app, writer)

    assert ids == (55,)
    # The legacy single slot belongs to the request's primary LLM row and
    # must not be displaced by an embedding row.
    assert single is None


# --------------------------------------------------------------------------
# No-op and skip paths
# --------------------------------------------------------------------------


def test_no_contributions_touches_no_database():
    writer = _RecordingWriter()
    app = _make_app([])

    response, ids, _ = _run(app, writer)

    assert response.status_code == 200
    assert writer.calls == []
    assert ids == ()


def test_missing_attribution_skips_and_logs(caplog):
    writer = _RecordingWriter()
    app = _make_app([_contribution()], attribution=None)

    with caplog.at_level(logging.WARNING, logger=MODULE):
        response, ids, _ = _run(app, writer)

    assert response.status_code == 200
    assert writer.calls == []
    assert ids == ()
    assert "event=embedding_usage_persistence_skipped" in caplog.text
    assert "reason=no_attribution" in caplog.text


def test_no_persistable_partition_touches_no_database(caplog):
    writer = _RecordingWriter()
    app = _make_app(
        [
            _contribution(
                cost_state=COST_PROVIDER_ABSENT_RESOLVABLE, provider_cost=None
            )
        ]
    )

    with caplog.at_level(logging.WARNING, logger=MODULE):
        _run(app, writer)

    assert writer.calls == []
    assert "reason=no_persistable_partitions" in caplog.text


# --------------------------------------------------------------------------
# Cost states
# --------------------------------------------------------------------------


def test_no_provider_billing_persists_an_honest_zero():
    writer = _RecordingWriter(ids=[9])
    app = _make_app(
        [_contribution(cost_state=COST_NO_PROVIDER_BILLING, provider_cost=None)]
    )

    _run(app, writer)

    (entry,) = writer.calls[0]
    assert entry.cost == pytest.approx(0.0)
    assert entry.cost_origin == "no_provider_billing"


def test_unresolved_partition_is_skipped_while_others_still_persist(caplog):
    writer = _RecordingWriter(ids=[3])
    app = _make_app(
        [
            _contribution(
                operation_kind=OPERATION_DOCUMENT,
                cost_state=COST_PROVIDER_ABSENT_RESOLVABLE,
                provider_cost=None,
            ),
            _contribution(operation_kind=OPERATION_QUERY, provider_cost=0.2),
        ]
    )

    with caplog.at_level(logging.WARNING, logger=MODULE):
        _, ids, _ = _run(app, writer)

    assert "event=embedding_usage_cost_unresolved" in caplog.text
    assert len(writer.calls) == 1
    (entry,) = writer.calls[0]
    assert entry.embedding_operation_kind == OPERATION_QUERY
    assert entry.cost_origin == "provider_authoritative"
    assert ids == (3,)


# --------------------------------------------------------------------------
# Duplicate prevention across the two seams
# --------------------------------------------------------------------------


def test_after_request_persists_and_teardown_does_not_retry():
    writer = _RecordingWriter(ids=[1])
    app = _make_app([_contribution()])

    _run(app, writer)

    assert len(writer.calls) == 1


def test_failed_write_is_not_retried_by_the_teardown_fallback(caplog):
    writer = _RecordingWriter(exc=RuntimeError("db down"))
    app = _make_app([_contribution()])

    with caplog.at_level(logging.ERROR, logger=MODULE):
        response, ids, _ = _run(app, writer)

    assert response.status_code == 200
    assert len(writer.calls) == 1
    assert ids == ()
    assert "event=embedding_usage_persistence_failed" in caplog.text


def test_teardown_is_the_only_seam_when_after_request_is_skipped():
    writer = _RecordingWriter(ids=[8])
    app = _make_app([_contribution()])
    # The exception must propagate out of the WSGI call for after_request
    # to be skipped: a handled 500 still finalizes a response and runs it.
    app.config["PROPAGATE_EXCEPTIONS"] = True
    after_request_ran = []

    @app.after_request
    def _probe(response):
        after_request_ran.append(True)
        return response

    with patch(f"{MODULE}.log_resolved_cost_usage_batch", writer):
        with pytest.raises(RuntimeError, match="view exploded"):
            app.test_client().get("/boom")

    # The probe proves after_request never ran for this request, so the
    # single write below can only have come from the teardown fallback.
    assert after_request_ran == []
    assert len(writer.calls) == 1


def test_flag_is_set_before_the_write_attempt():
    seen = {}
    writer = _RecordingWriter(ids=[1])
    app = _make_app([_contribution()])

    def _spy(entries):
        from flask import g

        seen["flag"] = getattr(g, _G_PERSISTED_ATTR, False)
        return writer(entries)

    with patch(f"{MODULE}.log_resolved_cost_usage_batch", _spy):
        app.test_client().get("/ping")

    assert seen["flag"] is True


# --------------------------------------------------------------------------
# Response preservation
# --------------------------------------------------------------------------


def test_aggregation_failure_leaves_the_response_unchanged(caplog):
    writer = _RecordingWriter()
    app = _make_app([_contribution()])

    with caplog.at_level(logging.ERROR, logger=MODULE), patch(
        f"{MODULE}.aggregate_embedding_contributions",
        side_effect=ValueError("bad aggregate"),
    ):
        with patch(f"{MODULE}.log_resolved_cost_usage_batch", writer):
            response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"
    assert writer.calls == []
    assert "event=embedding_usage_persistence_failed" in caplog.text


def test_writer_failure_leaves_the_response_unchanged():
    writer = _RecordingWriter(exc=RuntimeError("insert failed"))
    app = _make_app([_contribution()])

    response, _, _ = _run(app, writer)

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"


# --------------------------------------------------------------------------
# Registration and isolation
# --------------------------------------------------------------------------


def test_registration_is_idempotent():
    app = Flask(__name__)
    register_embedding_usage_persistence_hooks(app)
    after_count = len(app.after_request_funcs[None])
    teardown_count = len(app.teardown_request_funcs[None])

    register_embedding_usage_persistence_hooks(app)

    assert getattr(app, _HOOKS_MARKER) is True
    assert len(app.after_request_funcs[None]) == after_count
    assert len(app.teardown_request_funcs[None]) == teardown_count


def test_sequential_requests_share_no_state():
    writer = _RecordingWriter(ids=[1])
    app = _make_app([_contribution()])

    with patch(f"{MODULE}.log_resolved_cost_usage_batch", writer):
        client = app.test_client()
        client.get("/ping")
        client.get("/ping")

    # Two independent requests, each claiming its own single attempt: the
    # idempotence flag, the accumulator and the attribution are all
    # discarded with the request context.
    assert len(writer.calls) == 2
    assert [len(call) for call in writer.calls] == [1, 1]


# --------------------------------------------------------------------------
# Hook ordering
# --------------------------------------------------------------------------


def _order_test_app():
    """An app wired with the exact production registration order."""
    from utils.embedding_accounting_lifecycle import register_embedding_accounting_hooks
    from utils.logging_config import register_request_context_hooks
    from utils.request_duration import register_request_duration_hooks
    from utils.usage_duration_finalization import (
        register_usage_duration_finalization_hooks,
    )

    app = Flask(__name__)
    register_request_context_hooks(app)
    register_usage_duration_finalization_hooks(app)
    register_embedding_usage_persistence_hooks(app)
    register_request_duration_hooks(app)
    register_embedding_accounting_hooks(app)
    return app


def test_effective_after_request_order_is_duration_then_persistence_then_finalizer():
    """Prove the real execution order, not the source order.

    Flask runs ``after_request`` LIFO, so the required normal-path sequence
    is asserted by observing what each stage can actually see.
    """
    observed = []
    app = _order_test_app()

    @app.route("/ping")
    def ping():
        return "pong"

    @app.before_request
    def _seed():
        get_embedding_accumulator(create=True).add(_contribution())
        bind_usage_attribution(7, "/completion.json", "web")

    def _writer(entries):
        from utils.request_duration import get_request_duration_ms

        # The request duration must already be finalized when the
        # embedding rows are written.
        observed.append(("persist", get_request_duration_ms() is not None))
        return [77]

    def _update(log_id, duration_ms):
        # The finalizer must see the embedding id, i.e. it must run after
        # persistence registered it.
        observed.append(("finalize", log_id, duration_ms))
        return True

    with patch(f"{MODULE}.log_resolved_cost_usage_batch", _writer), patch(
        "utils.usage_duration_finalization.update_usage_duration", _update
    ):
        response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert observed[0][0] == "persist"
    assert observed[0][1] is True, "duration was not finalized before persistence"
    assert [entry[0] for entry in observed[1:]] == ["finalize"]
    assert observed[1][1] == 77
    assert isinstance(observed[1][2], int)


def test_embedding_ids_join_the_multi_id_duration_lifecycle():
    """An LLM row set by a route and the embedding rows share one duration."""
    from utils.usage_request_state import set_usage_log_id

    finalized = []
    app = _order_test_app()

    @app.route("/ping")
    def ping():
        set_usage_log_id(500)  # the route's own LLM Usage row
        return "pong"

    @app.before_request
    def _seed():
        get_embedding_accumulator(create=True).add(_contribution())
        bind_usage_attribution(7, "/completion.json", "web")

    def _update(log_id, duration_ms):
        finalized.append((log_id, duration_ms))
        return True

    with patch(f"{MODULE}.log_resolved_cost_usage_batch", lambda entries: [501]), patch(
        "utils.usage_duration_finalization.update_usage_duration", _update
    ):
        app.test_client().get("/ping")

    assert [log_id for log_id, _ in finalized] == [500, 501]
    assert len({duration for _, duration in finalized}) == 1


def test_teardown_order_keeps_the_accumulator_and_request_id_readable():
    """The sink unbinds before the fallback; request_id unwinds after it."""
    from utils.logging_config import CONTEXT_UNSET

    seen = {}
    app = _order_test_app()
    app.config["PROPAGATE_EXCEPTIONS"] = True

    @app.route("/boom")
    def boom():
        raise RuntimeError("view exploded")

    @app.before_request
    def _seed():
        get_embedding_accumulator(create=True).add(_contribution())
        bind_usage_attribution(7, "/completion.json", "web")

    def _writer(entries):
        seen["request_id"] = entries[0].request_id
        seen["entries"] = len(entries)
        return [1]

    with patch(f"{MODULE}.log_resolved_cost_usage_batch", _writer):
        with pytest.raises(RuntimeError, match="view exploded"):
            app.test_client().get("/boom")

    assert seen["entries"] == 1
    assert seen["request_id"] != CONTEXT_UNSET


def test_main_registers_the_hooks_in_the_required_order():
    """main.py's registration order is the contract the tests above assume.

    Asserted on the parsed module rather than on a live import, which would
    require the full runtime config and a database; the execution-order
    tests above prove what this order actually produces.
    """
    with open(os.path.join(REPO_ROOT, "main.py"), encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    expected = [
        "register_request_context_hooks",
        "register_usage_duration_finalization_hooks",
        "register_embedding_usage_persistence_hooks",
        "register_request_duration_hooks",
        "register_embedding_accounting_hooks",
        "register_operational_persistence",
    ]

    called = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in set(expected)
    ]

    assert called == expected


def test_production_attribution_binding_is_confined_to_the_approved_routes():
    """Only /completion.json, /agentchat, /storeragfile and the admin upload.

    This guard is exact, not existential: it names every production file
    allowed to reference the binder, so an accidental new binder anywhere
    in routes/, services/ or main.py fails here rather than silently
    widening the policy boundary.

    routes/ingestion.py joined the list when the ratified legacy Dino
    exception brought /storeragfile inside the boundary. routes/admin.py
    joined it when DC-ADMIN1 ratified a dedicated technical accounting
    identity for /admin/rag-files/upload; that route is the ONLY approved
    admin binder, which the per-file caller assertion below pins.

    routes/rag.py has since LEFT the list. /completion.json and /agentchat
    adopted the public attribution boundary, so the route no longer names
    the binding primitive at all - the strictest possible outcome for that
    file. The guard is therefore not weakened but re-pointed: rag.py is
    asserted to reference no binder, and to declare attribution through
    ``attribute_usage_to_user`` at exactly the same two approved routes it
    used to bind from. routes/ingestion.py and routes/admin.py migrate in
    later slices and are unchanged here.
    """
    import subprocess

    result = subprocess.run(
        ["grep", "-rnI", "--include=*.py", "bind_usage_attribution",
         "routes", "services", "main.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    files = {line.split(":", 1)[0] for line in result.stdout.splitlines() if line}
    assert files == {"routes/ingestion.py", "routes/admin.py"}

    with open(os.path.join(REPO_ROOT, "routes", "rag.py")) as handle:
        rag_source = handle.read()
    rag_tree = ast.parse(rag_source)

    # No binder, and no private ambient-attribution helper, anywhere in the
    # module - not merely absent from the two routes.
    assert "bind_usage_attribution" not in rag_source
    assert "_bind_embedding_usage_attribution" not in rag_source

    # Attribution is declared through the public boundary, from exactly the
    # two approved routes and nowhere else in the module.
    rag_attributors = {
        node.name
        for node in ast.walk(rag_tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Name)
            and call.func.id == "attribute_usage_to_user"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    }
    assert rag_attributors == {"completion_handler", "agentchat"}

    with open(os.path.join(REPO_ROOT, "routes", "ingestion.py")) as handle:
        ingestion_tree = ast.parse(handle.read())

    ingestion_binders = {
        node.name
        for node in ast.walk(ingestion_tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Name)
            and call.func.id == "bind_usage_attribution"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    }
    assert ingestion_binders == {"_bind_embedding_usage_attribution"}

    with open(os.path.join(REPO_ROOT, "routes", "admin.py")) as handle:
        admin_tree = ast.parse(handle.read())

    admin_binders = {
        node.name
        for node in ast.walk(admin_tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Name)
            and call.func.id == "bind_usage_attribution"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    }
    assert admin_binders == {"_bind_embedding_usage_attribution"}

    admin_callers = {
        node.name
        for node in ast.walk(admin_tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Name)
            and call.func.id in admin_binders
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    }
    assert admin_callers == {"admin_upload_rag_file"}
