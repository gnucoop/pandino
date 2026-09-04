"""Route wiring of embedding Usage attribution for the two approved flows.

Under test is only what the routes own: a best-effort, post-authentication
attribution bind that precedes any embedding work, never blocks the
endpoint, and never displaces the legacy LLM Usage lookup/write. The
lifecycle modules it enables are not re-tested here - they run for real in
the end-to-end tests below, with only the database boundaries patched.

/completion.json and /agentchat are asserted independently, because their
Usage lookup failure semantics differ: the completion route lets a Usage
lookup failure reach its blanket handler, while /agentchat contains it and
still answers.
"""

import ast
import logging
import os
from types import SimpleNamespace

from flask import Flask

from routes import rag as rag_route
from utils import usage_attribution, usage_recording
from utils.embedding_accounting import (
    COST_PROVIDER_AUTHORITATIVE,
    EmbeddingAccountingContribution,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_UNIT_INPUT_TOKENS,
)
from utils.embedding_accounting_lifecycle import register_embedding_accounting_hooks
from utils.embedding_accounting_sink import get_embedding_accounting_sink
from utils.embedding_operation_context import OPERATION_QUERY
from utils.embedding_usage_persistence import (
    register_embedding_usage_persistence_hooks,
)
from utils.logging_config import register_request_context_hooks
from utils.request_duration import register_request_duration_hooks
from utils.usage_attribution_state import get_usage_attribution
from utils.usage_duration_finalization import (
    register_usage_duration_finalization_hooks,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEGACY_LOG_ID = 4444
EMBEDDING_LOG_ID = 8888


def _contribution(input_quantity=120):
    return EmbeddingAccountingContribution(
        provider="deepinfra",
        model="intfloat/multilingual-e5-large-instruct",
        input_quantity=input_quantity,
        quantity_unit=QUANTITY_UNIT_INPUT_TOKENS,
        quantity_origin=ORIGIN_PROVIDER_REPORTED,
        cost_state=COST_PROVIDER_AUTHORITATIVE,
        operation_kind=OPERATION_QUERY,
        provider_cost=0.25,
    )


# --------------------------------------------------------------------------
# app fixtures
# --------------------------------------------------------------------------


def _config():
    return SimpleNamespace(
        completion_token_cost=1,
        rag=SimpleNamespace(default_namespace="default-ns", top_k=5, min_sim=0.5),
        models=SimpleNamespace(
            completion_model_provider="test-provider",
            completion_model="test-model",
            completion_embedding_model_provider="test-emb-provider",
            completion_embedding_model="test-emb-model",
        ),
    )


def _make_app(full_lifecycle=False):
    """A throwaway app carrying the rag blueprint.

    ``full_lifecycle`` additionally registers the production hook chain, in
    main.py's order, so the end-to-end tests exercise the real
    accumulator -> attribution -> persistence -> duration path.
    """
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = _config()
    register_request_context_hooks(app)
    if full_lifecycle:
        register_usage_duration_finalization_hooks(app)
        register_embedding_usage_persistence_hooks(app)
        register_request_duration_hooks(app)
        register_embedding_accounting_hooks(app)
    app.register_blueprint(rag_route.rag_bp)
    return app


def _patch_common(monkeypatch, captured, user=None, lookup=None):
    """Patch every route boundary to an offline default.

    :param user: the dict the user lookup returns.
    :param lookup: a replacement lookup callable, for failure scenarios.
    """
    if user is None:
        user = {"id": 42, "username": "user@example.com", "client": "coopi"}

    monkeypatch.setattr(rag_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(rag_route.database_pg, "get_user_tokens", lambda username: 10)

    def default_lookup(username):
        captured.setdefault("lookups", []).append(username)
        return user

    # The two lookups diverged when /completion.json and /agentchat adopted
    # the public attribution boundary: ambient attribution now resolves
    # identity inside utils.usage_attribution, while Explicit Usage Recording
    # still resolves its own in the route. Both targets get the SAME callable,
    # so the ordered ``captured["lookups"]`` trace still spans the whole
    # request - attribution first, recording second - and the failure
    # scenarios below can still single out the attribution attempt.
    effective_lookup = lookup or default_lookup
    monkeypatch.setattr(rag_route, "get_user_by_username", effective_lookup)
    monkeypatch.setattr(usage_attribution, "get_user_by_username", effective_lookup)
    monkeypatch.setattr(rag_route, "choose_emb_model", lambda *a, **k: object())
    monkeypatch.setattr(rag_route, "MauiVectorStore", lambda *a, **k: object())

    def fake_log_token_usage(**kwargs):
        captured.setdefault("log_token_usage_calls", []).append(kwargs)
        return LEGACY_LOG_ID

    monkeypatch.setattr(rag_route, "edit_tokens", lambda *a, **k: None)

    # Both routes record through utils.usage_recording, which binds the
    # writer and its identity reads by direct name import - patching
    # rag_route does not reach them, which is exactly the point.
    monkeypatch.setattr(usage_recording, "log_token_usage", fake_log_token_usage)
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_id",
        lambda user_id: {"id": user_id, "username": user["username"]},
    )
    monkeypatch.setattr(usage_recording, "get_user_by_username", lambda username: user)


def _patch_completion(monkeypatch, captured, contributions=(), exc=None, **kwargs):
    _patch_common(monkeypatch, captured, **kwargs)

    def fake_complete_chat(*a, **kw):
        captured["complete_chat_called"] = True
        captured["attribution_at_provider"] = get_usage_attribution()
        sink = get_embedding_accounting_sink()
        for contribution in contributions:
            sink(contribution)
        if exc is not None:
            raise exc
        return {
            "answer": "an answer",
            "vectors": [],
            "token_usage": {"input_tokens": 11, "output_tokens": 22},
        }

    monkeypatch.setattr(rag_route, "complete_chat", fake_complete_chat)


def _patch_agentchat(monkeypatch, captured, contributions=(), **kwargs):
    _patch_common(monkeypatch, captured, **kwargs)

    def fake_run_agentchat(chat, namespace, language, username, config):
        captured["run_agentchat_called"] = True
        captured["attribution_at_provider"] = get_usage_attribution()
        sink = get_embedding_accounting_sink()
        for contribution in contributions:
            sink(contribution)
        return {
            "payload": {
                "answer": "an answer",
                "metrics": {
                    "duration_ms": 42,
                    "token_usage": {"input": 11, "output": 22},
                },
                "tool_calls": [],
                "vectors": [],
                "follow_ups": [],
            },
            "model": "test-model",
            "provider": "test-provider",
        }

    monkeypatch.setattr(rag_route, "run_agentchat", fake_run_agentchat)


def _post_completion(app):
    return app.test_client().post(
        "/completion.json",
        json={
            "chat": ["hello"],
            "username": "user@example.com",
            "info": ["background"],
        },
        headers={"X-API-KEY": "test-key"},
    )


def _post_agentchat(app):
    return app.test_client().post(
        "/agentchat",
        json={"chat": ["hello"], "username": "user@example.com"},
        headers={"X-API-KEY": "test-key"},
    )


def _diagnostics(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if "event=embedding_usage_attribution_unavailable" in r.getMessage()
    ]


# --------------------------------------------------------------------------
# attribution success
# --------------------------------------------------------------------------


def test_completion_binds_attribution_before_provider_work(monkeypatch):
    captured = {}
    _patch_completion(monkeypatch, captured)

    response = _post_completion(_make_app())

    assert response.status_code == 200
    assert captured["complete_chat_called"] is True

    attribution = captured["attribution_at_provider"]
    assert attribution is not None
    assert attribution.user_id == 42
    assert attribution.service == "/completion.json"
    assert attribution.source == "coopi"


def test_agentchat_binds_attribution_before_provider_work(monkeypatch):
    captured = {}
    _patch_agentchat(
        monkeypatch,
        captured,
        user={"id": 7, "username": "user@example.com", "client": "dino"},
    )

    response = _post_agentchat(_make_app())

    assert response.status_code == 200
    assert captured["run_agentchat_called"] is True

    attribution = captured["attribution_at_provider"]
    assert attribution is not None
    assert attribution.user_id == 7
    assert attribution.service == "/agentchat"
    assert attribution.source == "dino"


def test_attribution_source_may_honestly_be_none(monkeypatch):
    captured = {}
    _patch_completion(
        monkeypatch,
        captured,
        user={"id": 42, "username": "user@example.com", "client": None},
    )

    assert _post_completion(_make_app()).status_code == 200
    assert captured["attribution_at_provider"].source is None


# --------------------------------------------------------------------------
# attribution failure is non-blocking
# --------------------------------------------------------------------------


def _one_shot_failing_lookup(captured, user):
    """A lookup that fails once - the attribution attempt - then succeeds."""

    def lookup(username):
        captured.setdefault("lookups", []).append(username)
        if len(captured["lookups"]) == 1:
            raise RuntimeError("db unavailable")
        return user

    return lookup


def test_completion_attribution_lookup_failure_does_not_block_provider(
    monkeypatch, caplog
):
    captured = {}
    user = {"id": 42, "username": "user@example.com", "client": "coopi"}
    _patch_completion(
        monkeypatch, captured, lookup=_one_shot_failing_lookup(captured, user)
    )

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post_completion(_make_app())

    # Legacy HTTP contract intact, provider flow really reached.
    assert response.status_code == 200
    assert captured["complete_chat_called"] is True
    assert response.get_json()["log_id"] == LEGACY_LOG_ID

    # Nothing bound, exactly one safe diagnostic.
    assert captured["attribution_at_provider"] is None
    messages = _diagnostics(caplog)
    assert len(messages) == 1
    assert "reason=lookup_failed" in messages[0]
    assert "service=/completion.json" in messages[0]
    assert "error_type=RuntimeError" in messages[0]


def test_agentchat_attribution_lookup_failure_does_not_block_provider(
    monkeypatch, caplog
):
    captured = {}
    user = {"id": 7, "username": "user@example.com", "client": "dino"}
    _patch_agentchat(
        monkeypatch, captured, lookup=_one_shot_failing_lookup(captured, user)
    )

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post_agentchat(_make_app())

    assert response.status_code == 200
    assert captured["run_agentchat_called"] is True
    assert response.get_json()["log_id"] == LEGACY_LOG_ID
    assert captured["attribution_at_provider"] is None

    messages = _diagnostics(caplog)
    assert len(messages) == 1
    assert "reason=lookup_failed" in messages[0]
    assert "service=/agentchat" in messages[0]


def test_attribution_diagnostic_carries_no_identity_or_payload(monkeypatch, caplog):
    captured = {}
    _patch_completion(
        monkeypatch,
        captured,
        lookup=_one_shot_failing_lookup(
            captured, {"id": 42, "username": "u", "client": "coopi"}
        ),
    )

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        _post_completion(_make_app())

    message = _diagnostics(caplog)[0]
    for forbidden in ("user@example.com", "test-key", "hello", "background"):
        assert forbidden not in message


def test_user_not_found_binds_nothing_and_keeps_legacy_behaviour(monkeypatch, caplog):
    captured = {}

    def lookup(username):
        captured.setdefault("lookups", []).append(username)
        return None

    _patch_completion(monkeypatch, captured, lookup=lookup)

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post_completion(_make_app())

    # Legacy behaviour for an unknown user: 200, no LLM Usage row, no log_id.
    assert response.status_code == 200
    assert "log_id" not in response.get_json()
    assert captured["attribution_at_provider"] is None
    assert captured.get("log_token_usage_calls") is None
    assert "reason=not_found" in _diagnostics(caplog)[0]


def test_invalid_user_id_binds_nothing(monkeypatch, caplog):
    captured = {}
    _patch_completion(
        monkeypatch,
        captured,
        user={"id": "not-an-int", "username": "u", "client": "coopi"},
    )

    with caplog.at_level(logging.WARNING, logger="utils.usage_attribution"):
        response = _post_completion(_make_app())

    assert response.status_code == 200
    assert captured["complete_chat_called"] is True
    assert captured["attribution_at_provider"] is None
    assert "reason=invalid_user_id" in _diagnostics(caplog)[0]


# --------------------------------------------------------------------------
# the early attempt never replaces the legacy lookup/write
# --------------------------------------------------------------------------


def test_completion_early_failure_does_not_suppress_legacy_lookup(monkeypatch):
    """A persistently failing lookup is still attempted by the legacy path.

    /completion.json has always let a raising user lookup reach its blanket
    handler, so the 500 here is the pre-existing behaviour, not a new one -
    what matters is that the legacy path performed its own lookup instead
    of inheriting the best-effort attempt's failure.
    """
    captured = {}

    def always_failing(username):
        captured.setdefault("lookups", []).append(username)
        raise RuntimeError("db unavailable")

    _patch_completion(monkeypatch, captured, lookup=always_failing)

    response = _post_completion(_make_app())

    assert captured["complete_chat_called"] is True
    assert len(captured["lookups"]) == 2
    assert response.status_code == 500
    assert response.get_json() == {"error": "An unexpected error occurred"}


def test_agentchat_early_failure_does_not_suppress_legacy_lookup(monkeypatch):
    """/agentchat swallows its Usage failure; that stays true, twice over."""
    captured = {}

    def always_failing(username):
        captured.setdefault("lookups", []).append(username)
        raise RuntimeError("db unavailable")

    _patch_agentchat(monkeypatch, captured, lookup=always_failing)

    response = _post_agentchat(_make_app())

    assert captured["run_agentchat_called"] is True
    assert len(captured["lookups"]) == 2
    assert response.status_code == 200
    assert "log_id" not in response.get_json()


def test_successful_attribution_leaves_legacy_write_unchanged(monkeypatch):
    captured = {}
    _patch_completion(monkeypatch, captured)

    response = _post_completion(_make_app())

    calls = captured["log_token_usage_calls"]
    assert len(calls) == 1
    assert calls[0]["service"] == "/completion.json"
    assert calls[0]["user_id"] == 42
    assert calls[0]["source"] == "coopi"
    assert calls[0]["request_id"] == response.headers["X-Request-ID"]
    assert response.get_json()["log_id"] == LEGACY_LOG_ID


# --------------------------------------------------------------------------
# end-to-end Usage contract
# --------------------------------------------------------------------------


def _patch_persistence_boundaries(monkeypatch, captured):
    """Patch only the two DB boundaries of the lifecycle chain."""

    def fake_batch(entries):
        entries = list(entries)
        captured.setdefault("batch_calls", []).append(entries)
        return [EMBEDDING_LOG_ID + i for i in range(len(entries))]

    monkeypatch.setattr(
        "utils.embedding_usage_persistence.log_resolved_cost_usage_batch", fake_batch
    )

    def fake_update(log_id, duration_ms):
        captured.setdefault("duration_updates", []).append((log_id, duration_ms))
        return True

    monkeypatch.setattr(
        "utils.usage_duration_finalization.update_usage_duration", fake_update
    )


def test_completion_end_to_end_writes_a_separate_embedding_usage_row(monkeypatch):
    captured = {}
    _patch_completion(monkeypatch, captured, contributions=[_contribution()])
    _patch_persistence_boundaries(monkeypatch, captured)

    response = _post_completion(_make_app(full_lifecycle=True))
    request_id = response.headers["X-Request-ID"]

    assert response.status_code == 200
    # The response log_id contract is untouched: still the legacy LLM row.
    assert response.get_json()["log_id"] == LEGACY_LOG_ID

    # Exactly one embedding row, not merged with the LLM row.
    batches = captured["batch_calls"]
    assert len(batches) == 1
    assert len(batches[0]) == 1
    entry = batches[0][0]
    assert entry.user_id == 42
    assert entry.service == "/completion.json"
    assert entry.source == "coopi"
    assert entry.provider == "deepinfra"
    assert entry.token_input == 120
    assert entry.embedding_operation_kind == OPERATION_QUERY

    # Same request_id as the legacy row, and no separate embedding one.
    assert entry.request_id == request_id
    assert captured["log_token_usage_calls"][0]["request_id"] == request_id

    # Both ids reached the duration finalizer with one shared duration.
    updates = captured["duration_updates"]
    assert {log_id for log_id, _ in updates} == {LEGACY_LOG_ID, EMBEDDING_LOG_ID}
    assert len({duration for _, duration in updates}) == 1


def test_agentchat_end_to_end_aggregates_multiple_query_contributions(monkeypatch):
    captured = {}
    _patch_agentchat(
        monkeypatch,
        captured,
        contributions=[_contribution(100), _contribution(40)],
        user={"id": 7, "username": "user@example.com", "client": "dino"},
    )
    _patch_persistence_boundaries(monkeypatch, captured)

    response = _post_agentchat(_make_app(full_lifecycle=True))

    assert response.status_code == 200
    assert response.get_json()["log_id"] == LEGACY_LOG_ID

    batches = captured["batch_calls"]
    assert len(batches) == 1
    assert len(batches[0]) == 1
    entry = batches[0][0]
    assert entry.user_id == 7
    assert entry.service == "/agentchat"
    assert entry.source == "dino"
    assert entry.token_input == 140
    assert entry.request_id == response.headers["X-Request-ID"]


def test_no_embedding_contribution_writes_no_row_despite_attribution(monkeypatch):
    captured = {}
    _patch_completion(monkeypatch, captured, contributions=())
    _patch_persistence_boundaries(monkeypatch, captured)

    response = _post_completion(_make_app(full_lifecycle=True))

    assert response.status_code == 200
    assert captured["attribution_at_provider"] is not None
    assert "batch_calls" not in captured
    assert [log_id for log_id, _ in captured["duration_updates"]] == [LEGACY_LOG_ID]


def test_provider_failure_after_consumption_still_persists_the_contribution(
    monkeypatch,
):
    captured = {}
    _patch_completion(
        monkeypatch,
        captured,
        contributions=[_contribution()],
        exc=RuntimeError("provider exploded"),
    )
    _patch_persistence_boundaries(monkeypatch, captured)

    response = _post_completion(_make_app(full_lifecycle=True))

    assert response.status_code == 500
    assert "log_id" not in response.get_json()

    batches = captured["batch_calls"]
    assert len(batches) == 1
    assert batches[0][0].user_id == 42
    assert batches[0][0].service == "/completion.json"


def test_unattributed_contribution_writes_no_row(monkeypatch, caplog):
    """The P6 skip path stays the only fallback; routes duplicate none of it."""
    captured = {}

    def lookup(username):
        captured.setdefault("lookups", []).append(username)
        return None

    _patch_completion(
        monkeypatch, captured, contributions=[_contribution()], lookup=lookup
    )
    _patch_persistence_boundaries(monkeypatch, captured)

    with caplog.at_level(logging.WARNING):
        response = _post_completion(_make_app(full_lifecycle=True))

    assert response.status_code == 200
    assert "batch_calls" not in captured
    assert any(
        "event=embedding_usage_persistence_skipped" in r.getMessage()
        and "reason=no_attribution" in r.getMessage()
        for r in caplog.records
    )


# --------------------------------------------------------------------------
# unapproved flows stay inactive
# --------------------------------------------------------------------------


def test_admin_upload_is_the_only_attributing_admin_route():
    """/admin/rag-files/upload is the ONE approved admin attributor.

    Asserted statically, on the module owning the admin routes, so the
    guard cannot be satisfied by an incidental early return in a route
    test.

    DC-ADMIN1 ratified a dedicated technical accounting identity for
    /admin/rag-files/upload, which puts that one route inside the policy
    boundary. The admin surface is large, so the invariant worth pinning is
    narrow: exactly one admin route declares attribution and every other one
    stays out. Its runtime behaviour lives in
    tests/test_admin_route_usage_attribution.py.

    The module is scanned for the declared intent rather than the binding
    primitive, since mechanics belong to utils.usage_attribution. That
    ownership is what makes the identity-safety assertions absolute: the
    configuration attribute naming the provisioned identity lives at the
    boundary, so routes/admin.py must not mention it at all.
    """
    with open(os.path.join(REPO_ROOT, "routes/admin.py")) as handle:
        source = handle.read()
    tree = ast.parse(source)

    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]

    def _calls(node, names):
        return {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in names
        }

    INTENTS = {
        "attribute_usage_to_user",
        "attribute_usage_to_policy",
        "declare_usage_unattributed",
    }

    # No private module-local binder survives, and nothing in the module
    # touches the binding primitive.
    assert "bind_usage_attribution" not in source
    assert "_bind_embedding_usage_attribution" not in source
    assert not {
        node.name for node in functions if _calls(node, {"bind_usage_attribution"})
    }

    # Exactly one function declares attribution, and it is the upload route.
    attributors = {node.name for node in functions if _calls(node, INTENTS)}
    assert attributors == {"admin_upload_rag_file"}

    # Stated the other way round, over every Flask-routed admin view: no
    # unrelated admin route reaches attribution.
    routed = [
        node
        for node in functions
        if any(
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == "route"
            for dec in node.decorator_list
        )
    ]
    assert "admin_upload_rag_file" in {node.name for node in routed}
    attributing_routes = {node.name for node in routed if _calls(node, INTENTS)}
    assert attributing_routes == {"admin_upload_rag_file"}

    # The intent is the ratified technical policy, never the authenticated
    # operator: no admin credential and no session value can become a Usage
    # identity, and the route cannot even name the provisioned username.
    assert "USAGE_POLICY_ADMIN_RAG_INGESTION" in source
    assert "attribute_usage_to_user" not in source
    assert "admin_rag_usage_username" not in source
    assert 'get_user_by_username(session[' not in source
    assert "get_user_by_username(config.admin.username)" not in source

    # And the route remains a non-writer.
    assert "log_resolved_cost_usage_batch" not in source
    assert "register_usage_log_id" not in source
    assert "log_token_usage" not in source


def test_routes_never_call_the_writer_or_register_log_ids_directly():
    """Persistence and row-id registration remain lifecycle-owned."""
    for module in os.listdir(os.path.join(REPO_ROOT, "routes")):
        if not module.endswith(".py"):
            continue
        with open(os.path.join(REPO_ROOT, "routes", module)) as handle:
            source = handle.read()
        assert "log_resolved_cost_usage_batch" not in source
        assert "register_usage_log_id" not in source
