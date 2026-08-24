"""SECOND ADOPTER SLICE C3 — the terminal Operational fact for /completion.json.

Fact under test:
    completion_uncontrolled_failure  (ERROR, error_type only)

emitted by the route-level blanket ``except Exception`` boundary in
routes/rag.py, and nowhere else.

The route fixture mirrors the existing one in
tests/test_usage_service_slice_b_call_sites.py, but the timeline tests here
drive the REAL complete_chat() so that the C1/C2 service facts and the C3
route fact appear on one request. No provider, embedding model, vector store
or database is ever contacted: every boundary is monkeypatched.
"""

import logging
from types import SimpleNamespace

import pytest
from flask import Flask

from routes import rag as rag_route
from services import completion_service
from utils.logging_config import register_request_context_hooks
from utils.operational_persistence import snapshot_from_record

_SENSITIVE_MARKERS = ("question=", "text=", "chat=", "namespace=", "error=")


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        completion_token_cost=1,
        rag=SimpleNamespace(default_namespace="default-ns", top_k=5, min_sim=0.5),
        models=SimpleNamespace(
            completion_model_provider="test-provider",
            completion_model="test-model",
            completion_embedding_model_provider="test-emb-provider",
            completion_embedding_model="test-emb-model",
        ),
    )
    register_request_context_hooks(app)
    app.register_blueprint(rag_route.rag_bp)
    return app


def _patch_route_boundaries(monkeypatch):
    """Patch every route-level dependency to a benign, offline default."""
    monkeypatch.setattr(rag_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(rag_route.database_pg, "get_user_tokens", lambda username: 10)
    monkeypatch.setattr(
        rag_route,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username, "client": "coopi"},
    )
    monkeypatch.setattr(rag_route, "choose_emb_model", lambda *a, **k: object())
    monkeypatch.setattr(rag_route, "MauiVectorStore", lambda *a, **k: object())
    monkeypatch.setattr(rag_route, "log_token_usage", lambda **kwargs: 4444)
    monkeypatch.setattr(rag_route, "set_usage_log_id", lambda log_id: None)
    monkeypatch.setattr(rag_route, "edit_tokens", lambda *a, **k: None)


def _post(app):
    return app.test_client().post(
        "/completion.json",
        # `info` is present so that a zero-vector retrieval still proceeds to
        # the provider seam instead of taking the no-context early return.
        json={
            "chat": ["hello"],
            "username": "user@example.com",
            "info": ["background"],
        },
        headers={"X-API-KEY": "test-key"},
    )


def _operational_records(caplog, event):
    return [
        r
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
        and getattr(r, "maui_event", None) == event
    ]


def _operational_timeline(caplog):
    return [
        r.maui_event
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
    ]


def _the_operational_record(caplog, event):
    records = _operational_records(caplog, event)
    assert len(records) == 1, (
        f"expected exactly one {event} record, got {len(records)}"
    )
    return records[0]


def _assert_free_of(record, forbidden):
    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    surfaces = [
        record.getMessage(),
        str(getattr(record, "maui_details", None)),
        str(getattr(record, "maui_message", None)),
        str(getattr(record, "maui_error_type", None)),
        str(snapshot.details_json),
        str(snapshot.message),
        str(snapshot.error_type),
    ]
    for surface in surfaces:
        for needle in forbidden:
            assert needle not in surface, (
                f"forbidden content {needle!r} reached Operational surface: "
                f"{surface!r}"
            )


class _VectorStoreReturning:
    def __init__(self, vectors):
        self._vectors = vectors

    def find_similar_vectors(self, text, top_k, min_similarity):
        return self._vectors


def _patch_service_prompt_boundary(monkeypatch):
    monkeypatch.setattr(
        completion_service, "load_prompt", lambda *a, **k: "system prompt"
    )
    monkeypatch.setattr(
        completion_service, "render_prompt", lambda template, **kwargs: template
    )


# ---------------------------------------------------------------------------
# T1 / T2 — the blanket boundary itself
# ---------------------------------------------------------------------------


class _RouteBoom(Exception):
    pass


def test_generic_route_failure_persists_error_type_only(monkeypatch, caplog):
    """T1 — an exception reaching the blanket boundary emits exactly one marked
    completion_uncontrolled_failure carrying the exception class name only, and
    the generic 500 response is unchanged."""
    sentinel = "SENSITIVE-ROUTE-ERROR-secret-4242"
    app = _make_app()
    _patch_route_boundaries(monkeypatch)

    def exploding_complete_chat(*a, **k):
        raise _RouteBoom(f"failed talking to {sentinel}")

    monkeypatch.setattr(rag_route, "complete_chat", exploding_complete_chat)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    # Unchanged generic failure response.
    assert response.status_code == 500
    assert response.get_json() == {"error": "An unexpected error occurred"}

    record = _the_operational_record(caplog, "completion_uncontrolled_failure")
    assert record.levelno == logging.ERROR
    assert record.name == "routes.rag"
    assert record.maui_error_type == "_RouteBoom"
    assert not hasattr(record, "maui_details")
    assert not hasattr(record, "maui_provider")
    assert not hasattr(record, "maui_model")
    assert not hasattr(record, "maui_duration_ms")
    assert record.getMessage() == (
        "event=completion_uncontrolled_failure error_type=_RouteBoom"
    )

    _assert_free_of(record, (sentinel,) + _SENSITIVE_MARKERS)


def test_legacy_completion_request_failed_is_no_longer_emitted(monkeypatch, caplog):
    """T2 — one real failure boundary, one terminal LogRecord. The legacy
    runtime event=completion_request_failed line (which interpolated str(e))
    was replaced, not doubled."""
    sentinel = "SENSITIVE-ROUTE-ERROR-secret-4242"
    app = _make_app()
    _patch_route_boundaries(monkeypatch)
    monkeypatch.setattr(
        rag_route,
        "complete_chat",
        lambda *a, **k: (_ for _ in ()).throw(_RouteBoom(sentinel)),
    )

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500

    route_records = [r for r in caplog.records if r.name == "routes.rag"]
    assert not [
        r for r in route_records if "completion_request_failed" in r.getMessage()
    ], "the legacy runtime event must no longer be emitted at this boundary"

    # Exactly one terminal record at the boundary — not one legacy plus one new.
    assert len(route_records) == 1
    assert route_records[0].maui_event == "completion_uncontrolled_failure"
    assert sentinel not in route_records[0].getMessage()


# ---------------------------------------------------------------------------
# Early failure before any phase-specific fact exists
# ---------------------------------------------------------------------------


def test_pre_retrieval_failure_yields_terminal_fact_only(monkeypatch, caplog):
    """The accepted diagnostic trade-off: a failure before complete_chat() can
    declare any retrieval fact leaves completion_uncontrolled_failure as the
    ONLY persistent fact for the request."""

    class EmbBoom(Exception):
        pass

    app = _make_app()
    _patch_route_boundaries(monkeypatch)

    def exploding_choose_emb_model(*a, **k):
        raise EmbBoom("embedding provider credentials missing")

    monkeypatch.setattr(rag_route, "choose_emb_model", exploding_choose_emb_model)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500

    record = _the_operational_record(caplog, "completion_uncontrolled_failure")
    assert record.maui_error_type == "EmbBoom"

    assert _operational_timeline(caplog) == ["completion_uncontrolled_failure"]
    for absent in (
        "completion_retrieval_completed",
        "completion_retrieval_failed",
        "completion_provider_completed",
        "completion_provider_failed",
    ):
        assert _operational_records(caplog, absent) == []


# ---------------------------------------------------------------------------
# Full timelines across the service and route seams
# ---------------------------------------------------------------------------


def test_provider_failure_timeline_ends_in_the_terminal_fact(monkeypatch, caplog):
    """retrieval succeeds → provider invoke raises → the propagated exception
    reaches the route blanket boundary."""
    sentinel = "SENSITIVE-PROVIDER-BODY-secret-4242"

    class ProviderBoom(Exception):
        pass

    class FakeLlm:
        def invoke(self, messages):
            raise ProviderBoom(f"upstream said {sentinel}")

    app = _make_app()
    _patch_route_boundaries(monkeypatch)
    _patch_service_prompt_boundary(monkeypatch)
    monkeypatch.setattr(completion_service, "choose_llm", lambda *a, **k: FakeLlm())
    monkeypatch.setattr(
        rag_route, "MauiVectorStore", lambda *a, **k: _VectorStoreReturning([])
    )

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "An unexpected error occurred"}

    assert _operational_timeline(caplog) == [
        "completion_retrieval_completed",
        "completion_provider_failed",
        "completion_uncontrolled_failure",
    ]

    # The service wraps the provider failure in RuntimeError before it reaches
    # the route, and only the class name of what the route caught is persisted.
    terminal = _the_operational_record(caplog, "completion_uncontrolled_failure")
    assert terminal.maui_error_type == "RuntimeError"
    _assert_free_of(terminal, (sentinel,) + _SENSITIVE_MARKERS)

    # No legacy runtime failure event survives at either boundary.
    for legacy in ("event=completion_failed", "event=completion_request_failed"):
        assert not [r for r in caplog.records if legacy in r.getMessage()]


def test_post_provider_failure_timeline_distinguishes_a_paid_success(
    monkeypatch, caplog
):
    """The F6 signature: the provider returned successfully and the request
    failed AFTERWARD, in route-level accounting. Operational must be able to
    tell this apart from a provider failure."""

    class AccountingBoom(Exception):
        pass

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(content="an answer", response_metadata={})

    app = _make_app()
    _patch_route_boundaries(monkeypatch)
    _patch_service_prompt_boundary(monkeypatch)
    monkeypatch.setattr(completion_service, "choose_llm", lambda *a, **k: FakeLlm())
    monkeypatch.setattr(
        rag_route, "MauiVectorStore", lambda *a, **k: _VectorStoreReturning([])
    )

    def exploding_edit_tokens(*a, **k):
        raise AccountingBoom("token ledger unavailable")

    # The smallest existing seam after a successful provider response.
    monkeypatch.setattr(rag_route, "edit_tokens", exploding_edit_tokens)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "An unexpected error occurred"}

    assert _operational_timeline(caplog) == [
        "completion_retrieval_completed",
        "completion_provider_completed",
        "completion_uncontrolled_failure",
    ]

    terminal = _the_operational_record(caplog, "completion_uncontrolled_failure")
    assert terminal.maui_error_type == "AccountingBoom"
    assert _operational_records(caplog, "completion_provider_failed") == []


# ---------------------------------------------------------------------------
# Controlled branches must not gain the terminal fact
# ---------------------------------------------------------------------------


def test_successful_request_emits_no_terminal_fact(monkeypatch, caplog):
    """The happy path terminates in 200 and must carry no Fact E."""

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(content="an answer", response_metadata={})

    app = _make_app()
    _patch_route_boundaries(monkeypatch)
    _patch_service_prompt_boundary(monkeypatch)
    monkeypatch.setattr(completion_service, "choose_llm", lambda *a, **k: FakeLlm())
    monkeypatch.setattr(
        rag_route, "MauiVectorStore", lambda *a, **k: _VectorStoreReturning([])
    )

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 200
    assert _operational_records(caplog, "completion_uncontrolled_failure") == []
    assert _operational_timeline(caplog) == [
        "completion_retrieval_completed",
        "completion_provider_completed",
    ]


@pytest.mark.parametrize(
    "payload, headers, expected_status, expected_error",
    [
        # An empty (falsy) JSON body, not an absent one: a missing body makes
        # request.get_json() raise UnsupportedMediaType, which is genuinely the
        # uncontrolled boundary and is therefore not a controlled branch.
        ({}, {"X-API-KEY": "k"}, 400, "No JSON data provided"),
        (
            {"chat": ["hello"]},
            {"X-API-KEY": "k"},
            400,
            "Missing required keys: username",
        ),
        (
            {"chat": ["hello"], "username": "user@example.com"},
            {},
            400,
            "Missing X-API-KEY header",
        ),
    ],
)
def test_controlled_400_branches_emit_no_terminal_fact(
    monkeypatch, caplog, payload, headers, expected_status, expected_error
):
    """Controlled early returns are not the uncontrolled boundary: they must not
    emit Fact E, and their existing responses are unchanged."""
    app = _make_app()
    _patch_route_boundaries(monkeypatch)
    monkeypatch.setattr(rag_route, "complete_chat", lambda *a, **k: None)

    with caplog.at_level(logging.INFO):
        response = app.test_client().post(
            "/completion.json", json=payload, headers=headers
        )

    assert response.status_code == expected_status
    assert response.get_json()["error"] == expected_error
    assert _operational_records(caplog, "completion_uncontrolled_failure") == []


def test_controlled_quota_refusal_emits_no_terminal_fact(monkeypatch, caplog):
    """The quota/token refusal is a controlled branch delivered as a 500; it
    still must not emit the uncontrolled-failure fact."""
    app = _make_app()
    _patch_route_boundaries(monkeypatch)
    monkeypatch.setattr(rag_route.database_pg, "get_user_tokens", lambda username: 0)
    monkeypatch.setattr(rag_route, "complete_chat", lambda *a, **k: None)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Not enough tokens", "user_tokens": 0}
    assert _operational_records(caplog, "completion_uncontrolled_failure") == []


def test_controlled_token_lookup_failure_emits_no_terminal_fact(monkeypatch, caplog):
    """Token lookup returning None is an early controlled return, not the
    blanket boundary."""
    app = _make_app()
    _patch_route_boundaries(monkeypatch)
    monkeypatch.setattr(rag_route.database_pg, "get_user_tokens", lambda username: None)
    monkeypatch.setattr(rag_route, "complete_chat", lambda *a, **k: None)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Could not retrieve user tokens"}
    assert _operational_records(caplog, "completion_uncontrolled_failure") == []
