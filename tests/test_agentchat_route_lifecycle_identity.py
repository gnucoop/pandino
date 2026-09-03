"""Regression tests for POST /agentchat's lifecycle logging events.

The user_identity targeted inspection (docs/logging/requirements_reconciliation_
tranche_b_beta*.md) classified agentchat_request_started/agentchat_request_completed
as lifecycle-role, contextual-duplication content: within the same request the
identical username is already emitted by assert_valid_api_key/get_user_tokens/
get_user_by_username/edit_tokens, so the user field on these two lifecycle log
lines is redundant and was removed. The username itself still flows normally to
every business operation that requires it.
"""

import logging
from types import SimpleNamespace

from flask import Flask

from routes import rag as rag_route
from utils.logging_config import register_request_context_hooks
from utils.usage_request_state import set_usage_log_id

DISTINCTIVE_USERNAME = "distinctive-user@example.com"
RECORDED_LOG_ID = 999


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        rag=SimpleNamespace(default_namespace="default-ns"),
        completion_token_cost=1,
    )
    register_request_context_hooks(app)
    app.register_blueprint(rag_route.rag_bp)
    return app


def _patch_success_dependencies(monkeypatch, captured):
    monkeypatch.setattr(rag_route, "assert_valid_api_key", lambda *a, **k: None)

    def fake_get_user_tokens(username):
        captured["get_user_tokens_arg"] = username
        return 10

    monkeypatch.setattr(rag_route.database_pg, "get_user_tokens", fake_get_user_tokens)

    def fake_run_agentchat(chat, namespace, language, username, config):
        captured["run_agentchat_username"] = username
        return {
            "payload": {
                "answer": "an answer",
                "metrics": {"duration_ms": 42, "token_usage": {}},
                "tool_calls": ["t1"],
                "vectors": ["v1", "v2"],
                "follow_ups": [],
            },
            "model": "test-model",
            "provider": "test-provider",
        }

    monkeypatch.setattr(rag_route, "run_agentchat", fake_run_agentchat)

    def fake_get_user_by_username(username):
        captured["get_user_by_username_arg"] = username
        return {"id": 7, "username": username, "client": "dino"}

    monkeypatch.setattr(rag_route, "get_user_by_username", fake_get_user_by_username)

    def fake_record_token_consumption(**kwargs):
        captured.setdefault("record_calls", []).append(kwargs)
        # The boundary owns row-id registration; a recording adopter only
        # ever observes it through the read-back accessor.
        set_usage_log_id(RECORDED_LOG_ID)
        return True

    monkeypatch.setattr(
        rag_route, "record_token_consumption", fake_record_token_consumption
    )

    def fake_edit_tokens(username, amount):
        captured["edit_tokens_arg"] = username

    monkeypatch.setattr(rag_route, "edit_tokens", fake_edit_tokens)


def _post_agentchat(client):
    return client.post(
        "/agentchat",
        json={"chat": ["hello"], "username": DISTINCTIVE_USERNAME},
        headers={"X-API-KEY": "test-key"},
    )


def test_lifecycle_events_emitted_without_identity_but_with_technical_fields(
    monkeypatch, caplog
):
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger="routes.rag"):
        response = _post_agentchat(client)

    assert response.status_code == 200

    started_records = [
        r for r in caplog.records if "event=agentchat_request_started" in r.message
    ]
    completed_records = [
        r for r in caplog.records if "event=agentchat_request_completed" in r.message
    ]
    assert len(started_records) == 1
    assert len(completed_records) == 1

    started_message = started_records[0].message
    completed_message = completed_records[0].message

    # Non-identity technical fields are preserved.
    assert "namespace=default-ns" in started_message
    assert "language=ITA" in started_message
    assert "duration_ms=42" in completed_message
    assert "tools=1" in completed_message
    assert "vectors=2" in completed_message
    assert "follow_ups=0" in completed_message

    # Identity is absent from both lifecycle event messages.
    for message in (started_message, completed_message):
        assert DISTINCTIVE_USERNAME not in message
        assert "user=" not in message


def test_record_token_consumption_receives_the_agentchat_consumption_facts(
    monkeypatch,
):
    """/agentchat states what it consumed and nothing else: request
    correlation and client source are derived behind the boundary."""
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    response = _post_agentchat(app.test_client())

    assert response.status_code == 200
    record_calls = captured["record_calls"]
    assert len(record_calls) == 1
    assert record_calls[0] == {
        "user_id": 7,
        "provider": "test-provider",
        "model": "test-model",
        "service": "/agentchat",
        "token_input": 0,
        "token_output": 0,
    }


def test_agentchat_exposes_the_recorded_row_id_through_read_back(monkeypatch):
    """Successful recording keeps the existing response contract, read back
    from request state rather than returned by the recording operation."""
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    response = _post_agentchat(app.test_client())

    assert response.status_code == 200
    assert response.get_json()["log_id"] == RECORDED_LOG_ID


def test_agentchat_usage_recording_failure_omits_log_id(monkeypatch):
    """A fail-open recording result leaves the agent answer, the debit and
    the HTTP status untouched, and exposes no row id."""
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    monkeypatch.setattr(
        rag_route, "record_token_consumption", lambda **kwargs: False
    )

    response = _post_agentchat(app.test_client())

    assert response.status_code == 200
    body = response.get_json()
    assert "log_id" not in body
    assert body["answer"] == "an answer"
    assert captured["edit_tokens_arg"] == DISTINCTIVE_USERNAME


def test_agentchat_exposes_no_stale_log_id_when_recording_is_skipped(monkeypatch):
    """A missing or invalid accounting user skips recording entirely, and an
    unrelated row id already registered on the request must not leak into the
    response."""
    app = _make_app()

    for user_row in (None, {"id": "not-an-int", "username": DISTINCTIVE_USERNAME}):
        captured = {}
        _patch_success_dependencies(monkeypatch, captured)
        monkeypatch.setattr(
            rag_route, "get_user_by_username", lambda username: user_row
        )
        monkeypatch.setattr(
            rag_route,
            "get_usage_log_id",
            lambda: (_ for _ in ()).throw(
                AssertionError("read-back must not run when recording is skipped")
            ),
        )

        response = _post_agentchat(app.test_client())

        assert response.status_code == 200
        assert "log_id" not in response.get_json()
        assert "record_calls" not in captured


def test_agentchat_usage_user_lookup_failure_is_diagnosed_and_contained(
    monkeypatch, caplog
):
    """A raising accounting lookup skips recording entirely and leaves the
    agent response intact, diagnosed by one safe runtime WARNING naming only
    the exception class."""
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    class _LookupBoom(RuntimeError):
        pass

    def raising_lookup(username):
        raise _LookupBoom("connection to 10.0.0.1 refused for bob@example.com")

    monkeypatch.setattr(rag_route, "get_user_by_username", raising_lookup)
    monkeypatch.setattr(
        rag_route,
        "get_usage_log_id",
        lambda: (_ for _ in ()).throw(
            AssertionError("read-back must not run when recording is skipped")
        ),
    )

    with caplog.at_level(logging.WARNING, logger="routes.rag"):
        response = _post_agentchat(app.test_client())

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "an answer"
    assert "log_id" not in body
    assert "record_calls" not in captured

    records = [
        r
        for r in caplog.records
        if "event=agentchat_usage_user_lookup_failed" in r.getMessage()
    ]
    assert len(records) == 1
    message = records[0].getMessage()
    assert message == (
        "event=agentchat_usage_user_lookup_failed error_type=_LookupBoom"
    )
    assert "connection to 10.0.0.1 refused" not in message
    assert DISTINCTIVE_USERNAME not in message
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is None
    assert records[0].exc_text is None


def test_agentchat_normalizes_absent_runtime_token_counts_to_zero(monkeypatch):
    """serialize_runresult() reports absent runtime counts as None; the
    adopter seam normalizes them so the boundary receives real integers."""
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    def none_token_run_agentchat(chat, namespace, language, username, config):
        return {
            "payload": {
                "answer": "an answer",
                "metrics": {
                    "duration_ms": 42,
                    "token_usage": {"input": None, "output": None, "total": None},
                },
                "tool_calls": [],
                "vectors": [],
                "follow_ups": [],
            },
            "model": "test-model",
            "provider": "test-provider",
        }

    monkeypatch.setattr(rag_route, "run_agentchat", none_token_run_agentchat)

    response = _post_agentchat(app.test_client())

    assert response.status_code == 200
    call = captured["record_calls"][0]
    assert call["token_input"] == 0
    assert call["token_output"] == 0
    assert isinstance(call["token_input"], int)
    assert isinstance(call["token_output"], int)


def test_agentchat_passes_real_runtime_token_counts_through_unchanged(monkeypatch):
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    def counted_run_agentchat(chat, namespace, language, username, config):
        return {
            "payload": {
                "answer": "an answer",
                "metrics": {
                    "duration_ms": 42,
                    "token_usage": {"input": 120, "output": 34, "total": 154},
                },
                "tool_calls": [],
                "vectors": [],
                "follow_ups": [],
            },
            "model": "test-model",
            "provider": "test-provider",
        }

    monkeypatch.setattr(rag_route, "run_agentchat", counted_run_agentchat)

    response = _post_agentchat(app.test_client())

    assert response.status_code == 200
    call = captured["record_calls"][0]
    assert call["token_input"] == 120
    assert call["token_output"] == 34


def test_username_still_reaches_business_operations(monkeypatch, caplog):
    """The username is removed only from the two lifecycle log lines, not from
    authentication, token accounting, or agent execution."""
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger="routes.rag"):
        response = _post_agentchat(client)

    assert response.status_code == 200
    assert captured["get_user_tokens_arg"] == DISTINCTIVE_USERNAME
    assert captured["run_agentchat_username"] == DISTINCTIVE_USERNAME
    assert captured["get_user_by_username_arg"] == DISTINCTIVE_USERNAME
    assert captured["edit_tokens_arg"] == DISTINCTIVE_USERNAME
