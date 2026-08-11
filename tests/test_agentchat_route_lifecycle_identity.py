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

DISTINCTIVE_USERNAME = "distinctive-user@example.com"


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        rag=SimpleNamespace(default_namespace="default-ns"),
        completion_token_cost=1,
    )
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
        return {"id": 7, "username": username}

    monkeypatch.setattr(rag_route, "get_user_by_username", fake_get_user_by_username)
    monkeypatch.setattr(rag_route, "log_token_usage", lambda **kwargs: 999)

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
