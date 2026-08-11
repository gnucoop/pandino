"""Regression tests for POST /startdatachat's datachat_engine_bootstrap_started event.

The user_identity targeted inspection (docs/logging/requirements_reconciliation_
tranche_b_beta*.md) classified this event as lifecycle-role, contextual-
duplication content: the same request already carries the user's email through
assert_valid_api_key/get_user_tokens/edit_tokens, so the email field on this
specific log line is redundant and was removed. Everything else about how the
route reads/uses the email is unchanged.
"""

import io
import logging
from types import SimpleNamespace

from flask import Flask

from routes import datachat as datachat_route

DISTINCTIVE_EMAIL = "distinctive-user@example.com"


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        datachat_token_cost=1,
        datachat=SimpleNamespace(engine="default"),
        models=SimpleNamespace(
            datachat_model="test-model", datachat_provider="test-provider"
        ),
    )
    app.register_blueprint(datachat_route.datachat_bp)
    return app


class _StubBootstrapResult:
    suggested_questions_html = None


class _StubEngine:
    def bootstrap(self, lang):
        return _StubBootstrapResult()


def _patch_success_dependencies(monkeypatch, captured):
    monkeypatch.setattr(datachat_route, "assert_valid_api_key", lambda *a, **k: None)

    def fake_get_user_tokens(user_email):
        captured["get_user_tokens_arg"] = user_email
        return 10

    monkeypatch.setattr(datachat_route, "get_user_tokens", fake_get_user_tokens)
    monkeypatch.setattr(
        datachat_route, "load_csv_to_dataframe", lambda file: object()
    )
    monkeypatch.setattr(datachat_route, "choose_llm", lambda *a, **k: object())
    monkeypatch.setattr(
        datachat_route, "createAgent", lambda *a, **k: _StubEngine()
    )

    def fake_edit_tokens(user_email, amount):
        captured["edit_tokens_arg"] = user_email

    monkeypatch.setattr(datachat_route, "edit_tokens", fake_edit_tokens)


def _post_start_chat(client):
    return client.post(
        "/startdatachat",
        data={
            "lang": "ENG",
            "file": (io.BytesIO(b"col_a,col_b\n1,2\n"), "data.csv"),
        },
        content_type="multipart/form-data",
        headers={
            "X-API-KEY": "test-key",
            "X-USER-NAME": "Test User",
            "X-USER-EMAIL": DISTINCTIVE_EMAIL,
        },
    )


def test_bootstrap_event_still_emitted_with_language_but_no_identity(
    monkeypatch, caplog
):
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger="routes.datachat"):
        response = _post_start_chat(client)

    assert response.status_code == 200

    bootstrap_records = [
        r
        for r in caplog.records
        if "event=datachat_engine_bootstrap_started" in r.message
    ]
    assert len(bootstrap_records) == 1
    message = bootstrap_records[0].message

    # Non-identity field is preserved.
    assert "language=ENG" in message

    # Identity is absent from the event message entirely.
    assert DISTINCTIVE_EMAIL not in message
    assert "user=" not in message


def test_email_still_reaches_token_check_and_debit(monkeypatch, caplog):
    """The email is removed only from the log line, not from application logic."""
    app = _make_app()
    captured = {}
    _patch_success_dependencies(monkeypatch, captured)

    client = app.test_client()
    with caplog.at_level(logging.INFO, logger="routes.datachat"):
        response = _post_start_chat(client)

    assert response.status_code == 200
    assert captured["get_user_tokens_arg"] == DISTINCTIVE_EMAIL
    assert captured["edit_tokens_arg"] == DISTINCTIVE_EMAIL
