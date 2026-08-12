"""Focused tests proving the three Usage writer call sites that previously had
no test seam (audio_form_compile, prompt_handler, completion_handler) pass the
canonical Usage Service Slice B `service` literal to log_token_usage().

datachat, agentchat and compare_docs already had test seams, extended in
tests/test_datachat_route_request_id.py, tests/test_agentchat_route_lifecycle_identity.py
and tests/test_documents_route.py respectively.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

from flask import Flask

from routes import multimodal as multimodal_route
from routes import reporting as reporting_route
from routes import rag as rag_route
from utils.logging_config import register_request_context_hooks

_ROUTES_DIR = Path(__file__).resolve().parent.parent / "routes"


def _find_log_token_usage_calls():
    """Return every ast.Call node invoking log_token_usage() across routes/*.py."""
    calls = []
    for path in sorted(_ROUTES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "log_token_usage"
            ):
                calls.append((path.name, node))
    return calls


def test_exactly_six_production_log_token_usage_call_sites_pass_request_id():
    calls = _find_log_token_usage_calls()

    assert len(calls) == 6

    for filename, call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "request_id" in keywords, (
            f"log_token_usage() call in {filename} is missing request_id="
        )


def test_audio_form_compile_logs_usage_with_audioformcompilation_service(monkeypatch):
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        audio_form_token_cost="1",
        models=SimpleNamespace(audio_model="test-model", audio_provider="test-provider"),
    )
    register_request_context_hooks(app)
    app.register_blueprint(multimodal_route.multimodal_bp)

    log_calls = []

    monkeypatch.setattr(multimodal_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(
        multimodal_route.database_pg, "get_user_tokens", lambda user_email: 10
    )
    monkeypatch.setattr(
        multimodal_route.database_pg,
        "get_user_by_username",
        lambda user_email: {"id": 42, "username": user_email},
    )
    monkeypatch.setattr(
        multimodal_route,
        "audioFormPromptBuild",
        lambda *a, **k: {"userprompt": "u", "systemprompt": "s"},
    )
    monkeypatch.setattr(
        multimodal_route,
        "audioFormCompilation",
        lambda *a, **k: {
            "content": {"field": "value"},
            "token_usage": {"input_tokens": 5, "output_tokens": 3},
        },
    )
    monkeypatch.setattr(
        multimodal_route, "log_token_usage", lambda **kwargs: log_calls.append(kwargs)
    )
    monkeypatch.setattr(multimodal_route, "edit_tokens", lambda *a, **k: None)

    response = app.test_client().post(
        "/audioformcompilation",
        json={
            "name": "form-name",
            "exampledata": {"field": "example"},
            "choices": {},
            "transcribedAudio": "some transcribed audio",
        },
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )

    assert response.status_code == 200
    assert len(log_calls) == 1
    assert log_calls[0]["service"] == "/audioformcompilation"
    assert log_calls[0]["request_id"] == response.headers["X-Request-ID"]


def test_prompt_handler_logs_usage_with_prompt_txt_service(monkeypatch):
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        prompt_token_cost=1,
        models=SimpleNamespace(prompt_provider="test-provider", prompt_model="test-model"),
    )
    register_request_context_hooks(app)
    app.register_blueprint(reporting_route.reporting_bp)

    log_calls = []

    monkeypatch.setattr(reporting_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(
        reporting_route.database_pg, "get_user_tokens", lambda username: 10
    )
    monkeypatch.setattr(
        reporting_route.database_pg,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username},
    )
    monkeypatch.setattr(
        reporting_route,
        "reply_to_prompt",
        lambda *a, **k: {
            "content": "reply text",
            "token_usage": {"input_tokens": 5, "output_tokens": 3},
        },
    )
    monkeypatch.setattr(
        reporting_route, "log_token_usage", lambda **kwargs: log_calls.append(kwargs)
    )
    monkeypatch.setattr(reporting_route, "edit_tokens", lambda *a, **k: None)

    response = app.test_client().post(
        "/prompt.txt",
        data={"prompt": "hello", "username": "user@example.com"},
        headers={"X-API-KEY": "test-key"},
    )

    assert response.status_code == 200
    assert len(log_calls) == 1
    assert log_calls[0]["service"] == "/prompt.txt"
    assert log_calls[0]["request_id"] == response.headers["X-Request-ID"]


def test_completion_handler_logs_usage_with_completion_json_service(monkeypatch):
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

    log_calls = []

    monkeypatch.setattr(rag_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(rag_route.database_pg, "get_user_tokens", lambda username: 10)
    monkeypatch.setattr(
        rag_route,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username},
    )
    monkeypatch.setattr(rag_route, "choose_emb_model", lambda *a, **k: object())
    monkeypatch.setattr(rag_route, "MauiVectorStore", lambda *a, **k: object())
    monkeypatch.setattr(
        rag_route,
        "complete_chat",
        lambda *a, **k: {
            "answer": "an answer",
            "vectors": [],
            "token_usage": {"input_tokens": 5, "output_tokens": 3},
        },
    )
    monkeypatch.setattr(
        rag_route, "log_token_usage", lambda **kwargs: log_calls.append(kwargs)
    )
    monkeypatch.setattr(rag_route, "edit_tokens", lambda *a, **k: None)

    response = app.test_client().post(
        "/completion.json",
        json={"chat": ["hello"], "username": "user@example.com"},
        headers={"X-API-KEY": "test-key"},
    )

    assert response.status_code == 200
    assert len(log_calls) == 1
    assert log_calls[0]["service"] == "/completion.json"
    assert log_calls[0]["request_id"] == response.headers["X-Request-ID"]
