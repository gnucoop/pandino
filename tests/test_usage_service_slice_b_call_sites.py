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


def test_exactly_six_production_log_token_usage_call_sites_pass_source():
    calls = _find_log_token_usage_calls()

    assert len(calls) == 6

    for filename, call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "source" in keywords, (
            f"log_token_usage() call in {filename} is missing source="
        )


def _find_log_token_usage_assignments():
    """Return (filename, target_name) for every ``x = log_token_usage(...)``.

    Distinguishes capturing the return value (Usage Duration Slice B3
    handoff prerequisite) from merely passing arguments - a call site could
    pass ``service=``/``request_id=`` without ever binding the result to a
    name.
    """
    assignments = []
    for path in sorted(_ROUTES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "log_token_usage"
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                assignments.append((path.name, node.targets[0].id))
    return assignments


def test_all_six_call_sites_capture_log_id_locally():
    """Usage Duration Slice B3: every writer now binds the returned id.

    Three of the six previously discarded the return value
    (/compare_docs, /audioformcompilation, /prompt.txt); B3 requires all
    six to capture it internally, whether or not it is exposed publicly.
    """
    assignments = _find_log_token_usage_assignments()
    calls = _find_log_token_usage_calls()

    assert len(assignments) == len(calls) == 6


def test_all_six_call_sites_hand_off_log_id_to_usage_request_state():
    """Usage Duration Slice B3: every captured id is registered request-locally.

    For each ``x = log_token_usage(...)`` in a route file, the same file
    must contain a call ``set_usage_log_id(x)`` somewhere - proving the
    handoff exists without pinning down its exact line position relative
    to the assignment.
    """
    assignments = _find_log_token_usage_assignments()

    for filename, target_name in assignments:
        path = _ROUTES_DIR / filename
        tree = ast.parse(path.read_text(), filename=str(path))

        handed_off = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_usage_log_id"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == target_name
            for node in ast.walk(tree)
        )
        assert handed_off, (
            f"{filename} captures log_token_usage() into {target_name!r} "
            "but never hands it off via set_usage_log_id()"
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
        lambda user_email: {"id": 42, "username": user_email, "client": "dino"},
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
    def fake_log_token_usage(**kwargs):
        log_calls.append(kwargs)
        return 4242

    handoff_calls = []
    monkeypatch.setattr(multimodal_route, "log_token_usage", fake_log_token_usage)
    monkeypatch.setattr(
        multimodal_route, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
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
    assert log_calls[0]["source"] == "dino"
    # Slice B3: the returned id is now captured and handed off request-locally...
    assert handoff_calls == [4242]
    # ...but public response exposure is unchanged - /audioformcompilation
    # never returned log_id before B3 and must not start now.
    assert "log_id" not in response.get_json()


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
        lambda username: {"id": 42, "username": username, "client": None},
    )
    monkeypatch.setattr(
        reporting_route,
        "reply_to_prompt",
        lambda *a, **k: {
            "content": "reply text",
            "token_usage": {"input_tokens": 5, "output_tokens": 3},
        },
    )
    def fake_log_token_usage(**kwargs):
        log_calls.append(kwargs)
        return 4343

    handoff_calls = []
    monkeypatch.setattr(reporting_route, "log_token_usage", fake_log_token_usage)
    monkeypatch.setattr(
        reporting_route, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
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
    assert log_calls[0]["source"] is None
    # Slice B3: the returned id is now captured and handed off request-locally...
    assert handoff_calls == [4343]
    # ...but public response exposure is unchanged - /prompt.txt returns the
    # plain-text reply only, never log_id.
    assert response.get_data(as_text=True) == "reply text"


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
        lambda username: {"id": 42, "username": username, "client": "coopi"},
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
    def fake_log_token_usage(**kwargs):
        log_calls.append(kwargs)
        return 4444

    handoff_calls = []
    monkeypatch.setattr(rag_route, "log_token_usage", fake_log_token_usage)
    monkeypatch.setattr(
        rag_route, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
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
    assert log_calls[0]["source"] == "coopi"
    # Slice B3: the returned id is now handed off request-locally...
    assert handoff_calls == [4444]
    # ...and public response exposure is unchanged - /completion.json already
    # returned log_id before B3 and must keep doing so.
    assert response.get_json()["log_id"] == 4444
