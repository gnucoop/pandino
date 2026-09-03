"""Focused tests over the direct Usage writer call sites in routes/.

Originally proving that the three call sites without a test seam
(audio_form_compile, prompt_handler, completion_handler) pass the canonical
`service` literal to log_token_usage(); datachat, agentchat and compare_docs
already had seams, extended in tests/test_datachat_route_request_id.py,
tests/test_agentchat_route_lifecycle_identity.py and
tests/test_documents_route.py respectively.

/prompt.txt has since migrated to the explicit Usage adoption boundary
(utils.usage_recording), so it no longer calls the writer directly. The
structural counts below therefore cover the six adopters that remain
unmigrated, and prompt_handler is covered by boundary-shaped tests instead.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

from flask import Flask

from routes import multimodal as multimodal_route
from routes import reporting as reporting_route
from routes import rag as rag_route
from utils import usage_recording
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


def test_every_unmigrated_log_token_usage_call_site_passes_request_id():
    calls = _find_log_token_usage_calls()

    assert len(calls) == 5

    for filename, call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "request_id" in keywords, (
            f"log_token_usage() call in {filename} is missing request_id="
        )


def test_every_unmigrated_log_token_usage_call_site_passes_source():
    calls = _find_log_token_usage_calls()

    assert len(calls) == 5

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


def test_every_unmigrated_call_site_captures_log_id_locally():
    """Every remaining direct writer call binds the returned id.

    Some previously discarded the return value (/compare_docs,
    /audioformcompilation); all remaining direct call sites now capture it
    internally, whether or not it is exposed publicly. Adopters that have
    moved behind the Usage boundary never see an id at all and are
    deliberately outside this count.
    """
    assignments = _find_log_token_usage_assignments()
    calls = _find_log_token_usage_calls()

    assert len(assignments) == len(calls) == 5


def test_every_unmigrated_call_site_hands_off_log_id_to_usage_request_state():
    """Every captured id is registered request-locally.

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


def _build_audio_form_app(monkeypatch, token_usage):
    """Wire an /audioformcompilation app whose only unstubbed Usage step is
    the boundary - the mirror of :func:`_build_prompt_app` for the second
    adopter."""
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        audio_form_token_cost="1",
        models=SimpleNamespace(audio_model="test-model", audio_provider="test-provider"),
    )
    register_request_context_hooks(app)
    app.register_blueprint(multimodal_route.multimodal_bp)

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
            "token_usage": token_usage,
        },
    )
    monkeypatch.setattr(multimodal_route, "edit_tokens", lambda *a, **k: None)
    return app


def _post_audio_form(app):
    return app.test_client().post(
        "/audioformcompilation",
        json={
            "name": "form-name",
            "exampledata": {"field": "example"},
            "choices": {},
            "transcribedAudio": "some transcribed audio",
        },
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )


def test_audio_form_compile_records_usage_through_the_adoption_boundary(monkeypatch):
    """/audioformcompilation is the second adopter of the explicit Usage
    boundary, reusing it unchanged.

    The route supplies consumption facts only. request_id, source, the row
    id and its registration are all resolved behind the boundary, so this
    test observes them at the writer the boundary itself calls.
    """
    app = _build_audio_form_app(
        monkeypatch, {"input_tokens": 5, "output_tokens": 3}
    )

    log_calls = []
    monkeypatch.setattr(
        usage_recording,
        "log_token_usage",
        lambda **kwargs: (log_calls.append(kwargs), 4242)[1],
    )
    handoff_calls = []
    monkeypatch.setattr(
        usage_recording, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
    )
    _stub_boundary_user_lookups(monkeypatch, client="dino")

    response = _post_audio_form(app)

    assert response.status_code == 200
    assert response.get_json() == {"field": "value"}
    # ...and log_id exposure is unchanged: /audioformcompilation never
    # returned one, and adopting the boundary must not start now.
    assert "log_id" not in response.get_json()

    assert len(log_calls) == 1
    call = log_calls[0]
    assert call["user_id"] == 42
    assert call["provider"] == "test-provider"
    assert call["model"] == "test-model"
    assert call["service"] == "/audioformcompilation"
    assert call["token_input"] == 5
    assert call["token_output"] == 3
    # Derived behind the boundary, never supplied by this route.
    assert call["request_id"] == response.headers["X-Request-ID"]
    assert call["source"] == "dino"
    # Row-id registration is the boundary's bookkeeping, still performed.
    assert handoff_calls == [4242]


def test_audio_form_compile_keeps_its_zero_token_guard(monkeypatch):
    """The >0 guard stays a flow-level rule at the call site: the boundary
    accepts 0/0, so a recorded call here would prove the guard was lost."""
    app = _build_audio_form_app(
        monkeypatch, {"input_tokens": 0, "output_tokens": 0}
    )

    recorded = []
    monkeypatch.setattr(
        usage_recording, "log_token_usage", lambda **kwargs: recorded.append(kwargs)
    )

    response = _post_audio_form(app)

    assert response.status_code == 200
    assert recorded == []
    assert response.get_json() == {"field": "value"}


def test_audio_form_compile_response_survives_a_usage_failure(monkeypatch):
    """Accepted behaviour change, identical in shape to /prompt.txt: an
    accounting failure used to escape this route as a 500. It is now
    contained by the boundary, and the route's own token debit - which
    previously never ran on that path - now proceeds."""
    app = _build_audio_form_app(
        monkeypatch, {"input_tokens": 5, "output_tokens": 3}
    )

    def boom(**kwargs):
        raise RuntimeError("database is unreachable")

    monkeypatch.setattr(usage_recording, "log_token_usage", boom)
    _stub_boundary_user_lookups(monkeypatch, client="dino")

    debits = []
    monkeypatch.setattr(
        multimodal_route, "edit_tokens", lambda *a: debits.append(a)
    )

    response = _post_audio_form(app)

    assert response.status_code == 200
    assert response.get_json() == {"field": "value"}
    assert debits == [("user@example.com", -1)]


def _build_prompt_app(monkeypatch, token_usage):
    """Wire a /prompt.txt app whose only unstubbed Usage step is the boundary."""
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        prompt_token_cost=1,
        models=SimpleNamespace(prompt_provider="test-provider", prompt_model="test-model"),
    )
    register_request_context_hooks(app)
    app.register_blueprint(reporting_route.reporting_bp)

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
        lambda *a, **k: {"content": "reply text", "token_usage": token_usage},
    )
    monkeypatch.setattr(reporting_route, "edit_tokens", lambda *a, **k: None)
    return app


def _stub_boundary_user_lookups(monkeypatch, client=None):
    """Stub the boundary's own identity reads.

    The boundary binds these by direct name import, so patching
    ``database_pg`` for the route does not reach them - which is itself the
    point: source derivation is the boundary's business, not the route's.
    """
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_id",
        lambda user_id: {"id": user_id, "username": "user@example.com"},
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username, "client": client},
    )


def test_prompt_handler_records_usage_through_the_adoption_boundary(monkeypatch):
    """/prompt.txt is the first adopter of the explicit Usage boundary.

    The route now states consumption facts only: the writer, request_id,
    source, the row id and its registration all sit behind
    ``record_token_consumption``.
    """
    app = _build_prompt_app(
        monkeypatch, {"input_tokens": 5, "output_tokens": 3}
    )

    log_calls = []
    handoff_calls = []

    def fake_log_token_usage(**kwargs):
        log_calls.append(kwargs)
        return 4343

    monkeypatch.setattr(usage_recording, "log_token_usage", fake_log_token_usage)
    monkeypatch.setattr(
        usage_recording, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
    )
    _stub_boundary_user_lookups(monkeypatch)

    response = app.test_client().post(
        "/prompt.txt",
        data={"prompt": "hello", "username": "user@example.com"},
        headers={"X-API-KEY": "test-key"},
    )

    assert response.status_code == 200
    assert len(log_calls) == 1
    assert log_calls[0]["service"] == "/prompt.txt"
    assert log_calls[0]["provider"] == "test-provider"
    assert log_calls[0]["model"] == "test-model"
    assert log_calls[0]["token_input"] == 5
    assert log_calls[0]["token_output"] == 3
    # request_id and source are derived inside the boundary, not passed by
    # the route.
    assert log_calls[0]["request_id"] == response.headers["X-Request-ID"]
    assert log_calls[0]["source"] is None
    # The row id is registered by the boundary; the route never sees it...
    assert handoff_calls == [4343]
    # ...and public response exposure is unchanged - /prompt.txt returns the
    # plain-text reply only, never log_id.
    assert response.get_data(as_text=True) == "reply text"


def test_prompt_handler_keeps_its_zero_token_guard(monkeypatch):
    """The >0 guard stays a flow-level rule at the call site: the boundary
    accepts 0/0, so a recorded call here would prove the guard was lost."""
    app = _build_prompt_app(
        monkeypatch, {"input_tokens": 0, "output_tokens": 0}
    )

    recorded = []
    monkeypatch.setattr(
        usage_recording, "log_token_usage", lambda **kwargs: recorded.append(kwargs)
    )

    response = app.test_client().post(
        "/prompt.txt",
        data={"prompt": "hello", "username": "user@example.com"},
        headers={"X-API-KEY": "test-key"},
    )

    assert response.status_code == 200
    assert recorded == []
    assert response.get_data(as_text=True) == "reply text"


def test_prompt_handler_response_survives_a_usage_failure(monkeypatch):
    """Accepted behaviour change: an accounting failure used to escape this
    route as a 500. It is now contained by the boundary."""
    app = _build_prompt_app(
        monkeypatch, {"input_tokens": 5, "output_tokens": 3}
    )

    def boom(**kwargs):
        raise RuntimeError("database is unreachable")

    monkeypatch.setattr(usage_recording, "log_token_usage", boom)
    _stub_boundary_user_lookups(monkeypatch)

    response = app.test_client().post(
        "/prompt.txt",
        data={"prompt": "hello", "username": "user@example.com"},
        headers={"X-API-KEY": "test-key"},
    )

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "reply text"


_DIRECT_USAGE_NAMES = (
    "log_token_usage",
    "log_usage_with_resolved_cost",
    "set_usage_log_id",
)

# The migrated token adopters, as (route file, handler function). Scoped
# deliberately: the remaining adopters are intentionally unmigrated and
# still call the writer directly, so a repository-wide prohibition would be
# wrong today. Add a row here when an adopter migrates.
_MIGRATED_TOKEN_ADOPTERS = (
    ("reporting.py", "prompt_handler"),
    ("multimodal.py", "audio_form_compile"),
)


def _handler_ast(filename, function_name):
    tree = ast.parse((_ROUTES_DIR / filename).read_text(), filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return tree, node
    raise AssertionError(f"{function_name}() not found in routes/{filename}")


def test_migrated_token_adopters_do_not_persist_usage_directly():
    """Structural invariant over every migrated token adopter.

    Scoped to the handler function rather than the file, because
    routes/multimodal.py still hosts the unmigrated /transcribe flow and
    must keep its direct writer calls. A migrated handler may name none of
    the direct Usage persistence operations.
    """
    for filename, function_name in _MIGRATED_TOKEN_ADOPTERS:
        _, handler = _handler_ast(filename, function_name)

        called = {
            node.func.id
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } | {
            node.func.attr
            for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        for name in _DIRECT_USAGE_NAMES:
            assert name not in called, (
                f"{function_name}() in routes/{filename} calls {name}() "
                "directly; it must record Usage through the adoption boundary"
            )

        assert "record_token_consumption" in called, (
            f"{function_name}() in routes/{filename} no longer records Usage "
            "through record_token_consumption()"
        )


def test_migrated_token_adopters_import_the_adoption_boundary():
    """Each migrated adopter's module imports the boundary.

    A module hosting only migrated flows must additionally not import the
    direct writers at all; one that still hosts an unmigrated flow keeps
    them, and the call-level invariant above carries the guarantee.
    """
    hosts_unmigrated = {
        filename
        for filename, _ in _find_log_token_usage_calls()
    }

    for filename, _ in _MIGRATED_TOKEN_ADOPTERS:
        tree = ast.parse((_ROUTES_DIR / filename).read_text(), filename=filename)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        assert "record_token_consumption" in imported, (
            f"routes/{filename} no longer imports record_token_consumption"
        )

        if filename not in hosts_unmigrated:
            for name in _DIRECT_USAGE_NAMES:
                assert name not in imported, (
                    f"routes/{filename} hosts only migrated adopters but "
                    f"still imports {name}"
                )


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
