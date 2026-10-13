"""POST /datachat/conversation/close: explicit closure of one conversation.

Reuses the /datachat memory fixtures: a real SmolagentsEngine (no LLM) with a
fake agent and the real ConversationStore; collaborators patched at the route.
"""

import threading

import pytest

from datachat.conversation_memory import ConversationStore
from infrastructure import agent_manager
from routes import datachat as datachat_route
from tests.test_datachat_memory_routes import (  # noqa: F401  (autouse fixtures)
    HEADERS,
    _Agent,
    _engine,
    _Env,
    _start_conversation,
    _store,
    clean_busy_registry,
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)
from tests.test_datachat_route_request_id import _StubEngine

CLOSE = "/datachat/conversation/close"


def _close(env, body, headers=HEADERS):
    return env.app.test_client().post(CLOSE, headers=headers, json=body)


def _forbid(monkeypatch, *names):
    def forbidden(*a, **k):
        raise AssertionError("must not be called by the close endpoint")

    for name in names:
        monkeypatch.setattr(datachat_route, name, forbidden)


ACCOUNTING = (
    "get_user_tokens",
    "edit_tokens",
    "record_token_consumption",
    "record_additional_token_consumption",
    "_flush_provider_contributions",
    "try_acquire_run",
    "release_run",
    "createAgent",
    "deleteAgent",
    "purge_exports",
)


# --- success and idempotency -------------------------------------------------


def test_close_active_then_repeat_then_unknown(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    conversation_id = _start_conversation(env)
    other_id = _start_conversation(env)
    conv = _store(env).get(conversation_id)
    runs = len(agent.tasks)
    _forbid(monkeypatch, *ACCOUNTING)

    def forbidden_close():
        raise AssertionError("engine.close() must not be called")

    env.engine.close = forbidden_close

    first = _close(env, {"conversation_id": conversation_id})
    again = _close(env, {"conversation_id": conversation_id})
    unknown = _close(env, {"conversation_id": "unknown"})

    assert (first.status_code, first.get_json()) == (200, {"outcome": "closed"})
    assert (again.status_code, again.get_json()) == (200, {"outcome": "not_active"})
    assert (unknown.status_code, unknown.get_json()) == (200, {"outcome": "not_active"})
    assert conv.status == "closed"
    assert _store(env).get(other_id).status == "active"
    assert _store(env).active_count() == 1
    assert len(agent.tasks) == runs
    assert env.usage and len(env.usage) == 2  # only the two /datachat runs


def test_identifier_is_passed_unchanged(monkeypatch):
    env = _Env(monkeypatch, _engine())
    seen = []
    env.engine.close_memory_conversation = lambda cid: seen.append(cid) or "not_active"

    _close(env, {"conversation_id": "  Id-With Spaces  "})

    assert seen == ["  Id-With Spaces  "]


def test_close_releases_slot_at_limit(monkeypatch):
    env = _Env(monkeypatch, _engine(store=ConversationStore(max_conversations=1)))
    conversation_id = _start_conversation(env)
    assert env.post("q", memory=True).status_code == 409

    assert _close(env, {"conversation_id": conversation_id}).get_json() == {"outcome": "closed"}

    assert _store(env).active_count() == 0
    new_id = _start_conversation(env)
    assert new_id != conversation_id


def test_engine_stays_registered_and_usable(monkeypatch):
    env = _Env(monkeypatch, _engine())
    conversation_id = _start_conversation(env)
    other_id = _start_conversation(env)
    key = (agent_manager.DATACHAT, "test-key")
    monkeypatch.setitem(agent_manager.activeEngines, key, env.engine)
    monkeypatch.setattr(datachat_route, "getAgent", agent_manager.getAgent)

    _close(env, {"conversation_id": conversation_id})

    assert agent_manager.activeEngines[key] is env.engine
    assert env.post("q").status_code == 200  # memory OFF
    assert env.post("q", memory=True, conversation_id=other_id).get_json()[
        "conversation_id"
    ] == other_id
    gone = env.post("q", memory=True, conversation_id=conversation_id)
    assert (gone.status_code, gone.get_json()) == (404, {"error": "conversation_not_found"})


def test_memory_off_requests_close_nothing(monkeypatch):
    env = _Env(monkeypatch, _engine())
    conversation_id = _start_conversation(env)

    assert env.post("q").status_code == 200
    assert env.post("q", memory=False).status_code == 200

    assert _store(env).get(conversation_id).status == "active"


# --- input and authentication ------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"other": 1},
        {"conversation_id": None},
        {"conversation_id": 123},
        {"conversation_id": ["a"]},
        {"conversation_id": {"id": "a"}},
        {"conversation_id": True},
        {"conversation_id": ""},
        {"conversation_id": "   "},
        ["conversation_id"],
    ],
)
def test_invalid_body_is_400_before_lookup(monkeypatch, body):
    env = _Env(monkeypatch, _engine())
    _forbid(monkeypatch, "getAgent", *ACCOUNTING)

    response = _close(env, body)

    assert response.status_code == 400
    expected = "Missing JSON body" if body == {} else "invalid_memory_request"
    assert response.get_json() == {"error": expected}


def test_missing_json_body_is_400(monkeypatch):
    env = _Env(monkeypatch, _engine())
    _forbid(monkeypatch, "getAgent")

    response = env.app.test_client().post(CLOSE, headers=HEADERS)

    assert (response.status_code, response.get_json()) == (400, {"error": "Missing JSON body"})


@pytest.mark.parametrize(
    "headers, error",
    [
        ({"X-USER-EMAIL": "user@example.com"}, "Missing X-API-KEY header"),
        ({"X-API-KEY": "test-key"}, "Missing X-USER-EMAIL header"),
    ],
)
def test_missing_headers_are_400(monkeypatch, headers, error):
    env = _Env(monkeypatch, _engine())
    _forbid(monkeypatch, "getAgent", "assert_valid_api_key")

    response = _close(env, {"conversation_id": "x"}, headers=headers)

    assert (response.status_code, response.get_json()) == (400, {"error": error})


def test_failed_authentication_is_403_without_state_change(monkeypatch):
    from flask import abort

    env = _Env(monkeypatch, _engine())
    conversation_id = _start_conversation(env)
    monkeypatch.setattr(
        datachat_route, "assert_valid_api_key", lambda *a: abort(403, description="Invalid API key")
    )
    _forbid(monkeypatch, "getAgent")

    response = _close(env, {"conversation_id": conversation_id})

    assert response.status_code == 403
    assert _store(env).get(conversation_id).status == "active"


# --- engine ownership and lifecycle -------------------------------------------


def test_no_active_engine_is_not_active(monkeypatch):
    env = _Env(monkeypatch, _engine())
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: None)
    _forbid(monkeypatch, "createAgent")

    response = _close(env, {"conversation_id": "x"})

    assert (response.status_code, response.get_json()) == (200, {"outcome": "not_active"})


def test_engine_without_close_method_is_not_active(monkeypatch):
    env = _Env(monkeypatch, _engine())
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: _StubEngine())

    response = _close(env, {"conversation_id": "x"})

    assert (response.status_code, response.get_json()) == (200, {"outcome": "not_active"})


def test_conversation_of_another_engine_cannot_be_closed(monkeypatch):
    env = _Env(monkeypatch, _engine())
    other_engine = _engine()
    conversation_id = _start_conversation(env)  # lives in env.engine
    monkeypatch.setattr(
        datachat_route,
        "getAgent",
        lambda api_key: {"test-key": other_engine, "owner-key": env.engine}[api_key],
    )

    response = _close(env, {"conversation_id": conversation_id})

    assert response.get_json() == {"outcome": "not_active"}
    assert _store(env).get(conversation_id).status == "active"


# --- concurrency ---------------------------------------------------------------


def test_close_does_not_wait_for_held_run_guard(monkeypatch):
    env = _Env(monkeypatch, _engine())
    conversation_id = _start_conversation(env)
    assert agent_manager.try_acquire_run(env.engine)

    response = _close(env, {"conversation_id": conversation_id})

    assert response.get_json() == {"outcome": "closed"}
    assert not agent_manager.try_acquire_run(env.engine)  # still held by the "run"


def test_close_during_run_prevents_resurrection(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    conversation_id = _start_conversation(env)
    conv = _store(env).get(conversation_id)
    before = conv.turns
    usage_before, debits_before = len(env.usage), len(env.debits)
    closes = []

    def close_from_another_request():
        # A separate thread gives the close its own request context, as a
        # concurrent HTTP request would; joined, so the order is deterministic.
        t = threading.Thread(
            target=lambda: closes.append(_close(env, {"conversation_id": conversation_id}))
        )
        t.start()
        t.join()

    agent.on_run = close_from_another_request

    response = env.post("Next?", memory=True, conversation_id=conversation_id)

    assert [(r.status_code, r.get_json()) for r in closes] == [(200, {"outcome": "closed"})]
    assert response.status_code == 200
    assert "conversation_id" not in response.get_json()
    assert len(env.usage) == usage_before + 1
    assert len(env.debits) == debits_before + 1
    assert len(env.confirms) == 2  # confirmation attempted, without effect
    assert conv.status == "closed" and conv.turns == before
    assert _store(env).active_count() == 0
    agent.on_run = None
    later = env.post("q", memory=True, conversation_id=conversation_id)
    assert (later.status_code, later.get_json()) == (404, {"error": "conversation_not_found"})


def test_close_log_carries_no_conversation_id(monkeypatch):
    env = _Env(monkeypatch, _engine())
    conversation_id = _start_conversation(env)
    env.stream.truncate(0)

    _close(env, {"conversation_id": conversation_id})

    logged = env.stream.getvalue()
    assert "datachat_conversation_close" in logged and "outcome=closed" in logged
    assert conversation_id not in logged
