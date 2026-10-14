"""POST /datachat with optional conversational memory.

A real SmolagentsEngine (built via __new__, no LLM) with a fake agent and the
real ConversationStore; every other collaborator is patched at the route
module, following tests/test_datachat_route_request_id.py.
"""

import json
from types import SimpleNamespace

import pytest

from datachat.chart_registry import RecordedChart
from datachat.conversation_memory import HISTORY_HEADER, ConversationStore
from datachat.result_provenance import TrustedResult
from datachat.smolagents_engine import SmolagentsEngine
from infrastructure import agent_manager
from routes import datachat as datachat_route
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    _StubEngine,
    _make_app,
    _patch_success_dependencies,
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)

HEADERS = {"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"}
TEXT_OUT = '{"kind": "text", "text": "There are 42 rows."}'
TABLE_DATA = [{"region": "North", "n": 3}, {"region": "South", "n": 5}]
MEMORY_METHODS = ("memory_preflight", "chat_with_memory", "confirm_memory_turn")


@pytest.fixture(autouse=True)
def clean_busy_registry():
    agent_manager._busyEngines.clear()
    yield
    agent_manager._busyEngines.clear()


class _Agent:
    """Fake CodeAgent: records tasks; `on_run` simulates concurrent events."""

    def __init__(self, output=TEXT_OUT, state="success"):
        self.output = output
        self.state = state
        self.tasks = []
        self.on_run = None

    def run(self, task, **kwargs):
        assert kwargs == {"reset": True, "return_full_result": True}
        self.tasks.append(task)
        if self.on_run is not None:
            self.on_run()
        output = self.output() if callable(self.output) else self.output
        return SimpleNamespace(output=output, state=self.state)


def _engine(agent=None, store=None):
    engine = SmolagentsEngine.__new__(SmolagentsEngine)
    engine.api_key = "test-key"
    engine.user_name = "tester"
    engine.llm = None
    engine.data = None
    engine._last_run_result = None
    engine._last_run_duration_ms = None
    engine._agent = agent if agent is not None else _Agent()
    engine._conversations = store if store is not None else ConversationStore()
    return engine


class _Env:
    def __init__(self, monkeypatch, engine, debit=(True, "Tokens edited successfully")):
        self.app, self.stream, _ = _make_app()
        self.app.config["PROPAGATE_EXCEPTIONS"] = False
        self.engine = engine
        self.events = []
        self.debits = []
        self.usage = []
        self.token_reads = []
        _patch_success_dependencies(monkeypatch, engine, record_calls=self.usage)

        def get_user_tokens(email):
            self.token_reads.append(email)
            return 10

        def edit_tokens(*a, **k):
            self.events.append("debit")
            self.debits.append(a)
            if isinstance(debit, Exception):
                raise debit
            return debit

        real_normalize = datachat_route.normalize_datachat_response
        real_attach = datachat_route._attach_recorded_charts

        def normalize(*a, **k):
            self.events.append("normalize")
            return real_normalize(*a, **k)

        def attach(*a, **k):
            self.events.append("charts")
            return real_attach(*a, **k)

        monkeypatch.setattr(datachat_route, "get_user_tokens", get_user_tokens)
        monkeypatch.setattr(datachat_route, "edit_tokens", edit_tokens)
        monkeypatch.setattr(datachat_route, "normalize_datachat_response", normalize)
        monkeypatch.setattr(datachat_route, "_attach_recorded_charts", attach)
        monkeypatch.setattr(datachat_route, "get_recorded_charts", lambda: [])

        self.confirms = []
        if hasattr(engine, "confirm_memory_turn"):
            real_confirm = engine.confirm_memory_turn

            def confirm(result, **kwargs):
                self.events.append("confirm")
                self.confirms.append(kwargs)
                return real_confirm(result, **kwargs)

            engine.confirm_memory_turn = confirm

    def post(self, chat="Quante righe?", **fields):
        return self.app.test_client().post(
            "/datachat", json={"chat": chat, **fields}, headers=HEADERS
        )


def _store(env):
    return env.engine._conversations


def _start_conversation(env, chat="First?"):
    response = env.post(chat, memory=True)
    assert response.status_code == 200
    return response.get_json()["conversation_id"]


# --- memory OFF ------------------------------------------------------------


def _forbid_memory_methods(monkeypatch, engine):
    def forbidden(*a, **k):
        raise AssertionError("memory method called with memory OFF")

    for name in MEMORY_METHODS:
        monkeypatch.setattr(engine, name, forbidden, raising=False)


@pytest.mark.parametrize("fields", [{}, {"memory": False}])
def test_memory_off_uses_chat_and_keeps_response_shape(monkeypatch, fields):
    agent = _Agent()
    engine = _engine(agent)
    env = _Env(monkeypatch, engine, debit=None)
    _forbid_memory_methods(monkeypatch, engine)

    response = env.post("Quante righe?", **fields)

    assert response.status_code == 200
    assert set(response.get_json()) == {"response", "explanation", "log_id"}
    assert "conversation_id" not in response.get_json()
    assert agent.tasks == ["Quante righe?"]
    assert len(env.debits) == 1
    assert _store(env).active_count() == 0


def test_absent_and_false_memory_are_equivalent(monkeypatch):
    bodies = []
    for fields in ({}, {"memory": False}):
        env = _Env(monkeypatch, _engine())
        bodies.append(env.post("q", **fields).get_json())
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("fields", [{}, {"memory": False}])
def test_legacy_engine_without_memory_methods_still_works(monkeypatch, fields):
    engine = _StubEngine()
    env = _Env(monkeypatch, engine, debit=None)

    response = env.post("hello", **fields)

    assert response.status_code == 200
    assert engine.calls == [(("hello",), {})]
    assert "conversation_id" not in response.get_json()


# --- validation and capability --------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"memory": "true"},
        {"memory": 1},
        {"memory": None},
        {"conversation_id": "abc"},
        {"memory": False, "conversation_id": "abc"},
        {"memory": True, "conversation_id": ""},
        {"memory": True, "conversation_id": 7},
    ],
)
def test_invalid_memory_request_is_400_before_lookup(monkeypatch, fields):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    lookups = []
    monkeypatch.setattr(datachat_route, "getAgent", lambda k: lookups.append(k))

    response = env.post("q", **fields)

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid_memory_request"}
    assert lookups == []
    assert agent.tasks == [] and env.usage == [] and env.debits == []


def test_memory_on_unsupported_engine_is_400_without_work(monkeypatch):
    engine = _StubEngine()
    env = _Env(monkeypatch, engine)

    response = env.post("q", memory=True)

    assert response.status_code == 400
    assert response.get_json() == {"error": "memory_not_supported"}
    assert engine.calls == []
    assert env.usage == [] and env.debits == [] and env.token_reads == []
    assert agent_manager._busyEngines == {}


# --- preflight and lookup --------------------------------------------------


def _assert_no_work(env, agent):
    assert agent.tasks == []
    assert env.token_reads == []
    assert env.usage == []
    assert env.debits == []
    assert agent_manager._busyEngines == {}


def test_unknown_conversation_is_404_before_balance(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))

    response = env.post("q", memory=True, conversation_id="nope")

    assert response.status_code == 404
    assert response.get_json() == {"error": "conversation_not_found"}
    _assert_no_work(env, agent)


def test_closed_conversation_is_404(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    conversation_id = _start_conversation(env)
    _store(env).close(conversation_id)
    agent.tasks.clear()
    env.token_reads.clear(), env.usage.clear(), env.debits.clear()

    response = env.post("q", memory=True, conversation_id=conversation_id)

    assert response.status_code == 404
    assert response.get_json() == {"error": "conversation_not_found"}
    _assert_no_work(env, agent)


def test_limit_reached_is_409_but_active_conversation_still_usable(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent, ConversationStore(max_conversations=1)))
    conversation_id = _start_conversation(env)
    agent.tasks.clear()
    env.token_reads.clear(), env.usage.clear(), env.debits.clear()

    response = env.post("q", memory=True)

    assert response.status_code == 409
    assert response.get_json() == {"error": "conversation_limit_reached"}
    _assert_no_work(env, agent)

    response = env.post("Next?", memory=True, conversation_id=conversation_id)
    assert response.status_code == 200
    assert response.get_json()["conversation_id"] == conversation_id


def test_race_conversation_closed_after_preflight_is_404(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    conversation_id = _start_conversation(env)
    agent.tasks.clear()
    env.token_reads.clear(), env.usage.clear(), env.debits.clear()
    real_preflight = env.engine.memory_preflight

    def preflight_then_close(cid):
        real_preflight(cid)
        _store(env).close(conversation_id)

    monkeypatch.setattr(env.engine, "memory_preflight", preflight_then_close)

    response = env.post("q", memory=True, conversation_id=conversation_id)

    assert response.status_code == 404
    assert response.get_json() == {"error": "conversation_not_found"}
    assert agent.tasks == [] and env.usage == [] and env.debits == []
    assert agent_manager._busyEngines == {}


def test_race_slot_taken_after_preflight_is_409(monkeypatch):
    agent = _Agent()
    store = ConversationStore(max_conversations=1)
    env = _Env(monkeypatch, _engine(agent, store))
    real_preflight = env.engine.memory_preflight

    def preflight_then_fill(cid):
        real_preflight(cid)
        store.confirm(store.begin(None), SimpleNamespace())

    monkeypatch.setattr(env.engine, "memory_preflight", preflight_then_fill)

    response = env.post("q", memory=True)

    assert response.status_code == 409
    assert response.get_json() == {"error": "conversation_limit_reached"}
    assert agent.tasks == [] and env.usage == [] and env.debits == []
    assert agent_manager._busyEngines == {}


# --- successful conversations ----------------------------------------------


def test_new_conversation_is_activated_after_debit_and_reused(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))

    first = env.post("First?", memory=True)

    assert first.status_code == 200
    body = first.get_json()
    conversation_id = body["conversation_id"]
    assert set(body) == {"response", "explanation", "log_id", "conversation_id"}
    assert env.events == ["normalize", "charts", "debit", "confirm"]
    conv = _store(env).get(conversation_id)
    assert conv is not None and len(conv.turns) == 1
    assert agent.tasks == ["First?"]

    second = env.post("Next?", memory=True, conversation_id=conversation_id)

    assert second.status_code == 200
    assert second.get_json()["conversation_id"] == conversation_id
    assert agent.tasks[1].startswith(HISTORY_HEADER)
    assert "First?" in agent.tasks[1]
    assert agent.tasks[1].endswith("Next?")
    assert [t.question for t in conv.turns] == ["First?", "Next?"]
    assert _store(env).active_count() == 1
    assert env.confirms and len(env.confirms) == 2

    off = env.post("Plain?")
    assert "conversation_id" not in off.get_json()
    assert agent.tasks[2] == "Plain?"
    assert len(conv.turns) == 2


# --- deferred confirmation -------------------------------------------------


def test_usage_failure_does_not_block_confirmation(monkeypatch):
    env = _Env(monkeypatch, _engine())
    monkeypatch.setattr(datachat_route, "record_token_consumption", lambda **k: False)

    response = env.post("q", memory=True)

    assert response.status_code == 200
    assert "log_id" not in response.get_json()
    assert _store(env).get(response.get_json()["conversation_id"]) is not None


def _fail_normalize(*a, **k):
    raise RuntimeError("boom")


def _fail_charts(*a, **k):
    raise ValueError("boom")


@pytest.mark.parametrize("target, failure", [
    ("normalize_datachat_response", _fail_normalize),
    ("_attach_recorded_charts", _fail_charts),
])
def test_post_run_failure_prevents_confirmation(monkeypatch, target, failure):
    env = _Env(monkeypatch, _engine())
    conversation_id = _start_conversation(env)
    conv = _store(env).get(conversation_id)
    before = conv.turns
    monkeypatch.setattr(datachat_route, target, failure)

    for fields in ({}, {"conversation_id": conversation_id}):
        env.confirms.clear()
        response = env.post("q", memory=True, **fields)
        assert response.status_code == 500
        assert "conversation_id" not in (response.get_json() or {})
        assert env.confirms == []

    assert conv.turns == before
    assert _store(env).active_count() == 1


def test_debit_exception_prevents_confirmation(monkeypatch):
    env = _Env(monkeypatch, _engine(), debit=ValueError("db down"))

    response = env.post("q", memory=True)

    assert response.status_code == 500
    assert env.confirms == []
    assert _store(env).active_count() == 0


@pytest.mark.parametrize("debit", [(False, "Error while editing tokens"), None])
def test_explicit_debit_failure_keeps_http_but_skips_confirmation(monkeypatch, debit):
    ok_env = _Env(monkeypatch, _engine())
    ok_body = ok_env.post("q", memory=True).get_json()

    env = _Env(monkeypatch, _engine(), debit=debit)
    response = env.post("q", memory=True)

    assert response.status_code == 200
    body = response.get_json()
    assert body["response"] == ok_body["response"]
    assert "conversation_id" not in body
    assert env.confirms == []
    assert _store(env).active_count() == 0


def test_explicit_debit_failure_on_existing_conversation_keeps_id(monkeypatch):
    env = _Env(monkeypatch, _engine())
    conversation_id = _start_conversation(env)
    conv = _store(env).get(conversation_id)
    monkeypatch.setattr(datachat_route, "edit_tokens", lambda *a, **k: (False, "x"))
    env.confirms.clear()

    response = env.post("Next?", memory=True, conversation_id=conversation_id)

    assert response.status_code == 200
    assert response.get_json()["conversation_id"] == conversation_id
    assert env.confirms == []
    assert len(conv.turns) == 1


@pytest.mark.parametrize("output, state", [
    ("just prose", "success"),
    ('{"kind": "error", "message": "nope"}', "success"),
    (TEXT_OUT, "max_steps_error"),
])
def test_ineligible_new_candidate_gets_no_id_and_no_slot(monkeypatch, output, state):
    env = _Env(monkeypatch, _engine(_Agent(output, state)))

    response = env.post("q", memory=True)

    assert response.status_code == 200
    assert "conversation_id" not in response.get_json()
    assert env.confirms == []
    assert _store(env).active_count() == 0


def test_ineligible_turn_on_existing_conversation_keeps_id_and_history(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    conversation_id = _start_conversation(env)
    conv = _store(env).get(conversation_id)
    before = conv.turns
    agent.output = "just prose"
    env.confirms.clear()

    response = env.post("Next?", memory=True, conversation_id=conversation_id)

    assert response.status_code == 200
    assert response.get_json()["conversation_id"] == conversation_id
    assert env.confirms == []
    assert conv.turns == before
    assert conv.status == "active"


# --- trusted qualifiers ----------------------------------------------------


def _table_env(monkeypatch, trusted=None, charts=()):
    env = _Env(monkeypatch, _engine(_Agent(lambda: {"kind": "table", "data": list(TABLE_DATA)})))
    monkeypatch.setattr(datachat_route, "lookup_trusted_result", lambda r: trusted)
    monkeypatch.setattr(datachat_route, "get_recorded_charts", lambda: list(charts))
    return env


def _confirmed_turn(env, response):
    conv = _store(env).get(response.get_json()["conversation_id"])
    return conv.turns[-1]


def _chart(note=None, more=False):
    return RecordedChart(spec={"type": "bar"}, note=note, more_rows_available=more)


@pytest.mark.parametrize("trusted, charts, expected", [
    (TrustedResult(more_rows_available=True), (), (True, None)),
    (TrustedResult(note="Primary note."), (_chart(note="Chart note."),), (None, "Primary note.")),
    (None, (_chart(more=True),), (True, None)),
    (TrustedResult(), (_chart(note="Chart note."),), (None, "Chart note.")),
    (None, (), (None, None)),
])
def test_trusted_qualifiers_reach_the_confirmed_turn(monkeypatch, trusted, charts, expected):
    env = _table_env(monkeypatch, trusted, charts)

    response = env.post("q", memory=True)

    turn = _confirmed_turn(env, response)
    assert (turn.more_rows_available, turn.note) == expected
    body = response.get_json()["response"]
    # The response and the memory carry the same trusted caveats.
    assert body.get("more_rows_available") == (True if expected[0] else None)
    assert body.get("note") == expected[1]


def test_model_generated_provenance_is_ignored(monkeypatch):
    payload = {
        "kind": "table",
        "data": list(TABLE_DATA),
        "more_rows_available": True,
        "note": "Model says partial.",
    }
    env = _Env(monkeypatch, _engine(_Agent(lambda: dict(payload))))
    monkeypatch.setattr(datachat_route, "lookup_trusted_result", lambda r: None)

    turn = _confirmed_turn(env, env.post("q", memory=True))

    assert turn.more_rows_available is None
    assert turn.note is None


# --- outcomes and concurrency ----------------------------------------------


def test_closed_during_run_omits_id_and_does_not_resurrect(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    conversation_id = _start_conversation(env)
    conv = _store(env).get(conversation_id)
    before = conv.turns
    agent.on_run = lambda: _store(env).close(conversation_id)

    response = env.post("Next?", memory=True, conversation_id=conversation_id)

    assert response.status_code == 200
    assert "conversation_id" not in response.get_json()
    assert len(env.confirms) == 2  # the first request's and this one's
    assert conv.status == "closed" and conv.turns == before
    agent.on_run = None
    again = env.post("q", memory=True, conversation_id=conversation_id)
    assert again.status_code == 404


def test_ineligible_turn_closed_during_run_omits_id(monkeypatch):
    agent = _Agent()
    env = _Env(monkeypatch, _engine(agent))
    conversation_id = _start_conversation(env)
    agent.output = "just prose"
    agent.on_run = lambda: _store(env).close(conversation_id)

    response = env.post("Next?", memory=True, conversation_id=conversation_id)

    assert response.status_code == 200
    assert "conversation_id" not in response.get_json()


def test_stale_confirmation_omits_id(monkeypatch):
    engine = _engine()
    env = _Env(monkeypatch, engine)
    foreign = _engine()
    real_chat_with_memory = engine.chat_with_memory
    # E.g. a replaced engine: the run's conversation belongs to another store.
    monkeypatch.setattr(
        engine,
        "chat_with_memory",
        lambda chat, cid: foreign.chat_with_memory(chat, cid) if cid is None else real_chat_with_memory(chat, cid),
    )

    response = env.post("q", memory=True)

    assert response.status_code == 200
    assert "conversation_id" not in response.get_json()
    assert len(env.confirms) == 1
    assert _store(env).active_count() == 0
    assert foreign._conversations.active_count() == 0


def test_limit_reached_at_confirmation_omits_id(monkeypatch):
    agent = _Agent()
    store = ConversationStore(max_conversations=1)
    env = _Env(monkeypatch, _engine(agent, store))

    def take_last_slot():
        other = store.begin(None)
        store.confirm(other, SimpleNamespace())

    agent.on_run = take_last_slot

    response = env.post("q", memory=True)

    assert response.status_code == 200
    assert "conversation_id" not in response.get_json()
    assert len(env.confirms) == 1
    assert store.active_count() == 1


def test_session_busy_unchanged_with_memory(monkeypatch):
    env = _Env(monkeypatch, _engine())
    assert agent_manager.try_acquire_run(env.engine)

    response = env.post("q", memory=True)

    assert response.status_code == 409
    assert response.get_json()["error"] == "session_busy"
    assert env.confirms == [] and env.debits == []


# --- response isolation ----------------------------------------------------


def test_json_carries_no_internal_memory_objects(monkeypatch):
    env = _table_env(monkeypatch, TrustedResult(more_rows_available=True, note="n"))

    response = env.post("q", memory=True)

    body = response.get_json()
    assert set(body) == {"response", "explanation", "log_id", "conversation_id"}
    assert body["log_id"] == 999
    assert isinstance(body["conversation_id"], str)
    text = json.dumps(body)
    for leak in ("pending_turn", "CompactTurn", "MemoryChatResult", "Conversation", "ConfirmOutcome", "activated"):
        assert leak not in text
