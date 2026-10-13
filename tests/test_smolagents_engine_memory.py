"""SmolagentsEngine integration of the optional conversational memory.

Engines are built via __new__ so that no LLM model is constructed; the agent is
a fake recording its run() arguments. chat_with_memory() prepares turns and
never confirms them: confirmation belongs to the (future) route.
"""

from types import SimpleNamespace

import pytest

from datachat.conversation_memory import (
    HISTORY_HEADER,
    CompactTurn,
    ConversationLimitReached,
    ConversationNotFound,
    ConversationStore,
    compose_task,
    render_historical_context,
)
from datachat.smolagents_engine import MemoryChatResult, SmolagentsEngine

TEXT_OUT = '{"kind": "text", "text": "There are 42 rows."}'
TABLE_OUT = {"kind": "table", "data": [{"region": "North", "n": 3}, {"region": "South", "n": 5}]}


class _RecordingAgent:
    def __init__(self, output=TEXT_OUT, state="success", raises=None):
        self.output = output
        self.state = state
        self.raises = raises
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        if self.state is None:
            return SimpleNamespace(output=self.output)
        return SimpleNamespace(output=self.output, state=self.state)


def _make_engine(agent=None, with_store=True):
    engine = SmolagentsEngine.__new__(SmolagentsEngine)
    engine.api_key = "test-key"
    engine.user_name = "test-user"
    engine.llm = None
    engine.data = None
    engine._last_run_result = None
    engine._last_run_duration_ms = None
    engine._agent = agent
    if with_store:
        engine._conversations = ConversationStore()
    return engine


def _confirmed(engine, *turns):
    conv = engine._conversations.begin(None)
    for turn in turns:
        engine._conversations.confirm(conv, turn)
    return conv


def _turn(question="How many rows?", text="42 rows."):
    return CompactTurn(question=question, kind="text", text=text)


def _task(agent):
    args, kwargs = agent.calls[0]
    return args[0]


# --- memory OFF -------------------------------------------------------------


def test_chat_passes_original_task_once_with_reset():
    agent = _RecordingAgent()
    engine = _make_engine(agent, with_store=False)

    result = engine.chat("Quante righe?")

    assert result == {"kind": "text", "text": "There are 42 rows."}
    assert len(agent.calls) == 1
    args, kwargs = agent.calls[0]
    assert args == ("Quante righe?",)
    assert kwargs == {"reset": True, "return_full_result": True}


def test_chat_never_reads_conversation_store():
    # Built via __new__ without _conversations: any access would raise.
    agent = _RecordingAgent()
    engine = _make_engine(agent, with_store=False)

    engine.chat(123)

    assert _task(agent) == "123"
    assert not hasattr(engine, "_conversations")


def test_chat_error_responses_unchanged():
    engine = _make_engine(None, with_store=False)
    assert engine.chat("q")["code"] == "MISSING_CONFIG"

    engine = _make_engine(_RecordingAgent(raises=ValueError("boom")), with_store=False)
    assert engine.chat("q") == {
        "kind": "error",
        "message": "SmolagentsEngine failed to run: boom",
        "code": "RUN_FAILED",
    }


def test_chat_fallback_unchanged():
    engine = _make_engine(_RecordingAgent(output="just prose"), with_store=False)
    assert engine.chat("q") == {
        "kind": "text",
        "text": "just prose",
        "format": "plain",
    }


# --- task construction -----------------------------------------------------


def test_new_candidate_task_is_original_and_takes_no_slot():
    agent = _RecordingAgent()
    engine = _make_engine(agent)

    result = engine.chat_with_memory("Quante righe?")

    assert isinstance(result, MemoryChatResult)
    assert len(agent.calls) == 1
    args, kwargs = agent.calls[0]
    assert args == ("Quante righe?",)
    assert kwargs == {"reset": True, "return_full_result": True}
    assert result.conversation.status == "candidate"
    assert engine._conversations.active_count() == 0
    assert engine._conversations.get(result.conversation.conversation_id) is None


def test_active_conversation_without_turns_keeps_original_task(monkeypatch):
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    conv = _confirmed(engine, _turn())
    # An active conversation whose history is empty renders nothing.
    monkeypatch.setattr(type(conv), "turns", property(lambda self: ()))

    engine.chat_with_memory("Next?", conv.conversation_id)

    assert _task(agent) == "Next?"


def test_confirmed_history_augments_task_from_that_conversation_only():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    mine = _confirmed(engine, _turn("Mine?", "mine answer"))
    _confirmed(engine, _turn("Other?", "other answer"))

    result = engine.chat_with_memory("Follow-up?", mine.conversation_id)

    expected = compose_task("Follow-up?", render_historical_context(mine.turns))
    assert len(agent.calls) == 1
    args, kwargs = agent.calls[0]
    assert args == (expected,)
    assert kwargs == {"reset": True, "return_full_result": True}
    assert args[0].startswith(HISTORY_HEADER)
    assert "mine answer" in args[0]
    assert "other answer" not in args[0]
    assert result.conversation is mine


def test_pending_turn_question_is_original_message_not_task():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    conv = _confirmed(engine, _turn("Earlier?", "earlier answer"))

    result = engine.chat_with_memory("Follow-up?", conv.conversation_id)

    assert result.pending_turn.question == "Follow-up?"
    assert "earlier" not in result.pending_turn.question
    assert _task(agent) != "Follow-up?"


# --- eligibility -------------------------------------------------------------


def test_valid_text_prepares_text_turn():
    engine = _make_engine(_RecordingAgent(TEXT_OUT))
    turn = engine.chat_with_memory("q").pending_turn
    assert turn.kind == "text"
    assert turn.text == "There are 42 rows."


def test_valid_table_prepares_table_turn():
    engine = _make_engine(_RecordingAgent(dict(TABLE_OUT)))
    turn = engine.chat_with_memory("q").pending_turn
    assert turn.kind == "table"
    assert turn.columns == ("region", "n")
    assert turn.result_rows == 2


def test_valid_empty_table_is_eligible():
    engine = _make_engine(_RecordingAgent({"kind": "table", "data": []}))
    turn = engine.chat_with_memory("q").pending_turn
    assert turn is not None
    assert turn.kind == "table"
    assert turn.result_rows == 0


def test_max_steps_error_prepares_no_turn():
    engine = _make_engine(_RecordingAgent(TEXT_OUT, state="max_steps_error"))
    result = engine.chat_with_memory("q")
    assert result.response == {"kind": "text", "text": "There are 42 rows."}
    assert result.pending_turn is None


def test_unknown_state_prepares_no_turn():
    engine = _make_engine(_RecordingAgent(TEXT_OUT, state="weird"))
    assert engine.chat_with_memory("q").pending_turn is None


def test_none_state_remains_compatible():
    engine = _make_engine(_RecordingAgent(TEXT_OUT, state=None))
    assert engine.chat_with_memory("q").pending_turn is not None


@pytest.mark.parametrize(
    "output",
    [
        "just prose",  # fallback: no JSON object
        '{"text": "no kind"}',  # fallback: no kind
        '{"kind": "error", "message": "nope"}',  # valid contract, excluded kind
        '{"kind": "image_path", "path": "/tmp/x.png"}',
    ],
)
def test_contract_failures_and_excluded_kinds_prepare_no_turn(output):
    engine = _make_engine(_RecordingAgent(output))
    result = engine.chat_with_memory("q")
    assert result.pending_turn is None


def test_fallback_response_unchanged_and_no_turn():
    engine = _make_engine(_RecordingAgent("just prose"))
    result = engine.chat_with_memory("q")
    assert result.response == {"kind": "text", "text": "just prose", "format": "plain"}
    assert result.pending_turn is None


def test_missing_config_returns_error_and_no_turn():
    engine = _make_engine(None)
    result = engine.chat_with_memory("q")
    assert result.response["code"] == "MISSING_CONFIG"
    assert result.pending_turn is None


def test_run_failed_returns_error_and_no_turn():
    engine = _make_engine(_RecordingAgent(raises=RuntimeError("boom")))
    result = engine.chat_with_memory("q")
    assert result.response == {
        "kind": "error",
        "message": "SmolagentsEngine failed to run: boom",
        "code": "RUN_FAILED",
    }
    assert result.pending_turn is None


# --- deferred confirmation -------------------------------------------------


def test_prepared_turn_is_not_confirmed_on_candidate():
    engine = _make_engine(_RecordingAgent())
    result = engine.chat_with_memory("q")
    assert result.pending_turn is not None
    assert result.conversation.status == "candidate"
    assert result.conversation.turns == ()
    assert engine._conversations.active_count() == 0


def test_existing_conversation_unchanged_until_confirmed():
    engine = _make_engine(_RecordingAgent())
    conv = _confirmed(engine, _turn())
    before = conv.turns

    result = engine.chat_with_memory("Next?", conv.conversation_id)

    assert result.pending_turn is not None
    assert conv.turns == before
    assert conv.status == "active"


def test_response_dict_carries_no_memory_objects():
    engine = _make_engine(_RecordingAgent())
    result = engine.chat_with_memory("q")
    assert set(result.response) == {"kind", "text"}
    for value in result.response.values():
        assert not isinstance(value, (CompactTurn, MemoryChatResult))
        assert value is not result.conversation


# --- conversation lookup ---------------------------------------------------


def test_unknown_conversation_id_raises():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    with pytest.raises(ConversationNotFound):
        engine.chat_with_memory("q", "nope")
    assert agent.calls == []


def test_limit_reached_for_new_candidate_but_active_stays_usable():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    engine._conversations = ConversationStore(max_conversations=1)
    conv = _confirmed(engine, _turn())

    with pytest.raises(ConversationLimitReached):
        engine.chat_with_memory("q")
    assert agent.calls == []

    result = engine.chat_with_memory("q", conv.conversation_id)
    assert result.conversation is conv
    assert len(agent.calls) == 1


# --- ownership -------------------------------------------------------------


def test_dataclass_construction_gives_independent_stores(monkeypatch, tmp_path):
    # No model configured: __post_init__ exits early, no LLM is built.
    monkeypatch.setenv("DATACHAT_MODEL", "")
    monkeypatch.setenv("DATACHAT_PLOTS_DIR", str(tmp_path))

    a = SmolagentsEngine(api_key="a", user_name="u", llm=None, data=None)
    b = SmolagentsEngine(api_key="b", user_name="u", llm=None, data=None)

    assert isinstance(a._conversations, ConversationStore)
    assert a._conversations is not b._conversations
    assert "_conversations" not in vars(SmolagentsEngine)

    conv = a._conversations.begin(None)
    a._conversations.confirm(conv, _turn())
    assert a._conversations.active_count() == 1
    assert b._conversations.active_count() == 0
    with pytest.raises(ConversationNotFound):
        b._conversations.begin(conv.conversation_id)
