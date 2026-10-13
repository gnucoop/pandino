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
    ConfirmOutcome,
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


# --- memory_preflight ------------------------------------------------------


def test_preflight_new_conversation_reserves_nothing():
    agent = _RecordingAgent()
    engine = _make_engine(agent)

    for _ in range(3):
        assert engine.memory_preflight() is None
        assert engine.memory_preflight(None) is None

    assert engine._conversations.active_count() == 0
    assert engine._conversations.has_free_slot()
    assert agent.calls == []


def test_preflight_existing_active_conversation_leaves_it_untouched():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    conv = _confirmed(engine, _turn())
    before = conv.turns

    engine.memory_preflight(conv.conversation_id)

    assert conv.status == "active"
    assert conv.turns == before
    assert engine._conversations.get(conv.conversation_id) is conv
    assert agent.calls == []


def test_preflight_unknown_or_closed_id_raises_not_found():
    engine = _make_engine(_RecordingAgent())
    conv = _confirmed(engine, _turn())
    engine._conversations.close(conv.conversation_id)

    with pytest.raises(ConversationNotFound):
        engine.memory_preflight("nope")
    with pytest.raises(ConversationNotFound):
        engine.memory_preflight(conv.conversation_id)


def test_preflight_at_limit_rejects_new_but_accepts_active():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    engine._conversations = ConversationStore(max_conversations=1)
    conv = _confirmed(engine, _turn())

    with pytest.raises(ConversationLimitReached):
        engine.memory_preflight()
    engine.memory_preflight(conv.conversation_id)

    assert engine._conversations.active_count() == 1
    assert agent.calls == []


# --- confirm_memory_turn ---------------------------------------------------


def test_confirm_activates_candidate():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    result = engine.chat_with_memory("Quante righe?")
    assert engine._conversations.active_count() == 0  # no auto-confirmation

    outcome = engine.confirm_memory_turn(result)

    assert outcome is ConfirmOutcome.ACTIVATED
    conv = result.conversation
    assert conv.status == "active"
    assert engine._conversations.get(conv.conversation_id) is conv
    assert conv.turns == (result.pending_turn,)
    assert len(agent.calls) == 1  # only the chat_with_memory run


def test_confirm_appends_to_active_conversation():
    engine = _make_engine(_RecordingAgent())
    conv = _confirmed(engine, _turn())
    conv_id = conv.conversation_id

    result = engine.chat_with_memory("Next?", conv_id)
    outcome = engine.confirm_memory_turn(result)

    assert outcome is ConfirmOutcome.APPENDED
    assert conv.conversation_id == conv_id
    assert len(conv.turns) == 2
    assert conv.turns[-1].question == "Next?"


def test_no_pending_turn_confirms_nothing():
    engine = _make_engine(_RecordingAgent("just prose"))
    candidate = engine.chat_with_memory("q")
    assert candidate.pending_turn is None
    assert engine.confirm_memory_turn(candidate) is None
    assert candidate.conversation.status == "candidate"
    assert engine._conversations.active_count() == 0

    conv = _confirmed(engine, _turn())
    before = conv.turns
    existing = engine.chat_with_memory("q", conv.conversation_id)
    assert engine.confirm_memory_turn(existing, more_rows_available=True, note="n") is None
    assert conv.turns == before


def test_confirm_attaches_trusted_qualifiers_without_mutation():
    engine = _make_engine(_RecordingAgent(dict(TABLE_OUT)))
    result = engine.chat_with_memory("q")
    original = result.pending_turn

    engine.confirm_memory_turn(result, more_rows_available=True, note="Source capped.")

    stored = result.conversation.turns[-1]
    assert stored.more_rows_available is True
    assert stored.note == "Source capped."
    assert stored is not original
    assert original.more_rows_available is None
    assert original.note is None


@pytest.mark.parametrize("flag", [None, False])
def test_confirm_without_trusted_evidence_keeps_completeness_unknown(flag):
    engine = _make_engine(_RecordingAgent(dict(TABLE_OUT)))
    result = engine.chat_with_memory("q")

    engine.confirm_memory_turn(result, more_rows_available=flag)

    stored = result.conversation.turns[-1]
    assert stored.more_rows_available is None
    assert stored.note is None


def test_confirm_after_closure_returns_closed_and_does_not_resurrect():
    engine = _make_engine(_RecordingAgent())
    conv = _confirmed(engine, _turn())
    before = conv.turns
    result = engine.chat_with_memory("Next?", conv.conversation_id)
    engine._conversations.close(conv.conversation_id)

    outcome = engine.confirm_memory_turn(result)

    assert outcome is ConfirmOutcome.CLOSED
    assert conv.status == "closed"
    assert conv.turns == before
    assert engine._conversations.get(conv.conversation_id) is None
    with pytest.raises(ConversationNotFound):
        engine.memory_preflight(conv.conversation_id)


def test_result_from_another_engine_is_stale_and_changes_nothing():
    # E.g. an engine replaced for the same key: the old run's result is stale.
    old = _make_engine(_RecordingAgent())
    new = _make_engine(_RecordingAgent())
    old_active = _confirmed(old, _turn())

    for result in (
        old.chat_with_memory("q"),
        old.chat_with_memory("q", old_active.conversation_id),
    ):
        before = result.conversation.turns
        status = result.conversation.status
        assert new.confirm_memory_turn(result) is ConfirmOutcome.STALE
        assert result.conversation.turns == before
        assert result.conversation.status == status

    assert new._conversations.active_count() == 0
    assert old._conversations.active_count() == 1


def test_limit_reached_before_candidate_confirmation():
    engine = _make_engine(_RecordingAgent())
    engine._conversations = ConversationStore(max_conversations=1)
    first = engine.chat_with_memory("first?")
    second = engine.chat_with_memory("second?")

    assert engine.confirm_memory_turn(second) is ConfirmOutcome.ACTIVATED
    outcome = engine.confirm_memory_turn(first)

    assert outcome is ConfirmOutcome.LIMIT_REACHED
    assert first.conversation.status == "candidate"
    assert first.conversation.turns == ()
    assert engine._conversations.get(first.conversation.conversation_id) is None
    assert engine._conversations.active_count() == 1


def test_confirm_never_runs_the_agent():
    agent = _RecordingAgent()
    engine = _make_engine(agent)
    result = engine.chat_with_memory("q")
    agent.calls.clear()

    engine.confirm_memory_turn(result)
    engine.confirm_memory_turn(result)  # second confirm appends, still no run

    assert agent.calls == []
