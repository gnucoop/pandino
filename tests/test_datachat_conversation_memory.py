import threading

import pytest

from datachat.conversation_memory import (
    MAX_CONVERSATIONS,
    MAX_TURNS,
    CompactTurn,
    ConfirmOutcome,
    ConversationLimitReached,
    ConversationNotFound,
    ConversationStore,
)


def _turn(i=0):
    return CompactTurn(question=f"q{i}", kind="text", text=f"a{i}")


def _activate(store, i=0):
    conv = store.begin(None)
    assert store.confirm(conv, _turn(i)) is ConfirmOutcome.ACTIVATED
    return conv


def test_candidate_is_not_active_until_confirmed():
    store = ConversationStore()
    conv = store.begin(None)
    assert len(conv.conversation_id) == 32
    assert conv.status == "candidate"
    assert store.get(conv.conversation_id) is None
    assert store.active_count() == 0
    with pytest.raises(ConversationNotFound):
        store.begin(conv.conversation_id)
    assert store.close(conv.conversation_id) == "not_active"


def test_abandoned_candidates_consume_no_slot():
    store = ConversationStore()
    for _ in range(MAX_CONVERSATIONS * 3):
        store.begin(None)
    assert store.active_count() == 0
    assert store.has_free_slot()


def test_first_confirmation_activates_candidate():
    store = ConversationStore()
    conv = _activate(store)
    assert conv.status == "active"
    assert store.get(conv.conversation_id) is conv
    assert store.begin(conv.conversation_id) is conv
    assert conv.turns == (_turn(0),)


def test_candidate_ids_are_unique():
    store = ConversationStore()
    ids = {store.begin(None).conversation_id for _ in range(100)}
    assert len(ids) == 100


def test_conversation_limit():
    store = ConversationStore()
    convs = [_activate(store, i) for i in range(MAX_CONVERSATIONS)]
    assert not store.has_free_slot()
    with pytest.raises(ConversationLimitReached):
        store.begin(None)
    # Existing conversations are untouched and still usable.
    assert all(store.get(c.conversation_id) is c for c in convs)
    assert store.confirm(convs[0], _turn(9)) is ConfirmOutcome.APPENDED
    # Closing frees exactly one slot.
    store.close(convs[0].conversation_id)
    _activate(store)
    assert store.active_count() == MAX_CONVERSATIONS


def test_limit_rechecked_at_confirmation():
    store = ConversationStore()
    late = store.begin(None)
    for i in range(MAX_CONVERSATIONS):
        _activate(store, i)
    assert store.confirm(late, _turn()) is ConfirmOutcome.LIMIT_REACHED
    assert late.status == "candidate"
    assert store.get(late.conversation_id) is None


def test_sliding_window_keeps_last_five_turns():
    store = ConversationStore()
    conv = _activate(store, 0)
    for i in range(1, 8):
        store.confirm(conv, _turn(i))
    assert len(conv.turns) == MAX_TURNS
    assert [t.question for t in conv.turns] == ["q3", "q4", "q5", "q6", "q7"]


def test_conversations_are_independent():
    store = ConversationStore()
    a = _activate(store, 1)
    b = _activate(store, 2)
    store.confirm(a, _turn(3))
    assert [t.question for t in a.turns] == ["q1", "q3"]
    assert [t.question for t in b.turns] == ["q2"]
    store.close(a.conversation_id)
    assert store.get(b.conversation_id) is b


def test_close_is_idempotent():
    store = ConversationStore()
    conv = _activate(store)
    assert store.close(conv.conversation_id) == "closed"
    assert conv.status == "closed"
    assert store.close(conv.conversation_id) == "not_active"
    assert store.close("unknown") == "not_active"
    with pytest.raises(ConversationNotFound):
        store.begin(conv.conversation_id)


def test_late_confirmation_does_not_resurrect_closed_conversation():
    store = ConversationStore()
    conv = _activate(store)
    held = store.begin(conv.conversation_id)  # a run holds the object
    store.close(conv.conversation_id)
    assert store.confirm(held, _turn(1)) is ConfirmOutcome.CLOSED
    assert store.get(conv.conversation_id) is None
    assert store.active_count() == 0
    assert held.turns == (_turn(0),)


def test_stale_or_detached_objects_are_rejected():
    store = ConversationStore()
    conv = _activate(store)
    # Detached active object whose id now maps to another object.
    store._active[conv.conversation_id] = impostor = type(conv)(conv.conversation_id, store)
    impostor._status = "active"
    assert store.confirm(conv, _turn(1)) is ConfirmOutcome.STALE
    # Active object no longer registered at all.
    del store._active[conv.conversation_id]
    assert store.confirm(conv, _turn(1)) is ConfirmOutcome.STALE
    # A candidate never overwrites an existing entry for its id.
    other = _activate(store)
    clash = type(conv)(other.conversation_id, store)
    assert store.confirm(clash, _turn(2)) is ConfirmOutcome.STALE
    assert store.get(other.conversation_id) is other


def test_candidate_cannot_be_activated_by_another_store():
    owner, other = ConversationStore(), ConversationStore()
    cand = owner.begin(None)
    assert other.confirm(cand, _turn(1)) is ConfirmOutcome.STALE
    assert other.get(cand.conversation_id) is None
    assert other.active_count() == 0
    assert cand.status == "candidate"
    assert cand.turns == ()
    assert owner.confirm(cand, _turn(2)) is ConfirmOutcome.ACTIVATED
    assert owner.get(cand.conversation_id) is cand
    assert cand.turns == (_turn(2),)


def test_active_conversation_cannot_be_confirmed_by_another_store():
    owner, other = ConversationStore(), ConversationStore()
    conv = _activate(owner)
    assert other.confirm(conv, _turn(1)) is ConfirmOutcome.STALE
    assert other.active_count() == 0
    assert conv.status == "active"
    assert conv.turns == (_turn(0),)
    assert owner.confirm(conv, _turn(2)) is ConfirmOutcome.APPENDED


def test_rejected_turn_kind():
    with pytest.raises(ValueError):
        CompactTurn(question="q", kind="image_path")


def test_close_versus_confirm_is_atomic():
    # Whatever the interleaving, after both finish the conversation is not
    # active, and a turn is recorded only if confirmation won the race.
    for _ in range(200):
        store = ConversationStore()
        conv = _activate(store)
        barrier = threading.Barrier(2)
        results = {}

        def do_confirm():
            barrier.wait()
            results["confirm"] = store.confirm(conv, _turn(1))

        def do_close():
            barrier.wait()
            results["close"] = store.close(conv.conversation_id)

        threads = [threading.Thread(target=do_confirm), threading.Thread(target=do_close)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results["close"] == "closed"
        assert store.get(conv.conversation_id) is None
        assert conv.status == "closed"
        if results["confirm"] is ConfirmOutcome.APPENDED:
            assert len(conv.turns) == 2
        else:
            assert results["confirm"] is ConfirmOutcome.CLOSED
            assert len(conv.turns) == 1


def test_lock_serializes_transitions():
    store = ConversationStore()
    conv = _activate(store)
    done = threading.Event()

    def do_close():
        store.close(conv.conversation_id)
        done.set()

    with store._lock:
        t = threading.Thread(target=do_close)
        t.start()
        assert not done.wait(0.05)
        assert conv.status == "active"
    t.join()
    assert conv.status == "closed"
