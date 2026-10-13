import copy
import math
import threading

import pandas as pd
import pytest

from datachat.conversation_memory import (
    MAX_CELL_CHARS,
    MAX_CONVERSATIONS,
    MAX_NOTE_CHARS,
    MAX_QUESTION_CHARS,
    MAX_TABLE_COLUMNS,
    MAX_TABLE_ROWS,
    MAX_TEXT_CHARS,
    MAX_TURNS,
    TRUNCATION_MARKER,
    CompactTurn,
    ConfirmOutcome,
    ConversationLimitReached,
    ConversationNotFound,
    ConversationStore,
    build_compact_turn,
    with_trusted_qualifiers,
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


# --- CompactTurn construction ---------------------------------------------

_FROM_PAYLOAD = object()


def _build(payload, question="q", passed=True, final_kind=_FROM_PAYLOAD, run_state="success"):
    if final_kind is _FROM_PAYLOAD:
        final_kind = payload.get("kind")
    return build_compact_turn(
        question, payload, passed=passed, final_kind=final_kind, run_state=run_state
    )


def test_text_result():
    turn = _build({"kind": "text", "text": "answer", "format": "plain"}, question="how many?")
    assert turn == CompactTurn(question="how many?", kind="text", text="answer")
    assert turn.more_rows_available is None


def test_text_result_with_only_message():
    assert _build({"kind": "text", "message": "from message"}).text == "from message"


def test_text_precedence_text_content_message():
    turn = _build({"kind": "text", "text": "t", "content": "c", "message": "m"})
    assert turn.text == "t"
    assert _build({"kind": "text", "content": "c", "message": "m"}).text == "c"


def test_question_and_text_truncation_is_visible_and_deterministic():
    turn = _build({"kind": "text", "text": "x" * 5000}, question="y" * 2000)
    assert len(turn.question) == MAX_QUESTION_CHARS
    assert turn.question.endswith(TRUNCATION_MARKER)
    assert len(turn.text) == MAX_TEXT_CHARS
    assert turn.text.endswith(TRUNCATION_MARKER)
    assert turn == _build({"kind": "text", "text": "x" * 5000}, question="y" * 2000)


def test_list_table():
    data = [{"a": 1, "b": "x"}, {"b": "y", "c": 2.5}]
    turn = _build({"kind": "table", "data": data})
    assert turn.kind == "table"
    assert turn.columns == ("a", "b", "c")
    assert turn.rows == ((1, "x", None), (None, "y", 2.5))
    assert (turn.result_rows, turn.result_columns) == (2, 3)


def test_dataframe_table():
    df = pd.DataFrame({"n": [1, 2], "s": ["a", "b"]})
    turn = _build({"kind": "table", "data": df})
    assert turn.columns == ("n", "s")
    assert turn.rows == ((1, "a"), (2, "b"))
    assert type(turn.rows[0][0]) is int
    assert (turn.result_rows, turn.result_columns) == (2, 2)


def test_empty_list_table():
    turn = _build({"kind": "table", "data": []})
    assert turn.columns == () and turn.rows == ()
    assert (turn.result_rows, turn.result_columns) == (0, 0)


def test_empty_dataframe_keeps_columns():
    turn = _build({"kind": "table", "data": pd.DataFrame(columns=["a", "b"])})
    assert turn.columns == ("a", "b") and turn.rows == ()
    assert (turn.result_rows, turn.result_columns) == (0, 2)


@pytest.mark.parametrize("data", [{"a": [1]}, "a,b\n1,2", None, 3, [1, 2], [{"a": 1}, "x"]])
def test_unsupported_table_shapes_rejected(data):
    assert _build({"kind": "table", "data": data}) is None


@pytest.mark.parametrize(
    "payload, final_kind, passed",
    [
        ({"kind": "error", "message": "boom"}, "error", True),
        ({"kind": "image_path", "path": "/tmp/p.png"}, "image_path", True),
        ({"kind": "error", "message": "x", "code": "MISSING_CONFIG"}, None, False),
        ({"kind": "error", "message": "x", "code": "RUN_FAILED"}, None, False),
        ({"kind": "text", "text": "ok"}, "text", False),  # failed contract
        ({"kind": "text", "text": "fallback", "format": "plain"}, None, False),
        ({"kind": "text", "text": "fallback"}, None, True),  # final_kind=None
        ({"kind": "chart", "data": []}, "chart", True),
        ({"kind": "table", "data": []}, "text", True),  # kind mismatch
        ({"kind": "text", "text": "  "}, "text", True),
        ("not a dict", "text", True),
    ],
)
def test_ineligible_results_rejected(payload, final_kind, passed):
    assert _build(payload, final_kind=final_kind, passed=passed) is None


def test_max_steps_error_rejected_even_with_valid_payload():
    payload = {"kind": "text", "text": "looks fine"}
    assert _build(payload, run_state="max_steps_error") is None
    assert _build({"kind": "table", "data": [{"a": 1}]}, run_state="max_steps_error") is None
    assert _build(payload, run_state="something_else") is None
    # An unavailable state (test doubles without RunResult.state) is not a failure.
    assert _build(payload, run_state=None) is not None


def test_row_and_column_limits_keep_observed_dimensions():
    data = [{f"c{j}": i * 100 + j for j in range(18)} for i in range(150)]
    turn = _build({"kind": "table", "data": data})
    assert len(turn.rows) == MAX_TABLE_ROWS
    assert len(turn.columns) == MAX_TABLE_COLUMNS
    assert all(len(r) == MAX_TABLE_COLUMNS for r in turn.rows)
    assert turn.columns == tuple(f"c{j}" for j in range(10))
    assert turn.rows[9] == tuple(900 + j for j in range(10))
    assert (turn.result_rows, turn.result_columns) == (150, 18)
    assert turn.more_rows_available is None

    df = pd.DataFrame(data)
    assert _build({"kind": "table", "data": df}) == turn


def test_cell_and_column_name_truncation():
    long = "z" * 300
    turn = _build({"kind": "table", "data": [{long: long, "short": "ok"}]})
    assert len(turn.columns[0]) == MAX_CELL_CHARS
    assert turn.columns[0].endswith(TRUNCATION_MARKER)
    assert len(turn.rows[0][0]) == MAX_CELL_CHARS
    assert turn.rows[0][0].endswith(TRUNCATION_MARKER)
    assert turn.rows[0][1] == "ok"


def test_non_scalar_cells_converted_to_strings():
    turn = _build({"kind": "table", "data": [{"d": {"k": [1, 2]}, "l": [1, 2], "b": True}]})
    assert turn.rows[0] == ("{'k': [1, 2]}", "[1, 2]", True)


def test_nan_and_non_finite_values():
    turn = _build({"kind": "table", "data": [{"a": float("nan"), "b": float("inf"), "c": None}]})
    assert turn.rows[0][0] is None
    assert turn.rows[0][1] == math.inf  # kept as a float, like output_normalizer
    assert turn.rows[0][2] is None
    df_turn = _build({"kind": "table", "data": pd.DataFrame({"a": [1.0, float("nan")]})})
    assert df_turn.rows == ((1.0,), (None,))


def test_input_payload_not_mutated():
    nested = {"k": [1, 2]}
    data = [{"a": nested, 1: "x" * 300} for _ in range(20)]
    payload = {"kind": "table", "data": data}
    snapshot = copy.deepcopy(payload)
    data_id, row_ids = id(data), [id(r) for r in data]
    _build(payload)
    assert payload == snapshot
    assert payload["data"] is data and id(payload["data"]) == data_id
    assert [id(r) for r in data] == row_ids
    assert data[0]["a"] is nested

    df = pd.DataFrame({"a": [1.0, float("nan")]})
    df_payload = {"kind": "table", "data": df}
    before = df.copy()
    _build(df_payload)
    assert df_payload["data"] is df
    pd.testing.assert_frame_equal(df, before)


def test_trusted_qualifiers():
    base = _build({"kind": "table", "data": [{"a": 1}]})
    turn = with_trusted_qualifiers(base, more_rows_available=True, note="capped at 1000")
    assert turn.more_rows_available is True
    assert turn.note == "capped at 1000"
    assert turn.rows == base.rows and base.more_rows_available is None


def test_unknown_completeness_stays_unknown():
    base = _build({"kind": "table", "data": [{"a": 1}]})
    for value in (None, False):
        turn = with_trusted_qualifiers(base, more_rows_available=value, note=None)
        assert turn.more_rows_available is None
        assert turn.note is None


def test_note_truncation():
    base = _build({"kind": "text", "text": "t"})
    turn = with_trusted_qualifiers(base, more_rows_available=None, note="n" * 2000)
    assert len(turn.note) == MAX_NOTE_CHARS
    assert turn.note.endswith(TRUNCATION_MARKER)
