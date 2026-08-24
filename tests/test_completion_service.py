"""
Tests for services.completion_service.complete_chat.

These tests mock the vector store and LLM dependencies so they never perform
real retrieval, network, or provider calls.
"""

import logging
from types import SimpleNamespace

import pytest

from services import completion_service
from services.completion_service import CompletionRequest, complete_chat
from utils.operational_persistence import snapshot_from_record


def test_complete_chat_logs_question_received_without_question_content(
    monkeypatch, caplog
):
    """completion_question_received must be emitted as a content-free lifecycle
    event: the question still reaches retrieval, but must never appear in the
    log record, since it is arbitrary user-provided conversational content."""
    distinctive_question = "What is the sky-blue platypus protocol XK42?"
    captured = {}

    class FakeStore:
        def find_similar_vectors(self, text, top_k, min_similarity):
            captured["retrieval_text"] = text
            return []

    class FakeLlm:
        def invoke(self, messages):
            captured["messages"] = messages
            return SimpleNamespace(content="an answer", response_metadata={})

    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *args, **kwargs: FakeLlm()
    )
    monkeypatch.setattr(
        completion_service,
        "load_prompt",
        lambda *args, **kwargs: "system prompt",
    )
    monkeypatch.setattr(
        completion_service, "render_prompt", lambda template, **kwargs: template
    )

    req = CompletionRequest(
        username="alice",
        info=["some background info"],
        chat=[distinctive_question],
    )

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        result = complete_chat(req, FakeStore(), "OpenAI", "gpt-4o-mini")

    assert result["answer"] == "an answer"

    # The question is still used for retrieval and reaches the LLM prompt.
    assert captured["retrieval_text"] == distinctive_question
    assert any(
        distinctive_question in msg.get("content", "")
        for msg in captured["messages"]
    )

    received_records = [
        r for r in caplog.records if "event=completion_question_received" in r.message
    ]
    assert len(received_records) == 1
    received_message = received_records[0].message

    assert received_message == "event=completion_question_received"
    assert distinctive_question not in received_message
    # No replacement representation (length, hash, excerpt) was introduced.
    assert "question=" not in received_message


# ---------------------------------------------------------------------------
# SECOND ADOPTER SLICE C1 — persistent Operational retrieval facts.
#
# Facts under test:
#   completion_retrieval_completed  (INFO, vector_count/top_k/min_sim/info_present)
#   completion_retrieval_failed     (WARNING, error_type only)
#
# Emission is asserted through the marked-LogRecord contract and through the
# detached persistence snapshot, so both the stderr-visible message and the
# stored row are covered by the safe-data assertions.
# ---------------------------------------------------------------------------

_SENSITIVE_MARKERS = ("question=", "text=", "chat=", "info=", "namespace=")


def _operational_records(caplog, event):
    return [
        r
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
        and getattr(r, "maui_event", None) == event
    ]


def _the_operational_record(caplog, event):
    records = _operational_records(caplog, event)
    assert len(records) == 1, f"expected exactly one {event} record, got {len(records)}"
    return records[0]


def _assert_free_of(record, forbidden):
    """No forbidden substring may reach the record's rendered message, its
    maui_* metadata, or the detached snapshot that is what actually persists."""
    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    surfaces = [
        record.getMessage(),
        str(getattr(record, "maui_details", None)),
        str(getattr(record, "maui_message", None)),
        str(getattr(record, "maui_error_type", None)),
        str(snapshot.details_json),
        str(snapshot.message),
        str(snapshot.error_type),
    ]
    for surface in surfaces:
        for needle in forbidden:
            assert needle not in surface, (
                f"forbidden content {needle!r} reached Operational surface: {surface!r}"
            )


class _FakeLlm:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="an answer", response_metadata={})


def _patch_provider_boundary(monkeypatch):
    """Mock the existing provider boundary so no real provider is called."""
    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *args, **kwargs: _FakeLlm()
    )
    monkeypatch.setattr(
        completion_service, "load_prompt", lambda *args, **kwargs: "system prompt"
    )
    monkeypatch.setattr(
        completion_service, "render_prompt", lambda template, **kwargs: template
    )


def test_retrieval_completed_is_persisted_with_vectors(monkeypatch, caplog):
    """T1 — successful retrieval returning vectors emits exactly one marked
    completion_retrieval_completed carrying only counts, the retrieval
    configuration actually used, and the info_present boolean."""
    question = "What is the sky-blue platypus protocol XK42?"
    chunk_text = "CONFIDENTIAL retrieved chunk about platypus protocol"
    vectors = [
        {"metadata": {"text": chunk_text}, "similarity": 0.9},
        {"metadata": {"text": chunk_text + " II"}, "similarity": 0.8},
        {"metadata": {"text": chunk_text + " III"}, "similarity": 0.7},
    ]

    class FakeStore:
        def find_similar_vectors(self, text, top_k, min_similarity):
            return vectors

    _patch_provider_boundary(monkeypatch)

    req = CompletionRequest(
        username="alice", info=["background secret"], chat=[question]
    )

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        result = complete_chat(
            req, FakeStore(), "OpenAI", "gpt-4o-mini", top_k=3, min_sim=0.42
        )

    assert result["vectors"] == vectors

    record = _the_operational_record(caplog, "completion_retrieval_completed")
    assert record.levelno == logging.INFO
    assert record.name == "services.completion_service"
    assert record.maui_details == {
        "vector_count": len(vectors),
        "top_k": 3,
        "min_sim": 0.42,
        "info_present": True,
    }
    # No provider/model/duration/error metadata belongs to this fact.
    assert not hasattr(record, "maui_provider")
    assert not hasattr(record, "maui_model")
    assert not hasattr(record, "maui_duration_ms")
    assert not hasattr(record, "maui_error_type")

    _assert_free_of(
        record,
        (question, chunk_text, "background secret") + _SENSITIVE_MARKERS,
    )


def test_retrieval_completed_is_persisted_with_zero_vectors_and_no_info(
    monkeypatch, caplog
):
    """T2 — the degraded no-provider path. The fact must be emitted BEFORE the
    no-context early return, because degraded-success reconstruction reads the
    absence of a later provider fact against this record's presence."""
    question = "Totally unanswerable question ZQ99?"

    class FakeStore:
        def find_similar_vectors(self, text, top_k, min_similarity):
            return []

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("provider must not be reached on the early return")

    monkeypatch.setattr(completion_service, "choose_llm", _explode)

    req = CompletionRequest(username="alice", info=[], chat=[question])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        result = complete_chat(req, FakeStore(), "OpenAI", "gpt-4o-mini")

    # Early-return behavior is unchanged.
    assert result["answer"] == "No relevant information available."
    assert result["vectors"] == []
    assert result["is_no_info"] is True

    record = _the_operational_record(caplog, "completion_retrieval_completed")
    assert record.maui_details["vector_count"] == 0
    assert record.maui_details["info_present"] is False
    _assert_free_of(record, (question,) + _SENSITIVE_MARKERS)


def test_retrieval_completed_with_zero_vectors_and_info_present(monkeypatch, caplog):
    """T3 — zero vectors but info present: the ungrounded-completion branch.
    Only the boolean is persisted; the req.info value itself never is."""
    question = "Another question RR11?"
    info_value = "SENSITIVE-INFO-PAYLOAD-7788"

    class FakeStore:
        def find_similar_vectors(self, text, top_k, min_similarity):
            return []

    _patch_provider_boundary(monkeypatch)

    req = CompletionRequest(username="alice", info=[info_value], chat=[question])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        complete_chat(req, FakeStore(), "OpenAI", "gpt-4o-mini")

    record = _the_operational_record(caplog, "completion_retrieval_completed")
    assert record.maui_details["vector_count"] == 0
    assert record.maui_details["info_present"] is True
    _assert_free_of(record, (question, info_value) + _SENSITIVE_MARKERS)


def test_retrieval_failure_persists_error_type_only(monkeypatch, caplog):
    """T4 — retrieval failure emits completion_retrieval_failed with nothing but
    the exception class name, and the existing exception contract of
    complete_chat() is unchanged."""
    question = "Question that triggers a store failure MM55?"
    sentinel = "SENSITIVE-STORE-DSN-secret-token-4242"

    class StoreBoom(Exception):
        pass

    class FakeStore:
        def find_similar_vectors(self, text, top_k, min_similarity):
            raise StoreBoom(f"connection to {sentinel} refused")

    req = CompletionRequest(username="alice", info=["background"], chat=[question])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        with pytest.raises(RuntimeError) as excinfo:
            complete_chat(req, FakeStore(), "OpenAI", "gpt-4o-mini")

    # Unchanged public exception behavior: still RuntimeError, still wrapping
    # the original text on the raise path (that is runtime, not persistence).
    assert "Vector retrieval failed" in str(excinfo.value)
    assert sentinel in str(excinfo.value)

    record = _the_operational_record(caplog, "completion_retrieval_failed")
    assert record.levelno == logging.WARNING
    assert record.maui_error_type == "StoreBoom"
    assert not hasattr(record, "maui_details")
    assert record.getMessage() == "event=completion_retrieval_failed error_type=StoreBoom"

    # The sensitive sentinel must not appear anywhere in the Operational fact.
    _assert_free_of(record, (sentinel, question) + _SENSITIVE_MARKERS)

    # No success fact was emitted for a failed retrieval.
    assert _operational_records(caplog, "completion_retrieval_completed") == []


def test_former_runtime_retrieval_result_log_is_not_also_emitted(monkeypatch, caplog):
    """One real fact, one LogRecord: the persistent event replaced the former
    runtime completion_retrieval_result line rather than doubling it."""

    class FakeStore:
        def find_similar_vectors(self, text, top_k, min_similarity):
            return []

    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *a, **k: _FakeLlm()
    )

    req = CompletionRequest(username="alice", info=[], chat=["a question"])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        complete_chat(req, FakeStore(), "OpenAI", "gpt-4o-mini")

    assert not [
        r for r in caplog.records if "completion_retrieval_result" in r.getMessage()
    ]
    assert len(_operational_records(caplog, "completion_retrieval_completed")) == 1
