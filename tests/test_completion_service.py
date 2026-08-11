"""
Tests for services.completion_service.complete_chat.

These tests mock the vector store and LLM dependencies so they never perform
real retrieval, network, or provider calls.
"""

import logging
from types import SimpleNamespace

from services import completion_service
from services.completion_service import CompletionRequest, complete_chat


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
