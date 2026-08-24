"""
Tests for services.completion_service.complete_chat.

These tests mock the vector store and LLM dependencies so they never perform
real retrieval, network, or provider calls.
"""

import logging
import time
from types import SimpleNamespace

import pytest

from services import completion_service
from services.completion_service import CompletionRequest, complete_chat
from utils.logging_config import LOG_FORMAT, ContextDefaultsFilter, UtcIsoFormatter
from utils.operational_persistence import (
    OperationalPersistenceHandler,
    snapshot_from_record,
)


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


# ---------------------------------------------------------------------------
# SECOND ADOPTER SLICE C2 — persistent Operational provider facts.
#
# Facts under test:
#   completion_provider_completed  (INFO,  provider/model/duration_ms/is_no_info)
#   completion_provider_failed     (ERROR, provider/model/duration_ms/error_type)
#
# duration_ms brackets llm.invoke() and nothing else, so the timing source is
# monkeypatched with a deterministic sequence rather than measured from the
# wall clock.
# ---------------------------------------------------------------------------


# "info=" is deliberately absent here: the provider facts legitimately render
# the classifier boolean as "is_no_info=…", which is not req.info content.
_PROVIDER_SENSITIVE_MARKERS = ("question=", "text=", "chat=", "namespace=")


class _FakeClock:
    """Deterministic perf_counter replacement.

    Returns the scripted values in order and then holds the last one, so the
    number of unrelated perf_counter() reads elsewhere cannot make the test
    brittle — only the reads bracketing invoke() are scripted.
    """

    def __init__(self, values):
        self._values = list(values)
        self.reads = 0

    def __call__(self):
        self.reads += 1
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


def _patch_prompt_boundary(monkeypatch):
    monkeypatch.setattr(
        completion_service, "load_prompt", lambda *args, **kwargs: "system prompt"
    )
    monkeypatch.setattr(
        completion_service, "render_prompt", lambda template, **kwargs: template
    )


class _VectorStoreReturning:
    def __init__(self, vectors):
        self._vectors = vectors

    def find_similar_vectors(self, text, top_k, min_similarity):
        return self._vectors


def test_provider_completed_is_persisted_with_deterministic_duration(
    monkeypatch, caplog
):
    """T1 — a successful provider call emits exactly one marked
    completion_provider_completed carrying provider, model, the invoke-only
    duration and the classifier verdict, and nothing else."""
    question = "What is the sky-blue platypus protocol XK42?"
    chunk_text = "CONFIDENTIAL retrieved chunk"
    answer_text = "SENSITIVE-ANSWER-CONTENT-9911"

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(
                content=answer_text,
                response_metadata={
                    "token_usage": {"prompt_tokens": 11, "completion_tokens": 7}
                },
            )

    _patch_prompt_boundary(monkeypatch)
    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *a, **k: FakeLlm()
    )
    monkeypatch.setattr(time, "perf_counter", _FakeClock([10.0, 10.25]))

    req = CompletionRequest(
        username="alice", info=["background secret"], chat=[question]
    )
    store = _VectorStoreReturning([{"metadata": {"text": chunk_text}}])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        result = complete_chat(req, store, "OpenAI", "gpt-4o-mini")

    # Returned structure is unchanged by the instrumentation.
    assert result["answer"] == answer_text
    assert result["token_usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert result["is_no_info"] is False

    record = _the_operational_record(caplog, "completion_provider_completed")
    assert record.levelno == logging.INFO
    assert record.name == "services.completion_service"
    assert record.maui_provider == "OpenAI"
    assert record.maui_model == "gpt-4o-mini"
    assert record.maui_duration_ms == 250
    assert record.maui_details == {"is_no_info": False}
    # Usage owns token counts; no error metadata belongs to a success fact.
    assert not hasattr(record, "maui_error_type")
    assert "tokens" not in record.getMessage()

    # Neither the legacy runtime event nor the failure fact was emitted.
    assert not [
        r for r in caplog.records if "event=completion_failed" in r.getMessage()
    ]
    assert _operational_records(caplog, "completion_provider_failed") == []

    _assert_free_of(
        record,
        (question, chunk_text, answer_text, "background secret")
        + _PROVIDER_SENSITIVE_MARKERS,
    )


def test_provider_completed_carries_the_classifier_verdict(monkeypatch, caplog):
    """is_no_info is the current Maui classifier's boolean result and is
    persisted as a boolean only — never the matched phrase."""
    no_info_answer = "I have no information about this."

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(content=no_info_answer, response_metadata={})

    _patch_prompt_boundary(monkeypatch)
    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *a, **k: FakeLlm()
    )

    req = CompletionRequest(username="alice", info=["info"], chat=["a question"])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        result = complete_chat(req, _VectorStoreReturning([]), "Anthropic", "sonnet")

    assert result["is_no_info"] is True

    record = _the_operational_record(caplog, "completion_provider_completed")
    assert record.maui_details == {"is_no_info": True}
    assert record.maui_provider == "Anthropic"
    assert record.maui_model == "sonnet"
    _assert_free_of(record, (no_info_answer, "no information"))


def test_provider_duration_measures_only_the_invoke_interval(monkeypatch, caplog):
    """T-boundary — retrieval, prompt assembly, choose_llm, response parsing
    and the classifier all consume clock time in this test, yet duration_ms
    reflects exactly the invoke bracket."""
    ticks = iter([1.0, 2.0, 3.0, 100.000, 100.400, 500.0, 900.0])
    last = [0.0]

    def fake_perf_counter():
        try:
            last[0] = next(ticks)
        except StopIteration:
            pass
        return last[0]

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(content="an answer", response_metadata={})

    def slow_choose_llm(*args, **kwargs):
        # Burn scripted clock time BEFORE the invoke bracket opens.
        fake_perf_counter()
        fake_perf_counter()
        fake_perf_counter()
        return FakeLlm()

    _patch_prompt_boundary(monkeypatch)
    monkeypatch.setattr(completion_service, "choose_llm", slow_choose_llm)
    monkeypatch.setattr(time, "perf_counter", fake_perf_counter)

    req = CompletionRequest(username="alice", info=["info"], chat=["a question"])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        complete_chat(req, _VectorStoreReturning([]), "OpenAI", "gpt-4o-mini")

    record = _the_operational_record(caplog, "completion_provider_completed")
    assert record.maui_duration_ms == 400


def test_provider_failure_persists_error_type_only(monkeypatch, caplog):
    """T2 — when invoke() raises, exactly one marked completion_provider_failed
    is emitted at ERROR with the exception class name only, the legacy
    completion_failed line is gone, and exception propagation is unchanged."""
    sentinel = "SENSITIVE-PROVIDER-BODY-secret-4242"

    class ProviderBoom(Exception):
        pass

    class FakeLlm:
        def invoke(self, messages):
            raise ProviderBoom(f"upstream said {sentinel}")

    _patch_prompt_boundary(monkeypatch)
    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *a, **k: FakeLlm()
    )
    monkeypatch.setattr(time, "perf_counter", _FakeClock([5.0, 5.125]))

    req = CompletionRequest(username="alice", info=["info"], chat=["a question"])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        with pytest.raises(RuntimeError) as excinfo:
            complete_chat(req, _VectorStoreReturning([]), "OpenAI", "gpt-4o-mini")

    # Unchanged public exception behavior.
    assert "Chat completion failed" in str(excinfo.value)
    assert sentinel in str(excinfo.value)

    record = _the_operational_record(caplog, "completion_provider_failed")
    assert record.levelno == logging.ERROR
    assert record.maui_provider == "OpenAI"
    assert record.maui_model == "gpt-4o-mini"
    assert record.maui_duration_ms == 125
    assert record.maui_error_type == "ProviderBoom"
    assert not hasattr(record, "maui_details")
    assert record.getMessage() == (
        "event=completion_provider_failed provider=OpenAI model=gpt-4o-mini "
        "duration_ms=125 error_type=ProviderBoom"
    )

    _assert_free_of(record, (sentinel,) + _PROVIDER_SENSITIVE_MARKERS)

    # One boundary, one record: no legacy event, no duplicate provider fact.
    assert not [
        r for r in caplog.records if "event=completion_failed" in r.getMessage()
    ]
    assert _operational_records(caplog, "completion_provider_completed") == []
    # C1 still emitted its retrieval fact before the provider seam.
    assert len(_operational_records(caplog, "completion_retrieval_completed")) == 1


def test_pre_invoke_failure_is_not_a_provider_failure(monkeypatch, caplog):
    """The Fact D boundary: choose_llm() failing means the provider attempt
    never started, so no provider Operational fact may exist."""

    class SelectionBoom(Exception):
        pass

    def exploding_choose_llm(*args, **kwargs):
        raise SelectionBoom("unsupported llm type")

    _patch_prompt_boundary(monkeypatch)
    monkeypatch.setattr(completion_service, "choose_llm", exploding_choose_llm)

    req = CompletionRequest(username="alice", info=["info"], chat=["a question"])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        with pytest.raises(RuntimeError) as excinfo:
            complete_chat(req, _VectorStoreReturning([]), "OpenAI", "gpt-4o-mini")

    assert "Chat completion failed" in str(excinfo.value)
    assert "unsupported llm type" in str(excinfo.value)

    assert _operational_records(caplog, "completion_provider_failed") == []
    assert _operational_records(caplog, "completion_provider_completed") == []
    # The retrieval fact from C1 is unaffected.
    assert len(_operational_records(caplog, "completion_retrieval_completed")) == 1


def test_successful_flow_emits_retrieval_then_provider_fact(monkeypatch, caplog):
    """C1 regression: the expected ordered timeline for a normal success."""

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(content="an answer", response_metadata={})

    _patch_prompt_boundary(monkeypatch)
    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *a, **k: FakeLlm()
    )

    req = CompletionRequest(username="alice", info=["info"], chat=["a question"])

    with caplog.at_level(logging.INFO, logger="services.completion_service"):
        complete_chat(req, _VectorStoreReturning([]), "OpenAI", "gpt-4o-mini")

    events = [
        r.maui_event
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
    ]
    assert events == [
        "completion_retrieval_completed",
        "completion_provider_completed",
    ]


class _ListSink:
    def __init__(self):
        self.received = []

    def __call__(self, snapshot):
        self.received.append(snapshot)


def test_real_logger_exception_splits_traceback_from_snapshot(monkeypatch):
    """§12.1 regression — the production emission shape.

    One real logger.exception(message, extra=extra) call, raised from a real
    except block, must give stderr the full traceback while the Operational
    snapshot keeps only bounded fields. Both halves are asserted, because
    either alone would pass while the split silently broke.
    """
    sentinel = "SENSITIVE-PROVIDER-BODY-secret-4242"

    class ProviderBoom(Exception):
        pass

    class FakeLlm:
        def invoke(self, messages):
            raise ProviderBoom(f"upstream said {sentinel}")

    _patch_prompt_boundary(monkeypatch)
    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *a, **k: FakeLlm()
    )

    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    service_logger = logging.getLogger("services.completion_service")

    captured_records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    capture = _Capture()
    capture.addFilter(ContextDefaultsFilter())

    previous_level = service_logger.level
    service_logger.addHandler(handler)
    service_logger.addHandler(capture)
    service_logger.setLevel(logging.INFO)
    try:
        req = CompletionRequest(
            username="alice", info=["info"], chat=["a question"]
        )
        with pytest.raises(RuntimeError):
            complete_chat(
                req, _VectorStoreReturning([]), "OpenAI", "gpt-4o-mini"
            )
    finally:
        service_logger.removeHandler(handler)
        service_logger.removeHandler(capture)
        service_logger.setLevel(previous_level)

    failure_records = [
        r
        for r in captured_records
        if getattr(r, "maui_event", None) == "completion_provider_failed"
    ]
    assert len(failure_records) == 1
    record = failure_records[0]

    # --- stderr half: the real formatter keeps the traceback -----------------
    formatted = UtcIsoFormatter(LOG_FORMAT).format(record)
    assert "Traceback" in formatted
    assert "ProviderBoom" in formatted
    assert sentinel in formatted
    # Formatting has now mutated the shared record by caching exc_text.
    assert record.exc_text and sentinel in record.exc_text

    # --- Operational half: same record, bounded snapshot ---------------------
    provider_snapshots = [
        s for s in sink.received if s.event == "completion_provider_failed"
    ]
    assert len(provider_snapshots) == 1
    snapshot = provider_snapshots[0]
    assert snapshot.error_type == "ProviderBoom"
    assert snapshot.level == "ERROR"
    assert snapshot.logger == "services.completion_service"
    assert snapshot.provider == "OpenAI"
    assert snapshot.model == "gpt-4o-mini"
    assert isinstance(snapshot.duration_ms, int) and snapshot.duration_ms >= 0
    assert not hasattr(snapshot, "exc_info")
    assert not hasattr(snapshot, "traceback")
    for value in vars(snapshot).values() if hasattr(snapshot, "__dict__") else [
        getattr(snapshot, field) for field in snapshot.__slots__
    ]:
        text = str(value)
        assert sentinel not in text
        assert "Traceback" not in text

    # The snapshot taken again AFTER stderr formatting is still clean, proving
    # the shared-record exc_text mutation cannot leak into persistence.
    post_format_snapshot = snapshot_from_record(record)
    assert post_format_snapshot.error_type == "ProviderBoom"
    assert sentinel not in str(post_format_snapshot)
    assert "Traceback" not in str(post_format_snapshot)
