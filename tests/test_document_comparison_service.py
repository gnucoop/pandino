"""Tests for document comparison provider error translation."""

import json
import logging

import pytest

from services import document_comparison_service as comparison_service
from services.document_text_service import NormalizedDocument


class FakeProviderContextError(Exception):
    status_code = 400
    body = {
        "error": {
            "message": (
                "This model's maximum context length is 128000 tokens. "
                "Your prompt contains 606038 input tokens."
            ),
            "param": "input_tokens",
            "code": None,
        }
    }


class FakeProviderUnrelatedError(Exception):
    status_code = 400
    body = {
        "error": {
            "message": "Invalid API key",
            "param": None,
            "code": "invalid_api_key",
        }
    }


class RaisingFakeLlm:
    def __init__(self, error):
        self.error = error

    def invoke(self, messages):
        raise self.error


def _valid_documents() -> list[NormalizedDocument]:
    return [
        {"text": "First document", "filename": "first.txt", "role": "reference"},
        {"text": "Second document", "filename": "second.txt", "role": "candidate"},
    ]


def test_compare_documents_translates_context_window_provider_error(monkeypatch):
    provider_error = FakeProviderContextError("provider rejected prompt")

    monkeypatch.setattr(
        comparison_service,
        "load_prompt",
        lambda title, default_text="": default_text,
    )
    monkeypatch.setattr(
        comparison_service,
        "choose_llm",
        lambda *args, **kwargs: RaisingFakeLlm(provider_error),
    )

    with pytest.raises(
        comparison_service.DocumentComparisonPayloadTooLargeError
    ) as exc:
        comparison_service.compare_documents(
            documents=_valid_documents(),
            prompt="Compare these documents",
            llm_type="OpenAI",
            model="gpt-test",
        )

    assert str(exc.value) == comparison_service.CONTEXT_WINDOW_ERROR_MESSAGE
    assert exc.value.__cause__ is provider_error


def test_compare_documents_preserves_unrelated_provider_error(monkeypatch):
    provider_error = FakeProviderUnrelatedError("provider rejected request")

    monkeypatch.setattr(
        comparison_service,
        "load_prompt",
        lambda title, default_text="": default_text,
    )
    monkeypatch.setattr(
        comparison_service,
        "choose_llm",
        lambda *args, **kwargs: RaisingFakeLlm(provider_error),
    )

    with pytest.raises(FakeProviderUnrelatedError) as exc:
        comparison_service.compare_documents(
            documents=_valid_documents(),
            prompt="Compare these documents",
            llm_type="OpenAI",
            model="gpt-test",
        )

    assert exc.value is provider_error


# ---------------------------------------------------------------------------
# Persistent Operational events — PILOT SLICE P3 (E3, E4)
#
# E3 document_comparison_completed (INFO) fires only after the provider call
# succeeded AND the response parsed AND the schema validated. E4
# document_comparison_payload_too_large (WARNING) fires inside the existing
# context-window classification branch, immediately before the existing raise.
# Assertions are on LogRecord metadata captured via caplog; persistence has its
# own foundation tests.
# ---------------------------------------------------------------------------

SERVICE_LOGGER_NAME = comparison_service.__name__


def _persistent_records(caplog, event=None):
    records = [r for r in caplog.records if getattr(r, "maui_persist", False)]
    if event is not None:
        records = [r for r in records if getattr(r, "maui_event", None) == event]
    return records


def _rendered_text(record):
    """Every place a persisted event's content can surface, as one string."""
    return " ".join(
        [
            record.getMessage(),
            str(getattr(record, "maui_message", "")),
            str(getattr(record, "maui_details", {})),
            str(getattr(record, "maui_error_type", "")),
        ]
    )


class FakeResponse:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class SucceedingFakeLlm:
    def __init__(self, response):
        self.response = response
        self.captured_messages = None

    def invoke(self, messages):
        self.captured_messages = messages
        return self.response


def _patch_provider(monkeypatch, llm):
    monkeypatch.setattr(
        comparison_service,
        "load_prompt",
        lambda title, default_text="": default_text,
    )
    monkeypatch.setattr(
        comparison_service,
        "choose_llm",
        lambda *args, **kwargs: llm,
    )
    return llm


def _valid_response(summary="ok", reasoning="because"):
    return FakeResponse(
        json.dumps({"score": 91, "summary": summary, "reasoning": reasoning}),
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 34,
            "total_tokens": 154,
        },
    )


def test_successful_comparison_emits_one_completed_event(monkeypatch, caplog):
    llm = _patch_provider(monkeypatch, SucceedingFakeLlm(_valid_response()))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        result = comparison_service.compare_documents(
            documents=_valid_documents(),
            prompt="Compare these documents",
            llm_type="OpenAI",
            model="gpt-test",
        )

    assert result["comparison"]["score"] == 91

    completed = _persistent_records(caplog, "document_comparison_completed")
    assert len(completed) == 1

    record = completed[0]
    user_prompt = llm.captured_messages[1]["content"]

    assert record.levelno == logging.INFO
    assert record.name == SERVICE_LOGGER_NAME
    assert record.maui_persist is True
    assert record.maui_provider == "OpenAI"
    assert record.maui_model == "gpt-test"
    assert isinstance(record.maui_duration_ms, int)
    assert record.maui_duration_ms >= 0
    assert record.maui_details == {
        "document_count": 2,
        "prompt_chars": len(user_prompt),
        "input_tokens": 120,
        "output_tokens": 34,
        "total_tokens": 154,
    }
    assert "score" not in record.maui_details
    assert not hasattr(record, "maui_error_type")
    assert not hasattr(record, "maui_message")


def test_completed_event_persists_no_business_result_or_document_content(
    monkeypatch, caplog
):
    _patch_provider(
        monkeypatch,
        SucceedingFakeLlm(
            _valid_response(
                summary="SENTINEL-SUMMARY-do-not-persist",
                reasoning="SENTINEL-REASONING-do-not-persist",
            )
        ),
    )
    documents: list[NormalizedDocument] = [
        {
            "text": "SENTINEL-DOCUMENT-ONE",
            "filename": "SENTINEL-FIRST.txt",
            "role": "reference",
        },
        {
            "text": "SENTINEL-DOCUMENT-TWO",
            "filename": "SENTINEL-SECOND.txt",
            "role": "candidate",
        },
    ]

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        comparison_service.compare_documents(
            documents=documents,
            prompt="SENTINEL-PROMPT-do-not-persist",
            llm_type="OpenAI",
            model="gpt-test",
            additional_context="SENTINEL-CONTEXT-do-not-persist",
            language="SENTINEL-LANGUAGE-do-not-persist",
        )

    record = _persistent_records(caplog, "document_comparison_completed")[0]

    assert "SENTINEL" not in _rendered_text(record)
    assert set(record.maui_details) == {
        "document_count",
        "prompt_chars",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        json.dumps([1, 2, 3]),
        json.dumps({"score": "high", "summary": "ok", "reasoning": "because"}),
        json.dumps({"score": 101, "summary": "ok", "reasoning": "because"}),
        json.dumps({"score": 50, "summary": "  ", "reasoning": "because"}),
        json.dumps({"score": 50, "summary": "ok", "reasoning": ""}),
    ],
)
def test_invalid_model_response_emits_no_completed_event(
    monkeypatch, caplog, content
):
    _patch_provider(monkeypatch, SucceedingFakeLlm(FakeResponse(content)))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        with pytest.raises(ValueError):
            comparison_service.compare_documents(
                documents=_valid_documents(),
                prompt="Compare these documents",
                llm_type="OpenAI",
                model="gpt-test",
            )

    assert _persistent_records(caplog) == []


def test_provider_failure_emits_no_completed_event(monkeypatch, caplog):
    _patch_provider(
        monkeypatch, RaisingFakeLlm(FakeProviderUnrelatedError("provider rejected"))
    )

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        with pytest.raises(FakeProviderUnrelatedError):
            comparison_service.compare_documents(
                documents=_valid_documents(),
                prompt="Compare these documents",
                llm_type="OpenAI",
                model="gpt-test",
            )

    assert _persistent_records(caplog, "document_comparison_completed") == []


def test_context_window_error_emits_one_payload_too_large_event(monkeypatch, caplog):
    provider_error = FakeProviderContextError(
        "SENTINEL-PROVIDER-MESSAGE-do-not-persist"
    )
    _patch_provider(monkeypatch, RaisingFakeLlm(provider_error))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        with pytest.raises(
            comparison_service.DocumentComparisonPayloadTooLargeError
        ) as exc:
            comparison_service.compare_documents(
                documents=_valid_documents(),
                prompt="Compare these documents",
                llm_type="OpenAI",
                model="gpt-test",
            )

    # The existing translation behavior is unchanged.
    assert str(exc.value) == comparison_service.CONTEXT_WINDOW_ERROR_MESSAGE
    assert exc.value.__cause__ is provider_error

    too_large = _persistent_records(
        caplog, "document_comparison_payload_too_large"
    )
    assert len(too_large) == 1

    record = too_large[0]
    assert record.levelno == logging.WARNING
    assert record.name == SERVICE_LOGGER_NAME
    assert record.maui_persist is True
    assert record.maui_provider == "OpenAI"
    assert record.maui_model == "gpt-test"
    assert isinstance(record.maui_duration_ms, int)
    assert record.maui_duration_ms >= 0
    assert record.maui_error_type == "FakeProviderContextError"
    assert record.maui_details["document_count"] == 2
    assert isinstance(record.maui_details["prompt_chars"], int)
    assert record.maui_details["prompt_chars"] > 0
    assert set(record.maui_details) == {"document_count", "prompt_chars"}

    # Neither the provider's message nor Maui's own constant is persisted.
    rendered = _rendered_text(record)
    assert "SENTINEL" not in rendered
    assert comparison_service.CONTEXT_WINDOW_ERROR_MESSAGE not in rendered
    assert not hasattr(record, "maui_message")

    # A failed comparison never also reports completion.
    assert _persistent_records(caplog, "document_comparison_completed") == []


def test_unrelated_provider_error_emits_no_payload_too_large_event(
    monkeypatch, caplog
):
    provider_error = FakeProviderUnrelatedError(
        "SENTINEL-PROVIDER-MESSAGE-do-not-persist"
    )
    _patch_provider(monkeypatch, RaisingFakeLlm(provider_error))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        with pytest.raises(FakeProviderUnrelatedError) as exc:
            comparison_service.compare_documents(
                documents=_valid_documents(),
                prompt="Compare these documents",
                llm_type="OpenAI",
                model="gpt-test",
            )

    assert exc.value is provider_error
    # Generic provider failures are deliberately OUT OF PILOT in P3.
    assert _persistent_records(caplog) == []


def test_client_validation_errors_emit_no_events(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        with pytest.raises(ValueError):
            comparison_service.compare_documents(
                documents=_valid_documents()[:1],
                prompt="Compare these documents",
                llm_type="OpenAI",
                model="gpt-test",
            )

    assert _persistent_records(caplog) == []


def test_comparison_call_sites_declare_no_context_ownership_keys(
    monkeypatch, caplog
):
    _patch_provider(monkeypatch, SucceedingFakeLlm(_valid_response()))

    with caplog.at_level(logging.INFO, logger=SERVICE_LOGGER_NAME):
        comparison_service.compare_documents(
            documents=_valid_documents(),
            prompt="Compare these documents",
            llm_type="OpenAI",
            model="gpt-test",
        )

    record = _persistent_records(caplog, "document_comparison_completed")[0]
    maui_keys = {key for key in vars(record) if key.startswith("maui_")}

    assert maui_keys == {
        "maui_persist",
        "maui_event",
        "maui_provider",
        "maui_model",
        "maui_duration_ms",
        "maui_details",
    }
    assert not hasattr(record, "request_id")
    assert not hasattr(record, "app_id")
