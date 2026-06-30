"""Tests for document comparison provider error translation."""

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
