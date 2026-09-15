"""
Usage Slice 2 - ASR accounting normalization and cost resolution.

Covers infrastructure/asr_accounting.py in isolation from any HTTP request,
Flask context, AppConfig, or database access: it consumes plain dict
payloads representative of DeepInfra/Mistral ASR responses.
"""

import pytest

from infrastructure.asr_accounting import (
    AsrAccountingError,
    extract_deepinfra_cost,
    extract_mistral_audio_seconds,
    resolve_asr_cost,
)


# ---------------------------------------------------------------------------
# DeepInfra
# ---------------------------------------------------------------------------


def test_deepinfra_extracts_provider_reported_cost():
    payload = {
        "input_length_ms": 30000,
        "duration": 30.0,
        "inference_status": {"status": "succeeded", "cost": 0.0225},
    }

    assert extract_deepinfra_cost(payload) == pytest.approx(0.0225)


def test_deepinfra_missing_cost_fails_explicitly():
    payload = {"inference_status": {"status": "succeeded"}}

    with pytest.raises(AsrAccountingError):
        extract_deepinfra_cost(payload)


def test_deepinfra_missing_inference_status_fails_explicitly():
    payload = {"duration": 30.0}

    with pytest.raises(AsrAccountingError):
        extract_deepinfra_cost(payload)


def test_deepinfra_zero_cost_is_accepted_as_valid():
    payload = {"inference_status": {"cost": 0}}

    assert extract_deepinfra_cost(payload) == 0.0


def test_deepinfra_none_cost_fails_explicitly():
    payload = {"inference_status": {"cost": None}}

    with pytest.raises(AsrAccountingError):
        extract_deepinfra_cost(payload)


def test_resolve_asr_cost_deepinfra_passes_through():
    payload = {"inference_status": {"cost": 0.0225}}

    assert resolve_asr_cost("Deepinfra", payload) == pytest.approx(0.0225)


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------


def test_mistral_extracts_prompt_audio_seconds():
    payload = {
        "text": "hello",
        "usage": {
            "prompt_audio_seconds": 30,
            "prompt_tokens": 4,
            "completion_tokens": 88,
            "total_tokens": 92,
            "request_count": 1,
        },
    }

    assert extract_mistral_audio_seconds(payload) == 30.0


def test_mistral_missing_prompt_audio_seconds_fails_explicitly():
    payload = {"usage": {"prompt_tokens": 4, "total_tokens": 92}}

    with pytest.raises(AsrAccountingError):
        extract_mistral_audio_seconds(payload)


def test_mistral_missing_usage_fails_explicitly():
    payload = {"text": "hello"}

    with pytest.raises(AsrAccountingError):
        extract_mistral_audio_seconds(payload)


def test_mistral_zero_prompt_audio_seconds_is_valid():
    payload = {"usage": {"prompt_audio_seconds": 0}}

    assert extract_mistral_audio_seconds(payload) == 0.0


def test_resolve_asr_cost_mistral_computes_seconds_over_sixty_times_rate():
    payload = {"usage": {"prompt_audio_seconds": 30}}

    cost = resolve_asr_cost("Mistral", payload, mistral_price_per_minute=0.003)

    assert cost == pytest.approx(0.0015)


def test_resolve_asr_cost_mistral_zero_seconds_resolves_to_zero_cost():
    payload = {"usage": {"prompt_audio_seconds": 0}}

    cost = resolve_asr_cost("Mistral", payload, mistral_price_per_minute=0.003)

    assert cost == 0.0


def test_resolve_asr_cost_mistral_missing_prompt_audio_seconds_fails_explicitly():
    payload = {"usage": {"total_tokens": 92}}

    with pytest.raises(AsrAccountingError):
        resolve_asr_cost("Mistral", payload, mistral_price_per_minute=0.003)


def test_resolve_asr_cost_mistral_missing_rate_fails_explicitly():
    payload = {"usage": {"prompt_audio_seconds": 30}}

    with pytest.raises(AsrAccountingError):
        resolve_asr_cost("Mistral", payload)


def test_resolve_asr_cost_mistral_invalid_rate_fails_explicitly():
    payload = {"usage": {"prompt_audio_seconds": 30}}

    with pytest.raises(AsrAccountingError):
        resolve_asr_cost("Mistral", payload, mistral_price_per_minute="not-a-number")


# ---------------------------------------------------------------------------
# Unsupported provider
# ---------------------------------------------------------------------------


def test_resolve_asr_cost_unsupported_provider_fails_explicitly():
    with pytest.raises(AsrAccountingError):
        resolve_asr_cost("SelfHosted", {"text": "hello"})
