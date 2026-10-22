"""Tests for the LiteLLM model factory shared by the smolagents engines."""

from llm.litellm_factory import build_litellm_model


def test_mistral_greedy_sampling_sends_top_p_1():
    """Mistral refuses temperature 0 unless top_p is 1 (error 3054 on newer models)."""
    model = build_litellm_model(provider="Mistral", configured_model="mistral-large-4", api_key="k")

    assert model.model_id == "mistral/mistral-large-4"
    assert model.kwargs == {"temperature": 0.0, "top_p": 1.0}


def test_mistral_sampling_above_zero_leaves_top_p_alone():
    model = build_litellm_model(
        provider="Mistral", configured_model="mistral-large-4", api_key="k", temperature=0.7
    )
    assert "top_p" not in model.kwargs


def test_other_providers_keep_their_request_unchanged():
    model = build_litellm_model(provider="Deepinfra", configured_model="some/model", api_key="k")
    assert model.kwargs == {"temperature": 0.0}


def test_timeout_and_retries_reach_the_model_only_when_given():
    model = build_litellm_model(
        provider="Deepinfra", configured_model="some/model", api_key="k", timeout=180, num_retries=2
    )
    assert model.kwargs == {"temperature": 0.0, "timeout": 180, "num_retries": 2}
