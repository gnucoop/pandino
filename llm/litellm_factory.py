import os
from dataclasses import dataclass

from smolagents import LiteLLMModel


@dataclass(frozen=True)
class LiteLLMConfig:
    provider: str
    configured_model: str
    temperature: float = 0.0


def build_litellm_model(
    *,
    provider: str,
    configured_model: str,
    temperature: float = 0.0,
) -> LiteLLMModel:
    """
    Build a LiteLLMModel using Maui's provider->envvar convention.

    - model_id_for_llm must be: "{provider_lower}/{configured_model}"
    - api_key is read from provider-specific env var (map) or fallback "{PROVIDER}_API_KEY"
    """
    provider_clean = (provider or "").strip()
    model_clean = (configured_model or "").strip()

    if not provider_clean:
        raise ValueError("provider is empty")
    if not model_clean:
        raise ValueError("configured_model is empty")

    provider_env_var_map = {
        "Deepinfra": "DEEPINFRA_API_KEY",
        "Mistral": "MISTRAL_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Groq": "GROQ_API_KEY",
    }

    env_var_name = provider_env_var_map.get(provider_clean, f"{provider_clean.upper()}_API_KEY")
    api_key = os.getenv(env_var_name)

    if not api_key:
        raise ValueError(f"API key not found in env var: {env_var_name}")

    model_id_for_llm = f"{provider_clean.lower()}/{model_clean}"

    return LiteLLMModel(
        model_id=model_id_for_llm,
        api_key=api_key,
        temperature=temperature,
    )
