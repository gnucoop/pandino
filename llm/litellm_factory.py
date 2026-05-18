import os
from dataclasses import dataclass
from typing import Optional

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
    api_key: Optional[str] = None,
) -> LiteLLMModel:
    """
    Build a LiteLLMModel using Maui's provider->envvar convention.

    - model_id_for_llm must be: "{provider_lower}/{configured_model}"
    - api_key: explicit key to use. When None, the key is resolved from the
      environment via PROVIDER_API_KEY_MAP (imported from config) or the
      fallback pattern "{PROVIDER}_API_KEY".
    """
    provider_clean = (provider or "").strip()
    model_clean = (configured_model or "").strip()

    if not provider_clean:
        raise ValueError("provider is empty")
    if not model_clean:
        raise ValueError("configured_model is empty")

    env_var_name: Optional[str] = None
    if api_key is None:
        from config import PROVIDER_API_KEY_MAP
        env_var_name = PROVIDER_API_KEY_MAP.get(provider_clean, f"{provider_clean.upper()}_API_KEY")
        api_key = os.getenv(env_var_name)

    if not api_key:
        raise ValueError(f"API key not found in env var: {env_var_name}")

    model_id_for_llm = f"{provider_clean.lower()}/{model_clean}"

    return LiteLLMModel(
        model_id=model_id_for_llm,
        api_key=api_key,
        temperature=temperature,
    )
