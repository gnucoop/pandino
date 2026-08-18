"""
ASR provider accounting normalization and cost resolution.

Extracts the accounting quantity needed to bill an ASR transcription from a
provider response payload, and resolves it to an already-resolved monetary
cost. This module is provider-adjacent and intentionally narrow: it does not
perform the HTTP request, write Usage, touch the database, read Flask
context/AppConfig, or select the active provider. Callers own all of that
and pass in whatever this module needs explicitly.

DeepInfra returns an authoritative monetary cost (`inference_status.cost`)
that is passed through unchanged. Mistral reports billing-relevant audio
seconds (`usage.prompt_audio_seconds`) but no monetary cost, so the caller
must supply a governed per-minute rate for Maui-side resolution.
"""

from __future__ import annotations

from typing import Any, Optional


class AsrAccountingError(ValueError):
    """Raised when ASR provider accounting data is missing or malformed."""


def _require_numeric(value: Any, *, field: str) -> float:
    """Validate that `value` is present and a real number (bool excluded).

    Distinguishes a genuinely missing/invalid quantity from a valid zero -
    `0`/`0.0` must not be treated the same as `None`.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AsrAccountingError(f"{field} is missing or not numeric: {value!r}")
    return float(value)


def extract_deepinfra_cost(payload: dict) -> float:
    """Extract the provider-resolved monetary cost from a DeepInfra ASR payload.

    DeepInfra's `inference_status.cost` is authoritative and is never
    recomputed from `duration`/`input_length_ms`.
    """
    inference_status = payload.get("inference_status")
    if not isinstance(inference_status, dict):
        raise AsrAccountingError(
            "DeepInfra ASR payload is missing 'inference_status'"
        )

    return _require_numeric(
        inference_status.get("cost"), field="inference_status.cost"
    )


def extract_mistral_audio_seconds(payload: dict) -> float:
    """Extract the Mistral ASR billing basis: `usage.prompt_audio_seconds`.

    Token metrics reported alongside it (prompt_tokens, completion_tokens,
    total_tokens) are not the billing basis and are not extracted here.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise AsrAccountingError("Mistral ASR payload is missing 'usage'")

    return _require_numeric(
        usage.get("prompt_audio_seconds"), field="usage.prompt_audio_seconds"
    )


def resolve_asr_cost(
    provider: str,
    payload: dict,
    *,
    mistral_price_per_minute: Optional[float] = None,
) -> float:
    """Resolve an ASR transcription response to an already-resolved monetary cost.

    DeepInfra: passes through `inference_status.cost` unchanged.
    Mistral: `prompt_audio_seconds / 60 * mistral_price_per_minute`, using
    the provider-reported quantity as supplied (no rounding/truncation).

    :param provider: ASR provider identifier ('Deepinfra' or 'Mistral').
    :param payload: Parsed JSON body of the provider's ASR response.
    :param mistral_price_per_minute: Governed per-minute rate (USD),
        required only when `provider == 'Mistral'`. Must be supplied by the
        caller - this module never reads configuration itself.
    :raises AsrAccountingError: If the required accounting data is missing,
        malformed, or (for Mistral) the rate was not supplied/is not numeric.
    """
    if provider == "Deepinfra":
        return extract_deepinfra_cost(payload)

    if provider == "Mistral":
        if mistral_price_per_minute is None or isinstance(
            mistral_price_per_minute, bool
        ) or not isinstance(mistral_price_per_minute, (int, float)):
            raise AsrAccountingError(
                "mistral_price_per_minute is required and must be numeric "
                f"to resolve Mistral ASR cost, got: {mistral_price_per_minute!r}"
            )
        seconds = extract_mistral_audio_seconds(payload)
        return (seconds / 60.0) * float(mistral_price_per_minute)

    raise AsrAccountingError(
        f"Unsupported ASR provider for cost resolution: {provider!r}"
    )
