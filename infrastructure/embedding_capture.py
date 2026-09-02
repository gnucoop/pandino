"""Provider-adjacent embedding accounting capture (§13, DC1).

Extracts authoritative accounting metadata from a *native* provider
embedding response and normalizes it into exactly one
:class:`~utils.embedding_accounting.EmbeddingAccountingContribution` per
successful provider accounting response, delivered through the sink
ContextVar the foundation already publishes.

Currently DeepInfra only: it is the configured default provider and the only
one with `[V-RUNTIME]` provider-authoritative cost (§5.1). Other providers
have their own verified seams (§13) and are deliberately not implemented
here.

Narrow on purpose, in the shape ``infrastructure/asr_accounting.py``
established: this module does not write Usage, touch the database, read
Flask context or AppConfig, or select the active provider. It is
Flask-blind — it imports nothing from Flask — because capture runs on the
far side of ``PGVectorStore``'s background event-loop thread.

Observation-only: accounting capture must never turn a
successful embedding call into a user-visible failure. Every capture
failure degrades to "no contribution, one safe warning", and the vectors
the provider library computed are returned unchanged.
"""

import logging
from typing import Any, List, Optional

import requests
from langchain_community.embeddings import DeepInfraEmbeddings

from utils.embedding_accounting import (
    COST_PROVIDER_AUTHORITATIVE,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_UNIT_INPUT_TOKENS,
    EmbeddingAccountingContribution,
)
from utils.embedding_accounting_sink import get_embedding_accounting_sink
from utils.embedding_operation_context import get_embedding_operation

__all__ = [
    "EmbeddingCaptureError",
    "PROVIDER_DEEPINFRA",
    "extract_deepinfra_contribution",
    "DeepInfraAccountingEmbeddings",
]

logger = logging.getLogger(__name__)

#: Provider identity as Maui's configuration spells it
#: (``emb_llm_type == "Deepinfra"``, ``infrastructure/ai.py``), so the
#: normalized contribution carries the same identity the rest of Maui uses.
PROVIDER_DEEPINFRA = "Deepinfra"


class EmbeddingCaptureError(ValueError):
    """Raised when provider accounting data is missing or malformed.

    Never escapes to the caller of an embedding method: the capture wrapper
    catches it and skips the contribution.
    """


def _require_int(value: Any, *, field: str) -> int:
    """Validate a non-negative integer quantity.

    ``bool`` is rejected explicitly (it is an ``int`` subclass; a flag is
    not a quantity), and a ``float`` is accepted only when it is exactly
    integral — a fractional token count would be a shape Maui does not
    understand, not something to round.
    """
    if value is None or isinstance(value, bool):
        raise EmbeddingCaptureError(f"{field} is missing or not numeric")
    if isinstance(value, int):
        quantity = value
    elif isinstance(value, float) and value.is_integer():
        quantity = int(value)
    else:
        raise EmbeddingCaptureError(f"{field} is missing or not numeric")
    if quantity < 0:
        raise EmbeddingCaptureError(f"{field} is negative")
    return quantity


def _require_numeric(value: Any, *, field: str) -> float:
    """Validate a present, non-negative real number (``bool`` excluded).

    A valid ``0.0`` must stay distinguishable from absence, the same
    distinction ``infrastructure/asr_accounting.py`` draws.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbeddingCaptureError(f"{field} is missing or not numeric")
    if value < 0:
        raise EmbeddingCaptureError(f"{field} is negative")
    return float(value)


def _optional_int(value: Any) -> Optional[int]:
    """Best-effort read of an optional integer field, else ``None``.

    Used for metadata that is not part of the accounting authority: losing
    it must not cost the whole contribution.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value) if value >= 0 else None


def _optional_identity(value: Any) -> Optional[str]:
    """Best-effort read of an optional non-empty string field, else ``None``."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def extract_deepinfra_contribution(
    payload: dict,
    *,
    model: str,
    operation_kind: str,
) -> EmbeddingAccountingContribution:
    """Normalize one native DeepInfra embedding response into a contribution.

    Authority: the input quantity is the provider's own
    ``input_tokens`` — never a Maui-side count, which would also be wrong,
    since DeepInfra counts the ``passage: ``/``query: `` prefixes the
    library prepends. The monetary cost is
    ``inference_status.cost``, passed through unchanged and never recomputed.

    ``inference_status.tokens_input`` duplicated ``input_tokens`` in the
    runtime probe. It is therefore used only as a *disagreement check*: on
    mismatch this raises rather than silently picking one or merging them.
    ``request_id`` and ``inference_status.runtime_ms`` are optional
    metadata.

    No native field name, and nothing from ``embeddings``, the request body
    or the raw payload, survives into the returned contribution.

    :raises EmbeddingCaptureError: if the authoritative fields are absent,
        malformed, or mutually contradictory.
    """
    if not isinstance(payload, dict):
        raise EmbeddingCaptureError("DeepInfra embedding payload is not an object")

    inference_status = payload.get("inference_status")
    if not isinstance(inference_status, dict):
        raise EmbeddingCaptureError(
            "DeepInfra embedding payload is missing 'inference_status'"
        )

    input_quantity = _require_int(payload.get("input_tokens"), field="input_tokens")

    corroborating = inference_status.get("tokens_input")
    if corroborating is not None:
        if (
            _require_int(corroborating, field="inference_status.tokens_input")
            != input_quantity
        ):
            raise EmbeddingCaptureError(
                "DeepInfra input token counts disagree between "
                "'input_tokens' and 'inference_status.tokens_input'"
            )

    provider_cost = _require_numeric(
        inference_status.get("cost"), field="inference_status.cost"
    )

    return EmbeddingAccountingContribution(
        provider=PROVIDER_DEEPINFRA,
        model=model,
        input_quantity=input_quantity,
        quantity_unit=QUANTITY_UNIT_INPUT_TOKENS,
        quantity_origin=ORIGIN_PROVIDER_REPORTED,
        cost_state=COST_PROVIDER_AUTHORITATIVE,
        operation_kind=operation_kind,
        provider_cost=provider_cost,
        provider_request_id=_optional_identity(payload.get("request_id")),
        provider_runtime_ms=_optional_int(inference_status.get("runtime_ms")),
    )


class DeepInfraAccountingEmbeddings(DeepInfraEmbeddings):
    """``DeepInfraEmbeddings`` that also reports what it consumed.

    Capture sits in ``_embed``, the sole provider choke point for both
    ``embed_documents`` and ``embed_query``. Neither of those methods
    is overridden, so batching (``batch_size``), the ``passage: ``/``query: ``
    prefixes, the request body, the endpoint and the error semantics remain
    entirely the parent's — and because ``langchain_core``'s
    ``aembed_query``/``aembed_documents`` fall back to the sync methods via
    ``run_in_executor``, this one override covers the async path
    ``PGVectorStore`` uses, asserted by test rather than assumed.

    The override reproduces the parent's request/parse/raise sequence
    verbatim rather than delegating to it, because the parent returns only
    ``t["embeddings"]`` and discards the accounting metadata in the same
    expression. One provider call is made, exactly as before; the payload is
    read once, in-process.
    """

    def _embed(self, input: List[str]) -> List[List[float]]:
        _model_kwargs = self.model_kwargs or {}
        # HTTP headers for authorization
        headers = {
            "Authorization": f"bearer {self.deepinfra_api_token}",
            "Content-Type": "application/json",
        }
        # send request
        try:
            res = requests.post(
                f"https://api.deepinfra.com/v1/inference/{self.model_id}",
                headers=headers,
                json={"inputs": input, "normalize": self.normalize, **_model_kwargs},
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (res.status_code, res.text)
            )
        try:
            t = res.json()
            embeddings = t["embeddings"]
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )

        # Accounting is observed after the vectors are in hand, and can only
        # ever be skipped — never propagated to the caller.
        self._emit_accounting_contribution(t)

        return embeddings

    def _emit_accounting_contribution(self, payload: Any) -> None:
        """Normalize ``payload`` and hand one contribution to the sink.

        Skips silently-but-diagnosably in the two cases §11 and §12 name:

        * **No operation context.** ``operation_kind`` is required by the
          contract and comes from ambient context, so there is
          nothing legitimate to record: inventing a kind would fabricate an
          attribution. Debug level, not warning — running outside a bound
          operation is a legitimate state, not a defect: direct construction
          and reusable non-HTTP infrastructure can embed with no ambient
          operation to report.
        * **Malformed accounting.** The vectors are valid and were billed;
          only the observation is unusable. Warning level, naming provider,
          model and the failure reason only — never the payload, the input
          texts or the vectors.

        The broad ``except Exception`` is deliberate: this runs on the far
        side of a thread hop, and no accounting defect may surface as an
        embedding failure.
        """
        operation_kind = get_embedding_operation()
        if operation_kind is None:
            logger.debug(
                "event=embedding_accounting_skipped_no_context provider=%s model=%s",
                PROVIDER_DEEPINFRA,
                self.model_id,
            )
            return

        try:
            contribution = extract_deepinfra_contribution(
                payload, model=self.model_id, operation_kind=operation_kind
            )
        except Exception as e:
            logger.warning(
                "event=embedding_accounting_capture_failed provider=%s model=%s "
                "error_type=%s",
                PROVIDER_DEEPINFRA,
                self.model_id,
                type(e).__name__,
            )
            return

        try:
            get_embedding_accounting_sink()(contribution)
        except Exception as e:
            logger.warning(
                "event=embedding_accounting_delivery_failed provider=%s model=%s "
                "error_type=%s",
                PROVIDER_DEEPINFRA,
                self.model_id,
                type(e).__name__,
            )
