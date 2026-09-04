"""Embedding accounting Slice 2 - DeepInfra provider capture.

Covers infrastructure/embedding_capture.py: extraction/normalization of the
native DeepInfra embedding payload (§5.1 runtime evidence), emission of one
contribution per provider response (DC3), and - equally load-bearing - the
guarantee that none of this changes embedding behaviour (DC2, §21.4).

No live provider call is made anywhere: ``requests.post`` inside the
capture module is replaced by a fake that records the request bodies it was
handed and returns canned native payloads.
"""

import asyncio
import logging
import threading

import pytest
import requests

from infrastructure.embedding_capture import (
    PROVIDER_DEEPINFRA,
    DeepInfraAccountingEmbeddings,
    EmbeddingCaptureError,
    extract_deepinfra_contribution,
)
from usage.embedding_accounting import (
    COST_PROVIDER_AUTHORITATIVE,
    ORIGIN_PROVIDER_REPORTED,
    QUANTITY_UNIT_INPUT_TOKENS,
)
from usage.embedding_accounting_sink import embedding_accounting_sink
from usage.embedding_operation_context import (
    OPERATION_DOCUMENT,
    OPERATION_PROBE,
    OPERATION_QUERY,
    embedding_operation,
)

MODEL_ID = "BAAI/bge-m3"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def native_payload(vectors, *, input_tokens=14, cost=1.4e-07, **overrides):
    """One native DeepInfra response, shaped exactly as the §5.1 probe."""
    payload = {
        "embeddings": vectors,
        "request_id": "RXkKK6uvvRn6FqWgilEeOHx6",
        "inference_status": {
            "status": "succeeded",
            "runtime_ms": 106,
            "cost": cost,
            "tokens_input": input_tokens,
        },
        "input_tokens": input_tokens,
    }
    payload.update(overrides)
    return payload


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        if self._payload is None:
            raise requests.exceptions.JSONDecodeError("no json", "", 0)
        return self._payload


class FakePost:
    """Records every call and returns the queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, url, headers=None, json=None, **kwargs):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if not self._responses:
            raise AssertionError("unexpected extra provider call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def collected():
    """Sink collecting contributions, bound for the duration of a test."""
    contributions = []
    with embedding_accounting_sink(contributions.append):
        yield contributions


def make_embeddings(monkeypatch, responses, **kwargs):
    post = FakePost(responses)
    monkeypatch.setattr(
        "infrastructure.embedding_capture.requests.post", post, raising=True
    )
    embeddings = DeepInfraAccountingEmbeddings(
        model_id=MODEL_ID, deepinfra_api_token="token-not-real", **kwargs
    )
    return embeddings, post


# ---------------------------------------------------------------------------
# Extraction / normalization (§10)
# ---------------------------------------------------------------------------


def test_extraction_maps_every_native_field():
    contribution = extract_deepinfra_contribution(
        native_payload([[0.1]]), model=MODEL_ID, operation_kind=OPERATION_QUERY
    )

    assert contribution.provider == PROVIDER_DEEPINFRA == "Deepinfra"
    assert contribution.model == MODEL_ID
    assert contribution.input_quantity == 14
    assert contribution.quantity_unit == QUANTITY_UNIT_INPUT_TOKENS
    assert contribution.quantity_origin == ORIGIN_PROVIDER_REPORTED
    assert contribution.cost_state == COST_PROVIDER_AUTHORITATIVE
    assert contribution.provider_cost == pytest.approx(1.4e-07)
    assert contribution.provider_request_id == "RXkKK6uvvRn6FqWgilEeOHx6"
    assert contribution.provider_runtime_ms == 106
    assert contribution.operation_kind == OPERATION_QUERY


def test_extraction_keeps_no_native_or_content_field():
    contribution = extract_deepinfra_contribution(
        native_payload([[0.1, 0.2]]), model=MODEL_ID, operation_kind=OPERATION_QUERY
    )

    fields = set(contribution.__slots__)
    assert not fields & {
        "embeddings",
        "inference_status",
        "tokens_input",
        "status",
        "payload",
        "input",
    }


def test_zero_cost_is_kept_and_not_read_as_absent():
    contribution = extract_deepinfra_contribution(
        native_payload([[0.1]], cost=0.0),
        model=MODEL_ID,
        operation_kind=OPERATION_PROBE,
    )
    assert contribution.provider_cost == 0.0
    assert contribution.cost_state == COST_PROVIDER_AUTHORITATIVE


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p: p.pop("input_tokens"), id="missing_input_tokens"),
        pytest.param(
            lambda p: p["inference_status"].pop("cost"), id="missing_cost"
        ),
        pytest.param(lambda p: p.pop("inference_status"), id="missing_status_object"),
        pytest.param(
            lambda p: p.update(input_tokens="fourteen"), id="non_numeric_quantity"
        ),
        pytest.param(lambda p: p.update(input_tokens=-1), id="negative_quantity"),
        pytest.param(lambda p: p.update(input_tokens=True), id="bool_quantity"),
        pytest.param(lambda p: p.update(input_tokens=1.5), id="fractional_quantity"),
        pytest.param(
            lambda p: p["inference_status"].update(cost="free"), id="non_numeric_cost"
        ),
        pytest.param(
            lambda p: p["inference_status"].update(cost=-0.1), id="negative_cost"
        ),
        pytest.param(
            lambda p: p["inference_status"].update(tokens_input=99),
            id="token_disagreement",
        ),
    ],
)
def test_extraction_rejects_malformed_accounting(mutate):
    payload = native_payload([[0.1]])
    mutate(payload)
    with pytest.raises(EmbeddingCaptureError):
        extract_deepinfra_contribution(
            payload, model=MODEL_ID, operation_kind=OPERATION_QUERY
        )


def test_absent_corroborating_token_field_is_tolerated():
    payload = native_payload([[0.1]])
    payload["inference_status"].pop("tokens_input")
    contribution = extract_deepinfra_contribution(
        payload, model=MODEL_ID, operation_kind=OPERATION_QUERY
    )
    assert contribution.input_quantity == 14


def test_optional_metadata_degrades_to_none_without_losing_contribution():
    payload = native_payload([[0.1]])
    payload["request_id"] = "   "
    payload["inference_status"]["runtime_ms"] = "fast"
    contribution = extract_deepinfra_contribution(
        payload, model=MODEL_ID, operation_kind=OPERATION_QUERY
    )
    assert contribution.provider_request_id is None
    assert contribution.provider_runtime_ms is None
    assert contribution.input_quantity == 14


# ---------------------------------------------------------------------------
# Vector behaviour preservation (DC2, §21.4)
# ---------------------------------------------------------------------------


def test_embed_query_returns_the_provider_vector_unchanged(monkeypatch, collected):
    vector = [0.1, 0.2, 0.3]
    embeddings, post = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([vector]))]
    )

    with embedding_operation(OPERATION_QUERY):
        result = embeddings.embed_query("what is beta")

    assert result == vector
    assert len(post.calls) == 1


def test_request_shape_matches_the_parent_class(monkeypatch, collected):
    embeddings, post = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.1]]))]
    )

    with embedding_operation(OPERATION_QUERY):
        embeddings.embed_query("beta")

    call = post.calls[0]
    assert call["url"] == f"https://api.deepinfra.com/v1/inference/{MODEL_ID}"
    assert call["json"] == {"inputs": ["query: beta"], "normalize": False}
    assert call["headers"]["Authorization"] == "bearer token-not-real"
    assert call["headers"]["Content-Type"] == "application/json"


def test_embed_documents_preserves_prefix_and_batch_boundaries(
    monkeypatch, collected
):
    embeddings, post = make_embeddings(
        monkeypatch,
        [
            FakeResponse(native_payload([[0.1], [0.2]], input_tokens=8)),
            FakeResponse(native_payload([[0.3]], input_tokens=4)),
        ],
        batch_size=2,
    )

    with embedding_operation(OPERATION_DOCUMENT):
        result = embeddings.embed_documents(["alpha", "beta", "gamma"])

    assert result == [[0.1], [0.2], [0.3]]
    assert [call["json"]["inputs"] for call in post.calls] == [
        ["passage: alpha", "passage: beta"],
        ["passage: gamma"],
    ]


def test_vectors_are_identical_to_the_unwrapped_parent_class(monkeypatch):
    from langchain_community.embeddings import DeepInfraEmbeddings

    payloads = [native_payload([[0.1], [0.2]]), native_payload([[0.3]])]

    def run(cls):
        post = FakePost([FakeResponse(p) for p in payloads])
        monkeypatch.setattr(
            "langchain_community.embeddings.deepinfra.requests.post",
            post,
            raising=True,
        )
        monkeypatch.setattr(
            "infrastructure.embedding_capture.requests.post", post, raising=True
        )
        instance = cls(
            model_id=MODEL_ID, deepinfra_api_token="token-not-real", batch_size=2
        )
        return instance.embed_documents(["a", "b", "c"]), post.calls

    parent_vectors, parent_calls = run(DeepInfraEmbeddings)
    child_vectors, child_calls = run(DeepInfraAccountingEmbeddings)

    assert child_vectors == parent_vectors
    assert child_calls == parent_calls


# ---------------------------------------------------------------------------
# Contribution cardinality (DC3)
# ---------------------------------------------------------------------------


def test_one_provider_response_yields_one_contribution(monkeypatch, collected):
    embeddings, _ = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.1]]))]
    )

    with embedding_operation(OPERATION_QUERY):
        embeddings.embed_query("beta")

    assert len(collected) == 1


def test_n_provider_responses_yield_n_unaggregated_contributions(
    monkeypatch, collected
):
    embeddings, post = make_embeddings(
        monkeypatch,
        [
            FakeResponse(native_payload([[0.1], [0.2]], input_tokens=8)),
            FakeResponse(native_payload([[0.3]], input_tokens=4)),
        ],
        batch_size=2,
    )

    with embedding_operation(OPERATION_DOCUMENT):
        embeddings.embed_documents(["alpha", "beta", "gamma"])

    assert len(post.calls) == 2
    assert [c.input_quantity for c in collected] == [8, 4]
    assert {c.operation_kind for c in collected} == {OPERATION_DOCUMENT}


# ---------------------------------------------------------------------------
# Operation context (DC4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", [OPERATION_QUERY, OPERATION_DOCUMENT, OPERATION_PROBE]
)
def test_ambient_context_determines_operation_kind(monkeypatch, collected, kind):
    embeddings, _ = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.1]]))]
    )

    with embedding_operation(kind):
        embeddings.embed_query("beta")

    assert [c.operation_kind for c in collected] == [kind]


def test_without_operation_context_no_contribution_but_vectors_returned(
    monkeypatch, collected, caplog
):
    embeddings, post = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.1, 0.9]]))]
    )

    with caplog.at_level(logging.WARNING):
        result = embeddings.embed_query("beta")

    assert result == [0.1, 0.9]
    assert collected == []
    assert len(post.calls) == 1
    # An absent operation context is the expected state, not an anomaly.
    assert caplog.records == []


# ---------------------------------------------------------------------------
# Sink behaviour (DC8)
# ---------------------------------------------------------------------------


def test_default_no_op_sink_leaves_the_embedding_successful(monkeypatch):
    embeddings, _ = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.5]]))]
    )

    with embedding_operation(OPERATION_QUERY):
        assert embeddings.embed_query("beta") == [0.5]


def test_a_raising_sink_does_not_break_the_embedding(monkeypatch, caplog):
    embeddings, _ = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.5]]))]
    )

    def boom(contribution):
        raise RuntimeError("sink exploded")

    with caplog.at_level(logging.WARNING):
        with embedding_accounting_sink(boom), embedding_operation(OPERATION_QUERY):
            assert embeddings.embed_query("beta") == [0.5]

    assert any(
        "embedding_accounting_delivery_failed" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Malformed accounting must not break a successful embedding (§12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p: p.pop("input_tokens"), id="missing_input_tokens"),
        pytest.param(lambda p: p["inference_status"].pop("cost"), id="missing_cost"),
        pytest.param(
            lambda p: p.update(input_tokens="fourteen"), id="invalid_numeric"
        ),
        pytest.param(
            lambda p: p["inference_status"].update(tokens_input=99),
            id="token_disagreement",
        ),
    ],
)
def test_malformed_accounting_skips_contribution_and_warns_safely(
    monkeypatch, collected, caplog, mutate
):
    payload = native_payload([[0.7, 0.8]])
    mutate(payload)
    embeddings, _ = make_embeddings(monkeypatch, [FakeResponse(payload)])

    with caplog.at_level(logging.WARNING):
        with embedding_operation(OPERATION_QUERY):
            result = embeddings.embed_query("secret query text")

    assert result == [0.7, 0.8]
    assert collected == []

    messages = [r.getMessage() for r in caplog.records]
    assert any("embedding_accounting_capture_failed" in m for m in messages)
    for message in messages:
        assert "secret query text" not in message
        assert "token-not-real" not in message
        assert "0.7" not in message
        assert "inference_status" not in message


# ---------------------------------------------------------------------------
# Provider failure (§16.2)
# ---------------------------------------------------------------------------


def test_non_200_raises_like_the_parent_and_emits_no_contribution(
    monkeypatch, collected
):
    embeddings, _ = make_embeddings(
        monkeypatch, [FakeResponse(None, status_code=500, text="upstream boom")]
    )

    with embedding_operation(OPERATION_QUERY):
        with pytest.raises(ValueError, match="Error raised by inference API HTTP code"):
            embeddings.embed_query("beta")

    assert collected == []


def test_request_exception_raises_like_the_parent_and_emits_no_contribution(
    monkeypatch, collected
):
    embeddings, _ = make_embeddings(
        monkeypatch, [requests.exceptions.ConnectionError("no route")]
    )

    with embedding_operation(OPERATION_QUERY):
        with pytest.raises(ValueError, match="Error raised by inference endpoint"):
            embeddings.embed_query("beta")

    assert collected == []


def test_undecodable_body_raises_like_the_parent(monkeypatch, collected):
    embeddings, _ = make_embeddings(
        monkeypatch, [FakeResponse(None, status_code=200, text="<html/>")]
    )

    with embedding_operation(OPERATION_QUERY):
        with pytest.raises(ValueError, match="Error raised by inference API"):
            embeddings.embed_query("beta")

    assert collected == []


# ---------------------------------------------------------------------------
# Async coverage - inherited run_in_executor fallback (§14.2)
# ---------------------------------------------------------------------------


def test_the_class_adds_no_async_override():
    # If a future langchain_community release adds native async methods,
    # this slice's single sync capture point stops covering them.
    from langchain_community.embeddings import DeepInfraEmbeddings
    from langchain_core.embeddings import Embeddings

    for name in ("aembed_query", "aembed_documents"):
        assert getattr(DeepInfraEmbeddings, name) is getattr(Embeddings, name)
        assert getattr(DeepInfraAccountingEmbeddings, name) is getattr(
            Embeddings, name
        )


def test_aembed_query_captures_through_the_inherited_fallback(
    monkeypatch, collected
):
    embeddings, post = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.4]]))]
    )
    async def scenario():
        with embedding_operation(OPERATION_QUERY):
            return await embeddings.aembed_query("beta")

    result = asyncio.run(scenario())

    assert result == [0.4]
    assert [c.operation_kind for c in collected] == [OPERATION_QUERY]
    assert len(post.calls) == 1


def test_aembed_documents_captures_one_contribution_per_batch(
    monkeypatch, collected
):
    embeddings, post = make_embeddings(
        monkeypatch,
        [
            FakeResponse(native_payload([[0.1], [0.2]], input_tokens=8)),
            FakeResponse(native_payload([[0.3]], input_tokens=4)),
        ],
        batch_size=2,
    )

    async def scenario():
        with embedding_operation(OPERATION_DOCUMENT):
            return await embeddings.aembed_documents(["alpha", "beta", "gamma"])

    result = asyncio.run(scenario())

    assert result == [[0.1], [0.2], [0.3]]
    assert [c.input_quantity for c in collected] == [8, 4]


def test_capture_runs_on_the_executor_thread(monkeypatch):
    """The async fallback really hops threads, and capture still lands."""
    embeddings, _ = make_embeddings(
        monkeypatch, [FakeResponse(native_payload([[0.1]]))]
    )
    threads = []

    def record(contribution):
        threads.append(threading.current_thread())

    async def scenario():
        with embedding_accounting_sink(record), embedding_operation(OPERATION_QUERY):
            await embeddings.aembed_query("beta")

    asyncio.run(scenario())

    assert len(threads) == 1
    assert threads[0] is not threading.current_thread()


# ---------------------------------------------------------------------------
# Factory integration (§15) - DeepInfra branch only
# ---------------------------------------------------------------------------


def test_factory_returns_the_capture_enabled_class():
    from infrastructure.ai import choose_emb_model

    instance = choose_emb_model("Deepinfra", MODEL_ID, api_key="token-not-real")

    assert isinstance(instance, DeepInfraAccountingEmbeddings)
    assert instance.model_id == MODEL_ID
