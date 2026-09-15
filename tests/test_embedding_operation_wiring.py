"""Embedding accounting Slice 3 - production operation-context wiring.

Proves that the four CURRENT production embedding seams establish the right
operation kind at the narrowest scope, restore it on both the success and
the exception path, and - through the request-lifecycle sink binding - that
a DeepInfra accounting response actually reaches the request's accumulator.

No network, no database, no live provider call: the vector store, the
embeddings object and ``requests.post`` are all replaced by fakes.
"""

import pytest
import requests
from flask import Flask

import infrastructure.vector_store as vector_store
import services.completion_service as completion_service
import services.retrieval_service as retrieval_service
from infrastructure.embedding_capture import DeepInfraAccountingEmbeddings
from services.completion_service import CompletionRequest, complete_chat
from services.retrieval_service import retrieve_from_collection
from usage.embedding_accounting_lifecycle import register_embedding_accounting_hooks
from usage.embedding_accounting_sink import (
    embedding_accounting_sink,
    get_embedding_accounting_sink,
    no_op_sink,
)
from usage.embedding_operation_context import (
    OPERATION_DOCUMENT,
    OPERATION_PROBE,
    OPERATION_QUERY,
    get_embedding_operation,
)
from usage.embedding_state import get_embedding_contributions

MODEL_ID = "BAAI/bge-m3"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingStore:
    """Vector store recording the operation kind in scope at each call."""

    def __init__(self, *, raises=None):
        self.observed = []
        self._raises = raises

    def find_similar_vectors(self, text, top_k, min_similarity):
        self.observed.append(get_embedding_operation())
        if self._raises is not None:
            raise self._raises
        return []

    def store_paragraphs(self, paragraphs):
        self.observed.append(get_embedding_operation())
        if self._raises is not None:
            raise self._raises


class RecordingEmbeddings:
    """Embeddings double recording the scope active during embed_query."""

    def __init__(self):
        self.observed = []

    def embed_query(self, text):
        self.observed.append(get_embedding_operation())
        return [0.0, 0.1, 0.2]

    def embed_documents(self, texts):
        self.observed.append(get_embedding_operation())
        return [[0.0, 0.1, 0.2] for _ in texts]


def native_payload(vectors, *, input_tokens=14, cost=1.4e-07):
    """One native DeepInfra response, shaped as the §5.1 runtime probe."""
    return {
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


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload


class FakePost:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, url, headers=None, json=None, **kwargs):
        self.calls.append({"url": url, "json": json})
        if not self._responses:
            raise AssertionError("unexpected extra provider call")
        return self._responses.pop(0)


def deepinfra(monkeypatch, responses):
    """A capture-enabled DeepInfra embeddings object with canned responses."""
    post = FakePost(responses)
    monkeypatch.setattr(requests, "post", post)
    embeddings = DeepInfraAccountingEmbeddings(
        model_id=MODEL_ID, deepinfra_api_token="token-not-real"
    )
    return embeddings, post


def fake_llm(monkeypatch):
    from types import SimpleNamespace

    class FakeLlm:
        def invoke(self, messages):
            return SimpleNamespace(content="an answer", response_metadata={})

    monkeypatch.setattr(
        completion_service, "choose_llm", lambda *a, **k: FakeLlm()
    )
    monkeypatch.setattr(
        completion_service, "load_prompt", lambda *a, **k: "system prompt"
    )
    monkeypatch.setattr(
        completion_service, "render_prompt", lambda template, **k: template
    )


@pytest.fixture(autouse=True)
def _clean_context():
    """No test may start or finish inside a leaked scope."""
    assert get_embedding_operation() is None
    yield
    assert get_embedding_operation() is None


@pytest.fixture
def collected():
    contributions = []
    with embedding_accounting_sink(contributions.append):
        yield contributions


# ---------------------------------------------------------------------------
# /completion.json retrieval seam -> query
# ---------------------------------------------------------------------------


def test_completion_retrieval_runs_under_query(monkeypatch):
    fake_llm(monkeypatch)
    store = RecordingStore()
    req = CompletionRequest(username="alice", info=["info"], chat=["a question"])

    complete_chat(req=req, store=store, llm_type="openai", model="gpt-4o")

    assert store.observed == [OPERATION_QUERY]


def test_completion_restores_context_after_success(monkeypatch):
    fake_llm(monkeypatch)
    complete_chat(
        req=CompletionRequest(username="alice", info=["info"], chat=["q"]),
        store=RecordingStore(),
        llm_type="openai",
        model="gpt-4o",
    )
    assert get_embedding_operation() is None


def test_completion_restores_context_after_retrieval_failure(monkeypatch):
    store = RecordingStore(raises=ValueError("boom"))
    with pytest.raises(RuntimeError):
        complete_chat(
            req=CompletionRequest(username="alice", info=["info"], chat=["q"]),
            store=store,
            llm_type="openai",
            model="gpt-4o",
        )
    assert store.observed == [OPERATION_QUERY]
    assert get_embedding_operation() is None


def test_completion_scope_does_not_reach_the_llm(monkeypatch):
    """The scope names one embedding operation, not the whole request."""
    seen = {}

    from types import SimpleNamespace

    class FakeLlm:
        def invoke(self, messages):
            seen["operation"] = get_embedding_operation()
            return SimpleNamespace(content="an answer", response_metadata={})

    monkeypatch.setattr(completion_service, "choose_llm", lambda *a, **k: FakeLlm())
    monkeypatch.setattr(
        completion_service, "load_prompt", lambda *a, **k: "system prompt"
    )
    monkeypatch.setattr(
        completion_service, "render_prompt", lambda template, **k: template
    )

    complete_chat(
        req=CompletionRequest(username="alice", info=["info"], chat=["q"]),
        store=RecordingStore(),
        llm_type="openai",
        model="gpt-4o",
    )
    assert seen["operation"] is None


def test_completion_return_shape_is_unchanged(monkeypatch):
    fake_llm(monkeypatch)
    result = complete_chat(
        req=CompletionRequest(username="alice", info=["info"], chat=["q"]),
        store=RecordingStore(),
        llm_type="openai",
        model="gpt-4o",
    )
    assert set(result) == {"answer", "vectors", "token_usage", "is_no_info"}
    assert result["vectors"] == []


# ---------------------------------------------------------------------------
# /agentchat retrieval seam -> query
# ---------------------------------------------------------------------------


def _retrieve(store, monkeypatch, embeddings=None):
    monkeypatch.setattr(
        retrieval_service, "choose_emb_model", lambda *a, **k: embeddings or object()
    )
    monkeypatch.setattr(
        retrieval_service, "MauiVectorStore", lambda embeddings, table_name: store
    )
    return retrieve_from_collection(
        question="a question",
        namespace="Dino",
        embedding_provider="deepinfra",
        embedding_model=MODEL_ID,
        top_k=5,
        min_sim=0.5,
    )


def test_agentchat_retrieval_runs_under_query(monkeypatch):
    store = RecordingStore()
    _retrieve(store, monkeypatch)
    assert store.observed == [OPERATION_QUERY]
    assert get_embedding_operation() is None


def test_agentchat_sequential_retrievals_each_get_a_fresh_scope(monkeypatch):
    """One /agentchat request drives 0..N RetrieverTool.forward calls."""
    store = RecordingStore()
    for _ in range(3):
        _retrieve(store, monkeypatch)
        assert get_embedding_operation() is None
    assert store.observed == [OPERATION_QUERY] * 3


def test_agentchat_failure_restores_context(monkeypatch):
    store = RecordingStore(raises=ValueError("boom"))
    with pytest.raises(RuntimeError):
        _retrieve(store, monkeypatch)
    assert store.observed == [OPERATION_QUERY]
    assert get_embedding_operation() is None


def test_agentchat_failed_retrieval_does_not_bleed_into_the_next(monkeypatch):
    failing = RecordingStore(raises=ValueError("boom"))
    with pytest.raises(RuntimeError):
        _retrieve(failing, monkeypatch)
    ok = RecordingStore()
    _retrieve(ok, monkeypatch)
    assert ok.observed == [OPERATION_QUERY]


# ---------------------------------------------------------------------------
# ingestion seam -> document
# ---------------------------------------------------------------------------


def test_store_paragraphs_runs_under_document():
    """The ingestion seam as written: scope wraps store_paragraphs only."""
    from usage.embedding_operation_context import embedding_operation

    store = RecordingStore()
    with embedding_operation(OPERATION_DOCUMENT):
        store.store_paragraphs([])
    assert store.observed == [OPERATION_DOCUMENT]
    assert get_embedding_operation() is None


def test_ingestion_wraps_store_paragraphs_and_not_namespace_preparation():
    """The `document` scope must contain store_paragraphs and nothing else.

    Asserted on the parsed source of ``process_rag_file`` rather than by
    executing it: the function also writes files and rows, and the property
    under test is exactly which call the scope encloses. Probe and document
    stay distinguishable only if ensure_pgvector_namespace_ready is outside.
    """
    import ast
    import inspect

    import services.rag_ingestion_service as ingestion

    tree = ast.parse(inspect.getsource(ingestion.process_rag_file))
    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "embedding_operation"
            for item in node.items
        )
    ]
    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.items[0].context_expr.args[0].id == "OPERATION_DOCUMENT"

    enclosed = {
        node.func.attr
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert enclosed == {"store_paragraphs"}

    enclosed_names = {
        node.func.id
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "ensure_pgvector_namespace_ready" not in enclosed_names
    assert "choose_emb_model" not in enclosed_names
    assert "MauiVectorStore" not in enclosed_names


def test_document_scope_restored_after_exception():
    from usage.embedding_operation_context import embedding_operation

    store = RecordingStore(raises=ValueError("boom"))
    with pytest.raises(ValueError):
        with embedding_operation(OPERATION_DOCUMENT):
            store.store_paragraphs([])
    assert get_embedding_operation() is None


def test_document_scope_tolerates_a_no_provider_work_path(collected, monkeypatch):
    """Deduplication may leave nothing to embed: zero contributions is valid."""
    from usage.embedding_operation_context import embedding_operation

    embeddings, post = deepinfra(monkeypatch, [])
    with embedding_operation(OPERATION_DOCUMENT):
        pass  # store_paragraphs embedded nothing
    assert post.calls == []
    assert collected == []


# ---------------------------------------------------------------------------
# namespace probe seam -> probe
# ---------------------------------------------------------------------------


def _prepare_namespace(monkeypatch, *, table_already_exists):
    class FakeEngine:
        def __init__(self):
            self.init_calls = []

        def init_vectorstore_table(self, **kwargs):
            self.init_calls.append(kwargs)

    engine = FakeEngine()
    monkeypatch.setattr(vector_store, "schema", "public")
    monkeypatch.setattr(vector_store, "create_pgvector_engine", lambda: engine)
    monkeypatch.setattr(
        vector_store, "table_exists", lambda schema, name: table_already_exists
    )
    return engine


def test_probe_runs_under_probe_when_namespace_is_missing(monkeypatch):
    engine = _prepare_namespace(monkeypatch, table_already_exists=False)
    embeddings = RecordingEmbeddings()

    vector_store.ensure_pgvector_namespace_ready(
        embeddings=embeddings, table_name="Dino"
    )

    assert embeddings.observed == [OPERATION_PROBE]
    assert engine.init_calls[0]["vector_size"] == 3
    assert get_embedding_operation() is None


def test_no_probe_context_and_no_provider_call_when_namespace_exists(monkeypatch):
    engine = _prepare_namespace(monkeypatch, table_already_exists=True)
    embeddings = RecordingEmbeddings()

    vector_store.ensure_pgvector_namespace_ready(
        embeddings=embeddings, table_name="Dino"
    )

    assert embeddings.observed == []
    assert engine.init_calls == []
    assert get_embedding_operation() is None


def test_probe_scope_restored_when_the_provider_probe_fails(monkeypatch):
    _prepare_namespace(monkeypatch, table_already_exists=False)

    class Failing:
        def embed_query(self, text):
            raise ValueError("provider down")

    with pytest.raises(ValueError):
        vector_store.ensure_pgvector_namespace_ready(
            embeddings=Failing(), table_name="Dino"
        )
    assert get_embedding_operation() is None


# ---------------------------------------------------------------------------
# probe + document sequence, end to end through DeepInfra capture
# ---------------------------------------------------------------------------


def test_probe_then_document_produce_distinct_contributions(monkeypatch, collected):
    """The high-value ordering test: one ingestion flow, two operation kinds."""
    from usage.embedding_operation_context import embedding_operation

    engine = _prepare_namespace(monkeypatch, table_already_exists=False)
    embeddings, post = deepinfra(
        monkeypatch,
        [
            FakeResponse(native_payload([[0.1, 0.2, 0.3]], input_tokens=3)),
            FakeResponse(
                native_payload([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], input_tokens=41)
            ),
        ],
    )

    vector_store.ensure_pgvector_namespace_ready(
        embeddings=embeddings, table_name="Dino"
    )
    with embedding_operation(OPERATION_DOCUMENT):
        embeddings.embed_documents(["chunk one", "chunk two"])

    assert [c.operation_kind for c in collected] == [
        OPERATION_PROBE,
        OPERATION_DOCUMENT,
    ]
    assert [c.input_quantity for c in collected] == [3, 41]
    assert len(post.calls) == 2
    assert get_embedding_operation() is None


# ---------------------------------------------------------------------------
# request lifecycle sink binding
# ---------------------------------------------------------------------------


def test_request_binds_a_sink_over_a_fresh_accumulator(monkeypatch):
    app = Flask(__name__)
    register_embedding_accounting_hooks(app)
    seen = {}

    @app.get("/x")
    def _x():
        embeddings, _ = deepinfra(
            monkeypatch, [FakeResponse(native_payload([[0.1, 0.2]], input_tokens=7))]
        )
        from usage.embedding_operation_context import embedding_operation

        with embedding_operation(OPERATION_QUERY):
            embeddings.embed_query("a question")
        seen["contributions"] = get_embedding_contributions()
        return "ok"

    client = app.test_client()
    assert client.get("/x").status_code == 200

    contributions = seen["contributions"]
    assert len(contributions) == 1
    assert contributions[0].operation_kind == OPERATION_QUERY
    assert contributions[0].input_quantity == 7


def test_requests_do_not_share_contributions(monkeypatch):
    app = Flask(__name__)
    register_embedding_accounting_hooks(app)
    seen = []

    @app.get("/x")
    def _x():
        embeddings, _ = deepinfra(
            monkeypatch, [FakeResponse(native_payload([[0.1, 0.2]], input_tokens=5))]
        )
        from usage.embedding_operation_context import embedding_operation

        with embedding_operation(OPERATION_QUERY):
            embeddings.embed_query("a question")
        seen.append(len(get_embedding_contributions()))
        return "ok"

    client = app.test_client()
    client.get("/x")
    client.get("/x")
    assert seen == [1, 1]


def test_sink_is_unbound_after_the_request(monkeypatch):
    app = Flask(__name__)
    register_embedding_accounting_hooks(app)

    @app.get("/x")
    def _x():
        assert get_embedding_accounting_sink() is not no_op_sink
        return "ok"

    app.test_client().get("/x")
    assert get_embedding_accounting_sink() is no_op_sink


def test_sink_is_unbound_after_an_unhandled_view_exception():
    app = Flask(__name__)
    register_embedding_accounting_hooks(app)

    @app.get("/boom")
    def _boom():
        raise ValueError("boom")

    # Flask turns an unhandled view exception into a 500; only teardown is
    # guaranteed to run on that path, which is the point of the assertion.
    assert app.test_client().get("/boom").status_code == 500
    assert get_embedding_accounting_sink() is no_op_sink


def test_registration_is_idempotent():
    app = Flask(__name__)
    register_embedding_accounting_hooks(app)
    register_embedding_accounting_hooks(app)
    assert len(app.before_request_funcs[None]) == 1


def test_contribution_survives_a_later_request_failure(monkeypatch):
    """§16: consumption already observed is independent of HTTP outcome."""
    app = Flask(__name__)
    register_embedding_accounting_hooks(app)
    seen = {}

    @app.get("/boom")
    def _boom():
        embeddings, _ = deepinfra(
            monkeypatch, [FakeResponse(native_payload([[0.1]], input_tokens=9))]
        )
        from usage.embedding_operation_context import embedding_operation

        with embedding_operation(OPERATION_QUERY):
            embeddings.embed_query("a question")
        seen["contributions"] = get_embedding_contributions()
        raise ValueError("the LLM failed afterwards")

    assert app.test_client().get("/boom").status_code == 500
    assert len(seen["contributions"]) == 1
    assert seen["contributions"][0].operation_kind == OPERATION_QUERY
