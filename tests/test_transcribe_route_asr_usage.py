"""Route-level tests for /transcribe audio ASR Usage integration (Usage Slice 3).

Covers only the audio branch of routes/multimodal.py::asr_parse: resolving
an already-resolved ASR monetary cost via infrastructure.asr_accounting and
persisting it via log_usage_with_resolved_cost(). Document and image
branches are proven to remain untouched (no resolved-cost Usage write).

No real provider network calls are made: asr_response() is monkeypatched to
return a fake requests.Response-like object.
"""

import io
from types import SimpleNamespace

from flask import Flask

from routes import multimodal as multimodal_route
from utils.logging_config import register_request_context_hooks


class FakeAsrResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        models=SimpleNamespace(
            asr_model="test-asr-model",
            asr_provider="Deepinfra",
            asr_base_url=None,
            asr_mistral_price_per_minute_usd=None,
        )
    )
    register_request_context_hooks(app)
    app.register_blueprint(multimodal_route.multimodal_bp)
    return app


def _post_audio(app, headers=None):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(b"fake-audio-bytes"), "audio.wav", "audio/wav")},
        content_type="multipart/form-data",
        headers=headers
        or {
            "X-API-KEY": "test-key",
            "X-USER-EMAIL": "user@example.com",
            "X-USER-NAME": "Test User",
        },
    )


def _common_monkeypatches(monkeypatch, *, get_user=None):
    monkeypatch.setattr(multimodal_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(
        multimodal_route.database_pg,
        "get_user_by_username",
        get_user
        or (lambda user_email: {"id": 42, "username": user_email, "client": "dino"}),
    )


def test_deepinfra_audio_success_logs_resolved_cost_usage(monkeypatch):
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {
        "text": "hello world",
        "inference_status": {"cost": 0.0225},
    }
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    log_calls = []

    def fake_log_usage_with_resolved_cost(**kwargs):
        log_calls.append(kwargs)
        return 777

    handoff_calls = []
    monkeypatch.setattr(
        multimodal_route, "log_usage_with_resolved_cost", fake_log_usage_with_resolved_cost
    )
    monkeypatch.setattr(
        multimodal_route, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
    )

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "hello world"}

    assert len(log_calls) == 1
    call = log_calls[0]
    assert call["user_id"] == 42
    assert call["cost"] == 0.0225
    assert call["provider"] == "Deepinfra"
    assert call["model"] == "test-asr-model"
    assert call["service"] == "/transcribe"
    assert call["request_id"] == response.headers["X-Request-ID"]
    assert call["source"] == "dino"

    assert handoff_calls == [777]


def test_mistral_audio_success_uses_configured_rate_and_logs_usage(monkeypatch):
    app = _make_app()
    app.config["MAUI_CONFIG"].models.asr_provider = "Mistral"
    app.config["MAUI_CONFIG"].models.asr_mistral_price_per_minute_usd = 0.003
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

    payload = {
        "text": "bonjour",
        "usage": {
            "prompt_audio_seconds": 30,
            "prompt_tokens": 4,
            "completion_tokens": 88,
            "total_tokens": 92,
        },
    }
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    log_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "log_usage_with_resolved_cost",
        lambda **kwargs: (log_calls.append(kwargs), 888)[1],
    )
    monkeypatch.setattr(multimodal_route, "set_usage_log_id", lambda log_id: None)

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "bonjour"}

    assert len(log_calls) == 1
    call = log_calls[0]
    assert call["provider"] == "Mistral"
    assert call["cost"] == (30 / 60.0) * 0.003
    assert call["service"] == "/transcribe"


def test_mistral_missing_configured_rate_does_not_call_provider_or_log_usage(monkeypatch):
    app = _make_app()
    app.config["MAUI_CONFIG"].models.asr_provider = "Mistral"
    app.config["MAUI_CONFIG"].models.asr_mistral_price_per_minute_usd = None
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

    provider_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "asr_response",
        lambda *a, **k: provider_calls.append(1) or FakeAsrResponse(200, {"text": "x"}),
    )
    log_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "log_usage_with_resolved_cost",
        lambda **kwargs: log_calls.append(kwargs),
    )

    response = _post_audio(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Missing ASR configuration"}
    assert provider_calls == []
    assert log_calls == []


def test_malformed_deepinfra_payload_does_not_break_transcription_response(monkeypatch):
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {"text": "hello world"}
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    log_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "log_usage_with_resolved_cost",
        lambda **kwargs: log_calls.append(kwargs),
    )

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "hello world"}
    assert log_calls == []


def test_resolved_cost_persistence_failure_does_not_break_transcription_response(
    monkeypatch,
):
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {"text": "hello world", "inference_status": {"cost": 0.01}}
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    def raise_db_failure(**kwargs):
        raise RuntimeError("db is down")

    handoff_calls = []
    monkeypatch.setattr(multimodal_route, "log_usage_with_resolved_cost", raise_db_failure)
    monkeypatch.setattr(
        multimodal_route, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
    )

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "hello world"}
    assert handoff_calls == []


def test_zero_resolved_cost_is_persisted_as_valid_usage(monkeypatch):
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {"text": "hello world", "inference_status": {"cost": 0.0}}
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    log_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "log_usage_with_resolved_cost",
        lambda **kwargs: (log_calls.append(kwargs), 999)[1],
    )
    monkeypatch.setattr(multimodal_route, "set_usage_log_id", lambda log_id: None)

    response = _post_audio(app)

    assert response.status_code == 200
    assert len(log_calls) == 1
    assert log_calls[0]["cost"] == 0.0


def test_document_branch_writes_no_resolved_cost_usage(monkeypatch):
    app = _make_app()
    _common_monkeypatches(monkeypatch)

    log_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "log_usage_with_resolved_cost",
        lambda **kwargs: log_calls.append(kwargs),
    )
    monkeypatch.setattr(
        multimodal_route,
        "extract_and_normalize_document",
        lambda doc_input: {"text": "extracted document text"},
    )

    response = app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(b"%PDF-fake"), "doc.pdf", "application/pdf")},
        content_type="multipart/form-data",
        headers={
            "X-API-KEY": "test-key",
            "X-USER-EMAIL": "user@example.com",
            "X-USER-NAME": "Test User",
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"text": "extracted document text"}
    assert log_calls == []


def _post_image(app, headers=None):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(b"fake-image-bytes"), "photo.png", "image/png")},
        content_type="multipart/form-data",
        headers=headers
        or {
            "X-API-KEY": "test-key",
            "X-USER-EMAIL": "user@example.com",
            "X-USER-NAME": "Test User",
        },
    )


def _make_image_app():
    app = _make_app()
    app.config["MAUI_CONFIG"].models.vision_provider = "test-vision-provider"
    app.config["MAUI_CONFIG"].models.vision_model = "test-vision-model"
    return app


def test_image_branch_never_writes_resolved_cost_usage(monkeypatch):
    """Vision accounting is token-based; log_usage_with_resolved_cost() must
    never be invoked by the image branch (that writer is ASR-specific)."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    resolved_cost_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "log_usage_with_resolved_cost",
        lambda **kwargs: resolved_cost_calls.append(kwargs),
    )
    monkeypatch.setattr(
        multimodal_route,
        "describe_image_with_usage",
        lambda *a, **k: {
            "description": "described image text",
            "token_usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        },
    )
    monkeypatch.setattr(multimodal_route, "log_token_usage", lambda **kwargs: 555)
    monkeypatch.setattr(multimodal_route, "set_usage_log_id", lambda log_id: None)

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}
    assert resolved_cost_calls == []


def test_image_success_logs_token_usage_and_hands_off_log_id(monkeypatch):
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    monkeypatch.setattr(
        multimodal_route,
        "describe_image_with_usage",
        lambda *a, **k: {
            "description": "described image text",
            "token_usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        },
    )

    log_calls = []
    monkeypatch.setattr(
        multimodal_route,
        "log_token_usage",
        lambda **kwargs: (log_calls.append(kwargs), 555)[1],
    )
    handoff_calls = []
    monkeypatch.setattr(
        multimodal_route, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
    )

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}
    # Public response is unchanged - no accounting metadata leaks into it.
    assert set(response.get_json().keys()) == {"text"}

    assert len(log_calls) == 1
    call = log_calls[0]
    assert call["user_id"] == 42
    assert call["token_input"] == 12
    assert call["token_output"] == 7
    assert call["provider"] == "test-vision-provider"
    assert call["model"] == "test-vision-model"
    assert call["service"] == "/transcribe"
    assert call["request_id"] == response.headers["X-Request-ID"]
    assert call["source"] == "dino"

    assert handoff_calls == [555]


def test_image_usage_persistence_failure_does_not_break_response(monkeypatch):
    """Vision succeeds, log_token_usage() fails: the request must still
    succeed with the original description, and the accounting failure must
    surface only as an ERROR log, not a re-raised exception."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    monkeypatch.setattr(
        multimodal_route,
        "describe_image_with_usage",
        lambda *a, **k: {
            "description": "described image text",
            "token_usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        },
    )

    def raise_db_failure(**kwargs):
        raise RuntimeError("db is down")

    handoff_calls = []
    monkeypatch.setattr(multimodal_route, "log_token_usage", raise_db_failure)
    monkeypatch.setattr(
        multimodal_route, "set_usage_log_id", lambda log_id: handoff_calls.append(log_id)
    )

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}
    assert handoff_calls == []


def test_image_usage_persistence_failure_is_logged_at_error(monkeypatch, caplog):
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    monkeypatch.setattr(
        multimodal_route,
        "describe_image_with_usage",
        lambda *a, **k: {
            "description": "described image text",
            "token_usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    )
    monkeypatch.setattr(
        multimodal_route, "log_token_usage", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db is down"))
    )

    with caplog.at_level("ERROR", logger=multimodal_route.logger.name):
        response = _post_image(app)

    assert response.status_code == 200
    assert any(
        "event=transcribe_usage_accounting_failed" in record.getMessage()
        for record in caplog.records
    )


def test_image_vision_failure_is_not_swallowed_by_usage_containment(monkeypatch):
    """An actual Vision/provider failure must still surface as a 500 and must
    not be masked by the Usage-accounting failure containment."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    def raise_vision_failure(*a, **k):
        raise RuntimeError("vision provider is down")

    log_calls = []
    monkeypatch.setattr(
        multimodal_route, "describe_image_with_usage", raise_vision_failure
    )
    monkeypatch.setattr(
        multimodal_route,
        "log_token_usage",
        lambda **kwargs: log_calls.append(kwargs),
    )

    response = _post_image(app)

    assert response.status_code == 500
    assert "vision provider is down" in response.get_json()["error"]
    assert log_calls == []
