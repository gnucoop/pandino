"""Route-level tests for /transcribe provider-backed Usage integration.

Covers both provider-backed branches of routes/multimodal.py::asr_parse.
Audio resolves an already-resolved ASR monetary cost via
infrastructure.asr_accounting and hands it to record_resolved_consumption();
image hands a provider-reported token pair to record_token_consumption().
The assertions here are the adopter's own - which consumption facts each
branch states, and that the response is never governed by accounting. What
the boundary then derives, hides and registers is its own contract, proven
in tests/test_usage_recording.py. The document branch, and the image branch,
are both proven not to reach the resolved-cost shape at all.

No real provider network calls are made: asr_response() and
describe_image_with_usage() are monkeypatched.
"""

import io
from types import SimpleNamespace

from flask import Flask

from routes import multimodal as multimodal_route
from utils import usage_recording
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


def _patch_boundary(monkeypatch, *, result=True):
    """Capture what the route states to the resolved-cost boundary."""
    recorded = []

    def _record(**kwargs):
        recorded.append(kwargs)
        return result

    monkeypatch.setattr(multimodal_route, "record_resolved_consumption", _record)
    return recorded


def test_deepinfra_audio_success_records_resolved_cost(monkeypatch):
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

    recorded = _patch_boundary(monkeypatch)

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "hello world"}

    assert len(recorded) == 1
    call = recorded[0]
    assert call["user_id"] == 42
    assert call["cost"] == 0.0225
    assert call["provider"] == "Deepinfra"
    assert call["model"] == "test-asr-model"
    assert call["service"] == "/transcribe"
    # The adopter states consumption facts only.
    assert set(call) == {"user_id", "provider", "model", "service", "cost"}


def test_mistral_audio_success_uses_configured_rate_and_records_cost(monkeypatch):
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

    recorded = _patch_boundary(monkeypatch)

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "bonjour"}

    assert len(recorded) == 1
    call = recorded[0]
    assert call["provider"] == "Mistral"
    assert call["cost"] == (30 / 60.0) * 0.003
    assert call["service"] == "/transcribe"


def test_mistral_missing_configured_rate_does_not_call_provider_or_record(monkeypatch):
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
    recorded = _patch_boundary(monkeypatch)

    response = _post_audio(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Missing ASR configuration"}
    assert provider_calls == []
    assert recorded == []


def test_malformed_deepinfra_payload_does_not_break_transcription_response(monkeypatch):
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {"text": "hello world"}
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    recorded = _patch_boundary(monkeypatch)

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "hello world"}
    # Cost resolution failed before the boundary, so nothing was recorded.
    assert recorded == []


def test_boundary_false_does_not_break_transcription_response(monkeypatch):
    """Usage is fail-open: an unrecorded row never governs HTTP success."""
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {"text": "hello world", "inference_status": {"cost": 0.01}}
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    recorded = _patch_boundary(monkeypatch, result=False)

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "hello world"}
    assert len(recorded) == 1


def test_zero_resolved_cost_is_still_recorded(monkeypatch):
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {"text": "hello world", "inference_status": {"cost": 0.0}}
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    recorded = _patch_boundary(monkeypatch)

    response = _post_audio(app)

    assert response.status_code == 200
    assert len(recorded) == 1
    assert recorded[0]["cost"] == 0.0


def test_document_branch_never_records_resolved_cost(monkeypatch):
    app = _make_app()
    _common_monkeypatches(monkeypatch)

    recorded = _patch_boundary(monkeypatch)
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
    assert recorded == []


def test_audio_records_through_the_real_boundary(monkeypatch):
    """One end-to-end pass with the real boundary in place: only the database
    writer is stubbed, so the route's own facts must survive derivation,
    registration and persistence."""
    app = _make_app()
    _common_monkeypatches(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")

    payload = {"text": "hello world", "inference_status": {"cost": 0.0225}}
    monkeypatch.setattr(
        multimodal_route, "asr_response", lambda *a, **k: FakeAsrResponse(200, payload)
    )

    writes = []
    monkeypatch.setattr(
        usage_recording,
        "log_usage_with_resolved_cost",
        lambda **kwargs: (writes.append(kwargs), 4242)[1],
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_id",
        lambda user_id: {"id": user_id, "username": "user@example.com"},
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username, "client": "dino"},
    )

    response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "hello world"}

    assert len(writes) == 1
    write = writes[0]
    assert write["user_id"] == 42
    assert write["cost"] == 0.0225
    assert write["provider"] == "Deepinfra"
    assert write["model"] == "test-asr-model"
    assert write["service"] == "/transcribe"
    # Derived by the boundary, never stated by the route.
    assert write["request_id"] == response.headers["X-Request-ID"]
    assert write["source"] == "dino"


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


def _patch_token_boundary(monkeypatch, *, result=True):
    """Capture what the image branch states to the token boundary."""
    recorded = []

    def _record(**kwargs):
        recorded.append(kwargs)
        return result

    monkeypatch.setattr(multimodal_route, "record_token_consumption", _record)
    return recorded


def _patch_vision(monkeypatch, token_usage, description="described image text"):
    monkeypatch.setattr(
        multimodal_route,
        "describe_image_with_usage",
        lambda *a, **k: {"description": description, "token_usage": token_usage},
    )


def test_image_branch_never_records_resolved_cost(monkeypatch):
    """Vision accounting is token-based; the resolved-cost boundary is the
    audio shape and must never be reached from the image branch."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    resolved_cost_calls = _patch_boundary(monkeypatch)
    token_calls = _patch_token_boundary(monkeypatch)
    _patch_vision(
        monkeypatch, {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
    )

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}
    assert resolved_cost_calls == []
    assert len(token_calls) == 1


def test_image_success_records_token_consumption(monkeypatch):
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    recorded = _patch_token_boundary(monkeypatch)
    _patch_vision(
        monkeypatch, {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
    )

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}
    # Public response is unchanged - no accounting metadata leaks into it.
    assert set(response.get_json().keys()) == {"text"}

    assert len(recorded) == 1
    call = recorded[0]
    assert call["user_id"] == 42
    assert call["provider"] == "test-vision-provider"
    assert call["model"] == "test-vision-model"
    assert call["service"] == "/transcribe"
    assert call["token_input"] == 12
    assert call["token_output"] == 7
    # The adopter states consumption facts only: request_id, source and the
    # row id are the boundary's, and are proven in tests/test_usage_recording.py.
    assert set(call) == {
        "user_id",
        "provider",
        "model",
        "service",
        "token_input",
        "token_output",
    }


def test_image_zero_token_usage_is_still_recorded(monkeypatch):
    """A zero/zero observation is real; the image branch has no >0 guard."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    recorded = _patch_token_boundary(monkeypatch)
    _patch_vision(
        monkeypatch, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}
    assert len(recorded) == 1
    assert recorded[0]["token_input"] == 0
    assert recorded[0]["token_output"] == 0


def test_image_recording_failure_does_not_break_response(monkeypatch):
    """Vision succeeds, the boundary reports no row: the request must still
    succeed with the original description."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    recorded = _patch_token_boundary(monkeypatch, result=False)
    _patch_vision(
        monkeypatch, {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
    )

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}
    assert len(recorded) == 1


def test_image_vision_failure_is_not_swallowed_by_usage_containment(monkeypatch):
    """An actual Vision/provider failure must still surface as a 500 and must
    not be masked by the Usage-accounting failure containment."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)

    def raise_vision_failure(*a, **k):
        raise RuntimeError("vision provider is down")

    recorded = _patch_token_boundary(monkeypatch)
    monkeypatch.setattr(
        multimodal_route, "describe_image_with_usage", raise_vision_failure
    )

    response = _post_image(app)

    assert response.status_code == 500
    assert "vision provider is down" in response.get_json()["error"]
    assert recorded == []


def test_image_records_through_the_real_boundary(monkeypatch):
    """The image mirror of the audio end-to-end pass: with the real boundary
    in place, only the token writer is stubbed."""
    app = _make_image_app()
    _common_monkeypatches(monkeypatch)
    _patch_vision(
        monkeypatch, {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}
    )

    writes = []
    monkeypatch.setattr(
        usage_recording,
        "log_token_usage",
        lambda **kwargs: (writes.append(kwargs), 4242)[1],
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_id",
        lambda user_id: {"id": user_id, "username": "user@example.com"},
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username, "client": "dino"},
    )

    response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": "described image text"}

    assert len(writes) == 1
    write = writes[0]
    assert write["user_id"] == 42
    assert write["token_input"] == 12
    assert write["token_output"] == 7
    assert write["provider"] == "test-vision-provider"
    assert write["model"] == "test-vision-model"
    assert write["service"] == "/transcribe"
    # Derived by the boundary, never stated by the route.
    assert write["request_id"] == response.headers["X-Request-ID"]
    assert write["source"] == "dino"
