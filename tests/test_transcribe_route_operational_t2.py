"""THIRD ADOPTER SLICE T2 — /transcribe audio primary-operation outcomes.

Facts under test, all owned by routes/multimodal.py::asr_parse() and all
scoped to the AUDIO branch:

    transcribe_operation_blocked    (WARNING, details.branch/reason)
    transcribe_operation_completed  (INFO,    details.branch)
    transcribe_operation_failed     (ERROR,   details.branch/reason)

Only external/shared seams are monkeypatched (shared auth, the shared
asr_response provider call, Usage writers). The audio guards, the provider
dispatch, the response interpretation, the HTTP return behaviour and the
Operational logging boundary are all exercised for real.

T5 owns transcribe_usage_accounting_failed; T2 must never emit it.
"""

import io
import logging
from types import SimpleNamespace

from flask import Flask

from routes import multimodal as multimodal_route
from utils.logging_config import register_request_context_hooks
from utils.operational_persistence import snapshot_from_record

_HEADERS = {
    "X-API-KEY": "test-key",
    "X-USER-EMAIL": "user@example.com",
    "X-USER-NAME": "Test User",
}

_PROVIDER = "Deepinfra"
_MODEL = "test-asr-model"
_TRANSCRIPT = "the quick brown fox was transcribed"
_PROVIDER_BODY = "provider-error-body-should-never-be-persisted"

# Everything T2 is forbidden to persist on ANY audio Operational surface.
# provider/model are deliberately NOT here: they are ratified fact-local
# metadata on the audio outcome events.
_FORBIDDEN = (
    # identity / credentials / headers
    "X-API-KEY",
    "X-USER-EMAIL",
    "X-USER-NAME",
    "test-key",
    "user@example.com",
    "Test User",
    "Test_User",
    # secrets, env-var names, config values, base URL, price/rate
    "DEEPINFRA_API_KEY",
    "MISTRAL_API_KEY",
    "fake-key",
    "http://asr.internal",
    "0.006",
    "asr_base_url",
    "asr_mistral_price_per_minute_usd",
    # content / result / provider body / raw exception text
    _TRANSCRIPT,
    _PROVIDER_BODY,
    "not json at all",
    "inference_status",
    # Usage / accounting values
    "input_tokens",
    "output_tokens",
    "777",
    "audio_seconds",
    "Traceback",
)

_OUTCOME_EVENTS = (
    "transcribe_operation_blocked",
    "transcribe_operation_completed",
    "transcribe_operation_failed",
)


class FakeAsrResponse:
    """Stands in for the requests.Response the shared provider call returns."""

    def __init__(self, status_code, payload=None, raise_on_json=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self._raise_on_json = raise_on_json
        self.text = text

    def json(self):
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._payload


def _make_app(
    *,
    asr_model=_MODEL,
    asr_provider=_PROVIDER,
    asr_base_url="http://asr.internal",
    asr_mistral_price_per_minute_usd=0.006,
):
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        models=SimpleNamespace(
            asr_model=asr_model,
            asr_provider=asr_provider,
            asr_base_url=asr_base_url,
            asr_mistral_price_per_minute_usd=asr_mistral_price_per_minute_usd,
            vision_provider="test-vision-provider",
            vision_model="test-vision-model",
        )
    )
    register_request_context_hooks(app)
    app.register_blueprint(multimodal_route.multimodal_bp)
    return app


def _patch_shared_seams(monkeypatch):
    """Only shared/external seams: auth and the Usage writers."""
    monkeypatch.setattr(multimodal_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(
        multimodal_route.database_pg,
        "get_user_by_username",
        lambda user_email: {"id": 42, "username": user_email, "client": "dino"},
    )
    monkeypatch.setattr(multimodal_route, "set_usage_log_id", lambda log_id: None)
    monkeypatch.setattr(
        multimodal_route, "record_resolved_consumption", lambda **kwargs: True
    )


def _patch_asr(monkeypatch, result):
    """Replace only the shared provider call; record whether it ran."""
    calls: list[tuple] = []

    def _asr_response(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(multimodal_route, "asr_response", _asr_response)
    return calls


def _post_audio(app):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(b"fake-audio-bytes"), "audio.wav", "audio/wav")},
        content_type="multipart/form-data",
        headers=_HEADERS,
    )


def _operational_records(caplog, event):
    return [
        r
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
        and getattr(r, "maui_event", None) == event
    ]


def _the_operational_record(caplog, event):
    records = _operational_records(caplog, event)
    assert len(records) == 1, (
        f"expected exactly one {event} record, got {len(records)}"
    )
    return records[0]


def _assert_exactly_one_outcome(caplog, event):
    """The audio branch produces exactly one primary outcome per request."""
    for other in _OUTCOME_EVENTS:
        if other == event:
            continue
        assert _operational_records(caplog, other) == [], (
            f"{other} must not accompany {event}"
        )
    return _the_operational_record(caplog, event)


def _assert_t1_branch_fact_intact(caplog):
    """T1 semantics: exactly one branch_selected, branch=audio, INFO."""
    record = _the_operational_record(caplog, "transcribe_branch_selected")
    assert record.levelno == logging.INFO
    assert record.maui_details == {"branch": "audio"}


def _assert_no_accounting_event(caplog):
    assert _operational_records(caplog, "transcribe_usage_accounting_failed") == [], (
        "transcribe_usage_accounting_failed is owned by T5, not T2"
    )


def _assert_free_of_forbidden(record):
    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    surfaces = [
        record.getMessage(),
        str(getattr(record, "maui_details", None)),
        str(getattr(record, "maui_message", None)),
        str(getattr(record, "maui_error_type", None)),
        str(snapshot.details_json),
        str(snapshot.message),
        str(snapshot.error_type),
        str(snapshot.provider),
        str(snapshot.model),
    ]
    for surface in surfaces:
        for needle in _FORBIDDEN:
            assert needle not in surface, (
                f"forbidden content {needle!r} reached an Operational surface: "
                f"{surface!r}"
            )
    # duration_ms is forbidden on every T2 fact.
    assert snapshot.duration_ms is None
    assert not hasattr(record, "maui_duration_ms")


def _assert_legacy_asr_request_failed_absent(caplog):
    for record in caplog.records:
        assert "asr_request_failed" not in record.getMessage(), (
            "the legacy asr_request_failed log was replaced by "
            "transcribe_operation_failed{branch=audio,reason=http_error} and "
            "must not be retained alongside it"
        )


# ---------------------------------------------------------------------------
# A. transcribe_operation_blocked — audio prerequisite guards
# ---------------------------------------------------------------------------


def test_missing_model_blocks_before_the_provider_operation(monkeypatch, caplog):
    app = _make_app(asr_model=None)
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    calls = _patch_asr(monkeypatch, FakeAsrResponse(200, {"text": _TRANSCRIPT}))

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Existing HTTP behaviour is unchanged.
    assert response.status_code == 500
    assert response.get_json() == {"error": "Missing ASR configuration"}
    assert calls == [], "the provider operation must not be invoked when blocked"

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_blocked")
    assert record.levelno == logging.WARNING
    assert record.maui_details == {"branch": "audio", "reason": "missing_model"}
    # The model is by definition the falsy value the guard just rejected;
    # persisting it would say nothing.
    assert not hasattr(record, "maui_model")
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_missing_api_key_blocks_before_the_provider_operation(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    calls = _patch_asr(monkeypatch, FakeAsrResponse(200, {"text": _TRANSCRIPT}))

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Missing ASR configuration"}
    assert calls == [], "the provider operation must not be invoked when blocked"

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_blocked")
    assert record.levelno == logging.WARNING
    assert record.maui_details == {"branch": "audio", "reason": "missing_api_key"}
    # The provider is the value the guard tested against, so it is meaningful.
    assert record.maui_provider == _PROVIDER
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_missing_price_blocks_before_the_provider_operation(monkeypatch, caplog):
    app = _make_app(
        asr_provider="Mistral", asr_mistral_price_per_minute_usd=None
    )
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")
    calls = _patch_asr(monkeypatch, FakeAsrResponse(200, {"text": _TRANSCRIPT}))

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Missing ASR configuration"}
    assert calls == [], "the provider operation must not be invoked when blocked"

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_blocked")
    assert record.levelno == logging.WARNING
    assert record.maui_details == {"branch": "audio", "reason": "missing_price"}
    assert record.maui_provider == "Mistral"
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# B. The ambiguous ValueError boundary — DESIGNED non-emission
# ---------------------------------------------------------------------------


def test_ambiguous_asr_value_error_is_intentionally_unclassified(
    monkeypatch, caplog
):
    """The silence on this path is DESIGNED, not an oversight.

    The route's `except ValueError` around asr_response is a semantic union:
    the shared provider call raises ValueError for a never-started
    missing-base-url precondition, but reachable requests exceptions
    (MissingSchema / InvalidSchema / InvalidURL / InvalidHeader) also subclass
    ValueError and are raised from INSIDE an attempted call.

    The boundary therefore cannot distinguish never-started from
    started-and-failed, so it persists NEITHER transcribe_operation_blocked
    (which would falsely claim the operation never started) NOR
    transcribe_operation_failed (which would falsely claim it did). There is
    deliberately no `missing_base_url` reason in the vocabulary.

    Do NOT "fix" this by emitting an outcome here: it would persist a false
    semantic classification for a real, reachable case. The granularity loss
    is accepted, and branch_selected with no primary outcome is the correct
    represented timeline.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(monkeypatch, ValueError("Missing base_url for self-hosted ASR"))

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Existing HTTP behaviour is unchanged.
    assert response.status_code == 500
    assert response.get_json() == {
        "error": "Missing base_url for self-hosted ASR"
    }

    # T1 still represents the request; no primary outcome is claimed.
    _assert_t1_branch_fact_intact(caplog)
    for event in _OUTCOME_EVENTS:
        assert _operational_records(caplog, event) == [], (
            f"{event} must not be emitted at the ambiguous ValueError boundary"
        )
    for record in caplog.records:
        assert "missing_base_url" not in record.getMessage()
        assert "missing_base_url" not in str(
            getattr(record, "maui_details", "")
        )
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# C. transcribe_operation_completed — audio success
# ---------------------------------------------------------------------------


def test_audio_success_persists_one_completion_with_authoritative_identity(
    monkeypatch, caplog
):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(
        monkeypatch,
        FakeAsrResponse(
            200,
            {"text": _TRANSCRIPT, "inference_status": {"cost": 0.01}},
        ),
    )

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Existing HTTP behaviour is unchanged.
    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_completed")
    assert record.levelno == logging.INFO
    assert record.maui_details == {"branch": "audio"}
    # asr_response uses exactly these route-selected values with no
    # re-selection and no default-model fallback, so both are authoritative.
    assert record.maui_provider == _PROVIDER
    assert record.maui_model == _MODEL
    assert not hasattr(record, "maui_error_type")
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_audio_completion_precedes_the_accounting_block(monkeypatch, caplog):
    """Ordering pin for the T5 degraded-success timeline.

    The primary-operation completion must already be emitted at the point the
    Usage-accounting block begins, so that a later accounting failure reads as
    `operation_completed -> usage_accounting_failed` rather than the reverse.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(
        monkeypatch,
        FakeAsrResponse(200, {"text": _TRANSCRIPT, "inference_status": {"cost": 0.01}}),
    )

    seen_at_accounting_time: list[int] = []

    def _failing_record(**kwargs):
        seen_at_accounting_time.append(
            len(_operational_records(caplog, "transcribe_operation_completed"))
        )
        return False

    monkeypatch.setattr(
        multimodal_route, "record_resolved_consumption", _failing_record
    )

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Usage stays fail-open: the caller still gets the transcription.
    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}

    assert seen_at_accounting_time == [1], (
        "the audio completion must be emitted BEFORE the accounting block runs"
    )
    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_completed")
    assert record.maui_details == {"branch": "audio"}
    # T5 now owns the accounting fact this ordering pin exists to protect. The
    # completion must still come FIRST; T5 must not replace or duplicate it.
    accounting = _operational_records(caplog, "transcribe_usage_accounting_failed")
    assert len(accounting) == 1
    assert accounting[0].maui_details == {
        "branch": "audio",
        "reason": "usage_not_recorded",
    }
    assert caplog.records.index(record) < caplog.records.index(accounting[0])


# ---------------------------------------------------------------------------
# D. transcribe_operation_failed — audio failure boundaries
# ---------------------------------------------------------------------------


def test_provider_non_200_persists_http_error_without_the_response_body(
    monkeypatch, caplog
):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(monkeypatch, FakeAsrResponse(503, text=_PROVIDER_BODY))

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Existing HTTP behaviour is unchanged.
    assert response.status_code == 500
    assert response.get_json() == {"error": "ASR transcription failed"}

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.levelno == logging.ERROR
    assert record.maui_details == {"branch": "audio", "reason": "http_error"}
    assert record.maui_provider == _PROVIDER
    assert record.maui_model == _MODEL
    # No exception exists at this boundary.
    assert not hasattr(record, "maui_error_type")
    _assert_free_of_forbidden(record)

    # The single legacy route log this slice replaces is gone, and with it the
    # unbounded provider response body it used to interpolate.
    _assert_legacy_asr_request_failed_absent(caplog)
    for other in caplog.records:
        assert _PROVIDER_BODY not in other.getMessage()

    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_unparseable_provider_body_persists_invalid_response_with_error_type(
    monkeypatch, caplog
):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(
        monkeypatch,
        FakeAsrResponse(200, raise_on_json=ValueError("not json at all")),
    )

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Existing HTTP behaviour is unchanged, str(e) included.
    assert response.status_code == 500
    assert response.get_json() == {"error": "Invalid JSON from ASR: not json at all"}

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.levelno == logging.ERROR
    assert record.maui_details == {"branch": "audio", "reason": "invalid_response"}
    assert record.maui_provider == _PROVIDER
    assert record.maui_model == _MODEL
    # Class name only — never str(e).
    assert record.maui_error_type == "ValueError"
    assert record.exc_info is None
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_payload_without_usable_text_persists_missing_result(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(monkeypatch, FakeAsrResponse(200, {"inference_status": {"cost": 0.01}}))

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Existing HTTP behaviour is unchanged.
    assert response.status_code == 500
    assert response.get_json() == {"error": "ASR response missing 'text' field"}

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.levelno == logging.ERROR
    assert record.maui_details == {"branch": "audio", "reason": "missing_result"}
    assert record.maui_provider == _PROVIDER
    assert record.maui_model == _MODEL
    # No exception occurred, so no error_type is claimed.
    assert not hasattr(record, "maui_error_type")
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)
