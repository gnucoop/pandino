"""THIRD ADOPTER SLICE T5 — /transcribe Usage-accounting failure.

The single fact under test, owned by routes/multimodal.py::asr_parse():

    transcribe_usage_accounting_failed  (ERROR, details.branch/reason)

Semantics: the primary provider-backed operation SUCCEEDED, the subsequent
Usage-accounting attempt FAILED, the failure was CONTAINED, and the
caller-visible primary result remains successful (HTTP 200, unchanged body).

T5 replaces the two legacy runtime logs `asr_usage_accounting_failed` and
`vision_usage_accounting_failed`. Both carried `exc_info=True`, so T5 must
preserve equivalent RUNTIME diagnostic depth while the PERSISTED Operational
snapshot stays traceback-free and message-safe. Those two projections are
independent and are asserted independently.

Only shared/external seams are monkeypatched (shared auth, the shared
asr_response / describe_image_with_usage provider calls, the Usage writers).
The route's pre-existing accounting try/except boundaries, its fail-open
behaviour, its HTTP returns and the Operational logging boundary are all
exercised for real.
"""

import io
import logging
from types import SimpleNamespace

import pytest
from flask import Flask

from routes import multimodal as multimodal_route
from utils.logging_config import (
    LOG_FORMAT,
    ContextDefaultsFilter,
    UtcIsoFormatter,
    register_request_context_hooks,
)
from utils.operational_persistence import (
    OperationalPersistenceHandler,
    snapshot_from_record,
)

_EVENT = "transcribe_usage_accounting_failed"

_HEADERS = {
    "X-API-KEY": "test-key",
    "X-USER-EMAIL": "user@example.com",
    "X-USER-NAME": "Test User",
}

_ASR_PROVIDER = "Deepinfra"
_ASR_MODEL = "test-asr-model"
_TRANSCRIPT = "the quick brown fox was transcribed"

_VISION_PROVIDER = "test-vision-provider"
_VISION_MODEL = "test-vision-model"
_DESCRIPTION = "A scanned receipt whose description must never be persisted."

_IMAGE_BYTES = b"fake-image-bytes-never-persisted"
_IMAGE_FILENAME = "receipt-scan.png"
_IMAGE_MIMETYPE = "image/png"

_RAW_EXCEPTION_TEXT = "accounting-internal-detail-should-never-be-persisted"

_DEEPINFRA_COST = 0.0123
_TOKEN_USAGE = {"input_tokens": 11, "output_tokens": 22}

# Everything T5 is forbidden to persist on the accounting Operational surface.
# provider/model/error_type are deliberately NOT here: they are ratified
# fact-local metadata on this event.
_FORBIDDEN = (
    # identity / credentials / headers
    "X-API-KEY",
    "X-USER-EMAIL",
    "X-USER-NAME",
    "test-key",
    "user@example.com",
    "Test User",
    "Test_User",
    "dino",
    # secrets, env-var names, config values, base URL, price/rate
    "DEEPINFRA_API_KEY",
    "MISTRAL_API_KEY",
    "fake-key",
    "http://asr.internal",
    "0.006",
    "asr_mistral_price_per_minute_usd",
    # Usage / accounting values and Usage row identity
    "cost",
    "0.0123",
    "token_input",
    "token_output",
    "input_tokens",
    "output_tokens",
    "token_usage",
    "audio_seconds",
    "prompt_audio_seconds",
    "inference_status",
    "log_id",
    "usage_log_id",
    "777",
    "778",
    "user_id",
    "42",
    # result content, filename, MIME, data URL
    _TRANSCRIPT,
    _DESCRIPTION,
    "scanned receipt",
    _IMAGE_FILENAME,
    "receipt-scan",
    _IMAGE_MIMETYPE,
    "base64",
    "data:image",
    "fake-image-bytes",
    # raw exception text / traceback
    _RAW_EXCEPTION_TEXT,
    "Traceback",
)


class FakeAsrResponse:
    """Stands in for the requests.Response the shared provider call returns."""

    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _make_app(*, asr_provider=_ASR_PROVIDER):
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        models=SimpleNamespace(
            asr_model=_ASR_MODEL,
            asr_provider=asr_provider,
            asr_base_url="http://asr.internal",
            asr_mistral_price_per_minute_usd=0.006,
            vision_provider=_VISION_PROVIDER,
            vision_model=_VISION_MODEL,
        )
    )
    register_request_context_hooks(app)
    app.register_blueprint(multimodal_route.multimodal_bp)
    return app


def _patch_auth_and_user(monkeypatch):
    """Only shared/external seams: auth and the user lookup."""
    monkeypatch.setattr(multimodal_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(
        multimodal_route.database_pg,
        "get_user_by_username",
        lambda user_email: {"id": 42, "username": user_email, "client": "dino"},
    )
    monkeypatch.setattr(multimodal_route, "set_usage_log_id", lambda log_id: None)


def _patch_asr(monkeypatch, result):
    calls: list[tuple] = []

    def _asr_response(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(multimodal_route, "asr_response", _asr_response)
    return calls


def _patch_vision(monkeypatch, result):
    calls: list[tuple] = []

    def _describe(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(multimodal_route, "describe_image_with_usage", _describe)
    return calls


def _audio_payload():
    """A real DeepInfra-shaped payload, so resolve_asr_cost runs for real."""
    return {
        "text": _TRANSCRIPT,
        "inference_status": {"cost": _DEEPINFRA_COST},
    }


def _post_audio(app):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(b"fake-audio-bytes"), "audio.wav", "audio/wav")},
        content_type="multipart/form-data",
        headers=_HEADERS,
    )


def _post_image(app):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(_IMAGE_BYTES), _IMAGE_FILENAME, _IMAGE_MIMETYPE)},
        content_type="multipart/form-data",
        headers=_HEADERS,
    )


def _post_document(app, filename="report.pdf"):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(b"%PDF-1.4 fake"), filename, "application/pdf")},
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


def _operational_event_sequence(caplog):
    return [
        r.maui_event
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
    ]


def _assert_no_accounting_event(caplog):
    assert _operational_records(caplog, _EVENT) == [], (
        f"{_EVENT} must not be emitted when accounting did not fail"
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
    # duration_ms is ABSENT on this fact (§8.6).
    assert snapshot.duration_ms is None
    assert not hasattr(record, "maui_duration_ms")
    # The snapshot never carries exception context at all.
    assert not hasattr(snapshot, "exc_info")
    assert not hasattr(snapshot, "traceback")


def _assert_accounting_fact(record, *, branch, provider, model, error_type):
    """The whole ratified positive payload of transcribe_usage_accounting_failed."""
    assert record.levelno == logging.ERROR
    assert record.maui_details == {"branch": branch, "reason": "accounting_error"}
    assert record.maui_provider == provider
    assert record.maui_model == model
    assert record.maui_error_type == error_type

    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    assert snapshot.level == "ERROR"
    assert snapshot.event == _EVENT
    assert snapshot.provider == provider
    assert snapshot.model == model
    assert snapshot.error_type == error_type

    _assert_free_of_forbidden(record)


def _assert_runtime_exception_context(record):
    """The legacy logs carried exc_info=True; the replacement must too."""
    assert record.exc_info is not None, (
        "the accounting failure must remain exception-aware at runtime"
    )
    assert record.exc_info[0] is not None


# ---------------------------------------------------------------------------
# A. audio accounting failure — the degraded-success shape
# ---------------------------------------------------------------------------


def test_audio_accounting_failure_persists_the_contained_fact(monkeypatch, caplog):
    """The ASR operation succeeded; the Usage write then failed and was contained."""
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    calls = _patch_asr(monkeypatch, FakeAsrResponse(200, _audio_payload()))

    def _boom(**kwargs):
        raise RuntimeError(_RAW_EXCEPTION_TEXT)

    monkeypatch.setattr(multimodal_route, "log_usage_with_resolved_cost", _boom)

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    # Fail-open: the caller-visible primary result is untouched.
    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}
    assert len(calls) == 1, "the shared provider call must actually have run"

    # The degraded-success timeline, in order.
    assert _operational_event_sequence(caplog) == [
        "transcribe_branch_selected",
        "transcribe_operation_completed",
        _EVENT,
    ]

    record = _the_operational_record(caplog, _EVENT)
    _assert_accounting_fact(
        record,
        branch="audio",
        provider=_ASR_PROVIDER,
        model=_ASR_MODEL,
        error_type="RuntimeError",
    )
    _assert_runtime_exception_context(record)

    # The accounting failure never erases or replaces the primary completion.
    completed = _the_operational_record(caplog, "transcribe_operation_completed")
    assert completed.maui_details == {"branch": "audio"}
    assert completed.levelno == logging.INFO
    assert _operational_records(caplog, "transcribe_operation_failed") == []
    assert _operational_records(caplog, "transcribe_operation_blocked") == []
    branch = _the_operational_record(caplog, "transcribe_branch_selected")
    assert branch.maui_details == {"branch": "audio"}


@pytest.mark.parametrize(
    "seam, exception, expected_error_type",
    [
        # The single existing handler fuses cost resolution, user lookup and
        # the Usage writer; the coarse reason is the same for all of them.
        ("log_usage_with_resolved_cost", RuntimeError(_RAW_EXCEPTION_TEXT), "RuntimeError"),
        ("set_usage_log_id", ValueError(_RAW_EXCEPTION_TEXT), "ValueError"),
    ],
)
def test_audio_accounting_error_type_follows_the_real_exception(
    monkeypatch, caplog, seam, exception, expected_error_type
):
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(monkeypatch, FakeAsrResponse(200, _audio_payload()))
    monkeypatch.setattr(multimodal_route, "log_usage_with_resolved_cost", lambda **k: 777)

    def _boom(*args, **kwargs):
        raise exception

    monkeypatch.setattr(multimodal_route, seam, _boom)

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}

    record = _the_operational_record(caplog, _EVENT)
    _assert_accounting_fact(
        record,
        branch="audio",
        provider=_ASR_PROVIDER,
        model=_ASR_MODEL,
        error_type=expected_error_type,
    )
    _assert_runtime_exception_context(record)


def test_audio_user_lookup_miss_is_the_same_contained_accounting_fact(
    monkeypatch, caplog
):
    """The RuntimeError the route itself raises on a user miss stays inside."""
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    monkeypatch.setattr(
        multimodal_route.database_pg, "get_user_by_username", lambda user_email: None
    )
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(monkeypatch, FakeAsrResponse(200, _audio_payload()))

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}

    record = _the_operational_record(caplog, _EVENT)
    _assert_accounting_fact(
        record,
        branch="audio",
        provider=_ASR_PROVIDER,
        model=_ASR_MODEL,
        error_type="RuntimeError",
    )
    # The user lookup failure message embeds the caller's e-mail; it must not
    # reach any Operational surface.
    _assert_runtime_exception_context(record)


# ---------------------------------------------------------------------------
# B. image accounting failure — the degraded-success shape
# ---------------------------------------------------------------------------


def test_image_accounting_failure_persists_the_contained_fact(monkeypatch, caplog):
    """The vision operation succeeded; the Usage write then failed and was contained."""
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    calls = _patch_vision(
        monkeypatch, {"description": _DESCRIPTION, "token_usage": _TOKEN_USAGE}
    )

    def _boom(**kwargs):
        raise RuntimeError(_RAW_EXCEPTION_TEXT)

    monkeypatch.setattr(multimodal_route, "log_token_usage", _boom)

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    # Fail-open: the caller-visible primary result is untouched.
    assert response.status_code == 200
    assert response.get_json() == {"text": _DESCRIPTION}
    assert len(calls) == 1, "the shared vision helper must actually have run"

    # The degraded-success timeline, in order. transcribe_operation_completed
    # marks completion of the PRIMARY image operation, so it precedes the
    # secondary accounting attempt (T4.1 corrected its placement).
    assert _operational_event_sequence(caplog) == [
        "transcribe_branch_selected",
        "transcribe_operation_completed",
        _EVENT,
    ]

    record = _the_operational_record(caplog, _EVENT)
    _assert_accounting_fact(
        record,
        branch="image",
        provider=_VISION_PROVIDER,
        model=_VISION_MODEL,
        error_type="RuntimeError",
    )
    _assert_runtime_exception_context(record)

    completed = _the_operational_record(caplog, "transcribe_operation_completed")
    assert completed.maui_details == {"branch": "image"}
    assert completed.levelno == logging.INFO
    assert _operational_records(caplog, "transcribe_operation_failed") == []
    assert _operational_records(caplog, "transcribe_operation_blocked") == []
    branch = _the_operational_record(caplog, "transcribe_branch_selected")
    assert branch.maui_details == {"branch": "image"}


def test_image_missing_token_usage_is_the_same_contained_accounting_fact(
    monkeypatch, caplog
):
    """A malformed vision result reaches the same single coarse handler."""
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    monkeypatch.setattr(multimodal_route, "log_token_usage", lambda **k: 778)
    _patch_vision(monkeypatch, {"description": _DESCRIPTION})

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": _DESCRIPTION}

    record = _the_operational_record(caplog, _EVENT)
    _assert_accounting_fact(
        record,
        branch="image",
        provider=_VISION_PROVIDER,
        model=_VISION_MODEL,
        error_type="KeyError",
    )
    _assert_runtime_exception_context(record)


# ---------------------------------------------------------------------------
# C. negative — a successful accounting attempt emits NO accounting fact
# ---------------------------------------------------------------------------


def test_audio_successful_accounting_emits_no_accounting_fact(monkeypatch, caplog):
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(monkeypatch, FakeAsrResponse(200, _audio_payload()))

    written: list[dict] = []
    monkeypatch.setattr(
        multimodal_route,
        "log_usage_with_resolved_cost",
        lambda **kwargs: written.append(kwargs) or 777,
    )

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}
    assert len(written) == 1, "the accounting write must actually have succeeded"
    assert _operational_event_sequence(caplog) == [
        "transcribe_branch_selected",
        "transcribe_operation_completed",
    ]
    _assert_no_accounting_event(caplog)


def test_image_successful_accounting_emits_no_accounting_fact(monkeypatch, caplog):
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    _patch_vision(
        monkeypatch, {"description": _DESCRIPTION, "token_usage": _TOKEN_USAGE}
    )

    written: list[dict] = []
    monkeypatch.setattr(
        multimodal_route,
        "log_token_usage",
        lambda **kwargs: written.append(kwargs) or 778,
    )

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": _DESCRIPTION}
    assert len(written) == 1, "the accounting write must actually have succeeded"
    assert _operational_event_sequence(caplog) == [
        "transcribe_branch_selected",
        "transcribe_operation_completed",
    ]
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# D. negative — the deliberate unsupported-ASR-provider accounting skip
# ---------------------------------------------------------------------------


def test_unsupported_asr_provider_skip_is_not_an_accounting_failure(
    monkeypatch, caplog
):
    """Accounting is deliberately skipped outside Deepinfra/Mistral.

    That is an accepted NON-event, not a contained failure, and the existing
    gap is explicitly out of scope for this adopter.
    """
    app = _make_app(asr_provider="SelfHosted")
    _patch_auth_and_user(monkeypatch)
    _patch_asr(monkeypatch, FakeAsrResponse(200, {"text": _TRANSCRIPT}))

    def _must_not_run(**kwargs):
        raise AssertionError("accounting must not be attempted for this provider")

    monkeypatch.setattr(
        multimodal_route, "log_usage_with_resolved_cost", _must_not_run
    )

    with caplog.at_level(logging.INFO):
        response = _post_audio(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}
    assert _operational_event_sequence(caplog) == [
        "transcribe_branch_selected",
        "transcribe_operation_completed",
    ]
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# E. negative — the document branch performs no accounting at all
# ---------------------------------------------------------------------------


def test_document_branch_never_emits_an_accounting_fact(monkeypatch, caplog):
    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    monkeypatch.setattr(
        multimodal_route,
        "extract_and_normalize_document",
        lambda doc_input: {"text": "extracted document text"},
    )

    with caplog.at_level(logging.INFO):
        response = _post_document(app)

    assert response.status_code == 200
    assert _operational_event_sequence(caplog) == [
        "transcribe_branch_selected",
        "transcribe_operation_completed",
    ]
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# F. the C8.2 two-projection split, at the REAL logging boundary
# ---------------------------------------------------------------------------


class _ListSink:
    def __init__(self):
        self.received = []

    def __call__(self, snapshot):
        self.received.append(snapshot)


def _run_through_the_real_logging_boundary(post, app):
    """Attach a real OperationalPersistenceHandler to the real route logger.

    Returns (captured_records, sink, response). Nothing about the emission is
    simulated: no hand-built LogRecord, no mocked logger, no fake snapshot.
    """
    sink = _ListSink()
    handler = OperationalPersistenceHandler(sink)
    route_logger = multimodal_route.logger

    captured_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    capture = _Capture()
    capture.addFilter(ContextDefaultsFilter())

    previous_level = route_logger.level
    route_logger.addHandler(handler)
    route_logger.addHandler(capture)
    route_logger.setLevel(logging.INFO)
    try:
        response = post(app)
    finally:
        route_logger.removeHandler(handler)
        route_logger.removeHandler(capture)
        route_logger.setLevel(previous_level)

    return captured_records, sink, response


def _assert_traceback_split(captured_records, sink, sentinel, *, branch,
                            provider, model, error_type):
    failures = [
        r for r in captured_records if getattr(r, "maui_event", None) == _EVENT
    ]
    assert len(failures) == 1
    record = failures[0]

    # --- stderr half: the real formatter keeps the traceback ----------------
    formatted = UtcIsoFormatter(LOG_FORMAT).format(record)
    assert "Traceback" in formatted
    assert error_type in formatted
    assert sentinel in formatted
    # Formatting has now mutated the shared record by caching exc_text.
    assert record.exc_text and sentinel in record.exc_text

    # --- Operational half: same record, bounded snapshot --------------------
    snapshots = [s for s in sink.received if s.event == _EVENT]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.level == "ERROR"
    assert snapshot.logger == multimodal_route.logger.name
    assert snapshot.provider == provider
    assert snapshot.model == model
    assert snapshot.error_type == error_type
    assert snapshot.duration_ms is None
    assert f'"branch": "{branch}"' in snapshot.details_json
    assert '"reason": "accounting_error"' in snapshot.details_json
    assert not hasattr(snapshot, "exc_info")
    assert not hasattr(snapshot, "traceback")

    text = str(vars(snapshot)) if hasattr(snapshot, "__dict__") else str(snapshot)
    assert sentinel not in text
    assert "Traceback" not in text

    # Taken again AFTER stderr formatting, the snapshot is still clean: the
    # shared-record exc_text mutation cannot leak into persistence.
    post_format_snapshot = snapshot_from_record(record)
    assert sentinel not in str(post_format_snapshot)
    assert "Traceback" not in str(post_format_snapshot)

    # The primary completion fact survives the accounting failure.
    completions = [
        s
        for s in sink.received
        if s.event == "transcribe_operation_completed"
    ]
    assert len(completions) == 1


def test_audio_accounting_failure_splits_traceback_from_snapshot(monkeypatch):
    sentinel = "SENSITIVE-ACCOUNTING-BODY-secret-4242"

    class AccountingBoom(Exception):
        pass

    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    _patch_asr(monkeypatch, FakeAsrResponse(200, _audio_payload()))

    def _boom(**kwargs):
        raise AccountingBoom(f"usage writer said {sentinel}")

    monkeypatch.setattr(multimodal_route, "log_usage_with_resolved_cost", _boom)

    captured, sink, response = _run_through_the_real_logging_boundary(
        _post_audio, app
    )

    assert response.status_code == 200
    assert response.get_json() == {"text": _TRANSCRIPT}

    _assert_traceback_split(
        captured,
        sink,
        sentinel,
        branch="audio",
        provider=_ASR_PROVIDER,
        model=_ASR_MODEL,
        error_type="AccountingBoom",
    )


def test_image_accounting_failure_splits_traceback_from_snapshot(monkeypatch):
    sentinel = "SENSITIVE-ACCOUNTING-BODY-secret-9191"

    class AccountingBoom(Exception):
        pass

    app = _make_app()
    _patch_auth_and_user(monkeypatch)
    _patch_vision(
        monkeypatch, {"description": _DESCRIPTION, "token_usage": _TOKEN_USAGE}
    )

    def _boom(**kwargs):
        raise AccountingBoom(f"usage writer said {sentinel}")

    monkeypatch.setattr(multimodal_route, "log_token_usage", _boom)

    captured, sink, response = _run_through_the_real_logging_boundary(
        _post_image, app
    )

    assert response.status_code == 200
    assert response.get_json() == {"text": _DESCRIPTION}

    _assert_traceback_split(
        captured,
        sink,
        sentinel,
        branch="image",
        provider=_VISION_PROVIDER,
        model=_VISION_MODEL,
        error_type="AccountingBoom",
    )
