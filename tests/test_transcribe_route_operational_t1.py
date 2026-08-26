"""THIRD ADOPTER SLICE T1 — /transcribe request rejection and dispatch.

Facts under test, both owned by routes/multimodal.py::asr_parse():

    transcribe_request_rejected   (WARNING, details.reason)
    transcribe_branch_selected    (INFO / WARNING, details.branch)

Only external/shared seams are monkeypatched (shared auth, provider calls,
document extraction, Usage writers). The route guards, the first-match-wins
dispatch, the HTTP return behaviour and the Operational logging boundary are
all exercised for real.

T1 implements no terminal fact: transcribe_operation_blocked / _completed /
_failed / transcribe_usage_accounting_failed must not appear.
"""

import io
import logging
from types import SimpleNamespace

import pytest
from flask import Flask

from routes import multimodal as multimodal_route
from utils.logging_config import register_request_context_hooks
from utils.operational_persistence import snapshot_from_record

_HEADERS = {
    "X-API-KEY": "test-key",
    "X-USER-EMAIL": "user@example.com",
    "X-USER-NAME": "Test User",
}

# Everything T1 is forbidden to persist: header names/values, credentials,
# identity, filename, MIME, extension, provider/model.
_FORBIDDEN = (
    "X-API-KEY",
    "X-USER-EMAIL",
    "X-USER-NAME",
    "test-key",
    "user@example.com",
    "Test User",
    "Test_User",
    "audio/wav",
    "image/png",
    "text/plain",
    "audio.wav",
    "picture.png",
    "report.pdf",
    "notes.txt",
    ".pdf",
    "Deepinfra",
    "test-asr-model",
    "test-vision-model",
    "matched_by",
)

_T1_EVENTS = {"transcribe_request_rejected", "transcribe_branch_selected"}
_LATER_SLICE_EVENTS = (
    "transcribe_operation_blocked",
    "transcribe_operation_completed",
    "transcribe_operation_failed",
    "transcribe_usage_accounting_failed",
)


class FakeAsrResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

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
            vision_provider="test-vision-provider",
            vision_model="test-vision-model",
        )
    )
    register_request_context_hooks(app)
    app.register_blueprint(multimodal_route.multimodal_bp)
    return app


def _patch_shared_seams(monkeypatch):
    """Only shared/external seams: auth, providers, extraction, Usage."""
    monkeypatch.setattr(multimodal_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(
        multimodal_route.database_pg,
        "get_user_by_username",
        lambda user_email: {"id": 42, "username": user_email, "client": "dino"},
    )
    monkeypatch.setattr(multimodal_route, "set_usage_log_id", lambda log_id: None)
    monkeypatch.setattr(
        multimodal_route, "log_usage_with_resolved_cost", lambda **kwargs: 777
    )
    monkeypatch.setattr(multimodal_route, "log_token_usage", lambda **kwargs: 778)


def _post(app, *, filename, mimetype, headers=None, data=None):
    payload = {"file": (io.BytesIO(b"fake-bytes"), filename, mimetype)}
    if data is not None:
        payload = data
    return app.test_client().post(
        "/transcribe",
        data=payload,
        content_type="multipart/form-data",
        headers=_HEADERS if headers is None else headers,
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


def _assert_no_later_slice_events(caplog, allowed=()):
    """Later-slice facts must not appear.

    `allowed` names events that a NOW-IMPLEMENTED later slice legitimately
    emits on the path under test, so that T1's own assertions stay scoped to
    T1 tokens instead of asserting the absence of implemented behaviour.
    """
    for event in _LATER_SLICE_EVENTS:
        if event in allowed:
            continue
        assert _operational_records(caplog, event) == [], (
            f"{event} belongs to a later slice and must not be emitted by T1"
        )


def _assert_free_of_forbidden(record):
    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    surfaces = [
        record.getMessage(),
        str(getattr(record, "maui_details", None)),
        str(getattr(record, "maui_message", None)),
        str(getattr(record, "maui_error_type", None)),
        str(getattr(record, "maui_provider", None)),
        str(getattr(record, "maui_model", None)),
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


# ---------------------------------------------------------------------------
# A. transcribe_request_rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_header, expected_body_error",
    [
        ("X-API-KEY", "Missing X-API-KEY header"),
        ("X-USER-EMAIL", "Missing X-USER-EMAIL header"),
        ("X-USER-NAME", "Missing X-USER-NAME header"),
    ],
)
def test_missing_required_header_persists_one_bounded_reason(
    monkeypatch, caplog, missing_header, expected_body_error
):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    headers = {k: v for k, v in _HEADERS.items() if k != missing_header}

    with caplog.at_level(logging.INFO):
        response = _post(
            app, filename="audio.wav", mimetype="audio/wav", headers=headers
        )

    # Existing HTTP behaviour is unchanged.
    assert response.status_code == 400
    assert response.get_json() == {"error": expected_body_error}

    record = _the_operational_record(caplog, "transcribe_request_rejected")
    assert record.levelno == logging.WARNING
    assert record.maui_details == {"reason": "missing_required_header"}
    _assert_free_of_forbidden(record)

    # Rejection is pre-dispatch: no branch was selected.
    assert _operational_records(caplog, "transcribe_branch_selected") == []
    _assert_no_later_slice_events(caplog)


def test_missing_file_persists_missing_file_reason(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = app.test_client().post(
            "/transcribe",
            data={"lang": "ENG"},
            content_type="multipart/form-data",
            headers=_HEADERS,
        )

    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing file"}

    record = _the_operational_record(caplog, "transcribe_request_rejected")
    assert record.levelno == logging.WARNING
    assert record.maui_details == {"reason": "missing_file"}
    _assert_free_of_forbidden(record)

    assert _operational_records(caplog, "transcribe_branch_selected") == []
    _assert_no_later_slice_events(caplog)


# ---------------------------------------------------------------------------
# B. transcribe_branch_selected
# ---------------------------------------------------------------------------


def _assert_single_branch_fact(caplog, branch, level, allowed_later_events=()):
    record = _the_operational_record(caplog, "transcribe_branch_selected")
    assert record.levelno == level
    assert record.maui_details == {"branch": branch}
    _assert_free_of_forbidden(record)
    assert _operational_records(caplog, "transcribe_request_rejected") == []
    _assert_no_later_slice_events(caplog, allowed=allowed_later_events)
    return record


def test_audio_dispatch_selects_audio_branch_at_info(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    monkeypatch.setattr(
        multimodal_route,
        "asr_response",
        lambda *a, **k: FakeAsrResponse(
            200, {"text": "hello", "inference_status": {"cost": 0.01}}
        ),
    )

    with caplog.at_level(logging.INFO):
        response = _post(app, filename="audio.wav", mimetype="audio/wav")

    assert response.status_code == 200
    # T2 owns the audio primary-operation outcome; it is legitimately present
    # on this successful audio path and is asserted by the T2 test module.
    _assert_single_branch_fact(
        caplog,
        "audio",
        logging.INFO,
        allowed_later_events=("transcribe_operation_completed",),
    )


def test_document_dispatch_selects_document_branch_at_info(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(
        multimodal_route,
        "extract_and_normalize_document",
        lambda doc_input: {"text": "extracted"},
    )

    with caplog.at_level(logging.INFO):
        response = _post(app, filename="report.pdf", mimetype="application/pdf")

    assert response.status_code == 200
    assert response.get_json() == {"text": "extracted"}
    # T3 is implemented, so the document completion legitimately accompanies
    # this branch fact.
    _assert_single_branch_fact(
        caplog,
        "document",
        logging.INFO,
        allowed_later_events=("transcribe_operation_completed",),
    )


def test_image_dispatch_selects_image_branch_at_info(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(
        multimodal_route,
        "describe_image_with_usage",
        lambda *a, **k: {
            "description": "a picture",
            "token_usage": {"input_tokens": 1, "output_tokens": 2},
        },
    )

    with caplog.at_level(logging.INFO):
        response = _post(app, filename="picture.png", mimetype="image/png")

    assert response.status_code == 200
    assert response.get_json() == {"text": "a picture"}
    _assert_single_branch_fact(
        caplog,
        "image",
        logging.INFO,
        allowed_later_events=("transcribe_operation_completed",),
    )


def test_fall_through_selects_reject_branch_at_warning(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = _post(app, filename="notes.txt", mimetype="text/plain")

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Unexpected file mimetype: text/plain"
    }
    _assert_single_branch_fact(caplog, "reject", logging.WARNING)


def test_document_extension_wins_over_image_mimetype(monkeypatch, caplog):
    """Ratified first-match-wins regression: document-extension dispatch
    precedes image-MIME dispatch, so an image-like MIME carrying a document
    extension must still select branch=document."""
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(
        multimodal_route,
        "extract_and_normalize_document",
        lambda doc_input: {"text": "extracted"},
    )

    def _explode(*a, **k):
        raise AssertionError("image branch must not run for a document extension")

    monkeypatch.setattr(multimodal_route, "describe_image_with_usage", _explode)

    with caplog.at_level(logging.INFO):
        response = _post(app, filename="report.pdf", mimetype="image/png")

    assert response.status_code == 200
    # T3 is implemented, so the document completion legitimately accompanies
    # this branch fact.
    _assert_single_branch_fact(
        caplog,
        "document",
        logging.INFO,
        allowed_later_events=("transcribe_operation_completed",),
    )
