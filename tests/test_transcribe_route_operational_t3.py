"""THIRD ADOPTER SLICE T3 — /transcribe document primary-operation outcomes.

Facts under test, all owned by routes/multimodal.py::asr_parse() and all
scoped to the DOCUMENT branch:

    transcribe_operation_completed  (INFO,  details.branch/extracted_chars)
    transcribe_operation_failed     (ERROR, details.branch/reason)

Only the shared document extractor is monkeypatched, as a leaf seam. The
dispatch gate, the route's existing try/except structure, the HTTP return
behaviour and the Operational logging boundary are all exercised for real.

The document branch is provider-free: `provider` and `model` are absent from
every fact here. T4 owns the image outcomes and T5 owns
transcribe_usage_accounting_failed; T3 must never emit either.
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

_FILENAME = "quarterly-report.pdf"
_MIMETYPE = "application/pdf"
_NORMALIZED_TEXT = "Normalized document body that must never be persisted."
_RAW_EXCEPTION_TEXT = "parser-internal-detail-should-never-be-persisted"

# Everything T3 is forbidden to persist on ANY document Operational surface.
# The document branch has no ratified provider/model, so both appear here.
_FORBIDDEN = (
    # identity / credentials / headers
    "X-API-KEY",
    "X-USER-EMAIL",
    "X-USER-NAME",
    "test-key",
    "user@example.com",
    "Test User",
    "Test_User",
    # document content, filename, MIME, extension, parser internals
    _NORMALIZED_TEXT,
    "Normalized document body",
    _FILENAME,
    "quarterly-report",
    _MIMETYPE,
    ".pdf",
    "pymupdf",
    "python-docx",
    "striprtf",
    # raw exception text / traceback
    _RAW_EXCEPTION_TEXT,
    "Traceback",
    # provider / model / Usage / accounting values
    "test-vision-provider",
    "test-vision-model",
    "Deepinfra",
    "input_tokens",
    "output_tokens",
    "cost",
    "usage_log_id",
)

_OUTCOME_EVENTS = (
    "transcribe_operation_blocked",
    "transcribe_operation_completed",
    "transcribe_operation_failed",
)


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        models=SimpleNamespace(
            asr_model="test-asr-model",
            asr_provider="Deepinfra",
            asr_base_url="http://asr.internal",
            asr_mistral_price_per_minute_usd=0.006,
            vision_provider="test-vision-provider",
            vision_model="test-vision-model",
        )
    )
    register_request_context_hooks(app)
    app.register_blueprint(multimodal_route.multimodal_bp)
    return app


def _patch_shared_seams(monkeypatch):
    monkeypatch.setattr(multimodal_route, "assert_valid_api_key", lambda *a, **k: None)


def _patch_extractor(monkeypatch, result):
    """Replace only the shared extractor leaf; record that it ran."""
    calls: list[dict] = []

    def _extract(doc_input):
        calls.append(doc_input)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(multimodal_route, "extract_and_normalize_document", _extract)
    return calls


def _post_document(app, filename=_FILENAME, mimetype=_MIMETYPE):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(b"fake-document-bytes"), filename, mimetype)},
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
    """The document branch produces exactly one primary outcome per request."""
    for other in _OUTCOME_EVENTS:
        if other == event:
            continue
        assert _operational_records(caplog, other) == [], (
            f"{other} must not accompany {event}"
        )
    return _the_operational_record(caplog, event)


def _assert_t1_branch_fact_intact(caplog):
    """T1 semantics: exactly one branch_selected, branch=document, INFO."""
    record = _the_operational_record(caplog, "transcribe_branch_selected")
    assert record.levelno == logging.INFO
    assert record.maui_details == {"branch": "document"}


def _assert_no_accounting_event(caplog):
    assert _operational_records(caplog, "transcribe_usage_accounting_failed") == [], (
        "transcribe_usage_accounting_failed is owned by T5 and never applies "
        "to the document branch"
    )


def _assert_provider_and_model_absent(record):
    """The document branch has no provider and no model to claim."""
    assert not hasattr(record, "maui_provider")
    assert not hasattr(record, "maui_model")
    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    assert snapshot.provider is None
    assert snapshot.model is None


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
    # duration_ms is forbidden on every T3 fact.
    assert snapshot.duration_ms is None
    assert not hasattr(record, "maui_duration_ms")


# ---------------------------------------------------------------------------
# A. transcribe_operation_completed — document success
# ---------------------------------------------------------------------------


def test_document_success_persists_one_completion_with_extracted_chars(
    monkeypatch, caplog
):
    """The success emission sits INSIDE the route's existing try block.

    That try's `except ValueError` arm would catch a programmer-contract
    ValueError raised by a malformed Operational builder call and silently
    convert this 200 into a 422. Control flow is deliberately NOT restructured
    to close that window, so this test is the control for the caveat: it pins
    both the exact emitted payload and the real HTTP 200 success path.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    calls = _patch_extractor(monkeypatch, {"text": _NORMALIZED_TEXT})

    with caplog.at_level(logging.INFO):
        response = _post_document(app)

    # The real success path still returns the current successful response.
    assert response.status_code == 200
    assert response.get_json() == {"text": _NORMALIZED_TEXT}
    assert len(calls) == 1, "the shared extractor must actually have run"

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_completed")
    assert record.levelno == logging.INFO
    assert record.maui_details == {
        "branch": "document",
        "extracted_chars": len(_NORMALIZED_TEXT),
    }
    # extracted_chars is a count, not a content proxy.
    assert isinstance(record.maui_details["extracted_chars"], int)
    assert not hasattr(record, "maui_error_type")
    _assert_provider_and_model_absent(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_document_success_extracted_chars_tracks_the_returned_text(
    monkeypatch, caplog
):
    """extracted_chars is exactly len(result["text"]), including the empty case."""
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_extractor(monkeypatch, {"text": ""})

    with caplog.at_level(logging.INFO):
        response = _post_document(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": ""}

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_completed")
    assert record.maui_details == {"branch": "document", "extracted_chars": 0}
    _assert_provider_and_model_absent(record)
    _assert_t1_branch_fact_intact(caplog)


# ---------------------------------------------------------------------------
# B. transcribe_operation_failed — the existing ValueError boundary
# ---------------------------------------------------------------------------


def test_value_error_extraction_boundary_persists_extraction_invalid(
    monkeypatch, caplog
):
    """A failure caught by the route's existing ValueError extraction boundary.

    The reason is deliberately NOT described as a user-validation rejection:
    parser-originated ValueError-shaped failures reach this same handler, and
    the boundary knows only the shape of what it caught.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_extractor(monkeypatch, ValueError(_RAW_EXCEPTION_TEXT))

    with caplog.at_level(logging.INFO):
        response = _post_document(app)

    # Existing HTTP behaviour is unchanged, str(e) included.
    assert response.status_code == 422
    assert response.get_json() == {"error": _RAW_EXCEPTION_TEXT}

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.levelno == logging.ERROR
    assert record.maui_details == {
        "branch": "document",
        "reason": "extraction_invalid",
    }
    # Class name only — never str(e), never a traceback.
    assert record.maui_error_type == "ValueError"
    assert record.exc_info is None
    _assert_provider_and_model_absent(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_value_error_subclass_reaches_the_same_extraction_boundary(
    monkeypatch, caplog
):
    """error_type stays faithful to the concrete class the boundary caught."""

    class ParserValueError(ValueError):
        pass

    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_extractor(monkeypatch, ParserValueError(_RAW_EXCEPTION_TEXT))

    with caplog.at_level(logging.INFO):
        response = _post_document(app)

    assert response.status_code == 422

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.maui_details == {
        "branch": "document",
        "reason": "extraction_invalid",
    }
    assert record.maui_error_type == "ParserValueError"
    _assert_provider_and_model_absent(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)


# ---------------------------------------------------------------------------
# C. transcribe_operation_failed — the existing generic boundary
# ---------------------------------------------------------------------------


def test_generic_extraction_boundary_persists_extraction_error(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_extractor(monkeypatch, RuntimeError(_RAW_EXCEPTION_TEXT))

    with caplog.at_level(logging.INFO):
        response = _post_document(app)

    # Existing HTTP behaviour is unchanged, str(e) included.
    assert response.status_code == 422
    assert response.get_json() == {
        "error": f"Error extracting text from file: {_RAW_EXCEPTION_TEXT}"
    }

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.levelno == logging.ERROR
    assert record.maui_details == {
        "branch": "document",
        "reason": "extraction_error",
    }
    assert record.maui_error_type == "RuntimeError"
    assert record.exc_info is None
    _assert_provider_and_model_absent(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# D. The unreachable NotImplementedError arm is deliberately unrepresented
# ---------------------------------------------------------------------------


def test_not_implemented_arm_remains_unrepresented(monkeypatch, caplog):
    """The 415 arm is unreachable from the current dispatch gate.

    No Operational reason is designed for it and the arm itself is untouched,
    so a NotImplementedError yields the existing 415 with no primary outcome.
    NotImplementedError is not a ValueError subclass, so it cannot arrive as
    extraction_invalid either.
    """
    assert not issubclass(NotImplementedError, ValueError)

    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_extractor(monkeypatch, NotImplementedError(_RAW_EXCEPTION_TEXT))

    with caplog.at_level(logging.INFO):
        response = _post_document(app)

    # Existing HTTP behaviour is unchanged.
    assert response.status_code == 415
    assert response.get_json() == {
        "error": f"Unsupported file format: {_FILENAME}"
    }

    _assert_t1_branch_fact_intact(caplog)
    for event in _OUTCOME_EVENTS:
        assert _operational_records(caplog, event) == [], (
            f"{event} must not be emitted at the unreachable 415 arm"
        )
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# E. T3 negative scope — the document dispatch gate is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["report.pdf", "report.docx", "report.rtf"])
def test_every_document_extension_reports_the_same_document_outcome(
    monkeypatch, caplog, filename
):
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_extractor(monkeypatch, {"text": _NORMALIZED_TEXT})

    with caplog.at_level(logging.INFO):
        response = _post_document(app, filename=filename, mimetype="application/pdf")

    assert response.status_code == 200
    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_completed")
    assert record.maui_details == {
        "branch": "document",
        "extracted_chars": len(_NORMALIZED_TEXT),
    }
    _assert_provider_and_model_absent(record)
    _assert_free_of_forbidden(record)


def test_document_outcomes_do_not_leak_into_the_image_branch(monkeypatch, caplog):
    """No document-branch fact may appear on an image request.

    T4 now legitimately emits the image primary outcome, so this test no
    longer asserts total silence; it asserts that every outcome fact on an
    image request carries branch=image and never branch=document.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(
        multimodal_route.database_pg,
        "get_user_by_username",
        lambda user_email: {"id": 42, "username": user_email, "client": "dino"},
    )
    monkeypatch.setattr(
        multimodal_route, "record_token_consumption", lambda **kwargs: True
    )
    monkeypatch.setattr(
        multimodal_route,
        "describe_image_with_usage",
        lambda *a, **k: {
            "description": "a picture",
            "token_usage": {"input_tokens": 1, "output_tokens": 2},
        },
    )

    with caplog.at_level(logging.INFO):
        response = app.test_client().post(
            "/transcribe",
            data={"file": (io.BytesIO(b"fake-bytes"), "picture.png", "image/png")},
            content_type="multipart/form-data",
            headers=_HEADERS,
        )

    assert response.status_code == 200
    assert response.get_json() == {"text": "a picture"}
    for event in _OUTCOME_EVENTS:
        for record in _operational_records(caplog, event):
            assert record.maui_details.get("branch") == "image", (
                f"{event} on an image request must never claim branch=document"
            )
    _assert_no_accounting_event(caplog)
