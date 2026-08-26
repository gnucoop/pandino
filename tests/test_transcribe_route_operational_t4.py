"""THIRD ADOPTER SLICE T4 — /transcribe image primary-operation outcomes.

Facts under test, all owned by routes/multimodal.py::asr_parse() and all
scoped to the IMAGE branch:

    transcribe_operation_completed  (INFO,  details.branch)
    transcribe_operation_failed     (ERROR, details.branch/reason)

Only shared/external seams are monkeypatched (shared auth, the shared
describe_image_with_usage vision helper, the Usage writers). The image
dispatch gate, the route's existing broad try/except, the HTTP return
behaviour and the Operational logging boundary are all exercised for real.

`execution_error` is deliberately COARSE: the route's single existing handler
fuses file read / base64 preparation, shared prompt-store access, choose_llm
failure and provider invocation failure. These tests must never assert that
the provider call was reached, and must never split the boundary.

T5 owns transcribe_usage_accounting_failed; T4 must never emit it.
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

_FILENAME = "receipt-scan.png"
_MIMETYPE = "image/png"
_IMAGE_BYTES = b"fake-image-bytes-never-persisted"
_DESCRIPTION = "A scanned receipt whose description must never be persisted."
_RAW_EXCEPTION_TEXT = "vision-internal-detail-should-never-be-persisted"

_PROVIDER = "test-vision-provider"
_MODEL = "test-vision-model"

# Everything T4 is forbidden to persist on ANY image Operational surface.
_FORBIDDEN = (
    # identity / credentials / headers
    "X-API-KEY",
    "X-USER-EMAIL",
    "X-USER-NAME",
    "test-key",
    "user@example.com",
    "Test User",
    "Test_User",
    # image content, filename, MIME, data URL, bytes
    _DESCRIPTION,
    "scanned receipt",
    _FILENAME,
    "receipt-scan",
    _MIMETYPE,
    ".png",
    "base64",
    "data:image",
    "fake-image-bytes",
    # raw exception text / traceback
    _RAW_EXCEPTION_TEXT,
    "Traceback",
    # Usage / accounting values
    "input_tokens",
    "output_tokens",
    "token_usage",
    "cost",
    "usage_log_id",
)

_OUTCOME_EVENTS = (
    "transcribe_operation_blocked",
    "transcribe_operation_completed",
    "transcribe_operation_failed",
)

_TOKEN_USAGE = {"input_tokens": 11, "output_tokens": 22}


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        models=SimpleNamespace(
            asr_model="test-asr-model",
            asr_provider="Deepinfra",
            asr_base_url="http://asr.internal",
            asr_mistral_price_per_minute_usd=0.006,
            vision_provider=_PROVIDER,
            vision_model=_MODEL,
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
    monkeypatch.setattr(multimodal_route, "log_token_usage", lambda **kwargs: 778)


def _patch_vision(monkeypatch, result):
    """Replace only the shared vision helper leaf; record whether it ran."""
    calls: list[tuple] = []

    def _describe(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(multimodal_route, "describe_image_with_usage", _describe)
    return calls


def _post_image(app, filename=_FILENAME, mimetype=_MIMETYPE):
    return app.test_client().post(
        "/transcribe",
        data={"file": (io.BytesIO(_IMAGE_BYTES), filename, mimetype)},
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
    """The image branch produces exactly one primary outcome per request."""
    for other in _OUTCOME_EVENTS:
        if other == event:
            continue
        assert _operational_records(caplog, other) == [], (
            f"{other} must not accompany {event}"
        )
    return _the_operational_record(caplog, event)


def _assert_t1_branch_fact_intact(caplog):
    """T1 semantics: exactly one branch_selected, branch=image, INFO."""
    record = _the_operational_record(caplog, "transcribe_branch_selected")
    assert record.levelno == logging.INFO
    assert record.maui_details == {"branch": "image"}


def _assert_no_accounting_event(caplog):
    assert _operational_records(caplog, "transcribe_usage_accounting_failed") == [], (
        "transcribe_usage_accounting_failed is owned by T5 and must not be "
        "emitted by T4"
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
    ]
    for surface in surfaces:
        for needle in _FORBIDDEN:
            assert needle not in surface, (
                f"forbidden content {needle!r} reached an Operational surface: "
                f"{surface!r}"
            )
    # duration_ms is forbidden on every T4 fact.
    assert snapshot.duration_ms is None
    assert not hasattr(record, "maui_duration_ms")


def _assert_authoritative_identity(record):
    """provider/model are the route-selected values, unchanged."""
    assert record.maui_provider == _PROVIDER
    assert record.maui_model == _MODEL
    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    assert snapshot.provider == _PROVIDER
    assert snapshot.model == _MODEL


# ---------------------------------------------------------------------------
# A. transcribe_operation_completed — image success
# ---------------------------------------------------------------------------


def test_image_success_persists_one_completion_with_provider_and_model(
    monkeypatch, caplog
):
    """The image branch produced a usable description result.

    The emission sits at the PRIMARY-SUCCESS boundary: immediately after a
    usable description has been obtained and BEFORE the Usage-accounting
    block, because the fact represents completion of the primary image
    operation, not of the whole HTTP route (T4.1).

    It remains OUTSIDE any try block, so a malformed builder call would still
    surface as a 500 rather than be silently reclassified.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    calls = _patch_vision(
        monkeypatch,
        {"description": _DESCRIPTION, "token_usage": _TOKEN_USAGE},
    )

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    # The real success path still returns the current successful response.
    assert response.status_code == 200
    assert response.get_json() == {"text": _DESCRIPTION}
    assert len(calls) == 1, "the shared vision helper must actually have run"

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_completed")
    assert record.levelno == logging.INFO
    assert record.maui_details == {"branch": "image"}
    # No exception is involved on the success path.
    assert not hasattr(record, "maui_error_type")
    assert record.exc_info is None
    _assert_authoritative_identity(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_image_success_completion_survives_a_failing_accounting_attempt(
    monkeypatch, caplog
):
    """Degraded success: the caller still gets the description and a 200.

    The accounting failure itself is T5's fact; T4 pins that it neither
    suppresses nor duplicates the primary completion, and that the completion
    is emitted BEFORE the accounting attempt runs (T4.1).
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    def _boom(**kwargs):
        raise RuntimeError(_RAW_EXCEPTION_TEXT)

    monkeypatch.setattr(multimodal_route, "log_token_usage", _boom)
    _patch_vision(
        monkeypatch,
        {"description": _DESCRIPTION, "token_usage": _TOKEN_USAGE},
    )

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    assert response.status_code == 200
    assert response.get_json() == {"text": _DESCRIPTION}

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_completed")
    assert record.maui_details == {"branch": "image"}
    _assert_authoritative_identity(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    # T5 legitimately emits exactly one accounting fact here. T4 asserts that
    # it accompanies, rather than replaces, the primary completion, and that
    # the primary completion comes FIRST (T4.1).
    accounting = _operational_records(caplog, "transcribe_usage_accounting_failed")
    assert len(accounting) == 1
    assert caplog.records.index(record) < caplog.records.index(accounting[0]), (
        "the image completion must be emitted BEFORE the accounting attempt"
    )


# ---------------------------------------------------------------------------
# B. transcribe_operation_failed — the existing broad image boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception, expected_error_type",
    [
        (RuntimeError(_RAW_EXCEPTION_TEXT), "RuntimeError"),
        (ValueError(_RAW_EXCEPTION_TEXT), "ValueError"),
    ],
)
def test_image_execution_boundary_persists_execution_error(
    monkeypatch, caplog, exception, expected_error_type
):
    """A representative failure reaching the route's existing broad handler.

    The injection is deliberately made at the shared vision seam, which is one
    of several sub-operations the single handler covers. The event does NOT
    claim the provider call started — only that the image execution boundary
    failed without producing a usable primary result.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_vision(monkeypatch, exception)

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    # Existing HTTP behaviour is unchanged, str(e) included.
    assert response.status_code == 500
    assert response.get_json() == {
        "error": f"Error extracting text from image: {_RAW_EXCEPTION_TEXT}"
    }

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.levelno == logging.ERROR
    assert record.maui_details == {"branch": "image", "reason": "execution_error"}
    # Class name only — never str(e), never a traceback.
    assert record.maui_error_type == expected_error_type
    assert record.exc_info is None
    _assert_authoritative_identity(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_image_failure_before_the_provider_seam_uses_the_same_reason(
    monkeypatch, caplog
):
    """The boundary is intentionally coarse.

    A failure raised during route-local image preparation — before the shared
    vision helper is ever consulted — is not distinguished from a provider
    failure. That fusion is ratified, so this test pins it rather than
    splitting the handler.
    """

    app = _make_app()
    _patch_shared_seams(monkeypatch)
    calls = _patch_vision(
        monkeypatch,
        {"description": _DESCRIPTION, "token_usage": _TOKEN_USAGE},
    )

    def _explode(_payload):
        raise OSError(_RAW_EXCEPTION_TEXT)

    # base64 preparation is route-local and sits inside the same broad handler.
    monkeypatch.setattr(multimodal_route.base64, "b64encode", _explode)

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    assert response.status_code == 500
    assert calls == [], "the shared vision helper must never have been reached"

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.levelno == logging.ERROR
    # Same coarse reason as a provider-side failure — deliberately.
    assert record.maui_details == {"branch": "image", "reason": "execution_error"}
    assert record.maui_error_type == "OSError"
    _assert_authoritative_identity(record)
    _assert_free_of_forbidden(record)
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


# ---------------------------------------------------------------------------
# C. T4 negative scope
# ---------------------------------------------------------------------------


def test_image_never_emits_blocked_semantics(monkeypatch, caplog):
    """Branch B performs no configuration validation, so `blocked` is
    structurally inexpressible. Empty vision configuration therefore still
    reaches the execution boundary and yields `execution_error`, never
    transcribe_operation_blocked.
    """
    app = _make_app()
    app.config["MAUI_CONFIG"].models.vision_provider = None
    app.config["MAUI_CONFIG"].models.vision_model = None
    _patch_shared_seams(monkeypatch)
    _patch_vision(monkeypatch, ValueError(_RAW_EXCEPTION_TEXT))

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    assert response.status_code == 500
    assert _operational_records(caplog, "transcribe_operation_blocked") == []

    record = _assert_exactly_one_outcome(caplog, "transcribe_operation_failed")
    assert record.maui_details == {"branch": "image", "reason": "execution_error"}
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_image_reason_vocabulary_is_exactly_execution_error(monkeypatch, caplog):
    """No image reason beyond `execution_error` exists in T4."""
    app = _make_app()
    _patch_shared_seams(monkeypatch)
    _patch_vision(monkeypatch, KeyError("token_usage"))

    with caplog.at_level(logging.INFO):
        response = _post_image(app)

    assert response.status_code == 500
    reasons = {
        r.maui_details.get("reason")
        for r in _operational_records(caplog, "transcribe_operation_failed")
    }
    assert reasons == {"execution_error"}
    _assert_t1_branch_fact_intact(caplog)
    _assert_no_accounting_event(caplog)


def test_shared_image_runtime_logs_are_not_owned_by_t4():
    """T4 must not remove or modify the shared runtime diagnostics.

    image_description_failed / image_description_started / llm_selected live
    in shared infrastructure and are KEEP-by-default for this adopter.
    """
    import inspect

    import infrastructure.ai as ai

    describe_source = inspect.getsource(ai.describe_image_with_usage)
    assert "image_description_started" in describe_source
    assert "image_description_failed" in describe_source
    assert "llm_selected" in inspect.getsource(ai.choose_llm)
    # No shared helper became adopter-aware.
    for source in (describe_source, inspect.getsource(ai.choose_llm)):
        assert "transcribe_" not in source
        assert "maui_persist" not in source
