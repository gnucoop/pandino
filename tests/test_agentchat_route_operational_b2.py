"""FOURTH ADOPTER SLICE B2 — /agentchat terminal failure facts.

Fact under test, owned by routes/rag.py::agentchat():

    agentchat_uncontrolled_failure   (ERROR, details.reason [, error_type])

One event token covers both arms of the route's EXISTING two-arm terminal
boundary, because they are one fact — *terminated at the catch-all* —
differing in a single classification:

    except RuntimeError  →  details.reason = "service_error"
    except Exception     →  details.reason = "unhandled" + error_type

The `error_type` asymmetry is ratified (§12.2): on the RuntimeError arm the
class is always the literal `RuntimeError`, so the field would answer no
question, while on the generic arm it is the genuine class and the single
most informative field available.

The logging-API asymmetry is equally ratified (§13.3.1) and is pinned here:

    service_error  →  logger.error      — the replaced legacy line carried no
                                          traceback, and C8.2's obligation is
                                          to PRESERVE runtime depth, not add it
    unhandled      →  logger.exception  — the two replaced legacy lines DID put
                                          a full traceback on stderr (badly, as
                                          an interpolated string), so stderr
                                          keeps it, properly

B2 replaces three legacy runtime lines — `agentchat_runtime_error`,
`agentchat_unexpected_error` and `agentchat_unexpected_error_trace` — with one
authoritative emission per arm. Raw `str(e)` and `traceback.format_exc()` leave
the log message entirely; only `error_type` represents the failure in the store.

B2 is diagnostic only: HTTP statuses, response bodies, exception propagation
and the six B1 guards are unchanged. Later-slice events (agentchat_agent_failed,
agentchat_audit_log_failed) must not appear.
"""

import json
import logging
from types import SimpleNamespace

import pytest
from flask import Flask

import routes.utils as routes_utils
from routes import rag as rag_route
from utils.logging_config import LOG_FORMAT, ContextDefaultsFilter, UtcIsoFormatter
from utils.operational_persistence import (
    OperationalPersistenceHandler,
    snapshot_from_record,
)

USERNAME = "distinctive-user@example.com"
API_KEY = "distinctive-test-key"
CHAT_CONTENT = "SUPERSECRETQUESTION"

_HEADERS = {"X-API-KEY": API_KEY}

_B2_EVENT = "agentchat_uncontrolled_failure"
_B1_EVENT = "agentchat_request_rejected"
_B3_EVENTS = ("agentchat_agent_failed", "agentchat_audit_log_failed")

# Legacy runtime tokens this slice replaces. They must no longer appear on any
# emitted record — the point of the slice is ONE authoritative record per arm.
_LEGACY_TOKENS = (
    "agentchat_runtime_error",
    "agentchat_unexpected_error",
    "agentchat_unexpected_error_trace",
)

# Distinctive values that must never reach an Operational surface.
_FORBIDDEN = (USERNAME, API_KEY, CHAT_CONTENT)


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        rag=SimpleNamespace(default_namespace="default-ns"),
        completion_token_cost=1,
    )
    # register_request_context_hooks is imported lazily here to mirror B1's
    # app construction exactly.
    from utils.logging_config import register_request_context_hooks

    register_request_context_hooks(app)
    app.register_blueprint(rag_route.rag_bp)
    return app


def _patch_shared_seams(monkeypatch, *, user_tokens=10):
    """Only shared/external seams: auth, ambient attribution, the balance
    lookup. Every route guard and both terminal arms stay real."""
    monkeypatch.setattr(rag_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(rag_route, "attribute_usage_to_user", lambda **k: None)
    monkeypatch.setattr(
        rag_route.database_pg, "get_user_tokens", lambda username: user_tokens
    )


def _post(app, *, body=None, headers=None):
    if body is None:
        body = {"chat": [CHAT_CONTENT], "username": USERNAME}
    return app.test_client().post(
        "/agentchat",
        json=body,
        headers=_HEADERS if headers is None else headers,
    )


def _operational_records(caplog, event):
    return [
        r
        for r in caplog.records
        if getattr(r, "maui_persist", None) is True
        and getattr(r, "maui_event", None) == event
    ]


def _the_failure_record(caplog):
    records = _operational_records(caplog, _B2_EVENT)
    assert len(records) == 1, (
        f"expected exactly one {_B2_EVENT} record, got {len(records)}"
    )
    return records[0]


def _assert_no_other_slice_events(caplog):
    for event in (_B1_EVENT,) + _B3_EVENTS:
        assert _operational_records(caplog, event) == [], (
            f"{event} does not belong to B2's terminal boundary and must "
            "not be emitted here"
        )


def _assert_legacy_tokens_gone(caplog):
    """The three replaced runtime lines are gone, not merely superseded.

    Asserted over EVERY captured record's rendered message, so a surviving
    duplicate emission on either arm fails here even though it would carry no
    maui_* metadata."""
    for record in caplog.records:
        rendered = record.getMessage()
        for token in _LEGACY_TOKENS:
            assert token not in rendered, (
                f"legacy runtime token {token!r} still emitted: {rendered!r}"
            )


def _assert_clean_payload(record, *, reason, error_type):
    """`details.reason` plus, on the unhandled arm only, `error_type`.

    Everything else the builder can carry is ABSENT (R25/R26), and neither the
    LogRecord's Operational metadata nor the persisted snapshot carries raw
    exception text, a traceback, or request content."""
    assert record.maui_details == {"reason": reason}

    absent = ["maui_provider", "maui_model", "maui_duration_ms", "maui_message"]
    if error_type is None:
        absent.append("maui_error_type")
    else:
        assert record.maui_error_type == error_type
    for field in absent:
        assert not hasattr(record, field), (
            f"{field} must be absent from {_B2_EVENT}(reason={reason})"
        )

    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    assert snapshot.event == _B2_EVENT
    assert snapshot.level == "ERROR"
    assert json.loads(snapshot.details_json) == {"reason": reason}
    assert snapshot.error_type == error_type
    assert snapshot.provider is None
    assert snapshot.model is None
    assert snapshot.duration_ms is None
    assert snapshot.message is None

    # request_id / app_id stay foundation-owned: the call site contributes the
    # single reason key and, where ratified, the class name — nothing else.
    surfaces = [
        record.getMessage(),
        str(getattr(record, "maui_details", None)),
        str(snapshot.details_json),
        str(snapshot.message),
        str(snapshot.error_type),
    ]
    for surface in surfaces:
        assert "Traceback" not in surface, (
            f"a traceback reached an Operational surface: {surface!r}"
        )
        for needle in _FORBIDDEN:
            assert needle not in surface, (
                f"forbidden content {needle!r} reached an Operational "
                f"surface: {surface!r}"
            )
    return snapshot


# ---------------------------------------------------------------------------
# 7. service_error — the except RuntimeError arm
# ---------------------------------------------------------------------------


def test_runtime_error_persists_service_error_without_error_type(monkeypatch, caplog):
    """§12.2 / §13.3.1 — the RuntimeError arm.

    `error_type` is deliberately ABSENT: the class is always RuntimeError here,
    so the field would carry zero information. The HTTP body keeps `str(e)`,
    exactly as CURRENT behaviour — that raw text in the RESPONSE is a separate,
    out-of-scope concern; what B2 guarantees is that it does not reach the
    persisted Operational payload.
    """
    boom = f"service exploded while handling {CHAT_CONTENT}"

    def _raise(**kwargs):
        raise RuntimeError(boom)

    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(rag_route, "run_agentchat", _raise)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    # HTTP behaviour is unchanged, including the raw message in the body.
    assert response.status_code == 500
    assert response.get_json() == {"error": boom}

    record = _the_failure_record(caplog)
    assert record.levelno == logging.ERROR
    _assert_clean_payload(record, reason="service_error", error_type=None)

    # Runtime depth is PRESERVED, not increased: the replaced line carried no
    # traceback, so logger.error must not have attached exception context.
    assert record.exc_info is None
    # caplog bypasses the app's handler chain, so the foundation's context
    # defaults are applied here before the real formatter runs.
    ContextDefaultsFilter().filter(record)
    assert "Traceback" not in UtcIsoFormatter(LOG_FORMAT).format(record)

    _assert_legacy_tokens_gone(caplog)
    _assert_no_other_slice_events(caplog)


# ---------------------------------------------------------------------------
# 8. unhandled — the except Exception arm, real logging boundary
# ---------------------------------------------------------------------------


def test_real_logger_exception_splits_traceback_from_snapshot(monkeypatch):
    """§13.3.1 / I8 — B2's boundary-faithful test for the unhandled arm.

    One real `logger.exception(message, extra=extra)` call, raised from the
    route's real `except Exception`, must give stderr the full traceback while
    the Operational snapshot keeps only `details.reason` and `error_type`.

    Both halves are asserted together, because either alone would pass while
    the split silently broke: mocking `logger.exception` away would prove
    neither. The real OperationalPersistenceHandler and the real stderr
    formatter are both attached to the route's own logger.
    """
    sentinel = "UPSTREAM-BODY-secret-4242"

    class AgentBoom(Exception):
        pass

    def _raise(**kwargs):
        raise AgentBoom(f"upstream said {sentinel}")

    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(rag_route, "run_agentchat", _raise)

    sink = []
    handler = OperationalPersistenceHandler(sink.append)
    route_logger = logging.getLogger(rag_route.__name__)

    captured_records = []

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
        response = _post(app)
    finally:
        route_logger.removeHandler(handler)
        route_logger.removeHandler(capture)
        route_logger.setLevel(previous_level)

    # HTTP behaviour unchanged: generic body, no leak of the upstream text.
    assert response.status_code == 500
    assert response.get_json() == {"error": "An unexpected error occurred"}

    failure_records = [
        r for r in captured_records if getattr(r, "maui_event", None) == _B2_EVENT
    ]
    assert len(failure_records) == 1
    record = failure_records[0]
    assert record.levelno == logging.ERROR

    # --- stderr half: the real formatter keeps the traceback -----------------
    assert record.exc_info is not None
    formatted = UtcIsoFormatter(LOG_FORMAT).format(record)
    assert "Traceback" in formatted
    assert "AgentBoom" in formatted
    assert sentinel in formatted
    # Formatting has now mutated the shared record by caching exc_text.
    assert record.exc_text and sentinel in record.exc_text

    # --- Operational half: same record, bounded snapshot ---------------------
    snapshots = [s for s in sink if s.event == _B2_EVENT]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.level == "ERROR"
    assert snapshot.error_type == "AgentBoom"
    assert json.loads(snapshot.details_json) == {"reason": "unhandled"}
    assert snapshot.provider is None
    assert snapshot.model is None
    assert snapshot.duration_ms is None
    assert snapshot.message is None

    for value in [
        snapshot.details_json,
        snapshot.message,
        snapshot.error_type,
        str(snapshot),
    ]:
        text = str(value)
        assert sentinel not in text
        assert "Traceback" not in text

    # The snapshot taken again AFTER stderr formatting is still clean, proving
    # the shared-record exc_text mutation cannot leak into persistence.
    post_format_snapshot = snapshot_from_record(record)
    assert post_format_snapshot.error_type == "AgentBoom"
    assert sentinel not in str(post_format_snapshot)
    assert "Traceback" not in str(post_format_snapshot)

    # Exactly one authoritative record, and none of the replaced legacy lines.
    assert len(snapshots) == 1
    for token in _LEGACY_TOKENS:
        for captured in captured_records:
            assert token not in captured.getMessage()


def test_unexpected_exception_persists_unhandled_with_real_class(monkeypatch, caplog):
    """§12.2 — the payload contract on the unhandled arm.

    Companion to the boundary-faithful test above: same arm, asserted through
    the shared payload-cleanliness helper so the field-by-field contract is
    pinned in one place.
    """

    def _raise(**kwargs):
        raise ValueError(f"bad value from {CHAT_CONTENT}")

    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(rag_route, "run_agentchat", _raise)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "An unexpected error occurred"}

    record = _the_failure_record(caplog)
    assert record.levelno == logging.ERROR
    _assert_clean_payload(record, reason="unhandled", error_type="ValueError")

    # Runtime depth PRESERVED on this arm: exception context is attached.
    assert record.exc_info is not None

    _assert_legacy_tokens_gone(caplog)
    _assert_no_other_slice_events(caplog)


# ---------------------------------------------------------------------------
# 9. HTTPException pass-through — invalid API key, malformed JSON, 415
# ---------------------------------------------------------------------------


def _assert_no_slice_events(caplog):
    """No B2 failure record, and none of the other slices' events either."""
    assert _operational_records(caplog, _B2_EVENT) == [], (
        f"{_B2_EVENT} must not be emitted for an HTTPException raised inside "
        "the route: those are already-classified HTTP responses"
    )
    _assert_no_other_slice_events(caplog)


def test_invalid_api_key_propagates_403_without_operational_event(
    monkeypatch, caplog
):
    """An invalid API key surfaces as HTTP 403, Operationally silent.

    `assert_valid_api_key` calls `abort(403)`, raising werkzeug's `Forbidden`.
    The route's `except HTTPException: raise` arm lets it reach Flask's
    default HTML error handling instead of being reclassified by the terminal
    `except Exception` arm. (Until that arm was added this returned 500 with
    `reason=unhandled`, `error_type=Forbidden`.)

    Auth failure is intentionally not part of either Operational vocabulary:
    it is neither a B1 ratified guard nor a B2 uncontrolled failure.
    """
    app = _make_app()
    # The real assert_valid_api_key and the real abort(403) both run; only the
    # database-backed key lookup is replaced.
    monkeypatch.setattr(
        routes_utils,
        "validate_api_key",
        lambda api_key, user_email: (False, "Invalid API key", None),
    )
    monkeypatch.setattr(rag_route, "attribute_usage_to_user", lambda **k: None)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 403
    assert b"Invalid API key" in response.data
    assert response.headers.get("X-Request-ID")

    _assert_legacy_tokens_gone(caplog)
    _assert_no_slice_events(caplog)


def test_malformed_json_propagates_400_without_operational_event(
    monkeypatch, caplog
):
    """Malformed JSON reaches `request.get_json()`, which raises BadRequest.

    Same boundary, same class of defect as the 403: the HTTPException arm
    restores Flask's 400 instead of a reclassified 500.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = app.test_client().post(
            "/agentchat",
            data="{not valid json",
            content_type="application/json",
            headers=_HEADERS,
        )

    assert response.status_code == 400
    assert response.headers.get("X-Request-ID")

    _assert_legacy_tokens_gone(caplog)
    _assert_no_slice_events(caplog)


def test_wrong_content_type_propagates_415_without_operational_event(
    monkeypatch, caplog
):
    """A non-JSON Content-Type makes `request.get_json()` raise
    UnsupportedMediaType, which must surface as 415, not 500."""
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = app.test_client().post(
            "/agentchat",
            data="chat=hello",
            content_type="text/plain",
            headers=_HEADERS,
        )

    assert response.status_code == 415
    assert response.headers.get("X-Request-ID")

    _assert_legacy_tokens_gone(caplog)
    _assert_no_slice_events(caplog)


# ---------------------------------------------------------------------------
# Cross-arm invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected_reason, expected_error_type",
    [
        (RuntimeError("boom"), "service_error", None),
        (KeyError("MAUI_CONFIG"), "unhandled", "KeyError"),
        (TypeError("not serializable"), "unhandled", "TypeError"),
    ],
)
def test_both_arms_share_one_event_token(
    monkeypatch, caplog, exc, expected_reason, expected_error_type
):
    """§12.2 — one token, two reasons.

    The two arms are one fact — *terminated at the catch-all* — differing in a
    single classification, so they must NOT be split into two event tokens.
    """

    def _raise(**kwargs):
        raise exc

    app = _make_app()
    _patch_shared_seams(monkeypatch)
    monkeypatch.setattr(rag_route, "run_agentchat", _raise)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500

    record = _the_failure_record(caplog)
    assert record.maui_event == _B2_EVENT
    assert record.maui_details == {"reason": expected_reason}
    assert getattr(record, "maui_error_type", None) == expected_error_type
