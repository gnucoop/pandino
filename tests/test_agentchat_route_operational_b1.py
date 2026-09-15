"""FOURTH ADOPTER SLICE B1 — /agentchat request-rejection facts.

Fact under test, owned by routes/rag.py::agentchat():

    agentchat_request_rejected   (WARNING, details.reason)

The event persists the reason a request-owned guard stopped the request
BEFORE agent execution began. Its `reason` is a closed, Maui-owned six-value
enum, one value per existing route guard, and it is the ONLY semantic field.

Only external/shared seams are monkeypatched (shared auth, the ambient Usage
attribution boundary, the token-balance lookup, run_agentchat). The six route
guards, their order, their HTTP statuses, their response bodies and the real
Operational logging boundary are all exercised for real.

B1 implements no other fact: the later-slice terminal and service events
(agentchat_uncontrolled_failure, agentchat_agent_failed,
agentchat_audit_log_failed) must not appear, and neither must any
success-path or start event — a healthy request emits nothing at all.

§14.5 hazard control: the entire view body sits inside one `try`, so a
malformed Operational builder call would be caught by the route's own
`except Exception` and silently convert the intended 400/403/500 into a
generic 500. Every test below therefore pins the unchanged HTTP status and
body alongside the event, which is what makes the exact call shape safe
without wrapping any emission in a local try.
"""

import json
import logging
from types import SimpleNamespace

import pytest
from flask import Flask

from routes import rag as rag_route
from utils.logging_config import register_request_context_hooks
from utils.operational_persistence import snapshot_from_record

USERNAME = "distinctive-user@example.com"
API_KEY = "distinctive-test-key"
CHAT_CONTENT = "SUPERSECRETQUESTION"

_HEADERS = {"X-API-KEY": API_KEY}

# Everything B1 is forbidden to persist: identity, key material, chat
# content and the token balance. Distinctive values only, so the needles
# cannot collide with the event token itself (which contains "chat") or with
# a legitimate reason literal.
_FORBIDDEN = (
    USERNAME,
    API_KEY,
    CHAT_CONTENT,
    "user_tokens",
    "X-API-KEY",
)

_B1_EVENT = "agentchat_request_rejected"
_LATER_SLICE_EVENTS = (
    "agentchat_uncontrolled_failure",
    "agentchat_agent_failed",
    "agentchat_audit_log_failed",
)


def _make_app():
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        rag=SimpleNamespace(default_namespace="default-ns"),
        completion_token_cost=1,
    )
    register_request_context_hooks(app)
    app.register_blueprint(rag_route.rag_bp)
    return app


def _patch_shared_seams(monkeypatch, *, user_tokens=10):
    """Only shared/external seams: auth, ambient attribution, the balance
    lookup. Every route guard stays real."""
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


def _all_operational_records(caplog):
    return [r for r in caplog.records if getattr(r, "maui_persist", None) is True]


def _the_rejection_record(caplog):
    records = _operational_records(caplog, _B1_EVENT)
    assert len(records) == 1, (
        f"expected exactly one {_B1_EVENT} record, got {len(records)}"
    )
    return records[0]


def _assert_no_later_slice_events(caplog):
    for event in _LATER_SLICE_EVENTS:
        assert _operational_records(caplog, event) == [], (
            f"{event} belongs to a later slice and must not be emitted by B1"
        )


def _assert_minimal_payload(record, reason):
    """`details.reason` is the ONE semantic field. Everything else the
    builder can carry is ABSENT (R25/R26), and no forbidden content reaches
    any Operational surface — the LogRecord's or the persisted snapshot's."""
    assert record.maui_details == {"reason": reason}
    for absent in (
        "maui_provider",
        "maui_model",
        "maui_duration_ms",
        "maui_error_type",
        "maui_message",
    ):
        assert not hasattr(record, absent), (
            f"{absent} must be absent from {_B1_EVENT}"
        )

    # request_id / app_id are foundation-owned and are never call-site
    # fields: the persisted details carry the single reason key and nothing
    # else.
    snapshot = snapshot_from_record(record)
    assert snapshot is not None
    assert json.loads(snapshot.details_json) == {"reason": reason}

    surfaces = [
        record.getMessage(),
        str(getattr(record, "maui_details", None)),
        str(snapshot.details_json),
        str(snapshot.message),
        str(snapshot.error_type),
        str(snapshot.provider),
        str(snapshot.model),
    ]
    for surface in surfaces:
        for needle in _FORBIDDEN:
            assert needle not in surface, (
                f"forbidden content {needle!r} reached an Operational "
                f"surface: {surface!r}"
            )


def _assert_rejection(caplog, reason):
    record = _the_rejection_record(caplog)
    assert record.levelno == logging.WARNING
    _assert_minimal_payload(record, reason)
    _assert_no_later_slice_events(caplog)
    return record


# ---------------------------------------------------------------------------
# 1. no_json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("falsy_body", [{}, []])
def test_absent_body_persists_no_json(monkeypatch, caplog, falsy_body):
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = _post(app, body=falsy_body)

    assert response.status_code == 400
    assert response.get_json() == {"error": "No JSON data provided"}
    _assert_rejection(caplog, "no_json")


# ---------------------------------------------------------------------------
# 2. missing_required_keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, expected_body_error",
    [
        ({"chat": [CHAT_CONTENT]}, "Missing required keys: username"),
        ({"username": USERNAME}, "Missing required keys: chat"),
    ],
)
def test_missing_required_keys_persists_one_bounded_reason(
    monkeypatch, caplog, body, expected_body_error
):
    """The response body still names the missing keys; the persisted event
    deliberately does not — `missing_required_keys` already names the guard."""
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = _post(app, body=body)

    assert response.status_code == 400
    assert response.get_json() == {"error": expected_body_error}
    record = _assert_rejection(caplog, "missing_required_keys")

    # The missing-key NAMES stay out of the persisted payload, even though
    # they are Maui-owned and safe: the reason already names the guard.
    details = str(record.maui_details)
    assert "username" not in details
    assert "'chat'" not in details


# ---------------------------------------------------------------------------
# 3. missing_api_key
# ---------------------------------------------------------------------------


def test_missing_api_key_header_persists_missing_api_key(monkeypatch, caplog):
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = _post(app, headers={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing X-API-KEY header"}
    _assert_rejection(caplog, "missing_api_key")


# ---------------------------------------------------------------------------
# 4. invalid_chat — both the empty list and a non-list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chat", [[], CHAT_CONTENT, {"turn": CHAT_CONTENT}, 7])
def test_invalid_chat_persists_invalid_chat(monkeypatch, caplog, chat):
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    with caplog.at_level(logging.INFO):
        response = _post(app, body={"chat": chat, "username": USERNAME})

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid 'chat': expected non-empty list"
    }
    _assert_rejection(caplog, "invalid_chat")


# ---------------------------------------------------------------------------
# 5. token_balance_unavailable
# ---------------------------------------------------------------------------


def test_unavailable_token_balance_persists_token_balance_unavailable(
    monkeypatch, caplog
):
    """A 500 rejection is still a locally controlled, classified rejection:
    C18 makes it WARNING, because severity describes the local fact and not
    the final fate of the request."""
    app = _make_app()
    _patch_shared_seams(monkeypatch, user_tokens=None)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Could not retrieve user tokens"}
    _assert_rejection(caplog, "token_balance_unavailable")


# ---------------------------------------------------------------------------
# 6. insufficient_tokens
# ---------------------------------------------------------------------------


def test_insufficient_tokens_persists_insufficient_tokens(monkeypatch, caplog):
    """The response body still exposes the balance; the persisted event
    carries neither the balance nor the cost."""
    app = _make_app()
    _patch_shared_seams(monkeypatch, user_tokens=0)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 403
    assert response.get_json() == {"error": "Not enough tokens", "user_tokens": 0}
    _assert_rejection(caplog, "insufficient_tokens")


# ---------------------------------------------------------------------------
# Negative B1 invariant — the healthy path
# ---------------------------------------------------------------------------


def test_a_healthy_request_emits_no_operational_record(monkeypatch, caplog):
    """ZERO Operational rows on a healthy request is the INTENDED design.

    Per §3 of the ratified fourth-adopter design, Operational Persistence is
    selective durable diagnostic memory of rejection, failure and degraded
    execution — it is NOT a durable copy of success-path facts and gives NO
    guarantee of a row per request. A healthy /agentchat request therefore
    MAY, and does, produce no Operational row at all.

    Do NOT read this silence as missing instrumentation and "fix" it by
    adding a success or start event: both were explicitly REJECTED in human
    review (§25.1, §25.2). The runtime lifecycle lines
    event=agentchat_request_started / _completed are KEPT and are Runtime
    Observability, not Operational Persistence.
    """
    app = _make_app()
    _patch_shared_seams(monkeypatch)

    monkeypatch.setattr(
        rag_route,
        "run_agentchat",
        lambda **kwargs: {
            "payload": {
                "answer": "an answer",
                "metrics": {"duration_ms": 42, "token_usage": {}},
                "tool_calls": [],
                "vectors": [],
                "follow_ups": [],
            },
            "model": "test-model",
            "provider": "test-provider",
        },
    )
    monkeypatch.setattr(
        rag_route, "get_user_by_username", lambda username: {"id": 7}
    )
    monkeypatch.setattr(rag_route, "record_token_consumption", lambda **k: False)
    monkeypatch.setattr(rag_route, "edit_tokens", lambda username, amount: None)

    with caplog.at_level(logging.INFO):
        response = _post(app)

    assert response.status_code == 200
    assert response.get_json()["answer"] == "an answer"

    assert _operational_records(caplog, _B1_EVENT) == []
    assert _all_operational_records(caplog) == [], (
        "a healthy /agentchat request must emit no Operational record at all"
    )

    # The KEPT runtime lifecycle lines are unaffected by this slice.
    messages = [r.getMessage() for r in caplog.records]
    assert any("event=agentchat_request_started" in m for m in messages)
    assert any("event=agentchat_request_completed" in m for m in messages)
