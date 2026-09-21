"""Usage → Operational Admin drill-down: dedicated Admin route.

Covers the three ratified states of GET
/admin/logs/<request_id>/operational - invalid key, valid key with an empty
timeline, and a contained Operational read failure - plus the admin
authentication gate and the runtime failure log.

Uses a bare Flask app with the admin blueprint registered (same seam style as
tests/test_documents_route.py); the Operational reader is monkeypatched, so no
live PostgreSQL is involved.
"""

import logging
import os

import pytest
from flask import Flask, session

from routes import admin as admin_route

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(REPO_ROOT, "templates")

REQUEST_ID = "9bf218009db0127d"
TIMELINE_URL = f"/admin/logs/{REQUEST_ID}/operational"


def _make_app() -> Flask:
    app = Flask(__name__, template_folder=TEMPLATES_DIR)
    app.secret_key = "test-secret"
    app.register_blueprint(admin_route.admin_bp)
    return app


def _logged_in_client(app):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["admin_logged_in"] = True
        sess["admin_username"] = "admin"
    return client


def _event(**overrides):
    event = {
        "event_time": "2026-08-27 14:22:31.482",
        "level": "INFO",
        "logger": "routes.multimodal",
        "event": "transcribe_started",
        "app_id": "app-1",
        "provider": "openai",
        "model": "whisper-1",
        "duration_ms": 842,
        "error_type": None,
        "details": None,
        "message": None,
    }
    event.update(overrides)
    return event


class _RecordingReader:
    def __init__(self, result=None, exception=None):
        self.result = result if result is not None else []
        self.exception = exception
        self.calls = []

    def __call__(self, request_id):
        self.calls.append(request_id)
        if self.exception is not None:
            raise self.exception
        return self.result


def _install_reader(monkeypatch, reader):
    monkeypatch.setattr(
        admin_route, "get_operational_events_by_request_id", reader
    )
    return reader


# --- 1. Authentication gate ---


def test_timeline_requires_admin_authentication(monkeypatch):
    reader = _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    app = _make_app()

    response = app.test_client().get(TIMELINE_URL)

    assert response.status_code == 302
    assert "/admin/login" in response.headers["Location"]


def test_timeline_does_not_read_operational_data_when_unauthenticated(monkeypatch):
    reader = _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    app = _make_app()

    app.test_client().get(TIMELINE_URL)

    assert reader.calls == []


# --- 2. Populated timeline ---


def test_populated_timeline_renders_ok(monkeypatch):
    _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    client = _logged_in_client(_make_app())

    response = client.get(TIMELINE_URL)
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "transcribe_started" in html
    assert REQUEST_ID in html


def test_populated_timeline_renders_rows_in_reader_order(monkeypatch):
    _install_reader(
        monkeypatch,
        _RecordingReader(
            result=[
                _event(event="first_event"),
                _event(event="second_event"),
            ]
        ),
    )
    client = _logged_in_client(_make_app())

    html = client.get(TIMELINE_URL).get_data(as_text=True)

    assert html.index("first_event") < html.index("second_event")


def test_reader_receives_the_request_id_from_the_path(monkeypatch):
    reader = _install_reader(monkeypatch, _RecordingReader(result=[]))
    client = _logged_in_client(_make_app())

    client.get(TIMELINE_URL)

    assert reader.calls == [REQUEST_ID]


def test_reader_receives_a_stripped_request_id(monkeypatch):
    reader = _install_reader(monkeypatch, _RecordingReader(result=[]))
    client = _logged_in_client(_make_app())

    client.get(f"/admin/logs/%20{REQUEST_ID}%20/operational")

    assert reader.calls == [REQUEST_ID]


# --- 3. Valid empty timeline ---


def test_valid_request_id_with_no_events_renders_empty_state(monkeypatch):
    _install_reader(monkeypatch, _RecordingReader(result=[]))
    client = _logged_in_client(_make_app())

    response = client.get(TIMELINE_URL)
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "No operational events" in html


def test_valid_empty_timeline_is_not_the_failure_state(monkeypatch):
    _install_reader(monkeypatch, _RecordingReader(result=[]))
    client = _logged_in_client(_make_app())

    html = client.get(TIMELINE_URL).get_data(as_text=True)

    assert "alert-danger" not in html
    assert "Unable to load the operational timeline" not in html


# --- 4. Contained read failure ---


def test_reader_failure_is_contained_and_still_renders_the_page(monkeypatch):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down"))
    )
    client = _logged_in_client(_make_app())

    response = client.get(TIMELINE_URL)

    assert response.status_code == 200


def test_reader_failure_renders_the_failure_alert(monkeypatch):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down"))
    )
    client = _logged_in_client(_make_app())

    html = client.get(TIMELINE_URL).get_data(as_text=True)

    assert "alert-danger" in html
    assert "Unable to load the operational timeline" in html


def test_reader_failure_does_not_render_the_valid_empty_state(monkeypatch):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down"))
    )
    client = _logged_in_client(_make_app())

    html = client.get(TIMELINE_URL).get_data(as_text=True)

    assert "No operational events" not in html


def test_reader_failure_does_not_redirect_to_the_usage_page(monkeypatch):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down"))
    )
    client = _logged_in_client(_make_app())

    response = client.get(TIMELINE_URL)

    assert response.status_code != 302


def test_reader_failure_does_not_leak_the_exception_text_to_the_ui(monkeypatch):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down secret"))
    )
    client = _logged_in_client(_make_app())

    html = client.get(TIMELINE_URL).get_data(as_text=True)

    assert "db down secret" not in html


# --- 5. Runtime failure log ---


def _failure_records(caplog):
    return [
        r
        for r in caplog.records
        if "admin_operational_timeline_read_failed" in r.getMessage()
    ]


def test_reader_failure_emits_one_runtime_log(monkeypatch, caplog):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down"))
    )
    client = _logged_in_client(_make_app())

    with caplog.at_level(logging.DEBUG):
        client.get(TIMELINE_URL)

    assert len(_failure_records(caplog)) == 1


def test_failure_log_carries_request_id_and_error_type(monkeypatch, caplog):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down"))
    )
    client = _logged_in_client(_make_app())

    with caplog.at_level(logging.DEBUG):
        client.get(TIMELINE_URL)

    message = _failure_records(caplog)[0].getMessage()
    assert "event=admin_operational_timeline_read_failed" in message
    assert f"request_id={REQUEST_ID}" in message
    assert "error_type=RuntimeError" in message


def test_failure_log_does_not_carry_the_driver_exception_string(monkeypatch, caplog):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down secret"))
    )
    client = _logged_in_client(_make_app())

    with caplog.at_level(logging.DEBUG):
        client.get(TIMELINE_URL)

    assert "db down secret" not in _failure_records(caplog)[0].getMessage()


def test_failure_log_is_not_a_persistent_operational_event(monkeypatch, caplog):
    _install_reader(
        monkeypatch, _RecordingReader(exception=RuntimeError("db down"))
    )
    client = _logged_in_client(_make_app())

    with caplog.at_level(logging.DEBUG):
        client.get(TIMELINE_URL)

    record = _failure_records(caplog)[0]
    assert not hasattr(record, "maui_persist")
    assert not hasattr(record, "maui_event")


def test_successful_read_emits_no_failure_log(monkeypatch, caplog):
    _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    client = _logged_in_client(_make_app())

    with caplog.at_level(logging.DEBUG):
        client.get(TIMELINE_URL)

    assert _failure_records(caplog) == []


# --- 6. Invalid correlation key ---


@pytest.mark.parametrize("raw", ["%20", "%20%20"])
def test_whitespace_only_request_id_redirects_to_usage(monkeypatch, raw):
    _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    client = _logged_in_client(_make_app())

    response = client.get(f"/admin/logs/{raw}/operational")

    assert response.status_code == 302
    assert "/admin/logs" in response.headers["Location"]


@pytest.mark.parametrize("raw", ["%20", "%20%20", "N/A", "N%2FA"])
def test_invalid_request_id_never_reaches_the_reader(monkeypatch, raw):
    reader = _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    client = _logged_in_client(_make_app())

    client.get(f"/admin/logs/{raw}/operational")

    assert reader.calls == []


@pytest.mark.parametrize("raw", ["N/A", "N%2FA"])
def test_na_sentinel_cannot_route_to_the_timeline_at_all(monkeypatch, raw):
    """The "N/A" Usage display sentinel contains a slash, which the default
    path converter refuses, so it 404s before the handler runs. The drill-down
    therefore cannot break on it even if such a URL is hand-typed."""
    _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    client = _logged_in_client(_make_app())

    assert client.get(f"/admin/logs/{raw}/operational").status_code == 404


def test_handler_rejects_the_na_sentinel_when_invoked_directly(monkeypatch):
    """Guards the caller-side validation itself, which HTTP routing shadows."""
    reader = _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    app = _make_app()

    with app.test_request_context():
        session["admin_logged_in"] = True
        response = admin_route.admin_operational_timeline("N/A")

    assert response.status_code == 302
    assert "/admin/logs" in response.headers["Location"]
    assert reader.calls == []


def test_handler_rejects_an_empty_request_id_when_invoked_directly(monkeypatch):
    reader = _install_reader(monkeypatch, _RecordingReader(result=[_event()]))
    app = _make_app()

    with app.test_request_context():
        session["admin_logged_in"] = True
        response = admin_route.admin_operational_timeline("   ")

    assert response.status_code == 302
    assert reader.calls == []


# --- 7. Method and Usage-page independence ---


def test_timeline_route_rejects_post(monkeypatch):
    _install_reader(monkeypatch, _RecordingReader(result=[]))
    client = _logged_in_client(_make_app())

    assert client.post(TIMELINE_URL).status_code == 405


def test_timeline_route_performs_no_usage_lookup(monkeypatch):
    """An Operational-only request_id (no Usage row) is a valid key."""
    reader = _install_reader(
        monkeypatch, _RecordingReader(result=[_event(event="manual_x10_check")])
    )

    def _fail(*args, **kwargs):
        raise AssertionError("the timeline route must not query Usage")

    monkeypatch.setattr(admin_route, "get_logs_for_admin", _fail)
    client = _logged_in_client(_make_app())

    response = client.get("/admin/logs/x10-20260820T152354Z/operational")

    assert response.status_code == 200
    assert reader.calls == ["x10-20260820T152354Z"]
