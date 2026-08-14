"""Source Slice D1: centralized Operational app_id binding in the shared
API-key authentication seam.

Covers:
- build_validate_api_key_query() widened to also select users.client, in
  the same round-trip, no second query.
- validate_api_key() extended return contract: (bool, str, Optional[str]),
  client only populated on a successful match, never leaked on failure.
- assert_valid_api_key() binds app_id via set_request_context() only after
  a successful validation, preserving request_id and existing 403
  contracts on every failure path.
- routes/auth.py::/validateapikey tolerates the widened tuple without
  exposing client in its response.

No live PostgreSQL: fake cursor/connection, same style as
tests/test_database_set_user_client_if_missing.py.
"""

from unittest.mock import patch

import pytest
from flask import Flask

from infrastructure import database_pg
from infrastructure.database_methods import build_validate_api_key_query
from routes import auth as auth_route
from routes import utils as routes_utils
from utils.logging_config import (
    CONTEXT_UNSET,
    _app_id_var,
    _request_id_var,
    get_request_id,
    register_request_context_hooks,
    set_request_context,
)


# ---------------------------------------------------------------------------
# build_validate_api_key_query
# ---------------------------------------------------------------------------


def test_build_validate_api_key_query_targets_users_by_username():
    query, params = build_validate_api_key_query("alice")

    query_str = query.as_string(None)
    assert "SELECT" in query_str
    assert '"users"' in query_str
    assert '"username" = %s' in query_str
    assert params == ("alice",)


def test_build_validate_api_key_query_selects_api_key_and_date_valid_until():
    query, _params = build_validate_api_key_query("alice")

    query_str = query.as_string(None)
    assert '"api_key"' in query_str
    assert '"date_valid_until"' in query_str


def test_build_validate_api_key_query_now_includes_client():
    query, _params = build_validate_api_key_query("alice")

    query_str = query.as_string(None)
    assert '"client"' in query_str


def test_build_validate_api_key_query_uses_placeholders_not_interpolation():
    query, params = build_validate_api_key_query("alice")

    query_str = query.as_string(None)
    assert "alice" not in query_str
    assert query_str.count("%s") == 1
    assert params == ("alice",)


# ---------------------------------------------------------------------------
# validate_api_key
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def _mock_cipher_suite(decrypted_plain: str):
    """Patch get_cipher_suite so decrypt() always returns decrypted_plain."""
    patcher = patch.object(database_pg, "get_cipher_suite")
    fake_factory = patcher.start()
    fake_factory.return_value.decrypt.return_value = decrypted_plain.encode()
    return patcher


def test_validate_api_key_valid_key_returns_true_match_and_client(monkeypatch):
    cursor = _FakeCursor([(b"encrypted", "2099-01-01 00:00:00", "dino")])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    patcher = _mock_cipher_suite("secret-key")
    try:
        result, message, client = database_pg.validate_api_key("secret-key", "alice")
    finally:
        patcher.stop()

    assert result is True
    assert message == "API key match found"
    assert client == "dino"
    assert conn.closed is True


def test_validate_api_key_valid_key_with_null_client_returns_none(monkeypatch):
    cursor = _FakeCursor([(b"encrypted", "2099-01-01 00:00:00", None)])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    patcher = _mock_cipher_suite("secret-key")
    try:
        result, message, client = database_pg.validate_api_key("secret-key", "alice")
    finally:
        patcher.stop()

    assert result is True
    assert message == "API key match found"
    assert client is None


def test_validate_api_key_invalid_key_returns_false_and_none_client(monkeypatch):
    cursor = _FakeCursor([(b"encrypted", "2099-01-01 00:00:00", "dino")])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    patcher = _mock_cipher_suite("secret-key")
    try:
        result, message, client = database_pg.validate_api_key("wrong-key", "alice")
    finally:
        patcher.stop()

    assert result is False
    assert message == "No matching API key found"
    assert client is None


def test_validate_api_key_expired_key_returns_false_and_none_client(monkeypatch):
    cursor = _FakeCursor([(b"encrypted", "2000-01-01 00:00:00", "dino")])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    patcher = _mock_cipher_suite("secret-key")
    try:
        result, message, client = database_pg.validate_api_key("secret-key", "alice")
    finally:
        patcher.stop()

    assert result is False
    assert message == "API key expired"
    assert client is None


def test_validate_api_key_no_rows_returns_false_and_none_client(monkeypatch):
    cursor = _FakeCursor([])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    result, message, client = database_pg.validate_api_key("secret-key", "ghost")

    assert result is False
    assert message == "No matching API key found"
    assert client is None


def test_validate_api_key_does_not_call_get_user_by_username(monkeypatch):
    """Zero-extra-query: the existing validation SELECT is the only query."""
    cursor = _FakeCursor([(b"encrypted", "2099-01-01 00:00:00", "dino")])
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    calls = []
    monkeypatch.setattr(
        database_pg,
        "get_user_by_username",
        lambda *a, **k: calls.append((a, k)),
    )
    patcher = _mock_cipher_suite("secret-key")
    try:
        database_pg.validate_api_key("secret-key", "alice")
    finally:
        patcher.stop()

    assert calls == []
    assert len(cursor.executed) == 1


# ---------------------------------------------------------------------------
# assert_valid_api_key — shared seam binding
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_context_vars():
    """Inter-test isolation only - not a lifecycle proof.

    This resets both ContextVars after every test in this module so a
    leak cannot make a later test order-dependent. It does not run until
    each test's assertions have already executed, so it never masks a
    within-test failure; it also runs after the real teardown_request hook
    has already fired inside the test client call, so it proves nothing
    about production cleanup either way. The sequential-request/teardown
    regression coverage for that lives in tests/test_logging_config.py,
    against the real hooks with no manual reset in the test body.
    """
    request_token = _request_id_var.set(CONTEXT_UNSET)
    app_token = _app_id_var.set(CONTEXT_UNSET)
    try:
        yield
    finally:
        _request_id_var.reset(request_token)
        _app_id_var.reset(app_token)


def _app_with_context():
    app = Flask(__name__)
    register_request_context_hooks(app)

    @app.route("/probe")
    def probe():
        set_request_context(request_id="fixed-request-id")
        before_app_id = _app_id_var.get()
        client = routes_utils.assert_valid_api_key("some-key", "alice")
        return {
            "request_id": get_request_id(),
            "before_app_id": before_app_id,
            "after_app_id": _app_id_var.get(),
            "returned_client": client,
        }

    return app


def test_assert_valid_api_key_valid_client_binds_app_id_and_preserves_request_id(
    monkeypatch,
):
    monkeypatch.setattr(
        routes_utils, "validate_api_key", lambda *a, **k: (True, "match", "dino")
    )
    app = _app_with_context()

    response = app.test_client().get("/probe")

    assert response.status_code == 200
    body = response.get_json()
    assert body["request_id"] == "fixed-request-id"
    assert body["before_app_id"] == CONTEXT_UNSET
    assert body["after_app_id"] == "dino"
    assert body["returned_client"] == "dino"


def test_assert_valid_api_key_valid_null_client_leaves_app_id_unset(monkeypatch):
    monkeypatch.setattr(
        routes_utils, "validate_api_key", lambda *a, **k: (True, "match", None)
    )
    app = _app_with_context()

    response = app.test_client().get("/probe")

    assert response.status_code == 200
    body = response.get_json()
    assert body["request_id"] == "fixed-request-id"
    assert body["after_app_id"] == CONTEXT_UNSET
    assert body["returned_client"] is None


def test_assert_valid_api_key_invalid_key_aborts_403_and_leaves_app_id_unset(
    monkeypatch,
):
    monkeypatch.setattr(
        routes_utils,
        "validate_api_key",
        lambda *a, **k: (False, "No matching API key found", None),
    )
    app = _app_with_context()

    response = app.test_client().get("/probe")

    assert response.status_code == 403
    assert "Invalid API key" in response.get_data(as_text=True)
    assert _app_id_var.get() == CONTEXT_UNSET


def test_assert_valid_api_key_expired_key_aborts_403_and_leaves_app_id_unset(
    monkeypatch,
):
    monkeypatch.setattr(
        routes_utils,
        "validate_api_key",
        lambda *a, **k: (False, "API key expired", None),
    )
    app = _app_with_context()

    response = app.test_client().get("/probe")

    assert response.status_code == 403
    assert "API key expired" in response.get_data(as_text=True)
    assert _app_id_var.get() == CONTEXT_UNSET


def test_assert_valid_api_key_missing_key_aborts_403_without_calling_validate(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        routes_utils,
        "validate_api_key",
        lambda *a, **k: calls.append((a, k)),
    )
    app = Flask(__name__)
    register_request_context_hooks(app)

    @app.route("/probe")
    def probe():
        return {"client": routes_utils.assert_valid_api_key("", "alice")}

    response = app.test_client().get("/probe")

    assert response.status_code == 403
    assert "Missing API key" in response.get_data(as_text=True)
    assert calls == []
    assert _app_id_var.get() == CONTEXT_UNSET


# ---------------------------------------------------------------------------
# /validateapikey — direct caller compatibility
# ---------------------------------------------------------------------------


def _auth_app():
    app = Flask(__name__)
    app.register_blueprint(auth_route.auth_bp)
    return app


def test_validateapikey_route_matches_key_response_unchanged(monkeypatch):
    monkeypatch.setattr(
        auth_route, "validate_api_key", lambda *a, **k: (True, "API key match found", "dino")
    )

    response = _auth_app().test_client().post(
        "/validateapikey",
        headers={"X-API-KEY": "secret", "X-USER-EMAIL": "alice"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"response": "API key match found"}
    assert "client" not in response.get_json()


def test_validateapikey_route_invalid_key_response_unchanged(monkeypatch):
    monkeypatch.setattr(
        auth_route,
        "validate_api_key",
        lambda *a, **k: (False, "No matching API key found", None),
    )

    response = _auth_app().test_client().post(
        "/validateapikey",
        headers={"X-API-KEY": "secret", "X-USER-EMAIL": "alice"},
    )

    assert response.status_code == 403
    assert response.get_json() == {"error": "Invalid API key"}


def test_validateapikey_route_expired_key_response_unchanged(monkeypatch):
    monkeypatch.setattr(
        auth_route,
        "validate_api_key",
        lambda *a, **k: (False, "API key expired", None),
    )

    response = _auth_app().test_client().post(
        "/validateapikey",
        headers={"X-API-KEY": "secret", "X-USER-EMAIL": "alice"},
    )

    assert response.status_code == 403
    assert response.get_json() == {"error": "API key expired"}
