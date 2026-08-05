"""Tests for GET /datachat/export/<token>.

The access-control property under test: a token is only resolvable through the engine
registered for the API key that produced it, and it is never used as a path.
"""

import logging
import os
import tempfile
from types import SimpleNamespace

import pytest
from flask import Flask

from routes import datachat as datachat_route


class FakeEngine:
    """Minimal stand-in exposing the engine's resolve_export contract."""

    def __init__(self) -> None:
        self._exports: dict[str, tuple[str, str]] = {}

    def add_export(self, token: str, path: str, download_name: str) -> None:
        self._exports[token] = (path, download_name)

    def resolve_export(self, token):
        entry = self._exports.get(str(token or ""))
        if entry and os.path.isfile(entry[0]):
            return entry
        return None


class EngineWithoutExports:
    """An engine predating the export feature: must 404, not blow up."""


@pytest.fixture()
def csv_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "deadbeef.csv")
        with open(path, "w") as handle:
            handle.write("a,b\n1,2\n3,4\n")
        yield path


def _make_app() -> Flask:
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(datachat_token_cost=1)
    app.config["DATACHAT_RUNTIME_LOGGER"] = logging.getLogger("test.datachat.runtime")
    app.register_blueprint(datachat_route.datachat_bp)
    return app


def _client(monkeypatch, agents: dict):
    """Wire a Flask test client whose getAgent resolves from `agents`."""
    monkeypatch.setattr(datachat_route, "assert_valid_api_key", lambda *args: None)
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: agents.get(api_key))
    return _make_app().test_client()


_HEADERS = {"X-API-KEY": "key-a", "X-USER-EMAIL": "a@example.com"}


def test_owner_downloads_the_csv(monkeypatch, csv_file):
    engine = FakeEngine()
    engine.add_export("tok-1", csv_file, "sentiment_reviews.csv")
    client = _client(monkeypatch, {"key-a": engine})

    response = client.get("/datachat/export/tok-1", headers=_HEADERS)

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert b"a,b\n1,2\n3,4" in response.data
    assert "sentiment_reviews.csv" in response.headers["Content-Disposition"]


def test_token_from_another_session_is_not_resolvable(monkeypatch, csv_file):
    """A token minted under key-a must be useless under key-b."""
    engine_a = FakeEngine()
    engine_a.add_export("tok-1", csv_file, "secret.csv")
    engine_b = FakeEngine()
    client = _client(monkeypatch, {"key-a": engine_a, "key-b": engine_b})

    response = client.get(
        "/datachat/export/tok-1",
        headers={"X-API-KEY": "key-b", "X-USER-EMAIL": "b@example.com"},
    )

    assert response.status_code == 404


def test_unknown_token_is_404(monkeypatch):
    client = _client(monkeypatch, {"key-a": FakeEngine()})

    response = client.get("/datachat/export/nope", headers=_HEADERS)

    assert response.status_code == 404


def test_missing_file_on_disk_is_404(monkeypatch):
    engine = FakeEngine()
    engine.add_export("tok-1", "/tmp/definitely-not-here-12345.csv", "gone.csv")
    client = _client(monkeypatch, {"key-a": engine})

    response = client.get("/datachat/export/tok-1", headers=_HEADERS)

    assert response.status_code == 404


def test_no_active_agent_is_400(monkeypatch):
    client = _client(monkeypatch, {})

    response = client.get("/datachat/export/tok-1", headers=_HEADERS)

    assert response.status_code == 400
    assert "not active" in response.get_json()["error"].lower()


def test_engine_without_export_support_is_404(monkeypatch):
    client = _client(monkeypatch, {"key-a": EngineWithoutExports()})

    response = client.get("/datachat/export/tok-1", headers=_HEADERS)

    assert response.status_code == 404


def test_missing_api_key_is_400(monkeypatch):
    client = _client(monkeypatch, {"key-a": FakeEngine()})

    response = client.get(
        "/datachat/export/tok-1", headers={"X-USER-EMAIL": "a@example.com"}
    )

    assert response.status_code == 400


def test_missing_user_email_is_400(monkeypatch):
    client = _client(monkeypatch, {"key-a": FakeEngine()})

    response = client.get("/datachat/export/tok-1", headers={"X-API-KEY": "key-a"})

    assert response.status_code == 400


def test_invalid_api_key_is_rejected(monkeypatch, csv_file):
    """assert_valid_api_key gates the route like every other DataChat endpoint."""
    from werkzeug.exceptions import Forbidden

    engine = FakeEngine()
    engine.add_export("tok-1", csv_file, "x.csv")

    def deny(*args):
        raise Forbidden("API key expired")

    monkeypatch.setattr(datachat_route, "assert_valid_api_key", deny)
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: engine)
    client = _make_app().test_client()

    response = client.get("/datachat/export/tok-1", headers=_HEADERS)

    assert response.status_code == 403


def test_traversal_in_the_token_does_not_reach_the_filesystem(monkeypatch):
    """The token is a dict key; a path-shaped one simply misses."""
    engine = FakeEngine()
    client = _client(monkeypatch, {"key-a": engine})

    response = client.get(
        "/datachat/export/..%2f..%2f..%2fetc%2fpasswd", headers=_HEADERS
    )

    assert response.status_code in (404, 308)
    assert b"root:" not in response.data
