"""Source Slice B2: authentication lifecycle persistence for users.client.

Covers the /checkpandinouser route wiring that connects the authenticated
client identity to the persisted Maui user record:

- new user: the authenticated client reaches the initial add_user() INSERT.
- existing user: set_user_client_if_missing() (Source Slice B1's atomic
  fill-if-empty primitive) is invoked without any route-level pre-read of
  the persisted client.
- authentication failure: no add_user, no set_user_client_if_missing.
- database failure during either path is surfaced, not swallowed into a
  fabricated success.

No network, no real authentication gateway, no real database: dino_authenticate,
external_authenticate, and database_pg are monkeypatched at the routes.auth
module namespace, same convention as
tests/test_agentchat_route_lifecycle_identity.py and
tests/test_documents_route.py.
"""

from flask import Flask

from routes import auth as auth_route

HEADERS_BASE = {
    "X-GRAPHQL-URL": "https://graphql.example.com",
    "X-AUTH-TOKEN": "token-123",
    "X-USER-EMAIL": "user@example.com",
}


def _make_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(auth_route.auth_bp)
    return app


def _client():
    return _make_app().test_client()


def test_new_dino_default_client_user_is_created_with_client_dino(monkeypatch):
    add_user_calls = []
    set_client_calls = []

    monkeypatch.setattr(auth_route, "dino_authenticate", lambda *a, **k: None)
    monkeypatch.setattr(auth_route.database_pg, "get_user_by_username", lambda u: None)

    def fake_add_user(username, api_key, date_valid_until, client):
        add_user_calls.append(
            {
                "username": username,
                "api_key": api_key,
                "date_valid_until": date_valid_until,
                "client": client,
            }
        )
        return None

    monkeypatch.setattr(auth_route.database_pg, "add_user", fake_add_user)
    monkeypatch.setattr(
        auth_route.database_pg,
        "set_user_client_if_missing",
        lambda *a, **k: set_client_calls.append((a, k)),
    )

    response = _client().post("/checkpandinouser", headers=HEADERS_BASE)

    assert response.status_code == 200
    body = response.get_json()
    assert body["response"]["user"]["user_email"] == "user@example.com"
    assert len(add_user_calls) == 1
    assert add_user_calls[0]["client"] == "dino"
    assert set_client_calls == []


def test_new_external_client_user_propagates_client_to_creation(monkeypatch):
    external_auth_calls = []
    add_user_calls = []

    def fake_external_authenticate(email, auth_token, client, graphql_url):
        external_auth_calls.append(
            {
                "email": email,
                "auth_token": auth_token,
                "client": client,
                "graphql_url": graphql_url,
            }
        )
        return None

    monkeypatch.setattr(
        auth_route, "external_authenticate", fake_external_authenticate
    )
    monkeypatch.setattr(auth_route.database_pg, "get_user_by_username", lambda u: None)

    def fake_add_user(username, api_key, date_valid_until, client):
        add_user_calls.append(client)
        return None

    monkeypatch.setattr(auth_route.database_pg, "add_user", fake_add_user)

    headers = {**HEADERS_BASE, "X-CLIENT": "acme"}
    response = _client().post("/checkpandinouser", headers=headers)

    assert response.status_code == 200
    assert len(external_auth_calls) == 1
    assert external_auth_calls[0]["client"] == "acme"
    assert external_auth_calls[0]["email"] == "user@example.com"
    assert add_user_calls == ["acme"]


def test_existing_user_calls_fill_if_empty_primitive_without_pre_read(monkeypatch):
    set_client_calls = []
    add_user_calls = []

    monkeypatch.setattr(auth_route, "dino_authenticate", lambda *a, **k: None)
    monkeypatch.setattr(
        auth_route.database_pg,
        "get_user_by_username",
        lambda u: {
            "username": u,
            "api_key": "existing-key",
            "date_valid_until": "2030-01-01 00:00:00",
        },
    )

    def fake_set_user_client_if_missing(username, client):
        set_client_calls.append((username, client))
        return True

    monkeypatch.setattr(
        auth_route.database_pg,
        "set_user_client_if_missing",
        fake_set_user_client_if_missing,
    )
    monkeypatch.setattr(
        auth_route.database_pg,
        "add_user",
        lambda *a, **k: add_user_calls.append((a, k)),
    )

    response = _client().post("/checkpandinouser", headers=HEADERS_BASE)

    assert response.status_code == 200
    assert set_client_calls == [("user@example.com", "dino")]
    assert add_user_calls == []


def test_existing_user_already_populated_client_is_not_overwritten(monkeypatch):
    add_user_calls = []

    monkeypatch.setattr(auth_route, "dino_authenticate", lambda *a, **k: None)
    monkeypatch.setattr(
        auth_route.database_pg,
        "get_user_by_username",
        lambda u: {
            "username": u,
            "api_key": "existing-key",
            "date_valid_until": "2030-01-01 00:00:00",
        },
    )
    # False means "client already set, nothing changed" - not a failure.
    monkeypatch.setattr(
        auth_route.database_pg, "set_user_client_if_missing", lambda *a, **k: False
    )
    monkeypatch.setattr(
        auth_route.database_pg,
        "add_user",
        lambda *a, **k: add_user_calls.append((a, k)),
    )

    response = _client().post("/checkpandinouser", headers=HEADERS_BASE)

    assert response.status_code == 200
    body = response.get_json()
    assert body["response"]["user"]["api_key"] == "existing-key"
    assert add_user_calls == []


def test_dino_authentication_failure_does_not_touch_users_table(monkeypatch):
    add_user_calls = []
    set_client_calls = []

    monkeypatch.setattr(
        auth_route, "dino_authenticate", lambda *a, **k: "Dino auth query not ok"
    )
    monkeypatch.setattr(
        auth_route.database_pg,
        "add_user",
        lambda *a, **k: add_user_calls.append((a, k)),
    )
    monkeypatch.setattr(
        auth_route.database_pg,
        "set_user_client_if_missing",
        lambda *a, **k: set_client_calls.append((a, k)),
    )

    response = _client().post("/checkpandinouser", headers=HEADERS_BASE)

    assert response.status_code == 403
    assert add_user_calls == []
    assert set_client_calls == []


def test_external_authentication_failure_does_not_touch_users_table(monkeypatch):
    add_user_calls = []
    set_client_calls = []

    monkeypatch.setattr(
        auth_route,
        "external_authenticate",
        lambda *a, **k: "External authentication failed",
    )
    monkeypatch.setattr(
        auth_route.database_pg,
        "add_user",
        lambda *a, **k: add_user_calls.append((a, k)),
    )
    monkeypatch.setattr(
        auth_route.database_pg,
        "set_user_client_if_missing",
        lambda *a, **k: set_client_calls.append((a, k)),
    )

    headers = {**HEADERS_BASE, "X-CLIENT": "acme"}
    response = _client().post("/checkpandinouser", headers=headers)

    assert response.status_code == 403
    assert add_user_calls == []
    assert set_client_calls == []


def test_db_failure_during_existing_user_client_persist_is_not_silently_false(
    monkeypatch,
):
    monkeypatch.setattr(auth_route, "dino_authenticate", lambda *a, **k: None)
    monkeypatch.setattr(
        auth_route.database_pg,
        "get_user_by_username",
        lambda u: {
            "username": u,
            "api_key": "existing-key",
            "date_valid_until": "2030-01-01 00:00:00",
        },
    )

    def raising_set_user_client_if_missing(username, client):
        raise RuntimeError("db down")

    monkeypatch.setattr(
        auth_route.database_pg,
        "set_user_client_if_missing",
        raising_set_user_client_if_missing,
    )

    response = _client().post("/checkpandinouser", headers=HEADERS_BASE)

    assert response.status_code == 500
    body = response.get_json()
    assert body != {
        "response": {
            "user": {
                "user_email": "user@example.com",
                "api_key": "existing-key",
                "expiration_date": "2030-01-01 00:00:00",
            }
        }
    }


def test_db_failure_during_new_user_creation_does_not_fabricate_success(monkeypatch):
    monkeypatch.setattr(auth_route, "dino_authenticate", lambda *a, **k: None)
    monkeypatch.setattr(auth_route.database_pg, "get_user_by_username", lambda u: None)
    monkeypatch.setattr(
        auth_route.database_pg,
        "add_user",
        lambda *a, **k: "Error adding new user: db down",
    )

    response = _client().post("/checkpandinouser", headers=HEADERS_BASE)

    assert response.status_code == 500
    body = response.get_json()
    assert body == {"error": "Error adding new user: db down"}
