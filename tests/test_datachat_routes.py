"""
Route-level tests for /startdatachat's data contract.

The CSV file became optional when the SQL datasource was added, because a
SQL-only session is legitimate. These tests pin the boundary: optional when a
database is available, still mandatory when it is not.

No database and no LLM are contacted — every collaborator the route reaches for
is patched.
"""

import io
from types import SimpleNamespace

import pandas as pd
import pytest
from flask import Flask

from routes import datachat as datachat_route


HEADERS = {
    "X-API-KEY": "key-123",
    "X-USER-EMAIL": "tester@example.com",
    "X-USER-NAME": "Test User",
}


def _make_app() -> Flask:
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        datachat_token_cost=1,
        datachat=SimpleNamespace(engine="smolagents"),
        models=SimpleNamespace(
            datachat_provider="Google",
            datachat_model="gemini-2.5-flash",
        ),
    )
    app.register_blueprint(datachat_route.datachat_bp)
    return app


@pytest.fixture
def created(monkeypatch):
    """Patch everything past validation; record what createAgent received."""
    calls = []

    def fake_create_agent(api_key, data, llm, user_name, engine_type=None):
        calls.append({"data": data, "user_name": user_name})
        engine = SimpleNamespace(
            bootstrap=lambda lang: SimpleNamespace(suggested_questions_html=None)
        )
        return engine

    monkeypatch.setattr(datachat_route, "assert_valid_api_key", lambda *a: None)
    monkeypatch.setattr(datachat_route, "get_user_tokens", lambda email: 10)
    monkeypatch.setattr(datachat_route, "edit_tokens", lambda *a, **kw: None)
    monkeypatch.setattr(datachat_route, "choose_llm", lambda *a, **kw: object())
    monkeypatch.setattr(datachat_route, "createAgent", fake_create_agent)
    monkeypatch.setattr(
        datachat_route,
        "load_csv_to_dataframe",
        lambda f: pd.DataFrame({"country": ["IT"], "sales": [1]}),
    )
    return calls


def _post(app, data=None):
    return app.test_client().post(
        "/startdatachat",
        headers=HEADERS,
        data=data or {},
        content_type="multipart/form-data",
    )


def _csv_payload():
    return {"file": (io.BytesIO(b"country,sales\nIT,1\n"), "data.csv")}


# ---------------------------------------------------------------------------
# The CSV contract
# ---------------------------------------------------------------------------

def test_a_csv_session_starts_and_the_engine_gets_the_dataframe(created, monkeypatch):
    monkeypatch.setattr(datachat_route, "get_sql_datasource", lambda: None)

    response = _post(_make_app(), _csv_payload())

    assert response.status_code == 200
    assert response.get_json() == {"Agent active": "active"}
    assert isinstance(created[0]["data"], pd.DataFrame)
    assert list(created[0]["data"].columns) == ["country", "sales"]


def test_no_file_and_no_sql_is_refused(created, monkeypatch):
    """
    Without a datasource this would start a session with no data and no tools.
    It must fail here, not obscurely on the user's first question.
    """
    monkeypatch.setattr(datachat_route, "get_sql_datasource", lambda: None)

    response = _post(_make_app())

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Missing parameters"
    assert "CSV file is required" in body["detail"]
    assert created == []  # no engine was built


def test_no_file_is_allowed_when_the_sql_datasource_is_available(created, monkeypatch):
    monkeypatch.setattr(datachat_route, "get_sql_datasource", lambda: object())

    response = _post(_make_app())

    assert response.status_code == 200
    assert created[0]["data"] is None


def test_the_other_required_parameters_are_still_checked(created, monkeypatch):
    """A missing user name is refused whatever the datasource situation is."""
    monkeypatch.setattr(datachat_route, "get_sql_datasource", lambda: object())

    response = _make_app().test_client().post(
        "/startdatachat",
        headers={k: v for k, v in HEADERS.items() if k != "X-USER-NAME"},
        data=_csv_payload(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing parameters"
    assert created == []
