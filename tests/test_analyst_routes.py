"""
Tests for starting and following analysis runs: the service's checks and the
HTTP contract of the analyst endpoints. No database, no LLM and no thread:
every collaborator is patched and the run is never actually spawned.
"""

import json
from types import SimpleNamespace

import pytest
from flask import Flask

import analyst.service as service
import infrastructure.agent_manager as agent_manager
from analyst.run import COMPLETED, AnalysisRun
from routes import analyst as analyst_route

HEADERS = {"X-API-KEY": "key-123", "X-USER-EMAIL": "tester@example.com", "X-USER-NAME": "Test User"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    state = SimpleNamespace(spent=[], spawned=[], tokens=10, brief={"brief": {"name": "B"}})
    config = SimpleNamespace(analyst_token_cost=3, analyst=SimpleNamespace(reports_dir=str(tmp_path)))

    monkeypatch.setattr(agent_manager, "activeEngines", {})
    monkeypatch.setattr(service, "get_sql_datasource", lambda: object())
    monkeypatch.setattr(service, "get_brief", lambda brief_id, user_email=None: state.brief)
    monkeypatch.setattr(service, "get_user_tokens", lambda email: state.tokens)
    monkeypatch.setattr(service, "edit_tokens", lambda email, qty: state.spent.append(qty))
    monkeypatch.setattr(service, "_spawn", lambda run, cfg: state.spawned.append(run))
    monkeypatch.setattr(analyst_route, "assert_valid_api_key", lambda *a: None)

    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = config
    app.register_blueprint(analyst_route.analyst_bp)
    state.client = app.test_client()
    state.config = config
    state.reports_dir = tmp_path
    return state


def start(env, body=None):
    return env.client.post("/start-analysis", headers=HEADERS, json=body if body is not None else {"brief_id": 7})


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


def test_start_registers_spawns_and_charges(env):
    response = start(env)

    assert response.status_code == 200
    analysis = response.get_json()["analysis"]
    assert analysis["status"] == "queued"
    run = agent_manager.getAnalyst("key-123")
    assert run.run_id == analysis["run_id"] and run.brief_id == 7
    assert env.spawned == [run]
    assert env.spent == [-3]


@pytest.mark.parametrize(
    "setup,code,status",
    [
        (lambda env, mp: mp.setattr(service, "get_sql_datasource", lambda: None), "SQL_UNAVAILABLE", 400),
        (lambda env, mp: setattr(env, "brief", None), "BRIEF_NOT_FOUND", 404),
        (lambda env, mp: setattr(env, "tokens", 1), "NOT_ENOUGH_TOKENS", 500),
    ],
)
def test_start_refusals_charge_nothing(env, monkeypatch, setup, code, status):
    setup(env, monkeypatch)
    response = start(env)

    assert response.status_code == status
    assert response.get_json()["analysis"]["code"] == code
    assert env.spent == [] and env.spawned == []
    assert agent_manager.getAnalyst("key-123") is None


def test_start_needs_an_integer_brief_id(env):
    assert start(env, {"brief_id": "abc"}).status_code == 400


def test_one_run_at_a_time_but_a_finished_one_is_replaced(env):
    start(env)
    busy = start(env)
    assert busy.status_code == 409
    assert busy.get_json()["analysis"]["code"] == "ANALYSIS_BUSY"

    first = agent_manager.getAnalyst("key-123")
    first.status = COMPLETED
    assert start(env).status_code == 200
    assert agent_manager.getAnalyst("key-123") is not first
    assert first.cancelled  # closed when replaced


# ---------------------------------------------------------------------------
# Status, report, end
# ---------------------------------------------------------------------------


def test_status_returns_new_events_from_the_cursor(env):
    start(env)
    run = agent_manager.getAnalyst("key-123")
    run.progress.emit("run_started", "Started")
    run.progress.emit("task_started", "Prep", task_id="prep")

    body = env.client.get("/analysis-status?since=1", headers=HEADERS).get_json()

    assert [e["type"] for e in body["events"]] == ["task_started"]
    assert body["next_since"] == 2
    assert body["run_id"] == run.run_id


def test_status_without_a_run_is_404(env):
    assert env.client.get("/analysis-status", headers=HEADERS).status_code == 404


def write_run(env, run_id, owner, report=True):
    folder = env.reports_dir / run_id
    folder.mkdir()
    (folder / "run_metadata.json").write_text(json.dumps({"user_email": owner}))
    if report:
        (folder / "report.html").write_text("<html>ok</html>")


def test_the_report_is_served_to_its_owner_only(env):
    run_id = "20261014_101500_b7_a1b2c3"
    write_run(env, run_id, "tester@example.com")

    response = env.client.get(f"/analysis-report?run_id={run_id}", headers=HEADERS)
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert b"<html>ok</html>" in response.data

    other = "20261014_101500_b7_d4e5f6"
    write_run(env, other, "someone@else.com")
    assert env.client.get(f"/analysis-report?run_id={other}", headers=HEADERS).status_code == 404


@pytest.mark.parametrize("run_id", ["../../etc", "", "20261014_101500_b7_zzzzzz"])
def test_report_paths_cannot_be_forged(env, run_id):
    assert env.client.get(f"/analysis-report?run_id={run_id}", headers=HEADERS).status_code == 404


def test_ending_cancels_and_forgets_the_run(env):
    start(env)
    run = agent_manager.getAnalyst("key-123")

    response = env.client.post("/end-analysis", headers=HEADERS)

    assert response.get_json() == {"run_id": run.run_id, "status": "cancelled"}
    assert run.cancelled
    assert agent_manager.getAnalyst("key-123") is None
    assert env.client.post("/end-analysis", headers=HEADERS).status_code == 404


def test_run_ids_name_the_brief():
    run_id = service.new_run_id(12)
    assert service._RUN_ID_RE.match(run_id) and "_b12_" in run_id


def test_a_run_is_owned_by_its_user():
    run = AnalysisRun(run_id="r", brief_id=1, api_key="k", user_email="a@b")
    assert run.status_payload()["status"] == "queued"
