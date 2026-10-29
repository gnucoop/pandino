"""
Tests for starting and following analysis runs: the service's checks and the
HTTP contract of the analyst endpoints. No database, no LLM and no thread:
every collaborator is patched and the run is never actually spawned.
"""

import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from flask import Flask

import analyst.service as service
import infrastructure.agent_manager as agent_manager
from analyst.run import COMPLETED, AnalysisRun
from routes import analyst as analyst_route

HEADERS = {"X-API-KEY": "key-123", "X-USER-EMAIL": "tester@example.com", "X-USER-NAME": "Test User"}


@pytest.fixture
def env(monkeypatch):
    state = SimpleNamespace(
        spent=[], spawned=[], tokens=10, brief={"brief": {"name": "B"}},
        # Stored runs: run_id -> {"user_email", "html", "listed"} (analysis_report_service).
        reports={}, key_checks=[],
    )
    config = SimpleNamespace(analyst_token_cost=3)

    monkeypatch.setattr(agent_manager, "activeEngines", {})
    monkeypatch.setattr(service, "get_sql_datasource", lambda: object())
    monkeypatch.setattr(service, "get_brief", lambda brief_id, user_email=None: state.brief)
    monkeypatch.setattr(service, "get_user_tokens", lambda email: state.tokens)
    monkeypatch.setattr(service, "edit_tokens", lambda email, qty: state.spent.append(qty))
    monkeypatch.setattr(service, "_spawn", lambda run, cfg: state.spawned.append(run))
    monkeypatch.setattr(analyst_route, "assert_valid_api_key", lambda *a: state.key_checks.append(a))

    def get_report_html(run_id, user_email):
        report = state.reports.get(run_id)
        return report["html"] if report and report["user_email"] == user_email else None

    def get_report(run_id, user_email):
        report = state.reports.get(run_id)
        if not report or report["user_email"] != user_email:
            return None
        return {"run_id": run_id, **report, "started_at": datetime(2026, 10, 20, 8, 0, tzinfo=timezone.utc)}

    def list_reports(user_email, brief_id=None, limit=50):
        return [
            {"run_id": run_id, "brief_id": r["brief_id"]}
            for run_id, r in state.reports.items()
            if r["user_email"] == user_email and brief_id in (None, r["brief_id"])
        ][:limit]

    monkeypatch.setattr(analyst_route, "get_report_html", get_report_html)
    monkeypatch.setattr(analyst_route, "get_report", get_report)
    monkeypatch.setattr(analyst_route, "list_reports", list_reports)

    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = config
    app.register_blueprint(analyst_route.analyst_bp)
    state.client = app.test_client()
    state.config = config
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


def store_run(env, run_id, owner, *, brief_id=7, html="<html>ok</html>"):
    env.reports[run_id] = {"user_email": owner, "brief_id": brief_id, "html": html}


def test_the_report_html_is_served_to_its_owner_only(env):
    store_run(env, "r_mine", "tester@example.com")
    store_run(env, "r_other", "someone@else.com")

    response = env.client.get("/analysis-report-html?run_id=r_mine", headers=HEADERS)
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert response.data == b"<html>ok</html>"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]
    assert env.key_checks == [("key-123", "tester@example.com")]

    for run_id in ("r_other", "unknown", ""):
        response = env.client.get(f"/analysis-report-html?run_id={run_id}", headers=HEADERS)
        assert response.status_code == 404 and response.get_json()["code"] == "NO_REPORT"


def test_a_run_without_its_report_html_yet_is_404(env):
    store_run(env, "r_running", "tester@example.com", html=None)
    assert env.client.get("/analysis-report-html?run_id=r_running", headers=HEADERS).status_code == 404


def test_the_report_row_is_served_to_its_owner_only(env):
    store_run(env, "r_mine", "tester@example.com")
    store_run(env, "r_other", "someone@else.com")

    response = env.client.get("/analysis-report?run_id=r_mine", headers=HEADERS)
    assert response.status_code == 200
    report = response.get_json()["report"]
    assert report["run_id"] == "r_mine" and report["brief_id"] == 7
    assert report["html"] == "<html>ok</html>"
    assert report["started_at"] == "2026-10-20T08:00:00+00:00"
    assert env.key_checks == [("key-123", "tester@example.com")]

    for run_id in ("r_other", "unknown", ""):
        response = env.client.get(f"/analysis-report?run_id={run_id}", headers=HEADERS)
        assert response.status_code == 404 and response.get_json()["code"] == "NO_REPORT"


def test_a_run_without_its_report_yet_is_served_without_html(env):
    store_run(env, "r_running", "tester@example.com", html=None)

    response = env.client.get("/analysis-report?run_id=r_running", headers=HEADERS)
    assert response.status_code == 200 and response.get_json()["report"]["html"] is None


def test_the_list_holds_the_user_runs_and_filters_by_brief(env):
    store_run(env, "r1", "tester@example.com", brief_id=7)
    store_run(env, "r2", "tester@example.com", brief_id=8)
    store_run(env, "r3", "someone@else.com", brief_id=7)

    body = env.client.get("/analysis-reports", headers=HEADERS).get_json()
    assert [r["run_id"] for r in body["reports"]] == ["r1", "r2"]

    body = env.client.get("/analysis-reports?brief_id=8", headers=HEADERS).get_json()
    assert [r["run_id"] for r in body["reports"]] == ["r2"]
    assert env.key_checks == [("key-123", "tester@example.com")] * 2


@pytest.mark.parametrize(
    "path", ["/analysis-reports", "/analysis-report?run_id=r1", "/analysis-report-html?run_id=r1"]
)
def test_the_report_endpoints_need_the_auth_headers(env, path):
    assert env.client.get(path, headers={"X-API-KEY": "key-123"}).status_code == 400
    assert env.key_checks == []


def test_a_database_failure_is_500(env, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(analyst_route, "list_reports", broken)
    monkeypatch.setattr(analyst_route, "get_report_html", broken)
    monkeypatch.setattr(analyst_route, "get_report", broken)
    assert env.client.get("/analysis-reports", headers=HEADERS).status_code == 500
    assert env.client.get("/analysis-report?run_id=r1", headers=HEADERS).status_code == 500
    assert env.client.get("/analysis-report-html?run_id=r1", headers=HEADERS).status_code == 500


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
    assert re.fullmatch(r"\d{8}_\d{6}_b12_[0-9a-f]{6}", run_id)


def test_a_run_is_owned_by_its_user():
    run = AnalysisRun(run_id="r", brief_id=1, api_key="k", user_email="a@b")
    assert run.status_payload()["status"] == "queued"
