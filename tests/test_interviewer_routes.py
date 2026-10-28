"""
Route-level tests for the analysis interviewer endpoints.

They pin the HTTP contract: the SQL datasource is required, starting is free
and tokens are spent only on usable turns, a draft is never stored, approval
stores the brief, returns its id and starts the analyst on it, and approving
or declining ends the session. No database and no LLM are contacted — the engine is a
scripted fake and every collaborator the routes reach for is patched.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from flask import Flask

import infrastructure.agent_manager as agent_manager
from routes import interviewer as interviewer_route

HEADERS = {
    "X-API-KEY": "key-123",
    "X-USER-EMAIL": "tester@example.com",
    "X-USER-NAME": "Test User",
}

QUESTION = {"kind": "question", "text": "Why?", "topic": "goal"}
DRAFT = {"kind": "brief_draft", "summary": "S", "brief": {"name": "B"}, "prompt_text": "P"}


class FakeEngine:
    """Scripted stand-in for InterviewerEngine: returns queued results."""

    model = "some/model"
    provider = "Deepinfra"

    def __init__(self, results=(), ready=True):
        self._results = list(results)
        self.is_ready = ready
        self.pending_draft = None
        self.transcript = [{"role": "user", "text": "hi"}]
        self.calls = []
        self.closed = False

    def bootstrap(self, lang):
        return SimpleNamespace(suggested_questions_html=f"<h2>{lang}</h2>")

    def _next(self, call):
        self.calls.append(call)
        result = self._results.pop(0)
        if result["kind"] == "brief_draft":
            self.pending_draft = {k: result[k] for k in ("summary", "brief", "prompt_text")}
        return result

    def chat(self, message):
        return self._next(("chat", message))

    def reject(self, feedback):
        self.pending_draft = None
        return self._next(("reject", feedback))

    def approve(self):
        self.calls.append(("approve",))
        draft, self.pending_draft = self.pending_draft, None
        return {"kind": "brief", **draft}

    def close(self):
        self.closed = True


def _make_app() -> Flask:
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        interviewer_token_cost=2,
        interviewer=SimpleNamespace(),
    )
    app.register_blueprint(interviewer_route.interviewer_bp)
    return app


@pytest.fixture
def env(monkeypatch):
    """Patch everything past validation; record token spending and saved briefs."""
    state = SimpleNamespace(
        spent=[], saved=[], analyses=[], engine=FakeEngine(), tokens=10,
        analysis=({"run_id": "run-1", "status": "queued"}, 200),
    )

    monkeypatch.setattr(agent_manager, "activeEngines", {})
    monkeypatch.setattr(interviewer_route, "assert_valid_api_key", lambda *a: None)
    monkeypatch.setattr(interviewer_route, "get_user_tokens", lambda email: state.tokens)
    monkeypatch.setattr(
        interviewer_route, "edit_tokens", lambda email, qty: state.spent.append(qty)
    )
    monkeypatch.setattr(interviewer_route, "get_sql_datasource", lambda: object())
    monkeypatch.setattr(
        interviewer_route, "create_interviewer_engine", lambda **kwargs: state.engine
    )

    def fake_save(**kwargs):
        state.saved.append(kwargs)
        return {"brief_id": 42, "name": kwargs["brief"]["name"]}

    monkeypatch.setattr(interviewer_route, "save_approved_brief", fake_save)

    def fake_start_analysis(**kwargs):
        state.analyses.append(kwargs)
        return state.analysis

    monkeypatch.setattr(interviewer_route, "start_analysis_payload", fake_start_analysis)
    state.app = _make_app()
    return state


def _start(env, json=None, headers=HEADERS):
    return env.app.test_client().post("/start-analysis-interview", headers=headers, json=json or {})


def _turn(env, json):
    return env.app.test_client().post("/analysis-interview", headers=HEADERS, json=json)


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


def test_start_greets_and_opens_the_session_for_free(env):
    response = _start(env, {"lang": "ITA"})

    assert response.status_code == 200
    assert response.get_json() == {"Agent active": "active", "welcome": "<h2>ITA</h2>"}
    assert agent_manager.getInterviewer("key-123") is env.engine
    assert env.engine.calls == []
    assert env.spent == []


def test_start_ignores_a_goal(env):
    response = _start(env, {"goal": "revenue by country"})

    assert "response" not in response.get_json()
    assert env.engine.calls == []


def test_start_accepts_form_data(env):
    response = env.app.test_client().post(
        "/start-analysis-interview", headers=HEADERS, data={"lang": "FRA"},
        content_type="multipart/form-data",
    )

    assert response.get_json()["welcome"] == "<h2>FRA</h2>"


def test_start_is_refused_without_the_sql_datasource(env, monkeypatch):
    monkeypatch.setattr(interviewer_route, "get_sql_datasource", lambda: None)

    response = _start(env)

    assert response.status_code == 400
    assert "SQL datasource" in response.get_json()["error"]
    assert agent_manager.getInterviewer("key-123") is None


def test_an_engine_that_cannot_run_is_not_kept(env):
    env.engine = FakeEngine(ready=False)

    response = _start(env)

    assert response.status_code == 500
    assert agent_manager.getInterviewer("key-123") is None
    assert env.engine.closed


@pytest.mark.parametrize("missing", ["X-API-KEY", "X-USER-EMAIL", "X-USER-NAME"])
def test_start_requires_every_header(env, missing):
    headers = {k: v for k, v in HEADERS.items() if k != missing}

    assert _start(env, headers=headers).status_code == 400


def test_start_requires_enough_tokens(env):
    env.tokens = 1

    response = _start(env)

    assert response.status_code == 500
    assert response.get_json()["error"] == "Not enough tokens"


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


@pytest.fixture
def started(env):
    _start(env)
    env.spent.clear()
    return env


def test_the_first_answer_states_the_goal_and_spends_tokens(started):
    started.engine._results = [QUESTION]

    response = _turn(started, {"answer": "revenue by country"})

    assert response.status_code == 200
    assert response.get_json()["response"] == {
        "type": "question",
        "value": {"text": "Why?", "topic": "goal"},
    }
    assert started.engine.calls == [("chat", "revenue by country")]
    assert started.spent == [-2]


def test_answer_is_the_default_action(started):
    started.engine._results = [QUESTION]

    _turn(started, {"action": "answer", "answer": "to set targets"})

    assert started.engine.calls == [("chat", "to set targets")]


def test_a_draft_is_returned_but_never_stored(started):
    started.engine._results = [DRAFT]

    response = _turn(started, {"answer": "a short report"})

    assert response.get_json()["response"] == {
        "type": "brief_draft",
        "value": {"summary": "S", "brief": {"name": "B"}, "prompt_text": "P"},
    }
    assert started.saved == []


def test_approval_stores_the_brief_and_returns_its_id(started):
    started.engine._results = [DRAFT]
    _turn(started, {"answer": "a short report"})
    started.spent.clear()

    response = _turn(started, {"action": "approve"})

    body = response.get_json()
    assert response.status_code == 200
    assert body["brief_id"] == 42
    assert body["response"]["type"] == "brief"
    assert started.saved[0]["brief"] == {"name": "B"}
    assert started.saved[0]["prompt_text"] == "P"
    assert started.saved[0]["transcript"] == started.engine.transcript
    assert started.saved[0]["user_email"] == "tester@example.com"
    assert started.spent == []  # approval is not a model call


def test_approval_hands_the_brief_to_the_analyst(started):
    started.engine._results = [DRAFT]
    _turn(started, {"answer": "a short report"})

    body = _turn(started, {"action": "approve"}).get_json()

    assert body["analysis"] == {"run_id": "run-1", "status": "queued"}
    assert started.analyses[0]["brief_id"] == 42
    assert started.analyses[0]["api_key"] == "key-123"
    assert started.analyses[0]["user_email"] == "tester@example.com"


def test_an_analysis_that_cannot_start_does_not_undo_the_approval(started):
    started.engine._results = [DRAFT]
    _turn(started, {"answer": "a short report"})
    started.analysis = ({"error": "Not enough tokens.", "code": "NOT_ENOUGH_TOKENS"}, 500)

    response = _turn(started, {"action": "approve"})

    assert response.status_code == 200
    assert response.get_json()["brief_id"] == 42
    assert response.get_json()["analysis"]["code"] == "NOT_ENOUGH_TOKENS"
    assert agent_manager.getInterviewer("key-123") is None


def test_approval_ends_the_session(started):
    started.engine._results = [DRAFT]
    _turn(started, {"answer": "a short report"})

    _turn(started, {"action": "approve"})

    assert started.engine.closed
    assert agent_manager.getInterviewer("key-123") is None


def test_a_failed_save_leaves_the_draft_pending(started, monkeypatch):
    started.engine._results = [DRAFT]
    _turn(started, {"answer": "a short report"})

    def failing_save(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(interviewer_route, "save_approved_brief", failing_save)

    response = _turn(started, {"action": "approve"})

    assert response.status_code == 500
    assert started.engine.pending_draft is not None
    assert ("approve",) not in started.engine.calls
    assert started.analyses == []
    assert agent_manager.getInterviewer("key-123") is started.engine


@pytest.mark.parametrize("action", ["approve", "decline"])
def test_deciding_without_a_draft_is_refused(started, action):
    response = _turn(started, {"action": action})

    assert response.status_code == 400
    assert response.get_json()["code"] == "NO_PENDING_DRAFT"
    assert agent_manager.getInterviewer("key-123") is started.engine


def test_revising_keeps_working_and_stores_nothing(started):
    started.engine._results = [DRAFT, QUESTION]
    _turn(started, {"answer": "a short report"})
    started.spent.clear()

    response = _turn(started, {"action": "revise", "answer": "add a monthly split"})

    assert response.get_json()["response"]["type"] == "question"
    assert started.engine.calls[-1] == ("reject", "add a monthly split")
    assert started.saved == []
    assert started.spent == [-2]
    assert agent_manager.getInterviewer("key-123") is started.engine


def test_revising_needs_no_reason(started):
    started.engine._results = [DRAFT, QUESTION]
    _turn(started, {"answer": "a short report"})

    _turn(started, {"action": "revise"})

    assert started.engine.calls[-1] == ("reject", None)


def test_declining_stores_nothing_and_ends_the_session(started):
    started.engine._results = [DRAFT]
    _turn(started, {"answer": "a short report"})
    started.spent.clear()

    response = _turn(started, {"action": "decline"})

    assert response.status_code == 200
    assert response.get_json()["response"] == {"type": "declined", "value": {}}
    assert started.saved == []
    assert started.spent == []
    assert started.engine.closed
    assert agent_manager.getInterviewer("key-123") is None


def test_an_answer_while_a_draft_is_pending_is_refused(started):
    started.engine._results = [
        DRAFT,
        {"kind": "error", "message": "decide first", "code": "DRAFT_PENDING"},
    ]
    _turn(started, {"answer": "a short report"})
    started.spent.clear()

    response = _turn(started, {"answer": "add a region"})

    assert response.status_code == 400
    assert response.get_json()["response"]["value"]["code"] == "DRAFT_PENDING"
    assert started.spent == []


def test_an_engine_error_does_not_spend_tokens(started):
    started.engine._results = [{"kind": "error", "message": "bad", "code": "INVALID_OUTPUT"}]

    response = _turn(started, {"answer": "hi"})

    assert response.status_code == 200
    assert response.get_json()["response"] == {
        "type": "error",
        "value": {"message": "bad", "code": "INVALID_OUTPUT"},
    }
    assert started.spent == []


@pytest.mark.parametrize(
    "body,error",
    [
        ({}, "Missing answer string"),
        ({"answer": "   "}, "Missing answer string"),
        ({"action": "answer"}, "Missing answer string"),
        ({"action": "maybe", "answer": "hi"}, "action must be one of answer, approve, revise, decline"),
    ],
)
def test_malformed_turns_are_refused(started, body, error):
    response = _turn(started, body)

    assert response.status_code == 400
    assert response.get_json()["error"] == error


def test_a_turn_without_a_session_is_refused(env):
    response = _turn(env, {"answer": "hi"})

    assert response.status_code == 400
    assert "not active" in response.get_json()["error"]


def test_a_datachat_session_is_not_an_interview(env):
    agent_manager.activeEngines[(agent_manager.DATACHAT, "key-123")] = FakeEngine()

    assert _turn(env, {"answer": "hi"}).status_code == 400


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------


def test_end_closes_the_session(started):
    response = started.app.test_client().post("/end-analysis-interview", headers=HEADERS)

    assert response.status_code == 200
    assert started.engine.closed
    assert agent_manager.getInterviewer("key-123") is None


def test_end_without_a_session_says_so(env):
    response = env.app.test_client().post("/end-analysis-interview", headers=HEADERS)

    assert "Agent was not active for this key" in response.get_json()


# ---------------------------------------------------------------------------
# Stored briefs
# ---------------------------------------------------------------------------


@pytest.fixture
def briefs(env, monkeypatch):
    """Stored briefs of two users; record the API key checks."""
    env.key_checks = []
    env.briefs = [
        {"id": 2, "user_email": "tester@example.com", "name": "B2", "status": "consumed",
         "approved_at": datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc)},
        {"id": 1, "user_email": "tester@example.com", "name": "B1", "status": "ready",
         "approved_at": datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc)},
        {"id": 3, "user_email": "someone@else.com", "name": "B3", "status": "ready",
         "approved_at": datetime(2026, 10, 3, 9, 30, tzinfo=timezone.utc)},
    ]

    def list_briefs(user_email=None, status=None, limit=50):
        return [
            {**b, "approved_at": b["approved_at"].isoformat()}
            for b in env.briefs
            if b["user_email"] == user_email and status in (None, b["status"])
        ][:limit]

    def get_brief(brief_id, user_email=None):
        return next(
            ({**b, "brief": {"name": b["name"]}, "prompt_text": "P"}
             for b in env.briefs if b["id"] == brief_id and b["user_email"] == user_email),
            None,
        )

    monkeypatch.setattr(interviewer_route, "assert_valid_api_key", lambda *a: env.key_checks.append(a))
    monkeypatch.setattr(interviewer_route, "list_briefs", list_briefs)
    monkeypatch.setattr(interviewer_route, "get_brief", get_brief)
    env.client = env.app.test_client()
    return env


def test_the_list_holds_the_user_briefs_and_filters_by_status(briefs):
    body = briefs.client.get("/analysis-briefs", headers=HEADERS).get_json()
    assert [b["id"] for b in body["briefs"]] == [2, 1]
    assert body["briefs"][0]["approved_at"] == "2026-10-02T09:30:00+00:00"

    body = briefs.client.get("/analysis-briefs?status=ready", headers=HEADERS).get_json()
    assert [b["id"] for b in body["briefs"]] == [1]
    assert briefs.key_checks == [("key-123", "tester@example.com")] * 2


@pytest.mark.parametrize("query", ["status=deleted", "limit=many"])
def test_the_list_refuses_a_bad_filter(briefs, query):
    assert briefs.client.get(f"/analysis-briefs?{query}", headers=HEADERS).status_code == 400


def test_a_brief_is_returned_with_its_content_to_its_owner_only(briefs):
    response = briefs.client.get("/analysis-brief?brief_id=1", headers=HEADERS)

    assert response.status_code == 200
    brief = response.get_json()["brief"]
    assert brief["brief"] == {"name": "B1"} and brief["prompt_text"] == "P"
    assert brief["approved_at"] == "2026-10-01T09:30:00+00:00"
    assert briefs.key_checks == [("key-123", "tester@example.com")]

    for brief_id in (3, 99):
        response = briefs.client.get(f"/analysis-brief?brief_id={brief_id}", headers=HEADERS)
        assert response.status_code == 404 and response.get_json()["code"] == "BRIEF_NOT_FOUND"


@pytest.mark.parametrize("query", ["", "?brief_id=", "?brief_id=abc"])
def test_a_brief_needs_an_integer_id(briefs, query):
    assert briefs.client.get(f"/analysis-brief{query}", headers=HEADERS).status_code == 400


@pytest.mark.parametrize("path", ["/analysis-briefs", "/analysis-brief?brief_id=1"])
def test_the_brief_endpoints_need_the_auth_headers(briefs, path):
    assert briefs.client.get(path, headers={"X-API-KEY": "key-123"}).status_code == 400
    assert briefs.key_checks == []


def test_a_database_failure_on_the_briefs_is_500(briefs, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(interviewer_route, "list_briefs", broken)
    monkeypatch.setattr(interviewer_route, "get_brief", broken)
    assert briefs.client.get("/analysis-briefs", headers=HEADERS).status_code == 500
    assert briefs.client.get("/analysis-brief?brief_id=1", headers=HEADERS).status_code == 500
