"""
Route-level tests for the analysis interviewer endpoints.

They pin the HTTP contract: the SQL datasource is required, starting is free
and tokens are spent only on usable turns, a draft is never stored, approval
stores the brief and returns its id, and approving or declining ends the
session. No database and no LLM are contacted — the engine is a
scripted fake and every collaborator the routes reach for is patched.
"""

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
    state = SimpleNamespace(spent=[], saved=[], engine=FakeEngine(), tokens=10)

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
