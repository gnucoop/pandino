"""Same-session concurrency guard and stale-engine safety on the interviewer.

One action (answer, revise, approve, decline) runs per engine: a concurrent
request on a busy engine gets 409 session_busy before any balance read,
engine run, token debit or brief save, and the guard is released on every
path. A request holding an engine that was ended or replaced never deletes
the newer session registered under the same Api Key.

Synchronisation uses threading.Event only; no sleeps.
"""

import threading

import pytest

from infrastructure import agent_manager
from routes import interviewer as interviewer_route
from tests.test_interviewer_routes import (  # noqa: F401  (fixtures)
    DRAFT,
    HEADERS,
    QUESTION,
    FakeEngine,
    _start,
    _turn,
    env,
    started,
)

TIMEOUT = 5
SESSION_BUSY = {
    "error": "session_busy",
    "message": "A request is already being processed for this session.",
}


@pytest.fixture(autouse=True)
def clean_busy_registry():
    agent_manager._busyEngines.clear()
    yield
    agent_manager._busyEngines.clear()


class _BlockingEngine(FakeEngine):
    """chat() signals entry, then waits until the test lets it finish."""

    def __init__(self, results=()):
        super().__init__(results)
        self.entered = threading.Event()
        self.release = threading.Event()

    def chat(self, message):
        self.entered.set()
        assert self.release.wait(TIMEOUT)
        return super().chat(message)


def _in_thread(target):
    results = []
    thread = threading.Thread(target=lambda: results.append(target()))
    thread.start()
    return thread, results


def _blocking_save(env, monkeypatch):
    """Make save_approved_brief pause until the test releases it."""
    entered, release = threading.Event(), threading.Event()

    def save(**kwargs):
        env.saved.append(kwargs)
        entered.set()
        assert release.wait(TIMEOUT)
        return {"brief_id": 42, "name": kwargs["brief"]["name"]}

    monkeypatch.setattr(interviewer_route, "save_approved_brief", save)
    return entered, release


def _count_token_reads(env, monkeypatch):
    reads = []

    def get_user_tokens(email):
        reads.append(email)
        return env.tokens

    monkeypatch.setattr(interviewer_route, "get_user_tokens", get_user_tokens)
    return reads


def _with_draft(env):
    env.engine._results = [DRAFT]
    _turn(env, {"answer": "a short report"})
    env.spent.clear()


# --- busy session ----------------------------------------------------------


def test_concurrent_answer_on_the_same_session_gets_409_and_does_no_work(env, monkeypatch):
    env.engine = _BlockingEngine([QUESTION, QUESTION])
    _start(env)
    env.spent.clear()
    reads = _count_token_reads(env, monkeypatch)

    thread_a, results = _in_thread(lambda: _turn(env, {"answer": "first"}))
    assert env.engine.entered.wait(TIMEOUT)

    response_b = _turn(env, {"answer": "second"})

    assert response_b.status_code == 409
    assert response_b.get_json() == SESSION_BUSY
    assert env.engine.calls == []
    assert len(reads) == 1
    assert env.spent == []

    env.engine.release.set()
    thread_a.join(TIMEOUT)
    assert results[0].status_code == 200
    assert env.engine.calls == [("chat", "first")]
    assert env.spent == [-2]
    assert agent_manager._busyEngines == {}

    # A request after A finished runs normally.
    assert _turn(env, {"answer": "third"}).status_code == 200


@pytest.mark.parametrize("action", ["revise", "decline"])
def test_other_actions_on_a_busy_session_get_409(env, action):
    env.engine = _BlockingEngine([QUESTION])
    _start(env)

    thread_a, results = _in_thread(lambda: _turn(env, {"answer": "first"}))
    assert env.engine.entered.wait(TIMEOUT)

    response_b = _turn(env, {"action": action})

    assert response_b.status_code == 409
    assert response_b.get_json() == SESSION_BUSY

    env.engine.release.set()
    thread_a.join(TIMEOUT)
    assert results[0].status_code == 200


def test_concurrent_approvals_store_the_brief_once(started, monkeypatch):
    _with_draft(started)
    entered, release = _blocking_save(started, monkeypatch)

    thread_a, results = _in_thread(lambda: _turn(started, {"action": "approve"}))
    assert entered.wait(TIMEOUT)

    response_b = _turn(started, {"action": "approve"})

    assert response_b.status_code == 409
    assert response_b.get_json() == SESSION_BUSY
    assert len(started.saved) == 1

    release.set()
    thread_a.join(TIMEOUT)
    assert results[0].status_code == 200
    assert results[0].get_json()["brief_id"] == 42
    assert len(started.saved) == 1
    assert started.engine.calls.count(("approve",)) == 1
    assert started.engine.closed
    assert agent_manager.getInterviewer("key-123") is None
    assert agent_manager._busyEngines == {}


def test_a_busy_datachat_session_does_not_block_the_interviewer(started):
    datachat_engine = object()
    agent_manager.activeEngines[(agent_manager.DATACHAT, "key-123")] = datachat_engine
    assert agent_manager.try_acquire_run(datachat_engine)
    started.engine._results = [QUESTION]

    assert _turn(started, {"answer": "revenue"}).status_code == 200

    assert agent_manager.try_acquire_run(datachat_engine) is False
    agent_manager.release_run(datachat_engine)


# --- guard release -----------------------------------------------------------


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    "setup, body, status",
    [
        (lambda env, mp: setattr(env, "tokens", 0), {"answer": "x"}, 500),
        (lambda env, mp: setattr(env, "tokens", None), {"answer": "x"}, 500),
        (lambda env, mp: mp.setattr(env.engine, "chat", _raise), {"answer": "x"}, 500),
        (lambda env, mp: None, {"action": "approve"}, 400),  # no pending draft
    ],
    ids=["not-enough-tokens", "unknown-balance", "engine-raises", "no-pending-draft"],
)
def test_the_guard_is_released_on_failures(started, monkeypatch, setup, body, status):
    setup(started, monkeypatch)

    assert _turn(started, body).status_code == status

    assert agent_manager._busyEngines == {}
    assert agent_manager.try_acquire_run(started.engine)


def test_the_guard_is_released_when_the_save_fails(started, monkeypatch):
    _with_draft(started)
    monkeypatch.setattr(interviewer_route, "save_approved_brief", _raise)

    assert _turn(started, {"action": "approve"}).status_code == 500

    assert agent_manager._busyEngines == {}
    assert agent_manager.getInterviewer("key-123") is started.engine


# --- stale engine versus replacement session ---------------------------------


def test_a_stale_decline_cannot_delete_the_replacement_session(started, monkeypatch):
    _with_draft(started)
    old = started.engine
    looked_up, resume = threading.Event(), threading.Event()
    real_get = interviewer_route.getInterviewer
    first_lookup = [True]

    def paused_get(api_key):
        engine = real_get(api_key)
        if first_lookup[0]:
            first_lookup[0] = False
            looked_up.set()
            assert resume.wait(TIMEOUT)
        return engine

    monkeypatch.setattr(interviewer_route, "getInterviewer", paused_get)

    # B holds the old engine, then A declines it and a new session starts.
    thread_b, results = _in_thread(lambda: _turn(started, {"action": "decline"}))
    assert looked_up.wait(TIMEOUT)
    assert _turn(started, {"action": "decline"}).status_code == 200
    new = started.engine = FakeEngine()
    _start(started)
    assert agent_manager.getInterviewer("key-123") is new

    resume.set()
    thread_b.join(TIMEOUT)

    assert results[0].status_code == 400
    assert results[0].get_json() == {"error": "Interviewer not active for this Api Key"}
    assert old.closed
    assert agent_manager.getInterviewer("key-123") is new
    assert not new.closed
    assert agent_manager._busyEngines == {}


def test_end_and_start_during_an_approval_keep_the_new_session(started, monkeypatch):
    _with_draft(started)
    old = started.engine
    entered, release = _blocking_save(started, monkeypatch)

    thread, results = _in_thread(lambda: _turn(started, {"action": "approve"}))
    assert entered.wait(TIMEOUT)

    client = started.app.test_client()
    ended = client.post("/end-analysis-interview", headers=HEADERS)
    assert ended.status_code == 200
    assert "Agent deleted succesfully" in ended.get_json()
    new = started.engine = FakeEngine()
    _start(started)

    release.set()
    thread.join(TIMEOUT)

    # Accepted current behaviour: the approval already under way completes.
    assert results[0].status_code == 200
    assert results[0].get_json()["brief_id"] == 42
    assert len(started.saved) == 1
    assert old.closed
    assert agent_manager.getInterviewer("key-123") is new
    assert not new.closed
    assert agent_manager._busyEngines == {}


class _SlowReadinessEngine(FakeEngine):
    """Not ready, and pauses when asked so a second start can register meanwhile."""

    def __init__(self):
        super().__init__(ready=False)
        self.checking = threading.Event()
        self.answer = threading.Event()

    @property
    def is_ready(self):
        self.checking.set()
        assert self.answer.wait(TIMEOUT)
        return False

    @is_ready.setter
    def is_ready(self, value):
        pass


def test_a_failed_start_does_not_delete_a_concurrent_successful_one(env, monkeypatch):
    failing, ready = _SlowReadinessEngine(), FakeEngine()
    gates = {id(failing): threading.Event(), id(ready): threading.Event()}
    built = threading.Semaphore(0)
    queue = [failing, ready]
    queue_lock = threading.Lock()

    def factory(**kwargs):
        with queue_lock:
            engine = queue.pop(0)
        built.release()
        assert gates[id(engine)].wait(TIMEOUT)
        return engine

    monkeypatch.setattr(interviewer_route, "create_interviewer_engine", factory)

    # Both starts see an empty slot before either registers.
    thread_failing, failing_results = _in_thread(lambda: _start(env))
    assert built.acquire(timeout=TIMEOUT)
    thread_ready, ready_results = _in_thread(lambda: _start(env))
    assert built.acquire(timeout=TIMEOUT)

    # The failing engine registers first, then the ready one replaces it.
    gates[id(failing)].set()
    assert failing.checking.wait(TIMEOUT)
    gates[id(ready)].set()
    thread_ready.join(TIMEOUT)
    assert ready_results[0].status_code == 200

    failing.answer.set()
    thread_failing.join(TIMEOUT)

    assert failing_results[0].status_code == 500
    assert failing_results[0].get_json()["code"] == "MISSING_CONFIG"
    assert agent_manager.getInterviewer("key-123") is ready
    assert not ready.closed
