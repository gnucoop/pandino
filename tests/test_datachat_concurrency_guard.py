"""Same-session concurrency guard on POST /datachat.

One /datachat run per engine: a concurrent request on a busy engine gets 409
session_busy before any balance read, engine run, Usage recording or token
debit, and the guard is released on every path after acquisition. Engines
are told apart by identity, never by Api Key or dataclass equality.

Synchronisation uses threading.Event only; no sleeps.
"""

import threading
from dataclasses import dataclass

import pytest

from infrastructure import agent_manager
from routes import datachat as datachat_route
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    _make_app,
    _patch_success_dependencies,
    _post_chat,
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)

TIMEOUT = 5


@pytest.fixture(autouse=True)
def clean_busy_registry():
    agent_manager._busyEngines.clear()
    yield
    agent_manager._busyEngines.clear()


@dataclass
class _EqualEngine:
    """Dataclass like SmolagentsEngine: unhashable and equal by fields."""

    api_key: str


class _BlockingEngine:
    """chat() signals entry, then waits until the test lets it finish."""

    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.entered.set()
        assert self.release.wait(TIMEOUT)
        return {"kind": "text", "text": "stub response"}

    def get_last_trace(self):
        return {"run_result": object()}


class _Engine:
    def __init__(self):
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"kind": "text", "text": "stub response"}

    def get_last_trace(self):
        return {"run_result": object()}


def _counting(monkeypatch, name, result=None):
    calls = []

    def fake(*a, **k):
        calls.append((a, k))
        return result

    monkeypatch.setattr(datachat_route, name, fake)
    return calls


def _start_in_thread(app, results):
    def run():
        results.append(_post_chat(app.test_client()))

    thread = threading.Thread(target=run)
    thread.start()
    return thread


# --- agent_manager --------------------------------------------------------


def test_same_engine_second_acquire_fails_until_released():
    engine = _Engine()
    assert agent_manager.try_acquire_run(engine) is True
    assert agent_manager.try_acquire_run(engine) is False
    agent_manager.release_run(engine)
    assert agent_manager.try_acquire_run(engine) is True
    agent_manager.release_run(engine)
    assert agent_manager._busyEngines == {}


def test_different_engines_are_independent():
    a, b = _Engine(), _Engine()
    assert agent_manager.try_acquire_run(a) is True
    assert agent_manager.try_acquire_run(b) is True


def test_field_equal_dataclass_engines_are_independent():
    old, new = _EqualEngine("same-key"), _EqualEngine("same-key")
    assert old == new
    assert agent_manager.try_acquire_run(old) is True
    assert agent_manager.try_acquire_run(new) is True
    assert agent_manager.try_acquire_run(old) is False


def test_release_is_idempotent():
    engine = _Engine()
    agent_manager.release_run(engine)
    assert agent_manager.try_acquire_run(engine) is True
    agent_manager.release_run(engine)
    agent_manager.release_run(engine)
    assert agent_manager._busyEngines == {}


def test_replacement_engine_for_same_api_key_is_not_busy(monkeypatch):
    monkeypatch.setattr(agent_manager, "activeEngines", {})
    monkeypatch.setattr(
        agent_manager, "create_engine", lambda **kw: _EqualEngine(kw["api_key"])
    )
    old = agent_manager.createAgent("k", None, None, "u", engine_type="x")
    assert agent_manager.try_acquire_run(old) is True

    old.close = lambda: None
    agent_manager.deleteAgent("k", "u")
    new = agent_manager.createAgent("k", None, None, "u", engine_type="x")

    assert new is not old
    assert agent_manager.try_acquire_run(new) is True
    assert agent_manager.try_acquire_run(old) is False


# --- route ----------------------------------------------------------------


def test_concurrent_request_on_same_engine_gets_409_and_does_no_work(monkeypatch):
    app, stream, _ = _make_app()
    engine = _BlockingEngine()
    record_calls = []
    _patch_success_dependencies(monkeypatch, engine, record_calls=record_calls)
    token_reads = _counting(monkeypatch, "get_user_tokens", result=10)
    debits = _counting(monkeypatch, "edit_tokens")

    results = []
    thread_a = _start_in_thread(app, results)
    assert engine.entered.wait(TIMEOUT)

    response_b = _post_chat(app.test_client())

    assert response_b.status_code == 409
    assert response_b.get_json() == {
        "error": "session_busy",
        "message": "A request is already being processed for this session.",
    }
    assert len(engine.calls) == 1
    assert len(token_reads) == 1
    assert record_calls == []
    assert debits == []
    assert "http_status=409" in stream.getvalue()
    assert "error_code=SESSION_BUSY" in stream.getvalue()

    engine.release.set()
    thread_a.join(TIMEOUT)
    assert results[0].status_code == 200
    assert len(record_calls) == 1
    assert len(debits) == 1
    assert agent_manager._busyEngines == {}

    # Request C after A finished runs normally.
    engine.release.set()
    response_c = _post_chat(app.test_client())
    assert response_c.status_code == 200
    assert len(engine.calls) == 2
    assert len(debits) == 2


def test_different_engines_do_not_block_each_other(monkeypatch):
    app, _, _ = _make_app()
    busy, other = _BlockingEngine(), _Engine()
    _patch_success_dependencies(monkeypatch, busy)
    current = {"engine": busy}
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: current["engine"])

    results = []
    thread_a = _start_in_thread(app, results)
    assert busy.entered.wait(TIMEOUT)

    current["engine"] = other
    assert _post_chat(app.test_client()).status_code == 200
    assert len(other.calls) == 1

    busy.release.set()
    thread_a.join(TIMEOUT)
    assert results[0].status_code == 200


def test_replacement_engine_same_api_key_is_not_blocked(monkeypatch):
    """/enddatachat + /startdatachat while the old run is still in flight."""
    app, _, _ = _make_app()
    old = _BlockingEngine()
    _patch_success_dependencies(monkeypatch, old)
    active = {"test-key": old}
    monkeypatch.setattr(datachat_route, "getAgent", lambda api_key: active[api_key])

    results = []
    thread_a = _start_in_thread(app, results)
    assert old.entered.wait(TIMEOUT)

    replacement = _Engine()
    active["test-key"] = replacement
    assert _post_chat(app.test_client()).status_code == 200
    assert len(replacement.calls) == 1

    old.release.set()
    thread_a.join(TIMEOUT)
    assert results[0].status_code == 200


@pytest.mark.parametrize(
    "name",
    ["adapt_engine_output", "record_token_consumption", "edit_tokens"],
)
def test_guard_released_when_guarded_path_raises(monkeypatch, name):
    app, _, _ = _make_app()
    app.config["PROPAGATE_EXCEPTIONS"] = False
    engine = _Engine()
    _patch_success_dependencies(monkeypatch, engine)

    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(datachat_route, name, boom)
    assert _post_chat(app.test_client()).status_code == 500
    assert agent_manager._busyEngines == {}
    assert agent_manager.try_acquire_run(engine) is True


@pytest.mark.parametrize(
    "tokens, expected_error",
    [(None, "Could not retrieve user tokens"), (0, "Not enough tokens")],
)
def test_guard_released_on_token_early_returns(monkeypatch, tokens, expected_error):
    app, _, _ = _make_app()
    engine = _Engine()
    _patch_success_dependencies(monkeypatch, engine)
    monkeypatch.setattr(datachat_route, "get_user_tokens", lambda e: tokens)

    response = _post_chat(app.test_client())

    assert response.status_code == 500
    assert response.get_json()["error"] == expected_error
    assert engine.calls == []
    assert agent_manager._busyEngines == {}


def test_guard_released_on_normalization_failure(monkeypatch):
    app, _, _ = _make_app()
    engine = _Engine()
    _patch_success_dependencies(monkeypatch, engine)

    def fail(response):
        raise RuntimeError("bad output")

    monkeypatch.setattr(datachat_route, "normalize_datachat_response", fail)
    response = _post_chat(app.test_client())

    assert response.status_code == 500
    assert agent_manager._busyEngines == {}
