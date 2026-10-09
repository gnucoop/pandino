"""
Tests that the agent registry keeps one session per agent kind, so a user can
run DataChat and the analysis interviewer at the same time under one Api Key.
"""

import threading
from unittest.mock import MagicMock

import pytest

import infrastructure.agent_manager as agent_manager

TIMEOUT = 5


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    monkeypatch.setattr(agent_manager, "activeEngines", {})


@pytest.fixture
def datachat_engine(monkeypatch):
    engine = MagicMock(name="datachat")
    monkeypatch.setattr(agent_manager, "create_engine", lambda **kwargs: engine)
    return engine


def _create_datachat(api_key="key"):
    return agent_manager.createAgent(api_key, None, None, "user", engine_type="smolagents")


def test_datachat_and_interviewer_sessions_do_not_collide(datachat_engine):
    interviewer = MagicMock(name="interviewer")

    _create_datachat()
    agent_manager.createInterviewer("key", lambda: interviewer)

    assert agent_manager.getAgent("key") is datachat_engine
    assert agent_manager.getInterviewer("key") is interviewer


def test_create_returns_the_existing_session_without_building_a_new_one():
    first = MagicMock(name="first")
    factory = MagicMock(return_value=first)

    agent_manager.createInterviewer("key", factory)
    again = agent_manager.createInterviewer("key", factory)

    assert again is first
    factory.assert_called_once()


def test_deleting_the_interviewer_leaves_datachat_running(datachat_engine):
    interviewer = MagicMock(name="interviewer")
    _create_datachat()
    agent_manager.createInterviewer("key", lambda: interviewer)

    deleted = agent_manager.deleteInterviewer("key", "user")

    assert deleted is interviewer
    interviewer.close.assert_called_once()
    assert agent_manager.getInterviewer("key") is None
    assert agent_manager.getAgent("key") is datachat_engine


def test_delete_without_a_user_name_keeps_the_session():
    interviewer = MagicMock(name="interviewer")
    agent_manager.createInterviewer("key", lambda: interviewer)

    assert agent_manager.deleteInterviewer("key", None) is None
    assert agent_manager.getInterviewer("key") is interviewer


def test_a_missing_api_key_finds_nothing():
    assert agent_manager.getAgent(None) is None
    assert agent_manager.getInterviewer("") is None


def test_list_keeps_datachat_keys_as_they_were(datachat_engine):
    _create_datachat("k1")
    agent_manager.createInterviewer("k1", MagicMock)

    assert agent_manager.listAgents() == {
        "k1": "engine_active",
        "interviewer:k1": "engine_active",
    }


# --- conditional interviewer delete ----------------------------------------


def test_conditional_delete_removes_and_closes_the_same_engine():
    interviewer = MagicMock(name="interviewer")
    agent_manager.createInterviewer("key", lambda: interviewer)

    assert agent_manager.deleteInterviewerIfCurrent("key", interviewer) is True
    interviewer.close.assert_called_once()
    assert agent_manager.getInterviewer("key") is None


def test_conditional_delete_leaves_a_different_engine_alone():
    old, new = MagicMock(name="old"), MagicMock(name="new")
    agent_manager.createInterviewer("key", lambda: new)

    assert agent_manager.deleteInterviewerIfCurrent("key", old) is False
    assert agent_manager.getInterviewer("key") is new
    new.close.assert_not_called()
    old.close.assert_not_called()


def test_conditional_delete_without_a_session_is_a_no_op():
    engine = MagicMock(name="engine")

    assert agent_manager.deleteInterviewerIfCurrent("key", engine) is False
    assert agent_manager.deleteInterviewerIfCurrent(None, engine) is False
    engine.close.assert_not_called()


def test_conditional_delete_leaves_datachat_running(datachat_engine):
    _create_datachat()

    assert agent_manager.deleteInterviewerIfCurrent("key", datachat_engine) is False
    assert agent_manager.getAgent("key") is datachat_engine
    datachat_engine.close.assert_not_called()


def _create_in_thread(engine, factory_entered, factory_release):
    def factory():
        factory_entered.set()
        assert factory_release.wait(TIMEOUT)
        return engine

    thread = threading.Thread(target=agent_manager.createInterviewer, args=("key", factory))
    thread.start()
    return thread


def test_a_late_registration_after_the_delete_is_kept():
    """A create that saw an empty slot registers after the old engine was removed."""
    old, new = MagicMock(name="old"), MagicMock(name="new")
    entered, release = threading.Event(), threading.Event()
    thread = _create_in_thread(new, entered, release)
    assert entered.wait(TIMEOUT)

    agent_manager.createInterviewer("key", lambda: old)
    assert agent_manager.deleteInterviewerIfCurrent("key", old) is True

    release.set()
    thread.join(TIMEOUT)
    assert agent_manager.getInterviewer("key") is new
    new.close.assert_not_called()


def test_a_registration_that_replaced_the_engine_is_never_deleted():
    """A create that saw an empty slot overwrites the old engine before the delete."""
    old, new = MagicMock(name="old"), MagicMock(name="new")
    entered, release = threading.Event(), threading.Event()
    thread = _create_in_thread(new, entered, release)
    assert entered.wait(TIMEOUT)

    agent_manager.createInterviewer("key", lambda: old)
    release.set()
    thread.join(TIMEOUT)

    assert agent_manager.deleteInterviewerIfCurrent("key", old) is False
    assert agent_manager.getInterviewer("key") is new
    new.close.assert_not_called()


class _ReentrantCloseEngine:
    """close() uses the registry, which would deadlock if it ran under its lock."""

    def __init__(self):
        self.lock_held_on_close = None

    def close(self):
        self.lock_held_on_close = agent_manager._registryLock.locked()
        agent_manager.createInterviewer("other", MagicMock)
        agent_manager.getAgent("other")


@pytest.mark.parametrize(
    "delete",
    [
        lambda engine: agent_manager.deleteInterviewerIfCurrent("key", engine),
        lambda engine: agent_manager.deleteInterviewer("key", "user"),
    ],
    ids=["conditional", "unconditional"],
)
def test_close_runs_outside_the_registry_lock(delete):
    engine = _ReentrantCloseEngine()
    agent_manager.createInterviewer("key", lambda: engine)

    thread = threading.Thread(target=delete, args=(engine,))
    thread.start()
    thread.join(TIMEOUT)

    assert not thread.is_alive()
    assert engine.lock_held_on_close is False
    assert agent_manager.getInterviewer("key") is None


def test_deleting_datachat_closes_and_returns_the_engine(datachat_engine):
    _create_datachat()

    assert agent_manager.deleteAgent("key", "user") is datachat_engine
    datachat_engine.close.assert_called_once()
    assert agent_manager.getAgent("key") is None
    assert agent_manager.deleteAgent("key", "user") is None
