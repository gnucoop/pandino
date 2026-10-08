"""
Tests that the agent registry keeps one session per agent kind, so a user can
run DataChat and the analysis interviewer at the same time under one Api Key.
"""

from unittest.mock import MagicMock

import pytest

import infrastructure.agent_manager as agent_manager


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
