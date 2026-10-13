"""Validation of the optional /datachat memory fields (not yet wired to the route)."""

import copy
import inspect

import pytest

import routes.datachat as datachat_route
from routes.datachat import InvalidMemoryRequest, _parse_memory_request


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"chat": "q"}, (False, None)),
        ({"chat": "q", "memory": False}, (False, None)),
        ({"chat": "q", "memory": True}, (True, None)),
        ({"chat": "q", "memory": True, "conversation_id": "abc123"}, (True, "abc123")),
    ],
)
def test_valid_requests(payload, expected):
    before = copy.deepcopy(payload)
    assert _parse_memory_request(payload) == expected
    assert payload == before


def test_valid_identifier_is_returned_unchanged():
    conversation_id = " 0f3a-ID "
    _, parsed = _parse_memory_request({"memory": True, "conversation_id": conversation_id})
    assert parsed is conversation_id


@pytest.mark.parametrize(
    "payload",
    [
        {"memory": "true"},
        {"memory": "false"},
        {"memory": 1},
        {"memory": 0},
        {"memory": None},
        {"memory": []},
        {"memory": {}},
        {"conversation_id": "abc123"},
        {"memory": False, "conversation_id": "abc123"},
        {"memory": True, "conversation_id": None},
        {"memory": True, "conversation_id": 123},
        {"memory": True, "conversation_id": []},
        {"memory": True, "conversation_id": {}},
        {"memory": True, "conversation_id": ""},
        {"memory": True, "conversation_id": "   "},
        {"memory": True, "conversation_id": "\t\n"},
        {"memory": "true", "conversation_id": "abc123"},
    ],
)
def test_invalid_requests_share_one_error(payload):
    before = copy.deepcopy(payload)
    with pytest.raises(InvalidMemoryRequest) as exc:
        _parse_memory_request(payload)
    assert exc.value.code == "invalid_memory_request"
    assert payload == before


def test_route_handler_does_not_use_validator_yet():
    assert "_parse_memory_request" not in inspect.getsource(datachat_route.dataChat)


def test_validator_touches_no_engine_or_database(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("validator must stay pure")

    for name in ("getAgent", "try_acquire_run", "get_user_tokens", "edit_tokens", "get_user_by_username"):
        monkeypatch.setattr(datachat_route, name, forbidden)

    assert _parse_memory_request({"memory": True, "conversation_id": "x"}) == (True, "x")
