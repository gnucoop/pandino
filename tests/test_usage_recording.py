"""Contract tests for the explicit Usage recording boundary.

Covers the public promises of
``utils.usage_recording.record_token_consumption``: what an adopter must
supply, what the boundary derives, what it hides, and how it behaves when
the world underneath it fails.
"""

import inspect
import logging

import pytest
from flask import Flask

from utils import usage_recording
from utils.usage_recording import record_token_consumption
from utils.usage_attribution_state import bind_usage_attribution
from utils.usage_request_state import get_usage_log_id, get_usage_log_ids


@pytest.fixture
def app():
    return Flask(__name__)


@pytest.fixture
def recorded(monkeypatch):
    """Capture writer calls and row-id hand-offs, with a stub user row."""
    calls = {"writes": [], "handoffs": []}

    def fake_log_token_usage(**kwargs):
        calls["writes"].append(kwargs)
        return 4343

    monkeypatch.setattr(usage_recording, "log_token_usage", fake_log_token_usage)
    monkeypatch.setattr(
        usage_recording,
        "set_usage_log_id",
        lambda log_id: calls["handoffs"].append(log_id),
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_id",
        lambda user_id: {"id": user_id, "username": "user@example.com"},
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username, "client": "dino"},
    )
    return calls


def _valid(**overrides):
    kwargs = {
        "user_id": 42,
        "provider": "test-provider",
        "model": "test-model",
        "service": "/prompt.txt",
        "token_input": 5,
        "token_output": 3,
    }
    kwargs.update(overrides)
    return kwargs


# --- public contract shape -------------------------------------------------


def test_operation_is_keyword_only():
    signature = inspect.signature(record_token_consumption)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_operation_exposes_no_request_id_or_source_parameter():
    """Absent from the signature, so supplying either is a TypeError rather
    than merely discouraged."""
    parameters = set(inspect.signature(record_token_consumption).parameters)

    assert parameters == {
        "user_id",
        "provider",
        "model",
        "service",
        "token_input",
        "token_output",
    }


def test_supplying_request_id_is_a_type_error(app, recorded):
    with app.test_request_context("/prompt.txt"):
        with pytest.raises(TypeError):
            record_token_consumption(**_valid(), request_id="deadbeef")


def test_supplying_source_is_a_type_error(app, recorded):
    with app.test_request_context("/prompt.txt"):
        with pytest.raises(TypeError):
            record_token_consumption(**_valid(), source="dino")


# --- happy path ------------------------------------------------------------


def test_persists_the_supplied_consumption_facts(app, recorded):
    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is True
    assert len(recorded["writes"]) == 1
    write = recorded["writes"][0]
    assert write["user_id"] == 42
    assert write["provider"] == "test-provider"
    assert write["model"] == "test-model"
    assert write["service"] == "/prompt.txt"
    assert write["token_input"] == 5
    assert write["token_output"] == 3


def test_returns_true_and_never_a_row_id(app, recorded):
    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is True
    assert isinstance(result, bool)


def test_request_id_is_derived_internally(app, recorded, monkeypatch):
    monkeypatch.setattr(usage_recording, "get_request_id", lambda: "cafebabe")

    with app.test_request_context("/prompt.txt"):
        record_token_consumption(**_valid())

    assert recorded["writes"][0]["request_id"] == "cafebabe"


def test_source_is_derived_from_the_persisted_user_row(app, recorded):
    with app.test_request_context("/prompt.txt"):
        record_token_consumption(**_valid())

    assert recorded["writes"][0]["source"] == "dino"


def test_source_none_is_a_legitimate_derived_value(app, recorded, monkeypatch):
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username, "client": None},
    )

    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is True
    assert recorded["writes"][0]["source"] is None


def test_bound_attribution_supplies_source_without_a_lookup(app, recorded, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("bound attribution must not trigger a user lookup")

    monkeypatch.setattr(usage_recording, "get_user_by_id", fail)
    monkeypatch.setattr(usage_recording, "get_user_by_username", fail)

    with app.test_request_context("/prompt.txt"):
        bind_usage_attribution(42, "/prompt.txt", "coopi")
        result = record_token_consumption(**_valid())

    assert result is True
    assert recorded["writes"][0]["source"] == "coopi"


def test_attribution_for_a_different_user_falls_back_to_the_row(app, recorded):
    with app.test_request_context("/prompt.txt"):
        bind_usage_attribution(999, "/other", "coopi")
        record_token_consumption(**_valid(user_id=42))

    assert recorded["writes"][0]["source"] == "dino"


# --- row-id ownership ------------------------------------------------------


def test_row_id_is_registered_internally(app, recorded):
    with app.test_request_context("/prompt.txt"):
        record_token_consumption(**_valid())

    assert recorded["handoffs"] == [4343]


def test_registration_reaches_both_duration_list_and_latest_slot(app, monkeypatch):
    """Uses the real request-state primitive: the row must be visible to the
    duration finaliser *and* to the latest-id compatibility reader."""
    monkeypatch.setattr(usage_recording, "log_token_usage", lambda **kwargs: 777)
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_id",
        lambda user_id: {"id": user_id, "username": "user@example.com"},
    )
    monkeypatch.setattr(
        usage_recording,
        "get_user_by_username",
        lambda username: {"id": 42, "username": username, "client": None},
    )

    with app.test_request_context("/prompt.txt"):
        record_token_consumption(**_valid())

        assert get_usage_log_ids() == (777,)
        assert get_usage_log_id() == 777


# --- zero-token acceptance -------------------------------------------------


def test_zero_zero_is_accepted_and_recorded(app, recorded):
    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid(token_input=0, token_output=0))

    assert result is True
    assert recorded["writes"][0]["token_input"] == 0
    assert recorded["writes"][0]["token_output"] == 0


# --- programmer-contract misuse -------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"token_input": -1},
        {"token_output": -1},
        {"token_input": 1.5},
        {"token_output": "3"},
        {"token_input": True},
        {"user_id": "42"},
        {"user_id": None},
        {"user_id": True},
        {"provider": ""},
        {"provider": "   "},
        {"provider": None},
        {"model": ""},
        {"model": 7},
        {"service": ""},
        {"service": "  "},
        {"service": None},
    ],
)
def test_invalid_public_fields_raise(app, recorded, overrides):
    with app.test_request_context("/prompt.txt"):
        with pytest.raises(ValueError):
            record_token_consumption(**_valid(**overrides))

    assert recorded["writes"] == []
    assert recorded["handoffs"] == []


def test_missing_required_field_is_a_type_error(app, recorded):
    kwargs = _valid()
    del kwargs["service"]

    with app.test_request_context("/prompt.txt"):
        with pytest.raises(TypeError):
            record_token_consumption(**kwargs)


# --- runtime fail-open -----------------------------------------------------


def _boom(*args, **kwargs):
    raise RuntimeError("database is unreachable")


def test_write_failure_returns_false_without_raising(app, recorded, monkeypatch):
    monkeypatch.setattr(usage_recording, "log_token_usage", _boom)

    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is False
    assert recorded["handoffs"] == []


def test_missing_pricing_returns_false(app, recorded, monkeypatch):
    def no_pricing(**kwargs):
        raise ValueError("Cost not found for provider: p and model: m")

    monkeypatch.setattr(usage_recording, "log_token_usage", no_pricing)

    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is False


def test_source_lookup_failure_returns_false_and_writes_nothing(
    app, recorded, monkeypatch
):
    """A failed lookup is not an absent client: no row is written rather
    than one claiming source=None."""
    monkeypatch.setattr(usage_recording, "get_user_by_id", _boom)

    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is False
    assert recorded["writes"] == []


def test_unknown_user_returns_false(app, recorded, monkeypatch):
    monkeypatch.setattr(usage_recording, "get_user_by_id", lambda user_id: None)

    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is False
    assert recorded["writes"] == []


def test_outside_a_request_context_degrades_rather_than_raising(recorded):
    result = record_token_consumption(**_valid())

    assert result is False


def test_registration_failure_returns_false_without_raising(app, recorded, monkeypatch):
    monkeypatch.setattr(usage_recording, "set_usage_log_id", _boom)

    with app.test_request_context("/prompt.txt"):
        result = record_token_consumption(**_valid())

    assert result is False


def test_failure_is_not_retried(app, recorded, monkeypatch):
    attempts = []

    def counting_failure(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("commit failed")

    monkeypatch.setattr(usage_recording, "log_token_usage", counting_failure)

    with app.test_request_context("/prompt.txt"):
        record_token_consumption(**_valid())

    assert len(attempts) == 1


def test_failure_emits_one_safe_diagnostic(app, recorded, monkeypatch, caplog):
    monkeypatch.setattr(usage_recording, "log_token_usage", _boom)

    with caplog.at_level(logging.WARNING, logger=usage_recording.__name__):
        with app.test_request_context("/prompt.txt"):
            record_token_consumption(**_valid())

    records = [
        record
        for record in caplog.records
        if "usage_token_recording_failed" in record.getMessage()
    ]
    assert len(records) == 1

    message = records[0].getMessage()
    assert "service=/prompt.txt" in message
    assert "provider=test-provider" in message
    assert "error_type=RuntimeError" in message
    # The exception text may carry request-adjacent detail; only its type
    # is reported.
    assert "database is unreachable" not in message
