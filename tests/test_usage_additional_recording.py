"""Contract tests for ``usage.recording.record_additional_token_consumption``.

The additional writer shares the primary token writer's persistence path;
these tests pin what differs (registration only, never the primary slot)
and that everything else - pricing writer, derived request_id/source,
fail-open - is the same.
"""

import inspect
import logging

import pytest
from flask import Flask

import usage.recording as usage_recording
from usage.recording import (
    record_additional_token_consumption,
    record_token_consumption,
)
from usage.request_state import get_usage_log_id, get_usage_log_ids
from utils.logging_config import register_request_context_hooks


@pytest.fixture
def writes(monkeypatch):
    """Stub the priced writer with sequential row ids and a stub user row."""
    calls = []

    def fake_log_token_usage(**kwargs):
        calls.append(kwargs)
        return 100 + len(calls)

    monkeypatch.setattr(usage_recording, "log_token_usage", fake_log_token_usage)
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
        "model": "tool-model",
        "service": "/datachat",
        "token_input": 7,
        "token_output": 2,
    }
    kwargs.update(overrides)
    return kwargs


def _in_request(fn):
    app = Flask(__name__)
    register_request_context_hooks(app)
    result = {}

    @app.route("/probe")
    def probe():
        from utils.logging_config import get_request_id

        result["request_id"] = get_request_id()
        result["value"] = fn()
        result["log_id"] = get_usage_log_id()
        result["log_ids"] = get_usage_log_ids()
        return "ok"

    app.test_client().get("/probe")
    return result


def test_signature_matches_primary_writer():
    assert inspect.signature(record_additional_token_consumption) == inspect.signature(
        record_token_consumption
    )


def test_request_id_and_source_cannot_be_supplied():
    with pytest.raises(TypeError):
        record_additional_token_consumption(**_valid(), request_id="x")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        record_additional_token_consumption(**_valid(), source="x")  # type: ignore[call-arg]


def test_writes_one_priced_row_with_exact_facts_and_derived_context(writes):
    result = _in_request(lambda: record_additional_token_consumption(**_valid()))

    assert result["value"] is True
    assert writes == [
        {
            "user_id": 42,
            "token_input": 7,
            "token_output": 2,
            "model": "tool-model",
            "provider": "test-provider",
            "service": "/datachat",
            "request_id": result["request_id"],
            "source": "dino",
        }
    ]


def test_registers_id_without_touching_the_primary_slot(writes):
    result = _in_request(lambda: record_additional_token_consumption(**_valid()))

    assert result["log_id"] is None
    assert result["log_ids"] == (101,)


@pytest.mark.parametrize("secondary_first", [True, False])
def test_primary_id_survives_either_write_order(writes, secondary_first):
    def run():
        if secondary_first:
            record_additional_token_consumption(**_valid())
            record_token_consumption(**_valid(model="agent-model"))
        else:
            record_token_consumption(**_valid(model="agent-model"))
            record_additional_token_consumption(**_valid())

    result = _in_request(run)

    primary = 102 if secondary_first else 101
    secondary = 101 if secondary_first else 102
    assert result["log_id"] == primary
    assert set(result["log_ids"]) == {primary, secondary}


def test_primary_writer_still_sets_the_primary_slot(writes):
    result = _in_request(lambda: record_token_consumption(**_valid()))

    assert result["value"] is True
    assert result["log_id"] == 101
    assert result["log_ids"] == (101,)


@pytest.mark.parametrize("failure", ["db_or_pricing", "user_lookup"])
def test_runtime_failure_is_fail_open_and_registers_nothing(
    writes, monkeypatch, caplog, failure
):
    def boom(**kwargs):
        raise RuntimeError("secret prompt content")

    if failure == "user_lookup":
        monkeypatch.setattr(usage_recording, "get_user_by_id", lambda user_id: None)
    else:
        monkeypatch.setattr(usage_recording, "log_token_usage", boom)

    with caplog.at_level(logging.WARNING, logger="usage.recording"):
        result = _in_request(lambda: record_additional_token_consumption(**_valid()))

    assert result["value"] is False
    assert result["log_id"] is None
    assert result["log_ids"] == ()
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert messages[0].startswith("event=usage_token_recording_failed")
    assert "secret prompt content" not in messages[0]


def test_failure_does_not_disturb_an_existing_primary_id(writes, monkeypatch):
    def run():
        record_token_consumption(**_valid())
        monkeypatch.setattr(
            usage_recording,
            "log_token_usage",
            lambda **k: (_ for _ in ()).throw(RuntimeError("down")),
        )
        return record_additional_token_consumption(**_valid())

    result = _in_request(run)

    assert result["value"] is False
    assert result["log_id"] == 101
    assert result["log_ids"] == (101,)


def test_outside_request_context_degrades_to_false(writes, caplog):
    with caplog.at_level(logging.WARNING, logger="usage.recording"):
        assert record_additional_token_consumption(**_valid()) is False
    assert len(caplog.records) == 1


@pytest.mark.parametrize(
    "overrides",
    [{"user_id": True}, {"provider": " "}, {"token_input": -1}, {"token_output": 1.5}],
)
def test_programmer_misuse_raises_like_the_primary_writer(writes, overrides):
    with pytest.raises(ValueError):
        record_additional_token_consumption(**_valid(**overrides))
    assert writes == []
