"""Tool-side provider consumption on POST /datachat.

Pins the request-local collector (``datachat.provider_contributions``) and
the route's flush of it into additional Usage rows: each contribution is its
own row, correlated with the request, finalized with the request duration,
and never the response-facing ``log_id``.

The route runs against the real ``usage.recording`` boundary; only the
database writers and user lookups are stubbed.
"""

import dataclasses
import io
import logging
from types import SimpleNamespace

import pytest
from flask import Flask

import usage.duration_finalization as duration_finalization
import usage.recording as usage_recording
from datachat.provider_contributions import (
    ProviderContribution,
    get_provider_contributions,
    record_provider_contribution,
)
from routes import datachat as datachat_route
from usage.duration_finalization import register_usage_duration_finalization_hooks
from usage.request_state import get_usage_log_id, get_usage_log_ids
from utils.logging_config import (
    CONTEXT_UNSET,
    _request_id_var,
    register_request_context_hooks,
)

RUNTIME_LOGGER_NAME = "datachat.runtime"
AGENT_RUNS_LOGGER_NAME = "agent_runs"


def _contribution(**overrides):
    kwargs = {
        "provider": "tool-provider",
        "model": "tool-model",
        "token_input": 11,
        "token_output": 4,
    }
    kwargs.update(overrides)
    return kwargs


# --- collector -------------------------------------------------------------


def test_records_contributions_in_production_order_without_dedup():
    app = Flask(__name__)
    with app.test_request_context():
        record_provider_contribution(**_contribution())
        record_provider_contribution(**_contribution(model="other"))
        record_provider_contribution(**_contribution())

        assert get_provider_contributions() == [
            ProviderContribution(**_contribution()),
            ProviderContribution(**_contribution(model="other")),
            ProviderContribution(**_contribution()),
        ]


def test_contributions_are_request_isolated():
    app = Flask(__name__)
    with app.test_request_context():
        record_provider_contribution(**_contribution())
    with app.test_request_context():
        assert get_provider_contributions() == []


def test_outside_app_context_record_is_noop_and_lookup_empty():
    record_provider_contribution(**_contribution())
    assert get_provider_contributions() == []


def test_lookup_returns_a_copy():
    app = Flask(__name__)
    with app.test_request_context():
        record_provider_contribution(**_contribution())
        get_provider_contributions().clear()
        assert len(get_provider_contributions()) == 1


def test_contribution_carries_consumption_facts_only():
    assert [f.name for f in dataclasses.fields(ProviderContribution)] == [
        "provider",
        "model",
        "token_input",
        "token_output",
    ]


# --- route integration -----------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_loggers():
    saved = {}
    for name in (RUNTIME_LOGGER_NAME, AGENT_RUNS_LOGGER_NAME):
        logger = logging.getLogger(name)
        saved[name] = (list(logger.handlers), logger.level, logger.propagate)
        handler = logging.StreamHandler(io.StringIO())
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
    _request_id_var.set(CONTEXT_UNSET)
    try:
        yield
    finally:
        _request_id_var.set(CONTEXT_UNSET)
        for name, (handlers, level, propagate) in saved.items():
            logger = logging.getLogger(name)
            logger.handlers = handlers
            logger.level = level
            logger.propagate = propagate


class _Engine:
    """Records tool contributions during chat(), like a provider-calling tool."""

    def __init__(self, contributions, trace=True, response=None):
        self._contributions = contributions
        self._trace = trace
        self._response = response or {"kind": "text", "text": "ok"}

    def chat(self, message):
        for contribution in self._contributions:
            record_provider_contribution(**contribution)
        return self._response

    def get_last_trace(self):
        return {"run_result": object()} if self._trace else None


@pytest.fixture
def harness(monkeypatch):
    state = {
        "writes": [],
        "debits": [],
        "durations": [],
        "fail_models": set(),
        "observed": {},
    }

    def fake_log_token_usage(**kwargs):
        if kwargs["model"] in state["fail_models"]:
            raise RuntimeError("db down")
        state["writes"].append(kwargs)
        return 500 + len(state["writes"])

    user = {"id": 123, "username": "user@example.com", "client": "dino"}
    monkeypatch.setattr(usage_recording, "log_token_usage", fake_log_token_usage)
    monkeypatch.setattr(usage_recording, "get_user_by_id", lambda _id: user)
    monkeypatch.setattr(usage_recording, "get_user_by_username", lambda _u: user)
    monkeypatch.setattr(datachat_route, "get_user_by_username", lambda _u: user)
    monkeypatch.setattr(datachat_route, "assert_valid_api_key", lambda *a, **k: None)
    monkeypatch.setattr(datachat_route, "get_user_tokens", lambda _e: 10)
    monkeypatch.setattr(
        datachat_route,
        "edit_tokens",
        lambda email, delta: state["debits"].append((email, delta)),
    )
    # Token metrics for the primary CodeAgent row.
    monkeypatch.setattr(
        datachat_route,
        "serialize_runresult",
        lambda _r: {"metrics": {"token_usage": {"input": 30, "output": 9}}},
    )
    monkeypatch.setattr(datachat_route, "log_runresult", lambda *a, **k: None)
    monkeypatch.setattr(duration_finalization, "get_request_duration_ms", lambda: 777)
    monkeypatch.setattr(
        duration_finalization,
        "update_usage_duration",
        lambda log_id, ms: state["durations"].append((log_id, ms)) or True,
    )

    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        datachat_token_cost=3,
        models=SimpleNamespace(datachat_model="agent-model", datachat_provider="agent-provider"),
    )
    app.config["DATACHAT_RUNTIME_LOGGER"] = logging.getLogger(RUNTIME_LOGGER_NAME)
    register_request_context_hooks(app)
    register_usage_duration_finalization_hooks(app)
    app.register_blueprint(datachat_route.datachat_bp)

    @app.after_request
    def observe(response):
        state["observed"] = {"log_id": get_usage_log_id(), "log_ids": get_usage_log_ids()}
        return response

    def post(engine):
        monkeypatch.setattr(datachat_route, "getAgent", lambda _k: engine)
        return app.test_client().post(
            "/datachat",
            json={"chat": "hello"},
            headers={"X-API-KEY": "k", "X-USER-EMAIL": "user@example.com"},
        )

    state["post"] = post
    return state


def test_one_contribution_adds_one_correlated_secondary_row(harness):
    response = harness["post"](_Engine([_contribution()]))

    assert response.status_code == 200
    writes = harness["writes"]
    assert len(writes) == 2
    secondary, primary = writes  # flushed right after chat(), before the agent row
    assert secondary["model"] == "tool-model"
    assert (secondary["token_input"], secondary["token_output"]) == (11, 4)
    assert primary["model"] == "agent-model"
    assert {w["service"] for w in writes} == {"/datachat"}
    assert {w["user_id"] for w in writes} == {123}
    assert {w["source"] for w in writes} == {"dino"}
    assert {w["request_id"] for w in writes} == {response.headers["X-Request-ID"]}

    assert response.get_json()["log_id"] == 502
    assert harness["observed"] == {"log_id": 502, "log_ids": (501, 502)}
    assert harness["debits"] == [("user@example.com", -3)]


def test_many_contributions_each_get_their_own_row(harness):
    engine = _Engine([_contribution(model=f"m{i}") for i in range(3)])
    response = harness["post"](engine)

    assert [w["model"] for w in harness["writes"]] == ["m0", "m1", "m2", "agent-model"]
    assert response.get_json()["log_id"] == 504
    assert harness["observed"]["log_ids"] == (501, 502, 503, 504)
    assert harness["debits"] == [("user@example.com", -3)]


def test_every_registered_row_gets_the_request_duration(harness):
    harness["post"](_Engine([_contribution(), _contribution()]))

    assert harness["durations"] == [(501, 777), (502, 777), (503, 777)]


def test_secondary_failure_is_fail_open_and_remaining_rows_are_attempted(harness):
    harness["fail_models"].add("broken")
    engine = _Engine([_contribution(model="broken"), _contribution(model="fine")])
    response = harness["post"](engine)

    assert response.status_code == 200
    assert [w["model"] for w in harness["writes"]] == ["fine", "agent-model"]
    assert response.get_json()["log_id"] == 502
    assert harness["observed"] == {"log_id": 502, "log_ids": (501, 502)}
    assert harness["debits"] == [("user@example.com", -3)]


def test_failed_run_still_records_tool_consumption_without_primary(harness):
    engine = _Engine(
        [_contribution()],
        trace=False,
        response={"kind": "error", "message": "failed", "code": "RUN_FAILED"},
    )
    response = harness["post"](engine)

    assert [w["model"] for w in harness["writes"]] == ["tool-model"]
    assert "log_id" not in response.get_json()
    assert harness["observed"] == {"log_id": None, "log_ids": (501,)}
    assert harness["durations"] == [(501, 777)]


def test_no_contributions_keeps_single_primary_row(harness):
    response = harness["post"](_Engine([]))

    assert [w["model"] for w in harness["writes"]] == ["agent-model"]
    assert response.get_json()["log_id"] == 501
    assert harness["observed"] == {"log_id": 501, "log_ids": (501,)}
