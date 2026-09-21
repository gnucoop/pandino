"""Tests for utils.request_duration.

Focused on the module's own concern only: request-duration timer
lifecycle. Does not exercise utils.logging_config beyond registering its
hooks alongside these, to prove the two hook systems coexist without
interference.
"""

import re
from itertools import count
from unittest.mock import patch

import pytest

from utils.request_duration import (
    _G_DURATION_ATTR,
    _G_START_ATTR,
    _HOOKS_MARKER,
    get_request_duration_ms,
    register_request_duration_hooks,
)


def _make_app(observed=None):
    """Throwaway Flask app carrying only the duration hooks and a probe route.

    ``teardown_request`` always runs after ``after_request`` regardless of
    registration order, so it is used here to observe the finalized value
    that ``after_request`` computed - reading ``get_request_duration_ms()``
    from inside the view itself would always see ``None``, since the view
    runs before finalization.
    """
    from flask import Flask, g

    app = Flask(__name__)
    register_request_duration_hooks(app)
    observed = {} if observed is None else observed

    @app.teardown_request
    def _observe(exc=None):
        observed["duration_ms"] = getattr(g, "_maui_request_duration_ms", None)

    @app.route("/ping")
    def ping():
        return "pong"

    app.observed = observed
    return app


def _make_app_with_logging_hooks(observed=None):
    """Registers both hook systems, in the order main.py uses."""
    from flask import Flask, g

    from utils.logging_config import register_request_context_hooks

    app = Flask(__name__)
    register_request_context_hooks(app)
    register_request_duration_hooks(app)
    observed = {} if observed is None else observed

    @app.teardown_request
    def _observe(exc=None):
        observed["duration_ms"] = getattr(g, "_maui_request_duration_ms", None)

    @app.route("/ping")
    def ping():
        return "pong"

    app.observed = observed
    return app


# --------------------------------------------------------------------------
# Public API surface
# --------------------------------------------------------------------------


def test_public_api_is_exactly_registration_and_getter():
    import utils.request_duration as module

    assert set(module.__all__) == {
        "register_request_duration_hooks",
        "get_request_duration_ms",
    }


def test_raw_g_attribute_names_are_not_part_of_public_api():
    import utils.request_duration as module

    assert "_G_START_ATTR" not in module.__all__
    assert "_G_DURATION_ATTR" not in module.__all__


# --------------------------------------------------------------------------
# Timer contract: perf_counter, rounding, finalization
# --------------------------------------------------------------------------


def _patched_perf_counter(values):
    it = iter(values)
    return lambda: next(it)


def test_duration_is_none_before_finalization():
    app = _make_app()

    with app.test_request_context("/ping"):
        assert get_request_duration_ms() is None


def test_after_request_computes_rounded_whole_milliseconds():
    app = _make_app()

    # 100.02439683 - 100.000 = 0.02439683s -> 24.39683ms -> rounds to 24.
    with patch(
        "utils.request_duration.time.perf_counter",
        _patched_perf_counter([100.000, 100.02439683]),
    ):
        response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert app.observed["duration_ms"] == 24


def test_rounding_uses_round_half_to_even_not_truncation():
    app = _make_app()

    # 0.0245s = 24.5ms; proves rounding, not floor/truncation toward 24.
    with patch(
        "utils.request_duration.time.perf_counter",
        _patched_perf_counter([100.000, 100.0245]),
    ):
        app.test_client().get("/ping")

    duration = app.observed["duration_ms"]
    assert duration in (24, 25)  # round() banker's rounding; not truncated to 24 via int()
    assert duration != int(24.5)  # sanity: not accidentally truncating


# --------------------------------------------------------------------------
# Response passthrough
# --------------------------------------------------------------------------


def test_response_status_and_body_are_unchanged():
    app = _make_app()

    response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"


def test_x_request_id_header_still_present_with_both_hook_systems():
    app = _make_app_with_logging_hooks()

    response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"
    assert re.fullmatch(r"[0-9a-f]{16}", response.headers["X-Request-ID"])
    assert isinstance(app.observed["duration_ms"], int)


# --------------------------------------------------------------------------
# Request isolation
# --------------------------------------------------------------------------


def test_sequential_requests_get_independent_durations():
    app = _make_app()

    counter = count(100)

    def fake_perf_counter():
        return next(counter)

    with patch("utils.request_duration.time.perf_counter", fake_perf_counter):
        app.test_client().get("/ping")
        first = app.observed["duration_ms"]
        app.test_client().get("/ping")
        second = app.observed["duration_ms"]

    # Each request consumes exactly one (start, stop) pair of consecutive
    # integers from the counter, so both durations are 1000ms, independently
    # computed rather than leaking or accumulating across requests.
    assert first == 1000
    assert second == 1000


# --------------------------------------------------------------------------
# Missing-start robustness
# --------------------------------------------------------------------------


def test_missing_start_does_not_fabricate_zero_duration():
    from flask import Flask, g

    app = Flask(__name__)
    register_request_duration_hooks(app)

    observed = {}

    @app.before_request
    def _clear_start():
        # Runs after request_duration's own before_request (registered
        # first), simulating the start value being absent by the time
        # after_request runs.
        if hasattr(g, _G_START_ATTR):
            delattr(g, _G_START_ATTR)

    @app.teardown_request
    def _observe_after_finalization(exc=None):
        # teardown_request always runs after after_request, regardless of
        # registration order, so this proves what after_request left behind
        # without crashing.
        observed["duration_ms"] = getattr(g, _G_DURATION_ATTR, "SENTINEL_UNSET")

    @app.route("/cleared")
    def cleared():
        return "ok"

    response = app.test_client().get("/cleared")

    assert response.status_code == 200
    assert observed["duration_ms"] == "SENTINEL_UNSET"


# --------------------------------------------------------------------------
# Idempotent registration
# --------------------------------------------------------------------------


def test_register_request_duration_hooks_is_idempotent():
    from flask import Flask

    app = Flask(__name__)
    register_request_duration_hooks(app)
    register_request_duration_hooks(app)  # second call must be a no-op

    assert getattr(app, _HOOKS_MARKER, False) is True
    assert len(app.before_request_funcs[None]) == 1
    assert len(app.after_request_funcs[None]) == 1


def test_idempotent_registration_runs_single_timer_lifecycle():
    app = _make_app()
    register_request_duration_hooks(app)  # second call, must be a no-op

    with patch(
        "utils.request_duration.time.perf_counter",
        _patched_perf_counter([100.000, 100.010]),
    ):
        app.test_client().get("/ping")

    assert app.observed["duration_ms"] == 10
