"""Tests for utils.usage_request_state.

Focused on the module's own concern only: request-local Usage row identity
storage. Mirrors the harness style of tests/test_request_duration.py.
"""

from utils.usage_request_state import (
    _G_LOG_ID_ATTR,
    get_usage_log_id,
    set_usage_log_id,
)


def _make_app():
    from flask import Flask

    app = Flask(__name__)

    @app.route("/ping")
    def ping():
        return "pong"

    return app


# --------------------------------------------------------------------------
# Public API surface
# --------------------------------------------------------------------------


def test_public_api_is_exactly_setter_and_getter():
    import utils.usage_request_state as module

    assert set(module.__all__) == {
        "set_usage_log_id",
        "get_usage_log_id",
    }


def test_raw_g_attribute_name_is_not_part_of_public_api():
    import utils.usage_request_state as module

    assert "_G_LOG_ID_ATTR" not in module.__all__


# --------------------------------------------------------------------------
# Absence / registration contract
# --------------------------------------------------------------------------


def test_getter_returns_none_before_registration():
    app = _make_app()

    with app.test_request_context("/ping"):
        assert get_usage_log_id() is None


def test_getter_returns_registered_value_after_setter():
    app = _make_app()

    with app.test_request_context("/ping"):
        set_usage_log_id(123)
        assert get_usage_log_id() == 123


def test_setter_stores_on_the_documented_private_g_attribute():
    from flask import g

    app = _make_app()

    with app.test_request_context("/ping"):
        set_usage_log_id(456)
        assert getattr(g, _G_LOG_ID_ATTR) == 456


# --------------------------------------------------------------------------
# Request isolation
# --------------------------------------------------------------------------


def test_state_is_isolated_across_sequential_requests():
    app = _make_app()

    @app.before_request
    def _noop():
        pass

    with app.test_request_context("/ping"):
        assert get_usage_log_id() is None
        set_usage_log_id(1)
        assert get_usage_log_id() == 1

    # A fresh request context gets a fresh flask.g - no leakage from the
    # previous context's registered value.
    with app.test_request_context("/ping"):
        assert get_usage_log_id() is None
