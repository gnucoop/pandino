"""Tests for usage.request_state.

Focused on the module's own concern only: request-local Usage row identity
storage. Mirrors the harness style of tests/test_request_duration.py.
"""

from usage.request_state import (
    _G_LOG_ID_ATTR,
    _G_LOG_IDS_ATTR,
    get_usage_log_id,
    get_usage_log_ids,
    register_usage_log_id,
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


def test_public_api_is_exactly_the_setter_registrar_and_getters():
    import usage.request_state as module

    assert set(module.__all__) == {
        "set_usage_log_id",
        "get_usage_log_id",
        "register_usage_log_id",
        "get_usage_log_ids",
    }


def test_raw_g_attribute_name_is_not_part_of_public_api():
    import usage.request_state as module

    assert "_G_LOG_ID_ATTR" not in module.__all__
    assert "_G_LOG_IDS_ATTR" not in module.__all__


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
        register_usage_log_id(2)
        assert get_usage_log_id() == 1
        assert get_usage_log_ids() == (1, 2)

    # A fresh request context gets a fresh flask.g - no leakage from the
    # previous context's registered values, single slot or ordered list.
    with app.test_request_context("/ping"):
        assert get_usage_log_id() is None
        assert get_usage_log_ids() == ()


# --------------------------------------------------------------------------
# Ordered multi-id registration
# --------------------------------------------------------------------------


def test_no_ids_registered_reads_as_empty_tuple():
    app = _make_app()

    with app.test_request_context("/ping"):
        assert get_usage_log_ids() == ()


def test_legacy_setter_also_registers_the_id():
    app = _make_app()

    with app.test_request_context("/ping"):
        set_usage_log_id(10)
        assert get_usage_log_id() == 10
        assert get_usage_log_ids() == (10,)


def test_repeated_legacy_setter_keeps_latest_single_id_and_registers_both():
    app = _make_app()

    with app.test_request_context("/ping"):
        set_usage_log_id(10)
        set_usage_log_id(20)
        assert get_usage_log_id() == 20
        assert get_usage_log_ids() == (10, 20)


def test_register_only_does_not_touch_the_single_slot():
    app = _make_app()

    with app.test_request_context("/ping"):
        set_usage_log_id(10)
        register_usage_log_id(20)
        assert get_usage_log_id() == 10
        assert get_usage_log_ids() == (10, 20)


def test_register_without_any_legacy_set_leaves_the_single_slot_empty():
    app = _make_app()

    with app.test_request_context("/ping"):
        register_usage_log_id(20)
        assert get_usage_log_id() is None
        assert get_usage_log_ids() == (20,)


def test_duplicates_appear_once_in_first_seen_order():
    app = _make_app()

    with app.test_request_context("/ping"):
        set_usage_log_id(10)
        register_usage_log_id(20)
        register_usage_log_id(10)
        register_usage_log_id(30)
        assert get_usage_log_ids() == (10, 20, 30)


def test_order_is_registration_order_not_numeric_order():
    app = _make_app()

    with app.test_request_context("/ping"):
        register_usage_log_id(30)
        register_usage_log_id(10)
        register_usage_log_id(20)
        assert get_usage_log_ids() == (30, 10, 20)


def test_returned_collection_is_an_immutable_tuple():
    app = _make_app()

    with app.test_request_context("/ping"):
        register_usage_log_id(10)
        ids = get_usage_log_ids()
        assert isinstance(ids, tuple)
        # Mutating a snapshot must be impossible, so later registrations
        # cannot be smuggled in through a previously returned value.
        register_usage_log_id(20)
        assert ids == (10,)
        assert get_usage_log_ids() == (10, 20)


def test_registration_stores_on_the_documented_private_g_attribute():
    from flask import g

    app = _make_app()

    with app.test_request_context("/ping"):
        set_usage_log_id(10)
        register_usage_log_id(20)
        assert getattr(g, _G_LOG_IDS_ATTR) == [10, 20]
        assert getattr(g, _G_LOG_ID_ATTR) == 10
