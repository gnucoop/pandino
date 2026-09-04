"""Tests for usage.attribution_state.

Focused on the module's own concern only: request-local Usage attribution
metadata storage. Mirrors the harness style of
tests/test_usage_request_state.py.
"""

import dataclasses

import pytest

from usage.attribution_state import (
    _G_ATTRIBUTION_ATTR,
    UsageAttribution,
    bind_usage_attribution,
    get_usage_attribution,
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


def test_public_api_is_exactly_the_type_the_binder_and_the_getter():
    import usage.attribution_state as module

    assert set(module.__all__) == {
        "UsageAttribution",
        "bind_usage_attribution",
        "get_usage_attribution",
    }


def test_raw_g_attribute_name_is_not_part_of_public_api():
    import usage.attribution_state as module

    assert "_G_ATTRIBUTION_ATTR" not in module.__all__


# --------------------------------------------------------------------------
# Dataclass contract
# --------------------------------------------------------------------------


def test_fields_are_exactly_user_id_service_and_source():
    assert [f.name for f in dataclasses.fields(UsageAttribution)] == [
        "user_id",
        "service",
        "source",
    ]


def test_attribution_is_frozen():
    attribution = UsageAttribution(user_id=1, service="/completion.json", source="dino")

    with pytest.raises(dataclasses.FrozenInstanceError):
        attribution.user_id = 2


def test_attribution_is_slotted():
    attribution = UsageAttribution(user_id=1, service="/completion.json", source="dino")

    assert not hasattr(attribution, "__dict__")
    assert UsageAttribution.__slots__ == ("user_id", "service", "source")


def test_attribution_compares_by_value():
    assert UsageAttribution(user_id=7, service="/agentchat", source=None) == (
        UsageAttribution(user_id=7, service="/agentchat", source=None)
    )


# --------------------------------------------------------------------------
# Absence / bind contract
# --------------------------------------------------------------------------


def test_getter_returns_none_before_any_bind():
    app = _make_app()

    with app.test_request_context("/ping"):
        assert get_usage_attribution() is None


def test_getter_returns_the_bound_value():
    app = _make_app()

    with app.test_request_context("/ping"):
        bind_usage_attribution(42, "/completion.json", "dino")

        assert get_usage_attribution() == UsageAttribution(
            user_id=42,
            service="/completion.json",
            source="dino",
        )


def test_null_source_is_preserved_as_none():
    app = _make_app()

    with app.test_request_context("/ping"):
        bind_usage_attribution(42, "/agentchat", None)

        attribution = get_usage_attribution()

    assert attribution is not None
    assert attribution.source is None


def test_bind_stores_on_the_documented_private_g_attribute():
    app = _make_app()

    with app.test_request_context("/ping"):
        from flask import g

        bind_usage_attribution(42, "/completion.json", "dino")

        assert getattr(g, _G_ATTRIBUTION_ATTR) == UsageAttribution(
            user_id=42,
            service="/completion.json",
            source="dino",
        )


# --------------------------------------------------------------------------
# Rebinding semantics: last bind wins
# --------------------------------------------------------------------------


def test_second_bind_replaces_the_first():
    app = _make_app()

    with app.test_request_context("/ping"):
        bind_usage_attribution(1, "/completion.json", "dino")
        bind_usage_attribution(2, "/agentchat", None)

        assert get_usage_attribution() == UsageAttribution(
            user_id=2,
            service="/agentchat",
            source=None,
        )


def test_rebinding_does_not_raise():
    app = _make_app()

    with app.test_request_context("/ping"):
        bind_usage_attribution(1, "/completion.json", "dino")
        bind_usage_attribution(1, "/completion.json", "dino")

        assert get_usage_attribution().user_id == 1


# --------------------------------------------------------------------------
# Request isolation
# --------------------------------------------------------------------------


def test_state_is_isolated_across_sequential_requests():
    app = _make_app()

    with app.test_request_context("/ping"):
        bind_usage_attribution(42, "/completion.json", "dino")
        assert get_usage_attribution().user_id == 42

    with app.test_request_context("/ping"):
        assert get_usage_attribution() is None


def test_each_request_sees_only_its_own_bind():
    app = _make_app()

    with app.test_request_context("/ping"):
        bind_usage_attribution(1, "/completion.json", "dino")

    with app.test_request_context("/ping"):
        bind_usage_attribution(2, "/agentchat", "other")

        assert get_usage_attribution() == UsageAttribution(
            user_id=2,
            service="/agentchat",
            source="other",
        )
