"""Contract tests for the public Usage attribution boundary.

Covers the public promises of ``utils.usage_attribution``: what an adopter
declares, what the boundary derives and hides, how each declared intent
behaves when the world underneath it fails, and what a diagnostic is
allowed to say.
"""

import inspect
import logging

import pytest
from flask import Flask

from utils import usage_attribution
from utils.usage_attribution import (
    USAGE_POLICY_ADMIN_RAG_INGESTION,
    USAGE_POLICY_LEGACY_DINO_INGESTION,
    attribute_usage_to_policy,
    attribute_usage_to_user,
    declare_usage_unattributed,
)
from utils.usage_attribution_state import get_usage_attribution

_ROUTE = "/completion.json"
_REAL_USERNAME = "person@example.com"
_DINO_TECHNICAL_USERNAME = "__dino_legacy_ingestion__"
_ADMIN_TECHNICAL_USERNAME = "__admin_rag_ingestion__"


class _Config:
    """Stand-in for ``MAUI_CONFIG``, carrying only what this boundary reads."""

    def __init__(self, dino=None, admin=None):
        self.dino_legacy_usage_username = dino
        self.admin_rag_usage_username = admin


@pytest.fixture
def app():
    """A Flask app with a real registered route, so ``url_rule`` populates."""
    flask_app = Flask(__name__)

    @flask_app.route(_ROUTE, methods=["POST"])
    def completion():  # pragma: no cover - never actually dispatched
        return ""

    flask_app.config["MAUI_CONFIG"] = _Config()
    return flask_app


@pytest.fixture
def lookups(monkeypatch):
    """Record every users lookup and serve a configurable row per username."""
    calls = {"usernames": [], "rows": {}}

    def fake_get_user_by_username(username):
        calls["usernames"].append(username)
        row = calls["rows"].get(username, "__missing__")
        if row == "__missing__":
            return None
        if isinstance(row, Exception):
            raise row
        return row

    monkeypatch.setattr(
        usage_attribution, "get_user_by_username", fake_get_user_by_username
    )
    return calls


def _diagnostics(caplog):
    """The attribution-unavailable records emitted during a test."""
    return [
        record
        for record in caplog.records
        if "embedding_usage_attribution_unavailable" in record.getMessage()
    ]


# --------------------------------------------------------------------------
# Real authenticated user
# --------------------------------------------------------------------------


def test_real_user_binds_resolved_persistent_user_id(app, lookups):
    lookups["rows"][_REAL_USERNAME] = {"id": 42, "client": "dino"}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_user(username=_REAL_USERNAME)
        attribution = get_usage_attribution()

    assert attribution is not None
    assert attribution.user_id == 42


def test_real_user_source_derives_from_users_client(app, lookups):
    lookups["rows"][_REAL_USERNAME] = {"id": 42, "client": "dino"}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_user(username=_REAL_USERNAME)
        attribution = get_usage_attribution()

    assert attribution.source == "dino"


def test_real_user_service_is_the_active_registered_route_rule(app, lookups):
    lookups["rows"][_REAL_USERNAME] = {"id": 42, "client": None}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_user(username=_REAL_USERNAME)
        attribution = get_usage_attribution()

    assert attribution.service == _ROUTE


def test_real_user_returns_none(app, lookups):
    lookups["rows"][_REAL_USERNAME] = {"id": 42, "client": None}

    with app.test_request_context(_ROUTE, method="POST"):
        assert attribute_usage_to_user(username=_REAL_USERNAME) is None


def test_real_user_missing_row_binds_nothing_and_reports_not_found(
    app, lookups, caplog
):
    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_user(username=_REAL_USERNAME)
        assert get_usage_attribution() is None

    records = _diagnostics(caplog)
    assert len(records) == 1
    assert "reason=not_found" in records[0].getMessage()


def test_real_user_invalid_persistent_id_binds_nothing_and_reports(
    app, lookups, caplog
):
    lookups["rows"][_REAL_USERNAME] = {"id": "42", "client": None}

    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_user(username=_REAL_USERNAME)
        assert get_usage_attribution() is None

    records = _diagnostics(caplog)
    assert len(records) == 1
    assert "reason=invalid_user_id" in records[0].getMessage()


def test_real_user_lookup_exception_reports_lookup_failed_with_error_type(
    app, lookups, caplog
):
    lookups["rows"][_REAL_USERNAME] = RuntimeError("database down")

    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_user(username=_REAL_USERNAME)
        assert get_usage_attribution() is None

    records = _diagnostics(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "reason=lookup_failed" in message
    assert "error_type=RuntimeError" in message


def test_real_user_non_exception_reason_carries_no_error_type(app, lookups, caplog):
    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_user(username=_REAL_USERNAME)

    assert "error_type=None" in _diagnostics(caplog)[0].getMessage()


def test_real_user_binding_failure_is_contained(app, lookups, caplog, monkeypatch):
    """A failing bind degrades exactly like a failing lookup.

    The whole runtime contract is pinned here, diagnostic included: the
    binding call sits inside the shared tail's guard, so a raising binder
    must leave nothing bound, emit one safe ``lookup_failed`` record
    carrying the exception class, and return ``None`` without propagating.
    """
    lookups["rows"][_REAL_USERNAME] = {"id": 42, "client": None}
    monkeypatch.setattr(
        usage_attribution,
        "bind_usage_attribution",
        lambda *args: (_ for _ in ()).throw(RuntimeError("bind exploded")),
    )

    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        assert attribute_usage_to_user(username=_REAL_USERNAME) is None
        assert get_usage_attribution() is None

    records = _diagnostics(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "reason=lookup_failed" in message
    assert "error_type=RuntimeError" in message
    assert _REAL_USERNAME not in message
    assert "bind exploded" not in message


@pytest.mark.parametrize("username", ["", "   ", None, 42, b"person", ["person"]])
def test_real_user_invalid_username_raises_value_error(app, lookups, username):
    with app.test_request_context(_ROUTE, method="POST"):
        with pytest.raises(ValueError):
            attribute_usage_to_user(username=username)

    assert lookups["usernames"] == []


def test_real_user_diagnostics_never_contain_the_supplied_username(
    app, lookups, caplog
):
    for row, reason in (
        ("__missing__", "not_found"),
        ({"id": "nope"}, "invalid_user_id"),
        (RuntimeError("boom"), "lookup_failed"),
    ):
        caplog.clear()
        lookups["rows"] = {} if row == "__missing__" else {_REAL_USERNAME: row}

        with caplog.at_level(logging.WARNING), app.test_request_context(
            _ROUTE, method="POST"
        ):
            attribute_usage_to_user(username=_REAL_USERNAME)

        message = _diagnostics(caplog)[0].getMessage()
        assert reason in message
        assert _REAL_USERNAME not in message
        assert "person" not in message
        assert "example.com" not in message


# --------------------------------------------------------------------------
# Legacy Dino ingestion policy
# --------------------------------------------------------------------------


def test_legacy_dino_policy_resolves_its_configured_username(app, lookups):
    app.config["MAUI_CONFIG"] = _Config(dino=_DINO_TECHNICAL_USERNAME)
    lookups["rows"][_DINO_TECHNICAL_USERNAME] = {"id": 7, "client": "dino"}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)
        attribution = get_usage_attribution()

    assert lookups["usernames"] == [_DINO_TECHNICAL_USERNAME]
    assert attribution.user_id == 7


def test_legacy_dino_policy_source_derives_from_the_technical_rows_client(
    app, lookups
):
    app.config["MAUI_CONFIG"] = _Config(dino=_DINO_TECHNICAL_USERNAME)
    lookups["rows"][_DINO_TECHNICAL_USERNAME] = {"id": 7, "client": "dino"}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)
        attribution = get_usage_attribution()

    assert attribution.source == "dino"


def test_legacy_dino_policy_returns_none(app, lookups):
    app.config["MAUI_CONFIG"] = _Config(dino=_DINO_TECHNICAL_USERNAME)
    lookups["rows"][_DINO_TECHNICAL_USERNAME] = {"id": 7, "client": "dino"}

    with app.test_request_context(_ROUTE, method="POST"):
        assert (
            attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)
            is None
        )


@pytest.mark.parametrize("configured", [None, ""])
def test_legacy_dino_policy_absent_config_is_the_off_switch(
    app, lookups, caplog, configured
):
    app.config["MAUI_CONFIG"] = _Config(dino=configured)

    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)
        assert get_usage_attribution() is None

    assert lookups["usernames"] == []
    records = _diagnostics(caplog)
    assert len(records) == 1
    assert "reason=not_configured" in records[0].getMessage()


def test_legacy_dino_policy_never_logs_the_technical_username(app, lookups, caplog):
    app.config["MAUI_CONFIG"] = _Config(dino=_DINO_TECHNICAL_USERNAME)
    lookups["rows"][_DINO_TECHNICAL_USERNAME] = RuntimeError("boom")

    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)

    message = _diagnostics(caplog)[0].getMessage()
    assert "reason=lookup_failed" in message
    assert _DINO_TECHNICAL_USERNAME not in message
    assert "dino_legacy" not in message


# --------------------------------------------------------------------------
# Admin RAG ingestion policy
# --------------------------------------------------------------------------


def test_admin_rag_policy_resolves_its_configured_username(app, lookups):
    app.config["MAUI_CONFIG"] = _Config(admin=_ADMIN_TECHNICAL_USERNAME)
    lookups["rows"][_ADMIN_TECHNICAL_USERNAME] = {"id": 9, "client": None}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)
        attribution = get_usage_attribution()

    assert lookups["usernames"] == [_ADMIN_TECHNICAL_USERNAME]
    assert attribution.user_id == 9


def test_admin_rag_policy_binds_source_none(app, lookups):
    app.config["MAUI_CONFIG"] = _Config(admin=_ADMIN_TECHNICAL_USERNAME)
    lookups["rows"][_ADMIN_TECHNICAL_USERNAME] = {"id": 9, "client": None}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)
        attribution = get_usage_attribution()

    assert attribution.source is None


def test_admin_rag_policy_source_stays_none_even_when_the_row_carries_a_client(
    app, lookups
):
    """The bound value is a policy literal, not a property of the row.

    Ratified explicitly: admin RAG ingestion has no client concept to
    report, so a provisioned row that happens to carry one must not leak
    into the attribution.
    """
    app.config["MAUI_CONFIG"] = _Config(admin=_ADMIN_TECHNICAL_USERNAME)
    lookups["rows"][_ADMIN_TECHNICAL_USERNAME] = {"id": 9, "client": "something"}

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)
        attribution = get_usage_attribution()

    assert attribution.user_id == 9
    assert attribution.source is None


def test_admin_rag_policy_returns_none(app, lookups):
    app.config["MAUI_CONFIG"] = _Config(admin=_ADMIN_TECHNICAL_USERNAME)
    lookups["rows"][_ADMIN_TECHNICAL_USERNAME] = {"id": 9, "client": None}

    with app.test_request_context(_ROUTE, method="POST"):
        assert (
            attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION) is None
        )


@pytest.mark.parametrize("configured", [None, ""])
def test_admin_rag_policy_absent_config_is_the_off_switch(
    app, lookups, caplog, configured
):
    app.config["MAUI_CONFIG"] = _Config(admin=configured)

    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)
        assert get_usage_attribution() is None

    assert lookups["usernames"] == []
    records = _diagnostics(caplog)
    assert len(records) == 1
    assert "reason=not_configured" in records[0].getMessage()


def test_admin_rag_policy_never_logs_the_technical_username(app, lookups, caplog):
    app.config["MAUI_CONFIG"] = _Config(admin=_ADMIN_TECHNICAL_USERNAME)
    lookups["rows"][_ADMIN_TECHNICAL_USERNAME] = {"id": "nope"}

    with caplog.at_level(logging.WARNING), app.test_request_context(
        _ROUTE, method="POST"
    ):
        attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)

    message = _diagnostics(caplog)[0].getMessage()
    assert "reason=invalid_user_id" in message
    assert _ADMIN_TECHNICAL_USERNAME not in message
    assert "admin_rag" not in message


# --------------------------------------------------------------------------
# Policy isolation
# --------------------------------------------------------------------------


def test_the_two_policies_read_different_config_and_apply_different_source_rules(
    app, lookups
):
    """Guards against a future accidental global technical-source rule.

    Both policies are provisioned at once, with rows that would be
    indistinguishable under any single source rule: the legacy Dino row
    must report its client, the admin row must not report its own.
    """
    app.config["MAUI_CONFIG"] = _Config(
        dino=_DINO_TECHNICAL_USERNAME, admin=_ADMIN_TECHNICAL_USERNAME
    )
    lookups["rows"] = {
        _DINO_TECHNICAL_USERNAME: {"id": 7, "client": "dino"},
        _ADMIN_TECHNICAL_USERNAME: {"id": 9, "client": "dino"},
    }

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)
        legacy = get_usage_attribution()

    with app.test_request_context(_ROUTE, method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)
        admin = get_usage_attribution()

    assert lookups["usernames"] == [
        _DINO_TECHNICAL_USERNAME,
        _ADMIN_TECHNICAL_USERNAME,
    ]
    assert (legacy.user_id, legacy.source) == (7, "dino")
    assert (admin.user_id, admin.source) == (9, None)


# --------------------------------------------------------------------------
# Deliberate non-attribution
# --------------------------------------------------------------------------


def test_declare_usage_unattributed_binds_nothing_and_returns_none(app):
    with app.test_request_context(_ROUTE, method="POST"):
        assert declare_usage_unattributed() is None
        assert get_usage_attribution() is None


def test_declare_usage_unattributed_emits_no_log_record_at_all(app, caplog):
    """The silence is the distinguishing property, so it is asserted directly."""
    with caplog.at_level(logging.DEBUG), app.test_request_context(
        _ROUTE, method="POST"
    ):
        declare_usage_unattributed()

    assert caplog.records == []


def test_declare_usage_unattributed_outside_a_request_context_does_not_raise(caplog):
    with caplog.at_level(logging.DEBUG):
        assert declare_usage_unattributed() is None

    assert caplog.records == []


def test_declare_usage_unattributed_performs_no_users_lookup(app, lookups):
    with app.test_request_context(_ROUTE, method="POST"):
        declare_usage_unattributed()

    assert lookups["usernames"] == []


# --------------------------------------------------------------------------
# Request context and service derivation
# --------------------------------------------------------------------------


def test_service_equals_the_active_route_rule_for_a_second_route(app, lookups):
    @app.route("/admin/rag-files/upload", methods=["POST"])
    def upload():  # pragma: no cover - never actually dispatched
        return ""

    app.config["MAUI_CONFIG"] = _Config(admin=_ADMIN_TECHNICAL_USERNAME)
    lookups["rows"][_ADMIN_TECHNICAL_USERNAME] = {"id": 9, "client": None}

    with app.test_request_context("/admin/rag-files/upload", method="POST"):
        attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION)
        attribution = get_usage_attribution()

    assert attribution.service == "/admin/rag-files/upload"


def test_real_user_outside_a_request_context_reports_no_request_context(
    lookups, caplog
):
    with caplog.at_level(logging.WARNING):
        assert attribute_usage_to_user(username=_REAL_USERNAME) is None

    assert lookups["usernames"] == []
    records = _diagnostics(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "reason=no_request_context" in message
    assert "error_type=None" in message


def test_policy_outside_a_request_context_reports_no_request_context(lookups, caplog):
    """No app context either, so the config read must never be reached."""
    with caplog.at_level(logging.WARNING):
        assert (
            attribute_usage_to_policy(policy=USAGE_POLICY_ADMIN_RAG_INGESTION) is None
        )

    assert lookups["usernames"] == []
    records = _diagnostics(caplog)
    assert len(records) == 1
    assert "reason=no_request_context" in records[0].getMessage()


def test_request_context_without_a_matched_url_rule_reports_no_request_context(
    app, lookups, caplog
):
    with caplog.at_level(logging.WARNING), app.test_request_context(
        "/no/such/route", method="POST"
    ):
        from flask import request

        assert request.url_rule is None
        assert attribute_usage_to_user(username=_REAL_USERNAME) is None
        assert get_usage_attribution() is None

    assert lookups["usernames"] == []
    records = _diagnostics(caplog)
    assert len(records) == 1
    assert "reason=no_request_context" in records[0].getMessage()


# --------------------------------------------------------------------------
# Public shape
# --------------------------------------------------------------------------


def test_all_exposes_exactly_the_five_approved_public_names():
    assert set(usage_attribution.__all__) == {
        "attribute_usage_to_user",
        "attribute_usage_to_policy",
        "declare_usage_unattributed",
        "USAGE_POLICY_LEGACY_DINO_INGESTION",
        "USAGE_POLICY_ADMIN_RAG_INGESTION",
    }
    assert len(usage_attribution.__all__) == 5


def test_attribute_usage_to_user_takes_only_a_keyword_only_username():
    parameters = inspect.signature(attribute_usage_to_user).parameters

    assert list(parameters) == ["username"]
    assert parameters["username"].kind is inspect.Parameter.KEYWORD_ONLY


def test_attribute_usage_to_policy_takes_only_a_keyword_only_policy():
    parameters = inspect.signature(attribute_usage_to_policy).parameters

    assert list(parameters) == ["policy"]
    assert parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY


def test_declare_usage_unattributed_takes_no_arguments():
    assert list(inspect.signature(declare_usage_unattributed).parameters) == []


@pytest.mark.parametrize(
    "operation", [attribute_usage_to_user, attribute_usage_to_policy]
)
@pytest.mark.parametrize(
    "hidden", ["service", "source", "user_id", "request_id", "client"]
)
def test_no_public_operation_exposes_subsystem_mechanics(operation, hidden):
    assert hidden not in inspect.signature(operation).parameters


def test_no_public_operation_absorbs_kwargs():
    for operation in (
        attribute_usage_to_user,
        attribute_usage_to_policy,
        declare_usage_unattributed,
    ):
        kinds = {
            parameter.kind
            for parameter in inspect.signature(operation).parameters.values()
        }
        assert inspect.Parameter.VAR_KEYWORD not in kinds
        assert inspect.Parameter.VAR_POSITIONAL not in kinds


def test_positional_use_raises_type_error(app):
    with app.test_request_context(_ROUTE, method="POST"):
        with pytest.raises(TypeError):
            attribute_usage_to_user(_REAL_USERNAME)
        with pytest.raises(TypeError):
            attribute_usage_to_policy(USAGE_POLICY_ADMIN_RAG_INGESTION)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"user": _REAL_USERNAME},
        {"user_name": _REAL_USERNAME},
        {"username": _REAL_USERNAME, "service": _ROUTE},
        {"username": _REAL_USERNAME, "source": "dino"},
    ],
)
def test_wrong_keyword_on_the_real_user_operation_raises_type_error(app, kwargs):
    with app.test_request_context(_ROUTE, method="POST"):
        with pytest.raises(TypeError):
            attribute_usage_to_user(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policies": USAGE_POLICY_ADMIN_RAG_INGESTION},
        {"kind": USAGE_POLICY_ADMIN_RAG_INGESTION},
        {"policy": USAGE_POLICY_ADMIN_RAG_INGESTION, "user_id": 9},
    ],
)
def test_wrong_keyword_on_the_policy_operation_raises_type_error(app, kwargs):
    with app.test_request_context(_ROUTE, method="POST"):
        with pytest.raises(TypeError):
            attribute_usage_to_policy(**kwargs)


def test_declare_usage_unattributed_rejects_any_argument(app):
    with app.test_request_context(_ROUTE, method="POST"):
        with pytest.raises(TypeError):
            declare_usage_unattributed("anything")


@pytest.mark.parametrize(
    "policy",
    [
        "unknown_policy",
        "",
        "   ",
        "LEGACY_DINO_INGESTION",
        None,
        7,
        USAGE_POLICY_LEGACY_DINO_INGESTION.upper(),
    ],
)
def test_unknown_or_invalid_policy_raises_value_error(app, lookups, policy):
    with app.test_request_context(_ROUTE, method="POST"):
        with pytest.raises(ValueError):
            attribute_usage_to_policy(policy=policy)

    assert lookups["usernames"] == []


def test_unknown_policy_is_rejected_before_the_fail_open_guard(lookups):
    """Misuse raises even outside a request context, where degradation hides."""
    with pytest.raises(ValueError):
        attribute_usage_to_policy(policy="unknown_policy")

    assert lookups["usernames"] == []


def test_policy_constants_are_opaque_identifiers(app):
    assert USAGE_POLICY_LEGACY_DINO_INGESTION == "legacy_dino_ingestion"
    assert USAGE_POLICY_ADMIN_RAG_INGESTION == "admin_rag_ingestion"
