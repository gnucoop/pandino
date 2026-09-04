"""Tests for usage.lifecycle.

Focused on the module's own concern only: composing the three Usage-owned
lifecycle registrars, in the one order production supports, and owning
nothing else. The children's own contracts are settled and tested in
their own modules; here they are monkeypatched or observed only as the
things this boundary delegates to.

The effective runtime order this composition produces is proven
behaviourally in tests/test_embedding_usage_persistence.py, not here.
"""

import ast
import os

from flask import Flask

import usage.lifecycle as usage_lifecycle
from usage.lifecycle import register_usage_lifecycle_hooks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Every registrar the boundary composes, in the only order production
#: supports. register_request_duration_hooks is shared infrastructure, not a
#: Usage module, but its exact position between persistence and embedding
#: accounting is part of the Usage lifecycle contract.
EXPECTED_ORDER = [
    "register_usage_duration_finalization_hooks",
    "register_embedding_usage_persistence_hooks",
    "register_request_duration_hooks",
    "register_embedding_accounting_hooks",
]

#: Registrars main.py must keep: request context precedes the boundary, and
#: Operational Persistence is an unrelated subsystem.
EXTERNAL_REGISTRARS = [
    "register_request_context_hooks",
    "register_operational_persistence",
]

#: The registrars bootstrap must no longer sequence itself.
BOOTSTRAP_FORBIDDEN = EXPECTED_ORDER


def _parse(path):
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as handle:
        source = handle.read()
    return source, ast.parse(source, filename=path)


def _hook_counts(app):
    return {
        "before": sum(len(v) for v in app.before_request_funcs.values()),
        "after": sum(len(v) for v in app.after_request_funcs.values()),
        "teardown": sum(len(v) for v in app.teardown_request_funcs.values()),
    }


def _called_names(tree, of_interest):
    """Every direct ``name(...)`` call in source order, filtered."""
    return [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in set(of_interest)
    ]


# --------------------------------------------------------------------------
# Ordered delegation
# --------------------------------------------------------------------------


def test_the_composed_registrars_are_called_in_the_approved_order(monkeypatch):
    """The ordering contract, asserted on behaviour rather than on source.

    Flask's FIFO before_request and LIFO after_request/teardown semantics
    make this one sequence a correctness invariant for three chains at
    once; the behavioural order tests in
    tests/test_embedding_usage_persistence.py prove what it produces.
    """
    calls = []
    app = Flask(__name__)

    for name in EXPECTED_ORDER:
        monkeypatch.setattr(
            usage_lifecycle, name, lambda received, _name=name: calls.append((_name, received))
        )

    register_usage_lifecycle_hooks(app)

    assert [name for name, _ in calls] == EXPECTED_ORDER
    assert [received for _, received in calls] == [app] * len(EXPECTED_ORDER)


# --------------------------------------------------------------------------
# No external ownership
# --------------------------------------------------------------------------


def test_the_boundary_owns_no_external_subsystem_registrar():
    """Guards against facade creep, on the module's own source.

    The boundary composes the request timer because the Usage lifecycle
    needs it interleaved at one exact position. It must not reach further:
    request context is the caller's prerequisite, and Operational
    Persistence registers no request hook at all.
    """
    _, tree = _parse("usage/lifecycle.py")

    assert _called_names(tree, EXTERNAL_REGISTRARS) == []

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imported.isdisjoint(EXTERNAL_REGISTRARS)


def test_the_boundary_registers_no_flask_hook_of_its_own():
    """Delegation only: the child modules own every hook seam."""
    source, _ = _parse("usage/lifecycle.py")

    for seam in ("before_request", "after_request", "teardown_request"):
        assert f"app.{seam}" not in source


def test_the_boundary_adds_no_hook_beyond_its_children():
    """Composed app and hand-wired app must have identical hook counts."""
    from usage.embedding_accounting_lifecycle import (
        register_embedding_accounting_hooks,
    )
    from usage.embedding_persistence import (
        register_embedding_usage_persistence_hooks,
    )
    from utils.request_duration import register_request_duration_hooks
    from usage.duration_finalization import (
        register_usage_duration_finalization_hooks,
    )

    composed = Flask(__name__)
    register_usage_lifecycle_hooks(composed)

    hand_wired = Flask(__name__)
    register_usage_duration_finalization_hooks(hand_wired)
    register_embedding_usage_persistence_hooks(hand_wired)
    register_request_duration_hooks(hand_wired)
    register_embedding_accounting_hooks(hand_wired)

    assert _hook_counts(composed) == _hook_counts(hand_wired)



def test_the_boundary_binds_the_request_timer_it_composes():
    """The composed timer really runs: duration is finalized per request.

    The boundary contributes no timing semantics of its own; this asserts
    only that composing it leaves utils.request_duration doing its job.
    """
    from utils.request_duration import get_request_duration_ms

    app = Flask(__name__)
    register_usage_lifecycle_hooks(app)

    seen = {}

    @app.route("/ping")
    def ping():
        seen["during"] = get_request_duration_ms()
        return "pong"

    @app.after_request
    def _observe(response):
        # Registered last, so LIFO puts this first - after the timer's own
        # after_request has finalized nothing yet. Read on teardown instead.
        return response

    @app.teardown_request
    def _after(exc=None):
        seen["settled"] = get_request_duration_ms()

    assert app.test_client().get("/ping").status_code == 200
    assert seen["during"] is None
    assert isinstance(seen["settled"], int)


def test_the_boundary_does_not_bind_request_context():
    """request context stays the caller's prerequisite, not a child."""
    from utils.logging_config import CONTEXT_UNSET, get_request_id

    app = Flask(__name__)
    register_usage_lifecycle_hooks(app)

    seen = {}

    @app.route("/ping")
    def ping():
        seen["request_id"] = get_request_id()
        return "pong"

    response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert seen["request_id"] == CONTEXT_UNSET
    assert "X-Request-ID" not in response.headers


# --------------------------------------------------------------------------
# Idempotence by delegation
# --------------------------------------------------------------------------


def test_a_second_composition_call_registers_nothing_twice():
    """Idempotence comes from the children, not from a marker here.

    Asserted on the app's real hook tables, so this stays true however the
    children implement their own guards.
    """
    app = Flask(__name__)
    register_usage_lifecycle_hooks(app)
    first = _hook_counts(app)

    register_usage_lifecycle_hooks(app)

    assert _hook_counts(app) == first


def test_composition_after_a_direct_child_registration_still_registers_once():
    """The mixed case: one child wired directly, then the whole boundary."""
    from usage.duration_finalization import (
        register_usage_duration_finalization_hooks,
    )

    app = Flask(__name__)
    register_usage_duration_finalization_hooks(app)
    register_usage_lifecycle_hooks(app)

    reference = Flask(__name__)
    register_usage_lifecycle_hooks(reference)

    assert _hook_counts(app) == _hook_counts(reference)


# --------------------------------------------------------------------------
# Bootstrap composition guard
# --------------------------------------------------------------------------


def test_main_composes_usage_through_the_public_boundary():
    """main.py's ordering contract, reduced to subsystem composition.

    Asserted on the parsed module rather than on a live import, which
    would require the full runtime config and a database. Only the
    inter-subsystem prerequisites are pinned here; Usage's internal
    registration order is owned by usage.lifecycle and is asserted
    against that module instead.
    """
    _, tree = _parse("main.py")

    ordered = _called_names(
        tree,
        ["register_request_context_hooks", "register_usage_lifecycle_hooks"],
    )

    assert ordered == [
        "register_request_context_hooks",
        "register_usage_lifecycle_hooks",
    ]


def test_main_does_not_sequence_any_composed_registrar():
    """The anti-regression half: bootstrap must not re-acquire the order.

    Includes the request timer: bootstrap calling it directly is exactly
    the interleaving mistake the boundary exists to prevent.
    """
    _, tree = _parse("main.py")

    assert _called_names(tree, BOOTSTRAP_FORBIDDEN) == []


def test_main_imports_the_boundary_and_not_its_children():
    _, tree = _parse("main.py")

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "register_usage_lifecycle_hooks" in imported
    assert imported.isdisjoint(BOOTSTRAP_FORBIDDEN)


def test_operational_persistence_is_not_pinned_into_the_usage_ordering():
    """A separate subsystem: registered once, at no particular position.

    Its own bootstrap guards live in
    tests/test_operational_persistence_bootstrap.py; this only records
    that Usage composition does not claim it.
    """
    _, tree = _parse("main.py")

    assert _called_names(tree, ["register_operational_persistence"]) == [
        "register_operational_persistence"
    ]
