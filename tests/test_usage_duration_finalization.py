"""Tests for usage.duration_finalization.

Focused on the module's own concern only: best-effort composition of
get_request_duration_ms() (utils.request_duration), get_usage_log_ids()
(usage.request_state), and update_usage_duration()
(infrastructure.database_pg) at the after_request boundary. Does not
re-test any of those three settled modules' own contracts - they are
monkeypatched here exactly as this module consumes them.
"""

import re
from unittest.mock import patch

import pytest

from usage.duration_finalization import (
    _HOOKS_MARKER,
    register_usage_duration_finalization_hooks,
)

MODULE = "usage.duration_finalization"


def _make_app(duration_ms, log_id, update_result_or_exc=None):
    """Throwaway Flask app carrying only the B4 hook and a probe route.

    ``get_request_duration_ms``/``get_usage_log_ids`` are monkeypatched at
    module scope (imported names, not attribute lookups on flask.g) since
    B4 imports them as plain functions. ``update_usage_duration`` is
    likewise monkeypatched to either return a fixed value or raise.
    """
    from flask import Flask

    app = Flask(__name__)
    register_usage_duration_finalization_hooks(app)

    @app.route("/ping")
    def ping():
        return "pong"

    return app


def _as_ids(log_id):
    """Normalize a test's ``log_id`` shorthand into a registered-ids tuple.

    ``None`` means "no Usage row registered" -> ``()``; a bare int means a
    single registered row; an iterable is passed through as-is so a test can
    exercise several ids (including duplicates) in registration order.
    """
    if log_id is None:
        return ()
    if isinstance(log_id, int):
        return (log_id,)
    return tuple(log_id)


def _run_with_patches(app, duration_ms, log_id, update_side_effect):
    with patch(f"{MODULE}.get_request_duration_ms", return_value=duration_ms), patch(
        f"{MODULE}.get_usage_log_ids", return_value=_as_ids(log_id)
    ), patch(f"{MODULE}.update_usage_duration", side_effect=update_side_effect) as mock_update:
        response = app.test_client().get("/ping")
    return response, mock_update


# --------------------------------------------------------------------------
# Public API surface
# --------------------------------------------------------------------------


def test_public_api_is_exactly_registration_hook():
    import usage.duration_finalization as module

    assert set(module.__all__) == {"register_usage_duration_finalization_hooks"}


def test_hooks_marker_is_not_part_of_public_api():
    import usage.duration_finalization as module

    assert "_HOOKS_MARKER" not in module.__all__


# --------------------------------------------------------------------------
# Absence / no-op matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duration_ms,log_id",
    [
        (None, None),
        (123, None),
        (None, 456),
    ],
)
def test_absence_is_a_silent_no_op(duration_ms, log_id, caplog):
    app = _make_app(duration_ms, log_id)

    with caplog.at_level("WARNING"):
        response, mock_update = _run_with_patches(
            app, duration_ms, log_id, update_side_effect=AssertionError("must not be called")
        )

    mock_update.assert_not_called()
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"
    assert caplog.records == []


# --------------------------------------------------------------------------
# Successful update
# --------------------------------------------------------------------------


def test_successful_update_is_called_once_and_silent(caplog):
    app = _make_app(123, 456)

    with caplog.at_level("WARNING"):
        response, mock_update = _run_with_patches(
            app, duration_ms=123, log_id=456, update_side_effect=lambda *a, **k: True
        )

    mock_update.assert_called_once_with(456, 123)
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"
    assert caplog.records == []


def test_x_request_id_header_still_present_alongside_b4():
    from flask import Flask

    from utils.logging_config import register_request_context_hooks
    from utils.request_duration import register_request_duration_hooks

    app = Flask(__name__)
    register_request_context_hooks(app)
    register_usage_duration_finalization_hooks(app)
    register_request_duration_hooks(app)

    @app.route("/ping")
    def ping():
        return "pong"

    with patch(f"{MODULE}.get_usage_log_ids", return_value=()):
        response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{16}", response.headers["X-Request-ID"])


# --------------------------------------------------------------------------
# Missing Usage row (False)
# --------------------------------------------------------------------------


def test_missing_row_logs_warning_and_preserves_response(caplog):
    app = _make_app(123, 456)

    with caplog.at_level("WARNING"):
        response, mock_update = _run_with_patches(
            app, duration_ms=123, log_id=456, update_side_effect=lambda *a, **k: False
        )

    mock_update.assert_called_once_with(456, 123)
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert record.exc_info is None
    message = record.getMessage()
    assert "event=usage_duration_update_not_found" in message
    assert "log_id=456" in message
    assert "duration_ms=123" in message


# --------------------------------------------------------------------------
# DB exception
# --------------------------------------------------------------------------


def test_db_exception_logs_exception_and_preserves_response(caplog):
    app = _make_app(123, 456)

    with caplog.at_level("WARNING"):
        response, mock_update = _run_with_patches(
            app, duration_ms=123, log_id=456, update_side_effect=RuntimeError("boom")
        )

    mock_update.assert_called_once_with(456, 123)
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.exc_info is not None
    message = record.getMessage()
    assert "event=usage_duration_update_failed" in message
    assert "log_id=456" in message
    assert "duration_ms=123" in message
    assert "error_type=RuntimeError" in message
    assert "error=boom" in message


# --------------------------------------------------------------------------
# Idempotent registration
# --------------------------------------------------------------------------


def test_register_hooks_is_idempotent():
    from flask import Flask

    app = Flask(__name__)
    register_usage_duration_finalization_hooks(app)
    register_usage_duration_finalization_hooks(app)  # second call must be a no-op

    assert getattr(app, _HOOKS_MARKER, False) is True
    assert len(app.after_request_funcs[None]) == 1


def test_idempotent_registration_calls_update_exactly_once():
    app = _make_app(123, 456)
    register_usage_duration_finalization_hooks(app)  # second call, must be a no-op

    response, mock_update = _run_with_patches(
        app, duration_ms=123, log_id=456, update_side_effect=lambda *a, **k: True
    )

    mock_update.assert_called_once_with(456, 123)
    assert response.status_code == 200


# --------------------------------------------------------------------------
# Hook ordering: B4 must read duration only after B2 has finalized it
# --------------------------------------------------------------------------


def test_b4_reads_duration_finalized_by_b2_when_registered_before_it():
    """Proves the required main.py order: context -> B4 -> request_duration.

    Uses the real request_duration module (not monkeypatched) so B4 reads
    whatever B2's own after_request hook actually finalized during the same
    request - this fails if B4 were registered after request_duration's
    hooks, since then B4 would observe None (LIFO: B4 would run first).
    """
    from itertools import count

    from flask import Flask

    from utils.logging_config import register_request_context_hooks
    from utils.request_duration import register_request_duration_hooks

    app = Flask(__name__)
    register_request_context_hooks(app)
    register_usage_duration_finalization_hooks(app)
    register_request_duration_hooks(app)

    @app.route("/ping")
    def ping():
        return "pong"

    counter = count(100)

    def fake_perf_counter():
        return next(counter)

    with patch(
        "utils.request_duration.time.perf_counter", fake_perf_counter
    ), patch(f"{MODULE}.get_usage_log_ids", return_value=(456,)), patch(
        f"{MODULE}.update_usage_duration", return_value=True
    ) as mock_update:
        response = app.test_client().get("/ping")

    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{16}", response.headers["X-Request-ID"])
    # start=100, stop=101 -> 1000ms, observed by B4 as a non-None duration.
    mock_update.assert_called_once_with(456, 1000)


# --------------------------------------------------------------------------
# Response identity/preservation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "update_side_effect",
    [
        lambda *a, **k: True,
        lambda *a, **k: False,
        RuntimeError("boom"),
    ],
)
def test_response_status_body_and_headers_are_unchanged(update_side_effect):
    app = _make_app(123, 456)

    @app.after_request
    def _tag_header(response):
        response.headers["X-Existing"] = "kept"
        return response

    response, _ = _run_with_patches(
        app, duration_ms=123, log_id=456, update_side_effect=update_side_effect
    )

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"
    assert response.headers["X-Existing"] == "kept"


# --------------------------------------------------------------------------
# Multiple registered Usage ids
# --------------------------------------------------------------------------


def test_every_registered_id_receives_the_same_request_duration(caplog):
    app = _make_app(850, (10, 20, 30))

    with caplog.at_level("WARNING"):
        response, mock_update = _run_with_patches(
            app,
            duration_ms=850,
            log_id=(10, 20, 30),
            update_side_effect=lambda *a, **k: True,
        )

    # Same duration for every row, in registration order - not split, summed
    # or recomputed per row.
    assert [call.args for call in mock_update.call_args_list] == [
        (10, 850),
        (20, 850),
        (30, 850),
    ]
    assert response.status_code == 200
    assert caplog.records == []


def test_duplicate_ids_are_finalized_once():
    """De-duplication is get_usage_log_ids()' contract; the loop must not
    re-update a row just because it appears twice upstream."""
    from usage.request_state import (
        register_usage_log_id,
        set_usage_log_id,
    )

    app = _make_app(850, None)

    ids_seen = []

    with patch(f"{MODULE}.get_request_duration_ms", return_value=850), patch(
        f"{MODULE}.update_usage_duration", side_effect=lambda *a: ids_seen.append(a) or True
    ):
        # Real request state, not a stub: registration happens inside the
        # same request the finalizer runs in.
        @app.before_request
        def _register():
            set_usage_log_id(10)
            register_usage_log_id(20)
            register_usage_log_id(10)
            register_usage_log_id(30)

        response = app.test_client().get("/ping")

    assert ids_seen == [(10, 850), (20, 850), (30, 850)]
    assert response.status_code == 200


def test_failure_on_one_id_does_not_block_the_remaining_ids(caplog):
    app = _make_app(850, (10, 20, 30))

    def _update(log_id, duration_ms):
        if log_id == 20:
            raise RuntimeError("boom")
        return True

    with caplog.at_level("WARNING"):
        response, mock_update = _run_with_patches(
            app, duration_ms=850, log_id=(10, 20, 30), update_side_effect=_update
        )

    assert [call.args for call in mock_update.call_args_list] == [
        (10, 850),
        (20, 850),
        (30, 850),
    ]
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "pong"

    # Exactly one diagnostic, naming the failing id and nothing else.
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "ERROR"
    assert record.exc_info is not None
    message = record.getMessage()
    assert "event=usage_duration_update_failed" in message
    assert "log_id=20" in message
    assert "duration_ms=850" in message
    assert "error_type=RuntimeError" in message


def test_not_found_is_reported_per_id_and_does_not_stop_the_loop(caplog):
    app = _make_app(850, (10, 20, 30))

    def _update(log_id, duration_ms):
        return log_id != 20

    with caplog.at_level("WARNING"):
        response, mock_update = _run_with_patches(
            app, duration_ms=850, log_id=(10, 20, 30), update_side_effect=_update
        )

    assert [call.args for call in mock_update.call_args_list] == [
        (10, 850),
        (20, 850),
        (30, 850),
    ]
    assert response.status_code == 200

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    message = record.getMessage()
    assert "event=usage_duration_update_not_found" in message
    assert "log_id=20" in message
    assert "duration_ms=850" in message


def test_legacy_single_set_usage_log_id_flow_updates_exactly_once():
    """Regression guard: a route that only calls set_usage_log_id() once
    must be finalized exactly as before this module learned about many ids."""
    from usage.request_state import set_usage_log_id

    app = _make_app(850, None)

    with patch(f"{MODULE}.get_request_duration_ms", return_value=850), patch(
        f"{MODULE}.update_usage_duration", return_value=True
    ) as mock_update:

        @app.before_request
        def _register():
            set_usage_log_id(456)

        response = app.test_client().get("/ping")

    mock_update.assert_called_once_with(456, 850)
    assert response.status_code == 200
