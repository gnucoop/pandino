"""
Contract tests for utils/operational_event.py — FOUNDATION INTERVENTION I1.

Scope is deliberately narrow: only the emission contract (builder + validation
+ anti-context-override + no-logger-ownership + no-production-adoption yet).
No snapshot, no handler, no queue, no DB — those belong to later
interventions.
"""

import ast
import logging
import os
import sys
from pathlib import Path

import pytest

from utils.operational_event import build_operational_event

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Valid minimal event
# ---------------------------------------------------------------------------


def test_minimal_event_message_and_extra():
    message, extra = build_operational_event(event="flow_started")

    assert message == "event=flow_started"
    assert extra["maui_persist"] is True
    assert extra["maui_event"] == "flow_started"
    assert set(extra.keys()) == {"maui_persist", "maui_event"}


# ---------------------------------------------------------------------------
# 2. Complete valid event
# ---------------------------------------------------------------------------


def test_complete_event_field_order_and_metadata():
    message, extra = build_operational_event(
        event="provider_call_failed",
        provider="DeepInfra",
        model="some-model",
        duration_ms=42,
        error_type="TimeoutError",
        details={"b": 2, "a": 1},
        message="Provider did not respond within configured timeout",
    )

    assert message == (
        "event=provider_call_failed provider=DeepInfra model=some-model "
        "duration_ms=42 error_type=TimeoutError a=1 b=2 "
        "msg=Provider did not respond within configured timeout"
    )
    assert extra == {
        "maui_persist": True,
        "maui_event": "provider_call_failed",
        "maui_provider": "DeepInfra",
        "maui_model": "some-model",
        "maui_duration_ms": 42,
        "maui_error_type": "TimeoutError",
        "maui_details": {"b": 2, "a": 1},
        "maui_message": "Provider did not respond within configured timeout",
    }
    assert message.startswith(f"event={extra['maui_event']}")


# ---------------------------------------------------------------------------
# 3. Optional fields omitted
# ---------------------------------------------------------------------------


def test_omitted_optional_fields_are_absent_not_none():
    _, extra = build_operational_event(event="flow_started", provider="X")

    assert "maui_model" not in extra
    assert "maui_duration_ms" not in extra
    assert "maui_error_type" not in extra
    assert "maui_details" not in extra
    assert "maui_message" not in extra
    assert extra["maui_provider"] == "X"


# ---------------------------------------------------------------------------
# 4. Invalid event names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_event",
    ["", "Upper", "with space", "with-hyphen", 123, None],
)
def test_invalid_event_names_raise_value_error(bad_event):
    with pytest.raises(ValueError):
        build_operational_event(event=bad_event)


# ---------------------------------------------------------------------------
# 5. Invalid scalar metadata
# ---------------------------------------------------------------------------


def test_invalid_provider_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", provider=123)


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", model=123)


def test_invalid_error_type_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", error_type=123)


def test_invalid_message_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", message=123)


def test_duration_ms_float_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", duration_ms=1.5)


def test_duration_ms_bool_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", duration_ms=True)
    with pytest.raises(ValueError):
        build_operational_event(event="e", duration_ms=False)


# ---------------------------------------------------------------------------
# 6. Invalid details
# ---------------------------------------------------------------------------


def test_details_non_dict_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", details=["not", "a", "dict"])


def test_details_non_string_key_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", details={1: "x"})


def test_details_nested_dict_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", details={"a": {"nested": 1}})


def test_details_list_value_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", details={"a": [1, 2]})


def test_details_tuple_value_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", details={"a": (1, 2)})


def test_details_set_value_raises():
    with pytest.raises(ValueError):
        build_operational_event(event="e", details={"a": {1, 2}})


def test_details_arbitrary_object_raises():
    class Thing:
        pass

    with pytest.raises(ValueError):
        build_operational_event(event="e", details={"a": Thing()})


def test_details_valid_scalars_including_bool_and_none():
    _, extra = build_operational_event(
        event="e",
        details={"flag": True, "absent": None, "count": 3, "ratio": 1.5, "name": "x"},
    )
    assert extra["maui_details"] == {
        "flag": True,
        "absent": None,
        "count": 3,
        "ratio": 1.5,
        "name": "x",
    }


# ---------------------------------------------------------------------------
# 7. Context ownership — anti-override
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwarg", ["request_id", "app_id", "logger", "level", "timestamp"]
)
def test_unsupported_context_kwargs_raise_type_error(kwarg):
    with pytest.raises(TypeError):
        build_operational_event(event="e", **{kwarg: "x"})


# ---------------------------------------------------------------------------
# 8. No logger ownership
# ---------------------------------------------------------------------------


def test_module_defines_no_module_level_logger():
    import utils.operational_event as mod

    for name, value in vars(mod).items():
        assert not isinstance(value, logging.Logger), (
            f"utils.operational_event must own no logger; found {name!r}"
        )


def test_module_performs_no_logging_calls_in_source():
    source = (REPO_ROOT / "utils" / "operational_event.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in (
                "info",
                "warning",
                "error",
                "critical",
                "debug",
                "exception",
                "log",
            ):
                pytest.fail(
                    "utils/operational_event.py must perform no logging itself"
                )


# ---------------------------------------------------------------------------
# 9. Real logging compatibility
# ---------------------------------------------------------------------------


def test_real_logging_compatibility(capsys):
    test_logger_name = "tests.test_operational_event_contract.compat"
    logger = logging.getLogger(test_logger_name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    captured_records = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    capture = _CaptureHandler()
    logger.addHandler(capture)

    try:
        message, extra = build_operational_event(event="compat_check", provider="X")

        logger.info(message, extra=extra)
        with_extra_out = capsys.readouterr().err
        record = captured_records[-1]

        logger.info(message)
        without_extra_out = capsys.readouterr().err

        assert with_extra_out == without_extra_out
        assert record.name == test_logger_name
        assert record.name != "utils.operational_event"
        assert record.args == ()
        assert record.maui_persist is True
        assert record.maui_event == "compat_check"
        assert record.maui_provider == "X"
    finally:
        logger.removeHandler(handler)
        logger.removeHandler(capture)


# ---------------------------------------------------------------------------
# 10. No production adoption yet
# ---------------------------------------------------------------------------


def test_no_production_call_sites_use_build_operational_event():
    excluded_dirs = {"venv", ".venv", "__pycache__", ".git", "node_modules"}
    own_module = (REPO_ROOT / "utils" / "operational_event.py").resolve()
    own_test = Path(__file__).resolve()
    # FOUNDATION INTERVENTION I7: tests/test_logging_invariants.py implements
    # the O2/O5 static invariants, which must reference the identifier
    # "build_operational_event" as TEXT/AST pattern-matching material (to
    # detect the canonical builder-pairing form and to prove the persistence
    # subsystem never calls it) without itself becoming a call site. This is
    # the same sanctioned shape as this file's own self-exclusion below.
    sanctioned_static_analysis = (
        REPO_ROOT / "tests" / "test_logging_invariants.py"
    ).resolve()

    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in excluded_dirs for part in path.parts):
            continue
        resolved = path.resolve()
        if resolved in (own_module, own_test, sanctioned_static_analysis):
            continue
        text = path.read_text(errors="ignore")
        if "build_operational_event" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == [], (
        "build_operational_event must have zero call sites outside its own "
        f"module and this test in FOUNDATION INTERVENTION I1: {offenders}"
    )


# ---------------------------------------------------------------------------
# 11. Q4 bounding — runtime normalization (message)
# ---------------------------------------------------------------------------

MESSAGE_LIMIT = 1000
DETAILS_VALUE_LIMIT = 200
DETAILS_KEY_COUNT_LIMIT = 20
DETAILS_KEY_LENGTH_LIMIT = 64
MARKER = "...[truncated]"


@pytest.mark.parametrize("size", [1, MESSAGE_LIMIT - 1, MESSAGE_LIMIT])
def test_message_at_or_below_limit_is_unchanged(size):
    original = "m" * size

    _, extra = build_operational_event(event="e", message=original)

    assert extra["maui_message"] == original
    assert MARKER not in extra["maui_message"]


def test_message_above_limit_is_truncated_within_bound():
    original = "m" * 5000

    _, extra = build_operational_event(event="e", message=original)
    bounded = extra["maui_message"]

    assert len(bounded) == MESSAGE_LIMIT
    assert bounded.endswith(MARKER)
    assert bounded[: MESSAGE_LIMIT - len(MARKER)] == original[
        : MESSAGE_LIMIT - len(MARKER)
    ]


def test_message_normalization_is_deterministic():
    original = "abc" * 4000

    _, first = build_operational_event(event="e", message=original)
    _, second = build_operational_event(event="e", message=original)

    assert first["maui_message"] == second["maui_message"]


# ---------------------------------------------------------------------------
# 12. Q4 bounding — runtime normalization (details string values)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size", [1, DETAILS_VALUE_LIMIT - 1, DETAILS_VALUE_LIMIT]
)
def test_details_string_at_or_below_limit_is_unchanged(size):
    original = "v" * size

    _, extra = build_operational_event(event="e", details={"reason": original})

    assert extra["maui_details"]["reason"] == original
    assert MARKER not in extra["maui_details"]["reason"]


def test_details_string_above_limit_is_truncated_within_bound():
    original = "v" * 5000

    _, extra = build_operational_event(event="e", details={"reason": original})
    bounded = extra["maui_details"]["reason"]

    assert len(bounded) == DETAILS_VALUE_LIMIT
    assert bounded.endswith(MARKER)
    assert bounded[: DETAILS_VALUE_LIMIT - len(MARKER)] == original[
        : DETAILS_VALUE_LIMIT - len(MARKER)
    ]


def test_oversized_message_and_details_in_one_call_do_not_raise():
    _, extra = build_operational_event(
        event="e",
        message="m" * 100000,
        details={"reason": "v" * 100000, "other": "w" * 100000},
    )

    assert len(extra["maui_message"]) == MESSAGE_LIMIT
    assert len(extra["maui_details"]["reason"]) == DETAILS_VALUE_LIMIT
    assert len(extra["maui_details"]["other"]) == DETAILS_VALUE_LIMIT


def test_non_string_details_scalars_are_preserved_exactly():
    _, extra = build_operational_event(
        event="e",
        details={"i": 42, "f": 1.5, "b": True, "nb": False, "n": None},
    )

    details = extra["maui_details"]
    assert details == {"i": 42, "f": 1.5, "b": True, "nb": False, "n": None}
    assert details["b"] is True
    assert details["nb"] is False
    assert details["n"] is None
    assert isinstance(details["i"], int) and not isinstance(details["i"], bool)
    assert isinstance(details["f"], float)


# ---------------------------------------------------------------------------
# 13. Q4 bounding — programmer-contract shape limits
# ---------------------------------------------------------------------------


def test_details_key_count_at_limit_is_accepted():
    details = {f"k{i:02d}": i for i in range(DETAILS_KEY_COUNT_LIMIT)}

    _, extra = build_operational_event(event="e", details=details)

    assert extra["maui_details"] == details


def test_details_key_count_above_limit_raises():
    details = {f"k{i:02d}": i for i in range(DETAILS_KEY_COUNT_LIMIT + 1)}

    with pytest.raises(ValueError):
        build_operational_event(event="e", details=details)


def test_details_key_length_at_limit_is_accepted_verbatim():
    key = "k" * DETAILS_KEY_LENGTH_LIMIT

    _, extra = build_operational_event(event="e", details={key: "x"})

    assert list(extra["maui_details"]) == [key]


def test_details_key_length_above_limit_raises():
    key = "k" * (DETAILS_KEY_LENGTH_LIMIT + 1)

    with pytest.raises(ValueError):
        build_operational_event(event="e", details={key: "x"})


def test_details_keys_are_never_truncated_when_a_value_is_oversized():
    key = "k" * DETAILS_KEY_LENGTH_LIMIT

    _, extra = build_operational_event(
        event="e", details={key: "v" * 5000}
    )

    assert list(extra["maui_details"]) == [key]
    assert MARKER not in key
    assert all(MARKER not in k for k in extra["maui_details"])
    assert len(extra["maui_details"][key]) == DETAILS_VALUE_LIMIT


# ---------------------------------------------------------------------------
# 14. Q4 bounding — validation runs before normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event": "Bad Event"},
        {"event": "e", "details": {"a": {"nested": 1}}},
        {"event": "e", "details": {"a": [1, 2]}},
        {"event": "e", "details": {"a": (1, 2)}},
        {"event": "e", "details": {"a": {1, 2}}},
        {"event": "e", "details": {1: "x"}},
        {"event": "e", "duration_ms": 1.5},
        {"event": "e", "duration_ms": True},
        {"event": "e", "provider": 123},
    ],
)
def test_contract_violations_still_raise_with_oversized_content_present(kwargs):
    call = dict(kwargs)
    call.setdefault("message", "m" * 5000)

    with pytest.raises(ValueError):
        build_operational_event(**call)


def test_key_count_violation_is_not_rescued_by_normalization():
    details = {
        f"k{i:02d}": "v" * 5000 for i in range(DETAILS_KEY_COUNT_LIMIT + 1)
    }

    with pytest.raises(ValueError):
        build_operational_event(
            event="e", message="m" * 5000, details=details
        )


def test_caller_details_dict_is_not_mutated():
    original_value = "v" * 5000
    details = {"reason": original_value, "count": 3}

    _, extra = build_operational_event(event="e", details=details)

    assert details == {"reason": original_value, "count": 3}
    assert details["reason"] is original_value
    assert extra["maui_details"] is not details


# ---------------------------------------------------------------------------
# 15. Q4 bounding — rendered output uses the SAME normalized values
# ---------------------------------------------------------------------------


def test_rendered_message_uses_the_normalized_values():
    message, extra = build_operational_event(
        event="bounded_event",
        message="m" * 5000,
        details={"reason": "v" * 5000, "count": 7},
    )

    bounded_message = extra["maui_message"]
    bounded_reason = extra["maui_details"]["reason"]

    assert message == (
        f"event=bounded_event count=7 reason={bounded_reason} "
        f"msg={bounded_message}"
    )
    assert message.count(MARKER) == 2
    assert len(bounded_message) == MESSAGE_LIMIT
    assert len(bounded_reason) == DETAILS_VALUE_LIMIT


def test_builder_remains_silent_on_oversized_input(caplog):
    with caplog.at_level(logging.DEBUG):
        build_operational_event(
            event="e", message="m" * 100000, details={"a": "v" * 100000}
        )

    assert caplog.records == []
