"""Usage → Operational Admin drill-down: timeline template presentation.

Renders templates/admin/operational_timeline.html through a bare Jinja2
environment, stubbing only the globals admin/base.html references - the same
seam as tests/test_admin_logs_template_service_column.py, which records that
this repo has no Flask/template test seam for admin templates to extend.
"""

import re
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = "templates"

REQUEST_ID = "9bf218009db0127d"


def _render(events, request_id=REQUEST_ID, read_failed=False):
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
    env.globals["url_for"] = lambda *a, **k: "#"
    env.globals["get_flashed_messages"] = lambda **k: []
    env.globals["session"] = {"admin_logged_in": True, "admin_username": "admin"}
    env.globals["request"] = SimpleNamespace(endpoint="admin_operational_timeline")

    template = env.get_template("admin/operational_timeline.html")
    return template.render(
        events=events, request_id=request_id, read_failed=read_failed
    )


def _event(**overrides):
    event = {
        "event_time": "2026-08-27 14:22:31.482",
        "level": "INFO",
        "logger": "routes.multimodal",
        "event": "transcribe_started",
        "app_id": "app-1",
        "provider": "openai",
        "model": "whisper-1",
        "duration_ms": 842,
        "error_type": None,
        "details": None,
        "message": None,
    }
    event.update(overrides)
    return event


# --- 1. Page frame ---


def test_page_title_is_operational_timeline():
    assert "Operational Timeline - Admin Panel" in _render([_event()])


def test_request_id_is_rendered_once_at_page_level_in_code():
    html = _render([_event(), _event()])

    assert f"<code>{REQUEST_ID}</code>" in html
    assert html.count(REQUEST_ID) == 1


def test_back_link_to_usage_is_present():
    html = _render([_event()])

    assert "fa-arrow-left" in html
    assert "Back to Usage" in html
    assert "btn btn-outline-secondary" in html


def test_table_uses_existing_admin_conventions():
    html = _render([_event()])

    assert "table-responsive" in html
    assert "table table-hover" in html


# --- 2. Columns ---


def test_all_six_headers_are_present_in_the_ratified_order():
    html = _render([_event()])

    indexes = [
        html.index(f"<th>{header}</th>")
        for header in ("Time", "Level", "Event", "Context", "Duration", "Details")
    ]
    assert indexes == sorted(indexes)


def test_no_id_column_is_displayed():
    html = _render([_event()])

    assert "<th>ID</th>" not in html
    assert "<th>Id</th>" not in html


def test_event_time_is_rendered():
    assert "2026-08-27 14:22:31.482" in _render([_event()])


def test_event_name_is_rendered_as_code():
    assert "<code>transcribe_started</code>" in _render([_event()])


def test_rows_are_rendered_in_the_order_supplied():
    html = _render([_event(event="first_event"), _event(event="second_event")])

    assert html.index("first_event") < html.index("second_event")


# --- 3. Level badges ---


def test_error_level_uses_danger_badge():
    assert "badge bg-danger" in _render([_event(level="ERROR")])


def test_critical_level_uses_danger_badge():
    assert "badge bg-danger" in _render([_event(level="CRITICAL")])


def test_warning_level_uses_warning_badge():
    assert "badge bg-warning" in _render([_event(level="WARNING")])


def test_info_level_uses_info_badge():
    assert "badge bg-info" in _render([_event(level="INFO")])


def test_unknown_level_falls_back_to_secondary_badge():
    html = _render([_event(level="DEBUG")])

    assert "badge bg-secondary" in html
    assert "DEBUG" in html


# --- 4. Context column ---


def test_context_renders_present_fields():
    html = _render([_event()])

    assert "openai" in html
    assert "whisper-1" in html
    assert "routes.multimodal" in html
    assert "app-1" in html


def test_context_omits_absent_fields_without_na_placeholders():
    html = _render(
        [
            _event(
                logger=None,
                app_id=None,
                provider=None,
                model=None,
                error_type=None,
                duration_ms=842,
            )
        ]
    )

    # The only N/A-free row still has a real duration, so no N/A should appear.
    assert "N/A" not in html


def test_error_type_is_rendered_when_present():
    html = _render([_event(level="ERROR", error_type="TimeoutError")])

    assert "TimeoutError" in html


# --- 5. Duration rendering ---


def test_duration_none_renders_na():
    html = _render([_event(duration_ms=None)])

    assert "N/A" in html


def test_duration_zero_renders_zero_ms_and_is_not_treated_as_missing():
    html = _render([_event(duration_ms=0)])

    assert "0 ms" in html
    assert "N/A" not in html


def test_duration_below_one_second_renders_integer_milliseconds():
    assert "842 ms" in _render([_event(duration_ms=842)])


def test_duration_at_or_above_one_second_renders_seconds_with_one_decimal():
    assert "18.3 s" in _render([_event(duration_ms=18308)])


def test_duration_uses_the_same_rounding_as_the_usage_page():
    """The Operational Timeline reuses the existing Admin Usage
    duration-formatting semantics exactly; this slice introduces no new
    rounding policy. 18250 ms is an exact .5 tie at one decimal, and "%.1f"
    rounds half-to-even, so it renders 18.2 s - the same result
    templates/admin/logs.html produces for the same field. Pinned so that any
    future change to rounding is a deliberate choice applied to both pages."""
    assert "18.2 s" in _render([_event(duration_ms=18250)])


def test_missing_duration_never_renders_as_zero():
    html = _render([_event(duration_ms=None)])

    assert re.search(r"^\s*0 ms\s*$", html, re.MULTILINE) is None
    assert re.search(r"^\s*0\.0 s\s*$", html, re.MULTILINE) is None


# --- 6. Details rendering ---


def test_details_render_as_flat_key_value_pairs():
    html = _render([_event(details={"branch": "audio", "reason": "missing_model"})])

    assert "<code>branch</code>=audio" in html
    assert "<code>reason</code>=missing_model" in html


def test_details_keys_are_rendered_in_deterministic_sorted_order():
    html = _render([_event(details={"zeta": 1, "alpha": 2, "mid": 3})])

    assert html.index("alpha") < html.index("mid") < html.index("zeta")


def test_details_are_not_rendered_as_a_serialized_json_blob():
    html = _render([_event(details={"branch": "audio"})])

    assert '{"branch"' not in html
    assert "{'branch'" not in html
    assert "<pre" not in html


def test_message_is_rendered_above_the_structured_pairs():
    html = _render(
        [_event(message="Transcription started", details={"branch": "audio"})]
    )

    assert html.index("Transcription started") < html.index("<code>branch</code>")


def test_absent_message_and_details_render_nothing_extra():
    html = _render([_event(message=None, details=None)])

    assert "None" not in html


def test_details_values_are_autoescaped():
    html = _render([_event(details={"reason": "<script>x</script>"})])

    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


# --- 7. Empty and failure states ---


def test_valid_empty_timeline_renders_a_neutral_empty_state():
    html = _render([])

    assert "No operational events" in html
    assert "alert-danger" not in html


def test_valid_empty_state_does_not_claim_failure_or_expectation():
    """The empty timeline is a legitimate outcome of intentionally selective
    Operational coverage, not an error, so the copy must not imply one."""
    html = _render([])
    empty_state = html[html.index("No operational events") :]
    empty_state = empty_state[: empty_state.index("</div>")].lower()

    for forbidden in ("failed", "failure", "broken", "expected", "error"):
        assert forbidden not in empty_state, forbidden
    assert "Unable to load" not in html


def test_failure_state_renders_a_danger_alert():
    html = _render([], read_failed=True)

    assert "alert alert-danger" in html
    assert "Unable to load the operational timeline for this request." in html


def test_failure_state_does_not_render_the_table_or_the_empty_state():
    html = _render([], read_failed=True)

    assert "<th>Time</th>" not in html
    assert "No operational events" not in html


def test_failure_state_wins_over_present_rows():
    html = _render([_event()], read_failed=True)

    assert "alert alert-danger" in html
    assert "<th>Time</th>" not in html


def test_empty_state_and_failure_state_produce_different_markup():
    assert _render([]) != _render([], read_failed=True)


# --- 8. No new frontend architecture ---


def test_template_adds_no_script_or_style_of_its_own():
    """admin/base.html contributes the shared Bootstrap bundle and the sidebar
    CSS; this page must introduce neither JS nor CSS of its own."""
    with open("templates/admin/operational_timeline.html", encoding="utf-8") as fh:
        source = fh.read()

    assert "<script" not in source
    assert "<style" not in source
    assert "onclick" not in source
    assert "data-bs-toggle" not in source
