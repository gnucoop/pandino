"""Usage Admin Visibility: admin logs template shows Service, Source,
Request ID, Duration columns, and the Usage terminology labels.

Renders templates/admin/logs.html directly through a bare Jinja2 environment
(no Flask app/blueprint/session machinery), stubbing only the handful of
globals admin/base.html references (url_for, session, request,
get_flashed_messages). This repo has no existing Flask/template test seam for
admin templates to extend, so this is the lightest verification available.
"""

import re
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = "templates"


def _render_logs_template(logs):
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    env.globals["url_for"] = lambda *a, **k: "#"
    env.globals["get_flashed_messages"] = lambda **k: []
    env.globals["session"] = {"admin_logged_in": True, "admin_username": "admin"}
    env.globals["request"] = SimpleNamespace(endpoint="admin_logs")

    template = env.get_template("admin/logs.html")
    return template.render(
        logs=logs,
        stats=None,
        pagination=None,
        current_start_date="",
        current_end_date="",
        current_search="",
    )


def test_service_column_header_is_present():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "/datachat",
            "request_id": "9bf218009db0127d",
            "duration_ms": 18308,
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "<th>Service</th>" in html


def test_service_row_value_is_displayed_for_non_null_service():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "/datachat",
            "request_id": "9bf218009db0127d",
            "duration_ms": 18308,
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "/datachat" in html


def test_service_row_value_displays_n_a_for_historical_null_service():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "N/A",
            "request_id": "N/A",
            "duration_ms": "N/A",
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "N/A" in html


_SAMPLE_LOG = {
    "id": 1,
    "user_id": 10,
    "username": "alice",
    "date": "2026-08-12 00:00:00",
    "token_input": 5,
    "token_output": 3,
    "cost": 0.01,
    "model": "gpt-4",
    "provider": "openai",
    "service": "/agentchat",
    "source": "dino",
    "request_id": "9bf218009db0127d",
    "duration_ms": 18308,
}


def test_source_column_header_is_present():
    html = _render_logs_template(logs=[_SAMPLE_LOG])

    assert "<th>Source</th>" in html


def test_source_row_value_is_displayed_for_non_null_source():
    html = _render_logs_template(logs=[_SAMPLE_LOG])

    assert "dino" in html


def test_source_row_value_displays_n_a_for_historical_null_source():
    logs = [
        {
            **_SAMPLE_LOG,
            "source": "N/A",
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "N/A" in html


def test_column_order_is_service_then_source_then_request_id_then_duration():
    html = _render_logs_template(logs=[_SAMPLE_LOG])

    service_index = html.index("<th>Service</th>")
    source_index = html.index("<th>Source</th>")
    request_id_index = html.index("<th>Request ID</th>")
    duration_index = html.index("<th>Duration</th>")

    assert service_index < source_index < request_id_index < duration_index


def test_request_id_column_header_is_present():
    html = _render_logs_template(logs=[_SAMPLE_LOG])

    assert "<th>Request ID</th>" in html


def test_full_request_id_is_rendered_without_truncation():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "/agentchat",
            "request_id": "9bf218009db0127d",
            "duration_ms": 18308,
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "9bf218009db0127d" in html


def test_request_id_displays_n_a_for_historical_null_rows():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "N/A",
            "request_id": "N/A",
            "duration_ms": "N/A",
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "<td><code>N/A</code></td>" in html


def test_duration_column_header_is_present():
    html = _render_logs_template(logs=[_SAMPLE_LOG])

    assert "<th>Duration</th>" in html


def test_duration_at_or_above_one_second_renders_as_seconds_with_one_decimal():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "/agentchat",
            "request_id": "9bf218009db0127d",
            "duration_ms": 18308,
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "18.3 s" in html


def test_duration_below_one_second_renders_as_integer_milliseconds():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "/agentchat",
            "request_id": "9bf218009db0127d",
            "duration_ms": 842,
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "842 ms" in html


def test_missing_duration_displays_n_a_and_never_zero():
    logs = [
        {
            "id": 1,
            "user_id": 10,
            "username": "alice",
            "date": "2026-08-12 00:00:00",
            "token_input": 5,
            "token_output": 3,
            "cost": 0.01,
            "model": "gpt-4",
            "provider": "openai",
            "service": "N/A",
            "request_id": "N/A",
            "duration_ms": "N/A",
        }
    ]

    html = _render_logs_template(logs=logs)

    assert re.search(r"^\s*0 ms\s*$", html, re.MULTILINE) is None
    assert re.search(r"^\s*0\.0 s\s*$", html, re.MULTILINE) is None


def test_page_title_contains_usage():
    html = _render_logs_template(logs=[])

    assert "Usage - Admin Panel" in html


def test_card_caption_is_recent_usage():
    html = _render_logs_template(logs=[])

    assert "Recent Usage" in html


def test_sidebar_nav_label_is_usage():
    html = _render_logs_template(logs=[])

    assert "fa-list me-2\"></i> Usage" in html
