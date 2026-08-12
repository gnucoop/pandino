"""Usage Service Slice B: admin logs template shows the Service column.

Renders templates/admin/logs.html directly through a bare Jinja2 environment
(no Flask app/blueprint/session machinery), stubbing only the handful of
globals admin/base.html references (url_for, session, request,
get_flashed_messages). This repo has no existing Flask/template test seam for
admin templates to extend, so this is the lightest verification available.
"""

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
        }
    ]

    html = _render_logs_template(logs=logs)

    assert "N/A" in html
