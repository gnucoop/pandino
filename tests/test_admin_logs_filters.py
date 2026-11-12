"""Usage admin page filters: user / service / source / model / provider,
quick date ranges, page size, and stats that follow the same filters.

Covers the shared SQL filter clause and the builders that use it, the
database reader's page clamping, and the /admin/logs route (validation,
what it passes to the readers, chips, quick ranges, clickable badges).
Uses a bare Flask app with the admin blueprint; the database is faked.
"""

import os
from datetime import date, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from flask import Flask

from infrastructure import database_methods as dm
from infrastructure import database_pg
from routes import admin as admin_route

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(REPO_ROOT, "templates")


# --- shared filter clause & builders --------------------------------------


def test_filter_clause_empty_when_unfiltered():
    clause, params = dm.build_logs_filter_clause()

    assert clause.as_string(None) == ""
    assert params == []


def test_filter_clause_combines_all_filters_with_bound_params():
    clause, params = dm.build_logs_filter_clause(
        "2026-11-01",
        "2026-11-12",
        "ali",
        {
            "user_id": 7,
            "service": "/datachat",
            "source": "dino",
            "model": "gpt-4o",
            "provider": "openai",
        },
    )

    text = clause.as_string(None)
    assert text.startswith("WHERE ")
    for fragment in (
        "l.date::timestamp >= %s::timestamp",
        "u.username ILIKE %s",
        "l.user_id = %s",
        "l.service = %s",
        "l.source = %s",
        "l.model = %s",
        "l.provider = %s",
    ):
        assert fragment in text
    assert params == [
        "2026-11-01",
        "2026-11-12 23:59:59",
        "%ali%",
        "%ali%",
        "%ali%",
        7,
        "/datachat",
        "dino",
        "gpt-4o",
        "openai",
    ]


def test_filter_clause_ignores_unknown_keys_and_empty_values():
    clause, params = dm.build_logs_filter_clause(
        filters={"service": "", "l.id; DROP TABLE logs": "x", "model": None}
    )

    assert clause.as_string(None) == ""
    assert params == []


FILTERS = {"user_id": 3, "service": "/datachat"}


@pytest.mark.parametrize(
    "builder",
    [
        dm.build_get_total_logs_count_query,
        dm.build_get_total_log_stats_query,
        dm.build_get_top_users_by_token_usage_query,
        dm.build_get_daily_log_stats_query,
    ],
)
def test_count_and_stats_builders_apply_the_same_filters(builder):
    query, params = builder("2026-11-01", "2026-11-12", "ali", FILTERS)

    text = query.as_string(None)
    assert "LEFT JOIN" in text and "u.username ILIKE %s" in text
    assert "l.user_id = %s" in text and "l.service = %s" in text
    assert params[-2:] == (3, "/datachat")


def test_list_builder_applies_filters_before_limit_offset():
    query, params = dm.build_get_logs_for_admin_query(
        25, 50, "2026-11-01", "2026-11-12", None, FILTERS
    )

    assert "ORDER BY l.date DESC, l.id DESC" in query.as_string(None)
    assert params[-4:] == (3, "/datachat", 25, 50)


def test_daily_stats_keeps_default_window_with_filters_but_no_dates():
    query, params = dm.build_get_daily_log_stats_query(filters={"model": "m"})

    text = query.as_string(None)
    assert "WHERE l.model = %s AND l.date::timestamp >= CURRENT_DATE - INTERVAL '7 days'" in text
    assert params == ("m",)


def test_filter_values_query_lists_distinct_non_empty_values():
    query, params = dm.build_get_log_filter_values_query(
        "source", "2026-11-01", "2026-11-02"
    )

    text = query.as_string(None)
    assert "SELECT DISTINCT l.source" in text
    assert "l.source IS NOT NULL AND l.source <> ''" in text
    assert params == ("2026-11-01", "2026-11-02 23:59:59")


def test_filter_values_query_rejects_unknown_column():
    with pytest.raises(ValueError):
        dm.build_get_log_filter_values_query("cost")


def test_filter_users_query_joins_users():
    query, _params = dm.build_get_log_filter_users_query("2026-11-01", "2026-11-02")

    text = query.as_string(None)
    assert "SELECT DISTINCT u.id, u.username" in text
    assert "JOIN \"users\" u ON l.user_id = u.id" in text


# --- database reader -------------------------------------------------------


class FakeCursor:
    def __init__(self, total):
        self.total = total
        self.calls = []

    def execute(self, query, params=()):
        self.calls.append(params)

    def fetchone(self):
        return (self.total,)

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def test_get_logs_for_admin_clamps_page_past_end(monkeypatch):
    cursor = FakeCursor(total=120)
    monkeypatch.setattr(database_pg, "connect", lambda: FakeConnection(cursor))

    result = database_pg.get_logs_for_admin(page=99, limit=50, filters=FILTERS)

    assert result["page"] == 3
    assert result["total_pages"] == 3
    # count first (filters only), then the page (filters + limit/offset)
    assert cursor.calls[0] == (3, "/datachat")
    assert cursor.calls[1] == (3, "/datachat", 50, 100)


def test_get_logs_for_admin_empty_result_is_one_page(monkeypatch):
    cursor = FakeCursor(total=0)
    monkeypatch.setattr(database_pg, "connect", lambda: FakeConnection(cursor))

    result = database_pg.get_logs_for_admin(page=4)

    assert result["page"] == 1
    assert result["total_pages"] == 1


# --- route -----------------------------------------------------------------


def _client():
    app = Flask(__name__, template_folder=TEMPLATES_DIR)
    app.secret_key = "test-secret"
    app.register_blueprint(admin_route.admin_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["admin_logged_in"] = True
        sess["admin_username"] = "admin"
    return client


LOG_ROW = {
    "id": 1,
    "user_id": 7,
    "username": "alice",
    "date": "2026-11-10 10:00:00",
    "token_input": 1200,
    "token_output": 300,
    "cost": 0.0123,
    "model": "gpt-4o",
    "provider": "openai",
    "service": "/datachat",
    "source": "dino",
    "request_id": "9bf218009db0127d",
    "duration_ms": 842,
}

OPTIONS = {
    "users": [{"id": 7, "username": "alice"}, {"id": 8, "username": "bob"}],
    "service": ["/agentchat", "/datachat"],
    "source": ["dino"],
    "model": ["gpt-4o"],
    "provider": ["openai"],
}


class Readers:
    def __init__(self, total=1, rows=None):
        self.total = total
        self.rows = [LOG_ROW] if rows is None else rows
        self.logs_calls = []
        self.stats_calls = []
        self.options_calls = []

    def logs(self, page, limit, **kwargs):
        self.logs_calls.append({"page": page, "limit": limit, **kwargs})
        total_pages = max(1, -(-self.total // limit))
        return {
            "logs": self.rows,
            "page": min(page, total_pages),
            "total_pages": total_pages,
            "total_count": self.total,
        }

    def stats(self, **kwargs):
        self.stats_calls.append(kwargs)
        return {
            "total_requests": self.total,
            "total_input": 1200,
            "total_output": 300,
            "total_cost": 0.0123,
            "daily_stats": [],
            "top_users": [],
        }

    def options(self, start_date, end_date):
        self.options_calls.append((start_date, end_date))
        return {k: list(v) for k, v in OPTIONS.items()}


@pytest.fixture
def readers(monkeypatch):
    def install(**kwargs):
        r = Readers(**kwargs)
        monkeypatch.setattr(admin_route, "get_logs_for_admin", r.logs)
        monkeypatch.setattr(admin_route, "get_logs_stats", r.stats)
        monkeypatch.setattr(admin_route, "get_log_filter_options", r.options)
        return r

    return install


def _today():
    return date.today()


def test_default_view_is_last_seven_days_unfiltered(readers):
    r = readers()

    resp = _client().get("/admin/logs")

    assert resp.status_code == 200
    call = r.logs_calls[0]
    assert call["page"] == 1 and call["limit"] == 50
    assert call["end_date"] == _today().isoformat()
    assert call["start_date"] == (_today() - timedelta(days=6)).isoformat()
    assert call["search"] is None and call["filters"] == {}
    body = resp.get_data(as_text=True)
    assert "(default: last 7 days)" in body


def test_table_and_stats_receive_identical_filters(readers):
    r = readers()

    _client().get(
        "/admin/logs?start_date=2026-11-01&end_date=2026-11-12&user=7"
        "&service=/datachat&source=dino&model=gpt-4o&provider=openai&search=ali"
    )

    expected = {
        "start_date": "2026-11-01",
        "end_date": "2026-11-12",
        "search": "ali",
        "filters": {
            "user_id": 7,
            "service": "/datachat",
            "source": "dino",
            "model": "gpt-4o",
            "provider": "openai",
        },
    }
    logs_call = {k: v for k, v in r.logs_calls[0].items() if k not in ("page", "limit")}
    assert logs_call == expected
    assert r.stats_calls == [expected]
    assert r.options_calls == [("2026-11-01", "2026-11-12")]


@pytest.mark.parametrize(
    "query,start,end",
    [
        ("start_date=junk&end_date=2026-13-45", None, None),
        ("start_date=2026-11-12&end_date=2026-11-01", "2026-11-01", "2026-11-12"),
        ("end_date=2026-11-10", "2026-11-04", "2026-11-10"),
    ],
)
def test_dates_are_validated(readers, query, start, end):
    r = readers()

    _client().get("/admin/logs?" + query)

    call = r.logs_calls[0]
    if start is None:
        assert call["end_date"] == _today().isoformat()
        assert call["start_date"] == (_today() - timedelta(days=6)).isoformat()
    else:
        assert (call["start_date"], call["end_date"]) == (start, end)


@pytest.mark.parametrize(
    "query",
    ["user=abc", "user=-2", "user=0", "service=" + "x" * 201, "model=%20%20"],
)
def test_invalid_filter_values_are_ignored(readers, query):
    r = readers()

    _client().get("/admin/logs?" + query)

    assert r.logs_calls[0]["filters"] == {}


@pytest.mark.parametrize(
    "query,page,limit",
    [("page=x&per_page=y", 1, 50), ("page=-1&per_page=7", 1, 50), ("page=3&per_page=100", 3, 100)],
)
def test_page_and_page_size_are_validated(readers, query, page, limit):
    r = readers(total=1000)

    _client().get("/admin/logs?" + query)

    assert (r.logs_calls[0]["page"], r.logs_calls[0]["limit"]) == (page, limit)


def test_active_filter_chips_remove_only_their_own_filter(readers):
    readers()

    body = _client().get(
        "/admin/logs?user=7&service=/datachat&per_page=100"
    ).get_data(as_text=True)

    assert "User: alice" in body
    assert "Service: /datachat" in body
    # the Service chip's remove link keeps the user filter and page size
    assert 'href="/admin/logs?user=7&amp;per_page=100" class="ms-1" aria-label="Remove Service filter"' in body


def test_selected_value_missing_from_options_stays_selectable(readers):
    readers()

    body = _client().get("/admin/logs?service=/gone&user=99").get_data(as_text=True)

    assert '<option value="/gone" selected>/gone</option>' in body
    assert '<option value="99" selected>#99</option>' in body


def test_quick_ranges_keep_filters_and_mark_active(readers):
    readers()
    today = _today()

    body = _client().get("/admin/logs?model=gpt-4o").get_data(as_text=True)

    seven = (today - timedelta(days=6)).isoformat()
    assert (
        f'href="/admin/logs?model=gpt-4o&amp;start_date={seven}'
        f'&amp;end_date={today.isoformat()}" class="btn btn-primary">Last 7 days</a>'
    ) in body
    month_start = today.replace(day=1).isoformat()
    assert f"start_date={month_start}" in body


def test_row_badges_link_to_filtered_view(readers):
    readers()

    body = _client().get("/admin/logs?source=dino").get_data(as_text=True)

    assert 'href="/admin/logs?source=dino&amp;service=/datachat"' in body
    assert 'href="/admin/logs?source=dino&amp;user=7"' in body
    assert 'href="/admin/logs?source=dino&amp;model=gpt-4o"' in body


def test_na_values_are_not_links(readers):
    readers(rows=[{**LOG_ROW, "service": "N/A", "username": "Unknown", "user_id": None}])

    body = _client().get("/admin/logs").get_data(as_text=True)

    assert "service=N/A" not in body
    assert '<span class="badge bg-light text-muted border">N/A</span>' in body


def test_pagination_links_keep_filters(readers):
    readers(total=120)

    body = _client().get("/admin/logs?service=/datachat&page=2").get_data(as_text=True)

    assert "Page 2 of 3" in body
    assert "Showing 51&ndash;51 of 120 records" in body
    assert 'href="/admin/logs?page=3&amp;service=/datachat"' in body


def test_empty_filtered_result_offers_reset(readers):
    readers(total=0, rows=[])

    body = _client().get("/admin/logs?model=nope").get_data(as_text=True)

    assert "No usage records match these filters" in body


def test_reader_failure_renders_empty_page_with_filters_kept(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(admin_route, "get_logs_for_admin", boom)

    resp = _client().get("/admin/logs?service=/datachat")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "No logs found" in body
    assert '<option value="/datachat" selected>/datachat</option>' in body


def test_badge_class_is_stable_and_neutral_for_missing_values():
    assert admin_route.badge_class("/datachat") == admin_route.badge_class("/datachat")
    assert admin_route.badge_class("N/A") == "bg-light text-muted border"


def test_redirect_link_parse_roundtrip(readers):
    """Filter links carry exact values, including characters that need escaping."""
    readers(rows=[{**LOG_ROW, "model": "org/model:7b & co"}])

    body = _client().get("/admin/logs").get_data(as_text=True)

    start = body.index('href="/admin/logs?model=')
    href = body[start + 6 : body.index('"', start + 6)].replace("&amp;", "&")
    assert parse_qs(urlparse(href).query)["model"] == ["org/model:7b & co"]
