"""Server-side pagination of the Admin "RAG Files" indexed-files list.

Covers the SQL builders (LIMIT/OFFSET + COUNT), the database reader's page
clamping, the route's validation of ``page`` / ``per_page``, the rendered
controls, and that delete/upload redirects keep the viewer's page.

Uses a bare Flask app with the admin blueprint registered; the database layer
is faked, so no live PostgreSQL is involved.
"""

import os
from datetime import datetime

import pytest
from flask import Flask

from infrastructure import database_methods, database_pg
from routes import admin as admin_route

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(REPO_ROOT, "templates")


# --- SQL builders ----------------------------------------------------------


def test_page_query_uses_limit_offset_and_stable_order():
    query, params = database_methods.build_get_rag_files_page_query(25, 50)

    text = str(query)
    assert "FROM rag_files" in text
    assert "ORDER BY created_at DESC, id ASC" in text
    assert "LIMIT %s OFFSET %s" in text
    assert params == (25, 50)


def test_count_query_counts_rag_files():
    query, params = database_methods.build_count_rag_files_query()

    assert "SELECT COUNT(*) FROM rag_files" in str(query)
    assert params == ()


# --- database reader -------------------------------------------------------


class FakeCursor:
    def __init__(self, total, rows=None):
        self.total = total
        self.rows = rows or []
        self.calls = []

    def execute(self, query, params=()):
        self.calls.append((query, params))

    def fetchone(self):
        return (self.total,)

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


@pytest.fixture
def fake_db(monkeypatch):
    def install(total, rows=None):
        cursor = FakeCursor(total, rows)
        conn = FakeConnection(cursor)
        monkeypatch.setattr(database_pg, "connect", lambda: conn)
        monkeypatch.setattr(
            database_pg, "build_count_rag_files_query", lambda: ("count", ())
        )
        monkeypatch.setattr(
            database_pg,
            "build_get_rag_files_page_query",
            lambda limit, offset: ("page", (limit, offset)),
        )
        return cursor, conn

    return install


def test_reader_pages_with_limit_offset(fake_db):
    row = ("f1", "a.pdf", "docs", 3, None, datetime(2026, 1, 2, 3, 4))
    cursor, conn = fake_db(total=60, rows=[row])

    result = database_pg.get_rag_files_page(page=2, per_page=25)

    assert cursor.calls == [("count", ()), ("page", (25, 25))]
    assert result["page"] == 2
    assert result["per_page"] == 25
    assert result["total_count"] == 60
    assert result["total_pages"] == 3
    assert result["rag_files"] == [
        {
            "id": "f1",
            "file_name": "a.pdf",
            "namespace": "docs",
            "chunk_count": 3,
            "language": "-",
            "created_at": "2026-01-02 03:04",
        }
    ]
    assert conn.closed is True


def test_reader_clamps_page_past_end_to_last_page(fake_db):
    cursor, _conn = fake_db(total=51)

    result = database_pg.get_rag_files_page(page=99, per_page=25)

    assert result["page"] == 3
    assert result["total_pages"] == 3
    assert cursor.calls[-1] == ("page", (25, 50))


def test_reader_empty_table_skips_page_query(fake_db):
    cursor, _conn = fake_db(total=0)

    result = database_pg.get_rag_files_page(page=5, per_page=10)

    assert cursor.calls == [("count", ())]
    assert result == {
        "rag_files": [],
        "page": 1,
        "per_page": 10,
        "total_count": 0,
        "total_pages": 1,
    }


# --- route -----------------------------------------------------------------


def _make_app() -> Flask:
    app = Flask(__name__, template_folder=TEMPLATES_DIR)
    app.secret_key = "test-secret"
    app.register_blueprint(admin_route.admin_bp)
    return app


def _client():
    app = _make_app()
    # The upload route reads the config before validating the form.
    app.config["MAUI_CONFIG"] = object()
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["admin_logged_in"] = True
        sess["admin_username"] = "admin"
    return client


def _files(n, start=0):
    return [
        {
            "id": f"id-{i}",
            "file_name": f"file-{i}.pdf",
            "namespace": "docs",
            "chunk_count": 1,
            "language": "-",
            "created_at": "2026-01-01 00:00",
        }
        for i in range(start, start + n)
    ]


class RecordingReader:
    """Mimics get_rag_files_page over a table of ``total`` rows."""

    def __init__(self, total):
        self.total = total
        self.calls = []

    def __call__(self, page, per_page):
        self.calls.append((page, per_page))
        total_pages = max(1, -(-self.total // per_page))
        page = min(page, total_pages)
        start = (page - 1) * per_page
        count = max(0, min(per_page, self.total - start))
        return {
            "rag_files": _files(count, start),
            "page": page,
            "per_page": per_page,
            "total_count": self.total,
            "total_pages": total_pages,
        }


@pytest.fixture
def reader(monkeypatch):
    def install(total):
        r = RecordingReader(total)
        monkeypatch.setattr(admin_route, "get_rag_files_page", r)
        return r

    return install


def test_default_page_and_page_size(reader):
    r = reader(total=60)

    resp = _client().get("/admin/rag-files")

    assert resp.status_code == 200
    assert r.calls == [(1, admin_route.RAG_FILES_DEFAULT_PER_PAGE)]
    body = resp.get_data(as_text=True)
    assert "Page 1 of 3" in body
    assert "Showing 1&ndash;25 of 60 files" in body
    assert "file-0.pdf" in body and "file-25.pdf" not in body
    assert "/admin/rag-files?page=2&amp;per_page=25" in body


@pytest.mark.parametrize(
    "query,expected",
    [
        ("?page=abc&per_page=xyz", (1, 25)),
        ("?page=0", (1, 25)),
        ("?page=-3", (1, 25)),
        ("?per_page=7", (1, 25)),
        ("?per_page=100000", (1, 25)),
        ("?per_page=0", (1, 25)),
        ("?page=2&per_page=10", (2, 10)),
    ],
)
def test_invalid_params_fall_back_to_defaults(reader, query, expected):
    r = reader(total=60)

    resp = _client().get("/admin/rag-files" + query)

    assert resp.status_code == 200
    assert r.calls == [expected]


def test_page_past_end_shows_last_page(reader):
    reader(total=30)

    body = _client().get("/admin/rag-files?page=50&per_page=10").get_data(
        as_text=True
    )

    assert "Page 3 of 3" in body
    assert "Showing 21&ndash;30 of 30 files" in body


def test_single_page_hides_pager_but_shows_count(reader):
    reader(total=3)

    body = _client().get("/admin/rag-files").get_data(as_text=True)

    assert 'aria-label="Indexed files navigation"' not in body
    assert "Showing 1&ndash;3 of 3 files" in body


def test_empty_state(reader):
    reader(total=0)

    body = _client().get("/admin/rag-files").get_data(as_text=True)

    assert "No RAG files found" in body


def test_reader_failure_renders_empty_state(monkeypatch):
    def boom(page, per_page):
        raise RuntimeError("db down")

    monkeypatch.setattr(admin_route, "get_rag_files_page", boom)

    resp = _client().get("/admin/rag-files")

    assert resp.status_code == 200
    assert "No RAG files found" in resp.get_data(as_text=True)


def test_delete_forms_carry_current_page(reader):
    reader(total=60)

    body = _client().get("/admin/rag-files?page=2&per_page=10").get_data(
        as_text=True
    )

    assert '<input type="hidden" name="page" value="2">' in body
    assert '<input type="hidden" name="per_page" value="10">' in body


def test_delete_redirects_back_to_same_page(monkeypatch):
    calls = []

    def fake_delete(file_id, namespace):
        calls.append((file_id, namespace))
        return {"row_deleted": True, "chunks_deleted": 2}

    monkeypatch.setattr(admin_route, "delete_rag_file", fake_delete)

    resp = _client().post(
        "/admin/rag-files/delete",
        data={"file_id": "id-1", "namespace": "docs", "page": "3", "per_page": "10"},
    )

    assert calls == [("id-1", "docs")]
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/admin/rag-files?page=3&per_page=10")


def test_delete_with_default_paging_redirects_to_bare_url(monkeypatch):
    monkeypatch.setattr(
        admin_route,
        "delete_rag_file",
        lambda file_id, namespace: {"row_deleted": False, "chunks_deleted": 0},
    )

    resp = _client().post(
        "/admin/rag-files/delete",
        data={"file_id": "id-1", "namespace": "docs", "page": "junk"},
    )

    assert resp.headers["Location"].endswith("/admin/rag-files")


def test_upload_validation_failure_keeps_page_size():
    resp = _client().post("/admin/rag-files/upload", data={"per_page": "50"})

    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/admin/rag-files?per_page=50")


def test_delete_confirm_escapes_file_name(monkeypatch):
    nasty = "it's \"bad\" </form><script>alert(1)</script>.pdf"
    files = _files(1)
    files[0]["file_name"] = nasty
    monkeypatch.setattr(
        admin_route,
        "get_rag_files_page",
        lambda page, per_page: {
            "rag_files": files,
            "page": 1,
            "per_page": per_page,
            "total_count": 1,
            "total_pages": 1,
        },
    )

    body = _client().get("/admin/rag-files").get_data(as_text=True)

    # tojson escapes ', <, > and & as \uXXXX, so the single-quoted attribute
    # cannot be closed and no markup is injected.
    assert (
        "onsubmit='return confirm(\"Delete it\\u0027s \\\"bad\\\" "
        "\\u003c/form\\u003e\\u003cscript\\u003ealert(1)"
        "\\u003c/script\\u003e.pdf and all its indexed chunks?\");'"
    ) in body
    assert "<script>alert(1)" not in body
