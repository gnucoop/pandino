"""
Tests for analysis run storage: the query builders, the database_pg wrappers and
the service that wraps them. No database is touched; psycopg's Jsonb wrapper is
checked for the JSONB columns because a bare dict would fail to adapt at insert time.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
from psycopg.types.json import Jsonb

import infrastructure.database_pg as database_pg
import services.analysis_report_service as service
from infrastructure.database_methods import (
    build_get_analysis_report_html_query,
    build_get_analysis_report_query,
    build_get_previous_analysis_report_query,
    build_list_analysis_reports_query,
    build_upsert_analysis_report_query,
)

STARTED = datetime(2026, 10, 20, 8, 0, tzinfo=timezone.utc)


def upsert(**overrides):
    row = dict(
        run_id="r1", brief_id=7, user_email="a@b.c", title="T", status="running",
        metadata={"status": "running"}, plan={"title": "T"}, state=None, html=None,
        started_at=STARTED, finished_at=None,
    )
    row.update(overrides)
    return build_upsert_analysis_report_query(**row)


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def test_the_upsert_replaces_the_row_of_the_run():
    query, params = upsert()

    text = query.as_string(None)
    assert text.startswith('INSERT INTO "analysis_reports"')
    assert 'ON CONFLICT ("run_id") DO UPDATE SET' in text
    assert '"html" = EXCLUDED."html"' in text and '"updated_at" = NOW()' in text
    assert '"run_id" = EXCLUDED' not in text
    assert params[0] == "r1"
    assert isinstance(params[5], Jsonb) and isinstance(params[6], Jsonb)
    assert params[7] is None  # no state yet: NULL, not a JSON null


def test_the_report_is_read_for_its_owner_only():
    query, params = build_get_analysis_report_html_query("r1", "a@b.c")

    assert query.as_string(None) == (
        'SELECT "html" FROM "analysis_reports" WHERE "run_id" = %s AND "user_email" = %s'
    )
    assert params == ("r1", "a@b.c")


def test_the_run_is_read_whole_for_its_owner_only():
    query, params = build_get_analysis_report_query("r1", "a@b.c")

    text = query.as_string(None)
    assert text.startswith('SELECT "id", "run_id", "brief_id", "user_email", "title", "status"')
    assert '"metadata", "plan", "state", "html"' in text
    assert text.endswith(
        '"created_at", "updated_at" FROM "analysis_reports" WHERE "run_id" = %s AND "user_email" = %s'
    )
    assert params == ("r1", "a@b.c")


def test_the_list_never_reads_the_report_and_filters_by_brief():
    query, params = build_list_analysis_reports_query("a@b.c", brief_id=7, limit=5)

    text = query.as_string(None)
    assert '"html" IS NOT NULL' in text and 'SELECT "html"' not in text
    assert "\"metadata\" -> 'metrics'" in text
    assert 'ORDER BY "created_at" DESC' in text
    assert params == ("a@b.c", 7, 5)
    assert build_list_analysis_reports_query("a@b.c")[1] == ("a@b.c", 50)


def test_the_previous_run_is_the_latest_completed_other_run_of_the_brief():
    query, params = build_get_previous_analysis_report_query(7, "r2")

    text = query.as_string(None)
    assert '"brief_id" = %s AND "status" = %s AND "run_id" <> %s' in text
    assert 'ORDER BY "finished_at" DESC NULLS LAST LIMIT 1' in text
    assert params == (7, "completed", "r2")


# ---------------------------------------------------------------------------
# database_pg wrappers
# ---------------------------------------------------------------------------


def test_saving_commits(monkeypatch):
    conn = MagicMock()
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    database_pg.save_analysis_report(
        "r1", 7, "a@b.c", "T", "running", {}, None, None, None, STARTED, None
    )

    conn.commit.assert_called_once()
    conn.close.assert_called_once()


def test_saving_rolls_back_on_failure(monkeypatch):
    conn = MagicMock()
    conn.cursor.return_value.execute.side_effect = RuntimeError("boom")
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError):
        database_pg.save_analysis_report(
            "r1", 7, "a@b.c", "T", "running", {}, None, None, None, STARTED, None
        )

    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


def test_a_missing_report_or_previous_run_is_none(monkeypatch):
    conn = MagicMock()
    conn.cursor.return_value.fetchone.return_value = None
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    assert database_pg.get_analysis_report_html("r1", "a@b.c") is None
    assert database_pg.get_analysis_report("r1", "a@b.c") is None
    assert database_pg.get_previous_analysis_report_metadata(7, "r2") is None


def test_the_run_row_becomes_a_dictionary(monkeypatch):
    conn = MagicMock()
    cursor = conn.cursor.return_value
    cursor.fetchone.return_value = ("r1", "completed", "<html></html>")
    cursor.description = [("run_id",), ("status",), ("html",)]
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    assert database_pg.get_analysis_report("r1", "a@b.c") == {
        "run_id": "r1", "status": "completed", "html": "<html></html>",
    }
    conn.close.assert_called_once()


def test_the_list_rows_become_dictionaries(monkeypatch):
    conn = MagicMock()
    conn.cursor.return_value.fetchall.return_value = [
        ("r1", 7, "T", "completed", STARTED, STARTED, {"metrics": {"x": 1}}, True),
    ]
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    assert database_pg.list_analysis_reports("a@b.c") == [{
        "run_id": "r1", "brief_id": 7, "title": "T", "status": "completed",
        "started_at": STARTED.isoformat(), "finished_at": STARTED.isoformat(),
        "metrics": {"metrics": {"x": 1}}, "has_report": True,
    }]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def test_the_service_stores_plain_json_and_timestamps(monkeypatch):
    saved = {}
    monkeypatch.setattr(service, "save_analysis_report", lambda **kwargs: saved.update(kwargs))

    service.save_report(
        run_id="r1", brief_id=7, user_email="a@b.c", title="T", status="completed",
        metadata={"tokens": np.int64(5)}, state={"outputs": {"total": np.float64(2.5)}},
        html="<html></html>", started_at=STARTED.isoformat(), finished_at=None,
    )

    assert saved["metadata"] == {"tokens": "5"}  # numpy values kept as text
    assert saved["started_at"] == STARTED and saved["finished_at"] is None
    assert saved["plan"] is None and saved["html"] == "<html></html>"


def test_the_service_wraps_storage_errors(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("db down")

    for name in ("save_analysis_report", "get_analysis_report_html", "get_analysis_report",
                 "list_analysis_reports", "get_previous_analysis_report_metadata"):
        monkeypatch.setattr(service, name, broken)

    with pytest.raises(RuntimeError, match="db down"):
        service.save_report(run_id="r1", brief_id=7, user_email="a", title=None, status="s", metadata={})
    with pytest.raises(RuntimeError):
        service.get_report_html("r1", "a")
    with pytest.raises(RuntimeError):
        service.get_report("r1", "a")
    with pytest.raises(RuntimeError):
        service.list_reports("a")
    with pytest.raises(RuntimeError):
        service.previous_report_metadata(7, "r1")
