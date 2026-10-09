"""
Tests for analysis brief storage: the query builders and the service that wraps
them. No database is touched; psycopg's Jsonb wrapper is checked for the JSONB
columns because a bare dict would fail to adapt at insert time.
"""

from unittest.mock import MagicMock

import pytest
from psycopg.types.json import Jsonb

import infrastructure.database_pg as database_pg
import services.analysis_brief_service as service
from infrastructure.database_methods import (
    build_get_analysis_brief_query,
    build_insert_analysis_brief_query,
    build_list_analysis_briefs_query,
)

BRIEF = {"name": "Revenue by region", "mission": "..."}


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def test_insert_wraps_the_json_columns_and_returns_the_id():
    query, params = build_insert_analysis_brief_query(
        user_email="a@b.c",
        name="Revenue by region",
        brief=BRIEF,
        prompt_text="PROMPT",
        transcript=[{"role": "user", "text": "hi"}],
        model="m",
        provider="p",
    )

    assert "RETURNING" in query.as_string(None)
    assert isinstance(params[2], Jsonb) and params[2].obj == BRIEF
    assert isinstance(params[4], Jsonb)
    assert params[3] == "PROMPT"


def test_insert_stores_a_missing_transcript_as_null():
    _, params = build_insert_analysis_brief_query(
        user_email="a@b.c", name="n", brief=BRIEF, prompt_text="p",
        transcript=None, model=None, provider=None,
    )

    assert params[4] is None


def test_get_can_be_scoped_to_the_owner():
    unscoped, unscoped_params = build_get_analysis_brief_query(7)
    scoped, scoped_params = build_get_analysis_brief_query(7, "a@b.c")

    assert '"user_email" = %s' not in unscoped.as_string(None)
    assert '"user_email" = %s' in scoped.as_string(None)
    assert unscoped_params == (7,)
    assert scoped_params == (7, "a@b.c")


def test_list_applies_only_the_filters_given():
    query, params = build_list_analysis_briefs_query(status="ready", limit=5)

    text = query.as_string(None)
    assert '"status" = %s' in text
    assert '"user_email" = %s' not in text
    assert params == ("ready", 5)


# ---------------------------------------------------------------------------
# database_pg wrapper
# ---------------------------------------------------------------------------


def test_save_commits_and_returns_the_new_id(monkeypatch):
    conn = MagicMock()
    conn.cursor.return_value.fetchone.return_value = (42,)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    brief_id = database_pg.save_analysis_brief("a@b.c", "n", BRIEF, "p")

    assert brief_id == 42
    conn.commit.assert_called_once()
    conn.close.assert_called_once()


def test_save_rolls_back_on_failure(monkeypatch):
    conn = MagicMock()
    conn.cursor.return_value.execute.side_effect = RuntimeError("boom")
    monkeypatch.setattr(database_pg, "connect", lambda: conn)

    with pytest.raises(RuntimeError):
        database_pg.save_analysis_brief("a@b.c", "n", BRIEF, "p")

    conn.rollback.assert_called_once()
    conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def test_the_service_names_the_row_after_the_brief(monkeypatch):
    saved = {}

    def fake_save(**kwargs):
        saved.update(kwargs)
        return 9

    monkeypatch.setattr(service, "save_analysis_brief", fake_save)

    result = service.save_approved_brief(
        user_email="a@b.c", brief=BRIEF, prompt_text="p", transcript=[], model="m", provider="x"
    )

    assert result == {"brief_id": 9, "name": "Revenue by region"}
    assert saved["brief"] is BRIEF
    assert saved["model"] == "m"


def test_the_service_wraps_storage_errors(monkeypatch):
    def failing_save(**kwargs):
        raise ValueError("db down")

    monkeypatch.setattr(service, "save_analysis_brief", failing_save)

    with pytest.raises(RuntimeError, match="Failed to save analysis brief: db down"):
        service.save_approved_brief(user_email="a@b.c", brief=BRIEF, prompt_text="p")
