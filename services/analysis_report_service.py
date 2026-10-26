"""
analysis_report_service.py
--------------------------

Persistence of the data analyst's runs and reports.

A run is saved as it progresses (one row per run_id, updated in place): its
metadata, plan and state along the way, and its HTML report once there is one.
The frontend lists a user's runs and reads a report from here, and a new run of
a brief compares its metrics with the previous completed run found here.
Flask-free, so both the HTTP layer and the analyst's background thread can use it.
"""

import json
from datetime import datetime
from typing import Any, Optional

from infrastructure.database_pg import (
    get_analysis_report_html,
    get_previous_analysis_report_metadata,
    list_analysis_reports,
    save_analysis_report,
)


def _timestamp(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def _json(value: Any) -> Any:
    """Plain JSON values: anything else a run produced is stored as its text."""
    return json.loads(json.dumps(value, default=str)) if value is not None else None


def save_report(
    *,
    run_id: str,
    brief_id: int,
    user_email: str,
    title: Optional[str],
    status: str,
    metadata: dict[str, Any],
    plan: Optional[dict[str, Any]] = None,
    state: Optional[dict[str, Any]] = None,
    html: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> None:
    """
    Store a run as it is now, replacing what was stored for its run_id.

    :param started_at: ISO 8601 timestamp.
    :param finished_at: ISO 8601 timestamp, once the run ended.
    :raises RuntimeError: if the run could not be stored.
    """
    try:
        save_analysis_report(
            run_id=run_id,
            brief_id=brief_id,
            user_email=user_email,
            title=title,
            status=status,
            metadata=_json(metadata),
            plan=_json(plan),
            state=_json(state),
            html=html,
            started_at=_timestamp(started_at),
            finished_at=_timestamp(finished_at),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save analysis run {run_id}: {e}") from e


def get_report_html(run_id: str, user_email: str) -> Optional[str]:
    """
    The report of one run of the user.

    :return: the HTML, or None (unknown run, someone else's, no report yet).
    :raises RuntimeError: if the database could not be read.
    """
    try:
        return get_analysis_report_html(run_id, user_email)
    except Exception as e:
        raise RuntimeError(f"Failed to load the report of run {run_id}: {e}") from e


def list_reports(user_email: str, brief_id: Optional[int] = None, limit: int = 50) -> list[dict[str, Any]]:
    """
    The user's runs, newest first, without their reports.

    :raises RuntimeError: if the database could not be read.
    """
    try:
        return list_analysis_reports(user_email, brief_id, limit)
    except Exception as e:
        raise RuntimeError(f"Failed to list analysis runs: {e}") from e


def previous_report_metadata(brief_id: int, run_id: str) -> Optional[dict[str, Any]]:
    """
    The metadata of the latest completed run of the brief other than `run_id`,
    or None when there is none: the baseline of the run's metric changes.

    :raises RuntimeError: if the database could not be read.
    """
    try:
        return get_previous_analysis_report_metadata(brief_id, run_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load the previous run of brief {brief_id}: {e}") from e
