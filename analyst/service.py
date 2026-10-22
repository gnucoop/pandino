"""
service.py
----------

Starting, following and stopping analysis runs. Flask-free: the analyst routes
and the interviewer's approval both start runs through start_analysis().

One run per Api Key at a time, kept in agent_manager under the "analyst" kind
until it is ended or replaced by the next one. The run itself happens in a
background thread (a greenlet under gunicorn gevent): an analysis takes
minutes, far beyond a request. Its report and metadata stay on disk in its
run folder, so a report can be fetched after the run left the registry.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from analyst.metrics import METADATA_FILE
from analyst.orchestrator import run_analysis
from analyst.run import AnalysisRun
from analyst.tools.save_html_report_tool import REPORT_FILE
from config import AppConfig
from datachat.sql_datasource import get_datasource as get_sql_datasource
from infrastructure.agent_manager import createAnalyst, deleteAnalyst, getAnalyst
from infrastructure.database_pg import edit_tokens, get_user_tokens
from services.analysis_brief_service import get_brief

# Run ids are interpolated into a path: anything else is refused before.
_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_b\d+_[0-9a-f]{6}$")


class AnalysisStartError(Exception):
    """The analysis could not start; `code` and `http_status` are for the client."""

    def __init__(self, message: str, code: str, http_status: int, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.extra = extra


def _spawn(run: AnalysisRun, config: AppConfig) -> None:
    threading.Thread(
        target=run_analysis, args=(run, config.analyst), name=f"analyst-{run.run_id}", daemon=True
    ).start()


def new_run_id(brief_id: int) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_b{int(brief_id)}_{secrets.token_hex(3)}"


def start_analysis(
    *,
    api_key: str,
    user_email: str,
    brief_id: int,
    config: AppConfig,
) -> AnalysisRun:
    """
    Start analysing a stored brief of the user; ANALYST_TOKEN_COST is spent here.

    :raises AnalysisStartError: when the run cannot start (no datasource, brief
        not found, not enough tokens, a run already in progress).
    """
    if get_sql_datasource() is None:
        raise AnalysisStartError("The analyst requires the SQL datasource (DATACHAT_SQL_ENABLED).",
                                 "SQL_UNAVAILABLE", 400)

    try:
        brief = get_brief(int(brief_id), user_email)
    except RuntimeError as e:
        raise AnalysisStartError(str(e), "BRIEF_UNAVAILABLE", 500) from e
    if brief is None:
        raise AnalysisStartError(f"Analysis brief {brief_id} not found.", "BRIEF_NOT_FOUND", 404)

    cost = int(config.analyst_token_cost)
    user_tokens = get_user_tokens(user_email)
    if user_tokens is None:
        raise AnalysisStartError("Could not retrieve user tokens.", "TOKENS_UNAVAILABLE", 500)
    if cost > user_tokens:
        raise AnalysisStartError("Not enough tokens.", "NOT_ENOUGH_TOKENS", 500, user_tokens=user_tokens)

    # Checked and replaced with no I/O in between: two starts cannot both pass.
    current = getAnalyst(api_key)
    if current is not None and not current.finished:
        raise AnalysisStartError("An analysis is already running for this key.", "ANALYSIS_BUSY", 409,
                                 run_id=current.run_id)
    if current is not None:
        deleteAnalyst(api_key, user_email)

    run = AnalysisRun(
        run_id=new_run_id(brief_id), brief_id=int(brief_id), api_key=str(api_key), user_email=user_email
    )
    createAnalyst(api_key, lambda: run)

    edit_tokens(user_email, -cost)
    _spawn(run, config)
    return run


def start_analysis_payload(
    *, api_key: str, user_email: str, brief_id: int, config: AppConfig
) -> tuple[dict[str, Any], int]:
    """start_analysis() for an HTTP response: ({"run_id", "status"}, 200) or ({"error", "code", ...}, status)."""
    try:
        run = start_analysis(api_key=api_key, user_email=user_email, brief_id=brief_id, config=config)
    except AnalysisStartError as e:
        return {"error": str(e), "code": e.code, **e.extra}, e.http_status
    return {"run_id": run.run_id, "status": run.status}, 200


def get_run(api_key: str) -> Optional[AnalysisRun]:
    return getAnalyst(api_key)


def end_analysis(api_key: str, user_name: str) -> Optional[AnalysisRun]:
    """Cancel the key's run if it is still going, and forget it."""
    return deleteAnalyst(api_key, user_name)


def report_path(reports_dir: str, run_id: str, user_email: str) -> Optional[Path]:
    """The report of a run of the user, or None (unknown run, someone else's, no report)."""
    if not _RUN_ID_RE.match(str(run_id or "")):
        return None
    run_dir = Path(reports_dir) / run_id
    try:
        meta = json.loads((run_dir / METADATA_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    path = run_dir / REPORT_FILE
    if meta.get("user_email") != user_email or not path.is_file():
        return None
    return path
