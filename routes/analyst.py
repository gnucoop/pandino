"""
Data analyst endpoints.

An analysis runs an approved brief (analysis_briefs) in the background and
ends with an HTML report:

- POST /start-analysis    starts analysing a brief of the user, by brief_id. The
                          interviewer starts one by itself when a brief is
                          approved; this endpoint runs a stored brief again.
- GET  /analysis-status   the run's state and its progress events; pass the
                          last `next_since` as `since` to get only new events
- GET  /analysis-report   the report of a run, as HTML
- POST /end-analysis      cancels the run if still going, and forgets it

One run per Api Key at a time.
"""

from typing import Optional

from flask import Blueprint, Response, current_app, jsonify, request, send_file

from analyst.service import end_analysis, get_run, report_path, start_analysis_payload
from routes.utils import assert_valid_api_key

analyst_bp = Blueprint("analyst", __name__)


def _read_headers() -> tuple[Optional[str], Optional[str], Optional[str]]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name_header = request.headers.get("X-USER-NAME")
    user_name = (
        user_name_header.replace(" ", "_").strip() if user_name_header is not None else None
    )
    return api_key, user_email, user_name


def _missing_header_response(api_key, user_email) -> Optional[tuple[Response, int]]:
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    return None


@analyst_bp.route("/start-analysis", methods=["POST"])
def startAnalysis() -> Response | tuple[Response, int]:
    api_key, user_email, _ = _read_headers()
    missing = _missing_header_response(api_key, user_email)
    if missing is not None:
        return missing

    assert_valid_api_key(api_key, user_email)

    body = request.get_json(silent=True) or {}
    try:
        brief_id = int(body.get("brief_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "brief_id must be an integer"}), 400

    payload, status = start_analysis_payload(
        api_key=str(api_key),
        user_email=str(user_email),
        brief_id=brief_id,
        config=current_app.config["MAUI_CONFIG"],
    )
    return jsonify({"analysis": payload}), status


@analyst_bp.route("/analysis-status", methods=["GET"])
def analysisStatus() -> Response | tuple[Response, int]:
    api_key, user_email, _ = _read_headers()
    missing = _missing_header_response(api_key, user_email)
    if missing is not None:
        return missing

    assert_valid_api_key(api_key, user_email)

    since = request.args.get("since", default=0, type=int) or 0
    run = get_run(str(api_key))
    if run is None or run.user_email != user_email:
        return jsonify({"error": "No analysis for this key", "code": "NO_ANALYSIS"}), 404
    return jsonify(run.status_payload(since)), 200


@analyst_bp.route("/analysis-report", methods=["GET"])
def analysisReport() -> Response | tuple[Response, int]:
    api_key, user_email, _ = _read_headers()
    missing = _missing_header_response(api_key, user_email)
    if missing is not None:
        return missing

    assert_valid_api_key(api_key, user_email)

    config = current_app.config["MAUI_CONFIG"]
    path = report_path(config.analyst.reports_dir, request.args.get("run_id", ""), str(user_email))
    if path is None:
        return jsonify({"error": "Report not found", "code": "NO_REPORT"}), 404
    return send_file(path.resolve(), mimetype="text/html")


@analyst_bp.route("/end-analysis", methods=["POST"])
def endAnalysis() -> Response | tuple[Response, int]:
    api_key, user_email, user_name = _read_headers()
    missing = _missing_header_response(api_key, user_email)
    if missing is not None:
        return missing

    assert_valid_api_key(api_key, user_email)

    run = end_analysis(str(api_key), user_name or str(user_email))
    if run is None:
        return jsonify({"error": "No analysis for this key", "code": "NO_ANALYSIS"}), 404
    return jsonify({"run_id": run.run_id, "status": "cancelled" if not run.finished else run.status}), 200
