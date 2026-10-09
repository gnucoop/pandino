"""
Analysis interviewer endpoints.

A session interviews the user about the analysis they want, grounded in the
SQL datasource, and ends with a brief the user approves:

- POST /start-analysis-interview  opens the session and returns the welcome,
                                  which asks the fixed goal question
- POST /analysis-interview        one step of the interview, by "action":
    answer   (default) answers the open questions; the first one states the goal
    approve  stores the pending draft and closes the session
    revise   keeps working on the pending draft: Q->A goes on
    decline  drops the pending draft and closes the session, storing nothing
- POST /end-analysis-interview    closes the session at any point

Only an approved brief is stored (analysis_briefs, status 'ready'); drafts live
in the session. The stored row, found by the returned brief_id, is what the
data analyst will run (see services.analysis_brief_service.get_brief).
"""

from typing import Any, Optional

from flask import Blueprint, Response, jsonify, request, current_app

from infrastructure.agent_manager import (
    getInterviewer,
    createInterviewer,
    deleteInterviewer,
    deleteInterviewerIfCurrent,
    try_acquire_run,
    release_run,
)
from infrastructure.database_pg import edit_tokens, get_user_tokens
from datachat.sql_datasource import get_datasource as get_sql_datasource
from interviewer.engine_factory import create_interviewer_engine
from services.analysis_brief_service import save_approved_brief
from routes.utils import assert_valid_api_key

interviewer_bp = Blueprint("interviewer", __name__)

# Engine error codes that are the caller's fault or the server's, rather than
# an interview turn that went wrong and can simply be retried.
_ERROR_HTTP_STATUS = {"NO_PENDING_DRAFT": 400, "DRAFT_PENDING": 400, "MISSING_CONFIG": 500}

_ACTIONS = ("answer", "approve", "revise", "decline")


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


def _not_enough_tokens_response(user_email: str, cost: int) -> Optional[tuple[Response, int]]:
    user_tokens = get_user_tokens(user_email)
    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500
    if int(cost) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500
    return None


def _to_response(result: dict[str, Any]) -> dict[str, Any]:
    """Engine result -> {"type", "value"} for the client."""
    kind = result.get("kind")
    if kind == "question":
        value = {"text": result["text"], "topic": result["topic"]}
    elif kind in ("brief_draft", "brief"):
        value = {
            "summary": result["summary"],
            "brief": result["brief"],
            "prompt_text": result["prompt_text"],
        }
    else:
        kind = "error"
        value = {"message": result.get("message", ""), "code": result.get("code")}
    return {"type": kind, "value": value}


def _run_turn(user_email: str, turn) -> tuple[Response, int]:
    """
    Run one interview turn; tokens are spent only on a usable turn.

    :param turn: callable running the turn on the engine and returning its result.
    """
    config = current_app.config["MAUI_CONFIG"]

    result = turn()
    payload: dict[str, Any] = {"response": _to_response(result)}

    if result.get("kind") == "error":
        return jsonify(payload), _ERROR_HTTP_STATUS.get(result.get("code"), 200)

    # Spends User's tokens
    edit_tokens(user_email, -int(config.interviewer_token_cost))
    return jsonify(payload), 200


@interviewer_bp.route("/start-analysis-interview", methods=["POST"])
def startAnalysisInterview() -> Response | tuple[Response, int]:
    config = current_app.config["MAUI_CONFIG"]

    api_key, user_email, user_name = _read_headers()
    missing = _missing_header_response(api_key, user_email)
    if missing is not None:
        return missing

    assert_valid_api_key(api_key, user_email)

    if not user_name:
        return jsonify({"error": "Missing X-USER-NAME header"}), 400

    body = request.get_json(silent=True) or request.form
    lang = body.get("lang") or "ENG"

    # The interview is about database data: without the datasource there is
    # nothing to explore and no schema to ground the brief in.
    if get_sql_datasource() is None:
        return (
            jsonify(
                {
                    "error": "SQL datasource unavailable",
                    "detail": "the analysis interviewer requires the SQL datasource (DATACHAT_SQL_ENABLED)",
                }
            ),
            400,
        )

    # Starting is free (no model call), but pointless without the tokens for
    # the first turn.
    not_enough = _not_enough_tokens_response(user_email, config.interviewer_token_cost)
    if not_enough is not None:
        return not_enough

    try:
        engine = createInterviewer(
            api_key,
            lambda: create_interviewer_engine(
                api_key=str(api_key),
                user_name=user_name,
                config=config.interviewer,
                lang=lang,
            ),
        )
    except Exception as e:
        return jsonify({"error": f"Failed to create Interviewer: {str(e)}"}), 500

    if not engine.is_ready:
        # Only this request's engine: a concurrent start may have registered
        # another one for the same Api Key.
        deleteInterviewerIfCurrent(api_key, engine)
        return jsonify({"error": "Interviewer not configured", "code": "MISSING_CONFIG"}), 500

    # The welcome ends with the fixed goal question: the user's first
    # /analysis-interview answer replies to it.
    agent_response: dict[str, Any] = {
        "Agent active": "active",
        "welcome": engine.bootstrap(lang).suggested_questions_html,
    }
    return jsonify(agent_response)


@interviewer_bp.route("/analysis-interview", methods=["POST"])
def analysisInterview() -> Response | tuple[Response, int]:
    config = current_app.config["MAUI_CONFIG"]

    api_key, user_email, user_name = _read_headers()
    missing = _missing_header_response(api_key, user_email)
    if missing is not None:
        return missing

    assert_valid_api_key(api_key, user_email)

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Missing JSON body"}), 400

    answer = str(body.get("answer") or "").strip()
    action = body.get("action") or "answer"
    if action not in _ACTIONS:
        return jsonify({"error": "action must be one of " + ", ".join(_ACTIONS)}), 400
    if action == "answer" and not answer:
        return jsonify({"error": "Missing answer string"}), 400

    engine = getInterviewer(api_key)
    if not engine:
        return jsonify({"error": "Interviewer not active for this Api Key"}), 400

    # Only one action per engine: a concurrent request is rejected, not queued.
    if not try_acquire_run(engine):
        return (
            jsonify(
                {
                    "error": "session_busy",
                    "message": "A request is already being processed for this session.",
                }
            ),
            409,
        )

    try:
        # The session may have been ended or replaced since it was looked up.
        if getInterviewer(api_key) is not engine:
            return jsonify({"error": "Interviewer not active for this Api Key"}), 400

        if action in ("approve", "decline"):
            response, status = _approve(engine, user_email) if action == "approve" else _decline(engine)
            if status == 200:
                # The interview is over. Only this engine is removed: if /end
                # and /start replaced it meanwhile, the new session stays.
                deleteInterviewerIfCurrent(api_key, engine)
        else:
            not_enough = _not_enough_tokens_response(user_email, config.interviewer_token_cost)
            if not_enough is not None:
                return not_enough
            response, status = _run_turn(
                user_email,
                (lambda: engine.reject(answer or None))
                if action == "revise"
                else (lambda: engine.chat(answer)),
            )

        return response, status
    finally:
        release_run(engine)


def _no_pending_draft_response(verb: str) -> tuple[Response, int]:
    return (
        jsonify({"error": f"There is no brief draft to {verb}", "code": "NO_PENDING_DRAFT"}),
        400,
    )


def _approve(engine, user_email: str) -> tuple[Response, int]:
    """
    Store the pending draft, then mark it approved.

    Stored first: should the database refuse it, the draft is still pending
    and the user can simply approve again. No model call, no tokens spent.
    The stored row (status 'ready') is the hand-off to the data analyst.
    """
    draft = engine.pending_draft
    if draft is None:
        return _no_pending_draft_response("approve")

    try:
        saved = save_approved_brief(
            user_email=user_email,
            brief=draft["brief"],
            prompt_text=draft["prompt_text"],
            transcript=engine.transcript,
            model=engine.model,
            provider=engine.provider,
        )
    except RuntimeError:
        return jsonify({"error": "Failed to save the analysis brief"}), 500

    result = engine.approve()
    return jsonify({"response": _to_response(result), "brief_id": saved["brief_id"]}), 200


def _decline(engine) -> tuple[Response, int]:
    """Drop the pending draft: nothing is stored. No model call, no tokens spent."""
    if engine.pending_draft is None:
        return _no_pending_draft_response("decline")

    return jsonify({"response": {"type": "declined", "value": {}}}), 200


@interviewer_bp.route("/end-analysis-interview", methods=["POST"])
def endAnalysisInterview() -> Response | tuple[Response, int]:
    api_key, user_email, user_name = _read_headers()
    missing = _missing_header_response(api_key, user_email)
    if missing is not None:
        return missing

    assert_valid_api_key(api_key, user_email)

    if not user_name:
        return jsonify({"error": "Missing X-USER-NAME header"}), 400

    deleted_engine = deleteInterviewer(api_key, user_name)
    if deleted_engine is not None:
        return jsonify({"Agent deleted succesfully": "active"})
    else:
        return jsonify({"Agent was not active for this key": api_key})
