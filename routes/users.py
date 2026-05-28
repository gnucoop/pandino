from flask import Blueprint, jsonify, request, current_app, Response
from infrastructure.database_pg import edit_tokens
import infrastructure.database_pg as database_pg
from routes.utils import assert_valid_api_key

users_bp = Blueprint("users", __name__, url_prefix="")


# Define a route for the '/edittokens' endpoint that accepts POST requests
@users_bp.route("/edittokens", methods=["POST"])
def editTokens() -> tuple[Response, int]:
    try:
        stripe_key = request.headers.get("X-STRIPE-KEY")

        # Check stripe_key is present and correct
        if not stripe_key:
            return jsonify({"error": "Missing X-STRIPE-KEY header"}), 400

        if stripe_key != current_app.config["MAUI_CONFIG"].stripe_key:
            return jsonify({"error": "Invalid STRIPE KEY"}), 403

        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required_keys = ["quantity", "useremail"]
        missing_keys = [key for key in required_keys if key not in r]

        if missing_keys:
            return (
                jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}),
                400,
            )

        result, message = edit_tokens(r["useremail"], r["quantity"])

        if result:
            return (
                jsonify(
                    {
                        "response": f"{message}: {r['quantity']} for user: {r['useremail']}"
                    }
                ),
                200,
            )
        elif not result:
            return (
                jsonify({"error": f"{message}"}),
                400,
            )

    except Exception as e:
        current_app.logger.error(f"Unexpected error in edit tokens: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

    return jsonify({"error": "Unhandled case in editTokens"}), 500


# Define a route for the '/edittokens' endpoint that accepts POST requests
@users_bp.route("/getusertokens", methods=["POST"])
def getUserTokens() -> tuple[Response, int]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    assert_valid_api_key(api_key, user_email)
    tokens = database_pg.get_user_tokens(user_email)
    return jsonify({"response": {"tokens": tokens}}), 200


@users_bp.route("/feedback", methods=["POST"])
def feedback_handler() -> Response | tuple[Response, int]:
    """
    Endpoint to collect user feedback on LLM-generated responses.

    This endpoint allows authenticated users to submit a positive or negative
    evaluation of a generated answer, optionally linking it to an existing log
    entry for future analysis. Feedback is stored persistently and does not
    interfere with token usage or response generation flows.
    """
    try:
        # === INPUT VALIDATION ===

        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required = ["username", "question", "answer", "feedback"]
        missing = [k for k in required if k not in r]
        if missing:
            return (
                jsonify({"error": f"Missing required keys: {', '.join(missing)}"}),
                400,
            )

        api_key = request.headers.get("X-API-KEY")
        if not api_key:
            return jsonify({"error": "Missing X-API-KEY header"}), 400

        # === AUTHENTICATION ===

        assert_valid_api_key(api_key, r["username"])

        # === SEMANTIC VALIDATION ===

        question = r["question"]
        answer = r["answer"]
        feedback_value = r["feedback"]

        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "Invalid 'question'"}), 400

        if not isinstance(answer, str) or not answer.strip():
            return jsonify({"error": "Invalid 'answer'"}), 400

        if feedback_value not in ("positive", "negative"):
            return jsonify({"error": "Invalid feedback value"}), 400

        log_id = r.get("log_id")
        if log_id is not None and not isinstance(log_id, int):
            return jsonify({"error": "Invalid 'log_id': expected integer"}), 400

        if log_id is not None:
            with database_pg.connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM logs WHERE id = %s", (log_id,))
                exists = cur.fetchone()

            if not exists:
                return jsonify({"error": f"log_id {log_id} does not exist"}), 400

        source = r.get("source")
        if source is not None and not isinstance(source, str):
            return jsonify({"error": "Invalid 'source'"}), 400

        # === PERSISTENCE ===

        feedback_id = database_pg.save_feedback(
            user_email=r["username"],
            question=question,
            answer=answer,
            feedback_value=feedback_value,
            log_id=log_id,
            source=source,
        )

        # === RESPONSE ===

        return jsonify({"feedback_id": feedback_id}), 200

    except Exception as e:
        current_app.logger.error(f"[feedback] Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# Define a route for the /datachat endpoint
@users_bp.route("/buyreport", methods=["POST"])
def buyReport() -> tuple[Response, int]:

    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400
    prompts = request.json.get("prompts")
    if not isinstance(prompts, int):
        return jsonify({"error": "Missing Prompts numeric parameter"}), 400

    user_tokens = database_pg.get_user_tokens(user_email)
    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500
    if prompts > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    result, message = edit_tokens(user_email, -prompts)

    return jsonify({"response": result, "message": f"{message}"}), 200
