# === Built-in ===
import os

os.environ["MPLBACKEND"] = "Agg"
import base64
import secrets
import tempfile
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Any, List, Union, Optional
import logging
import time
import json
from functools import wraps

# === Third-party ===
from flask import (
    Flask,
    request,
    Response,
    jsonify,
    abort,
    render_template,
    redirect,
    url_for,
    session,
    flash,
)
from flask_cors import CORS
import pandas as pd
import matplotlib
import pymupdf4llm
import bcrypt
import psutil

# === Local modules ===
from agent_manager import getAgent, createAgent, deleteAgent
from datachat.output_normalizer import normalize_datachat_response
from datachat.dataset_loader import load_csv_to_dataframe
from datachat.engine_output_adapter import (
    adapt_engine_output,
    consume_adapter_fallback_used,
)
from vector_store import MauiVectorStore
from rag_ingestion_service import process_rag_file
from document_comparison_service import compare_documents
from document_text_service import extract_and_normalize_document, DocumentInput
import database_pg
from database_pg import (
    edit_tokens,
    validate_api_key,
    get_users_for_admin,
    get_users_stats,
    get_logs_for_admin,
    get_logs_stats,
    update_user_tokens,
    get_user_by_id,
    get_all_prompts,
    get_prompt_by_id,
    add_prompt,
    update_prompt,
    delete_prompt,
    get_all_costs,
    add_cost,
    update_cost,
    delete_cost,
    get_cost_by_id,
    get_daily_stats,
    get_recent_activity,
    log_token_usage,
    get_user_by_username,
    get_feedback_for_admin,
    get_feedback_stats,
    get_all_rag_files,
)

from dino import dino_authenticate
from external_auth import external_authenticate
from ai import (
    describe_image,
    choose_llm,
    choose_emb_model,
    whisper_response,
)
from audio_form_service import audioFormCompilation, audioFormPromptBuild
from completion_service import complete_chat, CompletionRequest
from utils.agent_serialization import serialize_runresult
from utils.agent_logging import log_runresult, setup_agent_logger
from utils.runtime_logging import setup_datachat_runtime_logger
from dotenv import load_dotenv
from config import load_config, AppConfig, PROVIDER_API_KEY_MAP
from agentchat_service import run_agentchat

load_dotenv()  # Load environment variables from .env file
config: AppConfig = load_config()
database_pg.init(config)

# Initialize the Flask application
app = Flask(__name__)
# origins=["http://localhost:4200"]
CORS(app)

secret_key = os.environ.get("ENCRYPTION_KEY")
if not secret_key:
    raise RuntimeError("ENCRYPTION_KEY must be set in environment")
app.secret_key = secret_key

# Configure the agent run logger
setup_agent_logger()

DATACHAT_RUNTIME_LOGGER = setup_datachat_runtime_logger()

# Verify Matplotlib backend
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Removing Pandas read csv columns limitations to avoid truncated dataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


# Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            flash("Please log in to access the admin panel", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)

    return decorated_function


# Define a route for the '/' endpoint that returns a welcome message
@app.route("/")
def welcome() -> str:
    return "Welcome to Pandino! This is the root endpoint."


# Validates an API Key associated to an user email
def assert_valid_api_key(api_key: str, user_email: str) -> None:
    """
    Validate the provided API key for the given user email and abort the request if invalid.

    :param api_key: API key string to be validated.
    :param user_email: Email address of the user associated with the API key.
    :return: None
    :raises werkzeug.exceptions.HTTPException: Aborts with 403 if the API key is missing, expired, or invalid.
    """
    if not api_key:
        abort(403, description="Missing API key")
    result, message = validate_api_key(api_key, user_email)
    if not result:
        if "expired" in message:
            abort(403, description="API key expired")
        else:
            abort(403, description="Invalid API key")


# Define a route for the '/edittokens' endpoint that accepts POST requests
@app.route("/edittokens", methods=["POST"])
def editTokens() -> tuple[Response, int]:
    try:
        stripe_key = request.headers.get("X-STRIPE-KEY")

        # Check stripe_key is present and correct
        if not stripe_key:
            return jsonify({"error": "Missing X-STRIPE-KEY header"}), 400

        if stripe_key != config.stripe_key:
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
        app.logger.error(f"Unexpected error in edit tokens: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

    return jsonify({"error": "Unhandled case in editTokens"}), 500


# Define a route for the '/edittokens' endpoint that accepts POST requests
@app.route("/getusertokens", methods=["POST"])
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


# Define a route for the '/adduser' endpoint that accepts POST requests
@app.route("/checkpandinouser", methods=["POST"])
def addNewUser() -> Union[tuple[Response, int], tuple[str, int, dict[str, str]]]:
    graphql_url = request.headers.get("X-GRAPHQL-URL")
    auth_token = request.headers.get("X-AUTH-TOKEN")
    user_email = request.headers.get("X-USER-EMAIL")
    client = request.headers.get("X-CLIENT")

    # backward compatibility for Dino
    # TODO: remove this fallback once Dino sends X-CLIENT header
    if not client:
        client = "dino"

    if not auth_token:
        return jsonify({"error": "Missing X-AUTH-TOKEN header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    # Dino still requires graphql_url
    if client == "dino" and not graphql_url:
        return jsonify({"error": "Missing X-GRAPHQL-URL header"}), 400

    if client == "dino":
        err = dino_authenticate(graphql_url, auth_token)
    else:
        err = external_authenticate(user_email, auth_token, client, graphql_url)

    if err:
        return str(err), 403, {"Content-Type": "text/plain"}

    existingUser = database_pg.get_user_by_username(user_email)
    if not existingUser:
        generatedKey = secrets.token_urlsafe(8)
        currentDate = datetime.now()
        expirationDate = currentDate.replace(year=currentDate.year + 2)
        addUserResult = database_pg.add_user(
            user_email, generatedKey, expirationDate.strftime("%Y-%m-%d %H:%M:%S")
        )
        if addUserResult is None:
            return (
                jsonify(
                    {
                        "response": {
                            "user": {
                                "user_email": user_email,
                                "api_key": generatedKey,
                                "expiration_date": expirationDate,
                            }
                        }
                    }
                ),
                200,
            )
        else:
            return (
                jsonify({"error": addUserResult}),
                500,
            )
    else:
        print(existingUser)
        return (
            jsonify(
                {
                    "response": {
                        "user": {
                            "user_email": existingUser.get("username"),
                            "api_key": existingUser.get("api_key"),
                            "expiration_date": existingUser.get("date_valid_until"),
                        }
                    }
                }
            ),
            200,
        )


# Define a route for the '/validateapikey' endpoint that accepts POST requests
@app.route("/validateapikey", methods=["POST"])
def validate() -> tuple[Response, int]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    # Check if all required parameters are present
    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    result, message = validate_api_key(api_key, user_email)

    if not result:
        if "expired" in message:
            return jsonify({"error": "API key expired"}), 403
        else:
            return jsonify({"error": "Invalid API key"}), 403
    else:
        return jsonify({"response": "API key match found"}), 200


@app.route("/feedback", methods=["POST"])
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
        app.logger.error(f"[feedback] Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


# Define a route for the '/endchat' endpoint that accepts POST requests
@app.route("/enddatachat", methods=["POST"])
def endChat() -> Response | tuple[Response, int]:

    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name_header = request.headers.get("X-USER-NAME")
    user_name = (
        user_name_header.replace(" ", "_").strip() if user_name_header != None else None
    )

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400

    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    # Check if all required parameters are present
    if not user_name:
        return jsonify({"error": "Missing X-USER-NAME header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    deletedEngine = deleteAgent(api_key, user_name)
    if deletedEngine is not None:
        return jsonify({"Agent deleted succesfully": "active"})
    else:
        return jsonify({"Agent was not active for this key": api_key})


# Define a route for the '/startchat' endpoint that accepts POST requests
@app.route("/startdatachat", methods=["POST"])
def startChat() -> Response | tuple[Response, int]:
    api_key = request.headers.get("X-API-KEY")
    user_name_header = request.headers.get("X-USER-NAME")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name = (
        user_name_header.replace(" ", "_").strip()
        if user_name_header is not None
        else None
    )

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400

    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    # Extract necessary parameters from the request FORMDATA
    request_model_name = request.form.get("model_name")
    request_llm_type = request.form.get("llm_type")
    request_file = request.files.get("file")
    request_lang = request.form.get("lang")
    model_name = (
        request_model_name if request_model_name else config.models.datachat_model
    )
    llm_type = request_llm_type if request_llm_type else config.models.datachat_provider
    lang = request_lang if request_lang else "ENG"
    # Check if all required parameters are present
    if (
        not model_name
        or not llm_type
        or not user_name
        or not user_email
        or not request_file
    ):
        return jsonify({"error": "Missing parameters"}), 400

    # Checks if the User's tokens are enough for this operation
    user_tokens = database_pg.get_user_tokens(user_email)

    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    if int(config.datachat_token_cost) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Read the data from the provided CSV file
    data = load_csv_to_dataframe(request_file)

    # Initialize the language model based on the provided type
    llm = choose_llm(llm_type, model_name)

    # Initialize the agent with the data and configuration
    try:
        engine = createAgent(api_key, data, llm, user_name)

        if engine is None:
            return jsonify({"error": "Agent creation failed"}), 500

        agentResponse: dict[str, Any] = {"Agent active": "active"}

        # Language-aware prompt generation
        logging.info(
            f"Invoking startdatachat engine bootstrap with language={lang}, user={user_email}"
        )

        bootstrap = engine.bootstrap(lang)
        if bootstrap.suggested_questions_html is not None:
            agentResponse["suggested_questions"] = bootstrap.suggested_questions_html

        # Spends User's tokens
        edit_tokens(user_email, -int(config.datachat_token_cost))

        return jsonify(agentResponse)
    except Exception as e:
        return (
            jsonify({"error": f"Failed to create Agent: {str(e)}"}),
            500,
        )


# Define a route for the /datachat endpoint
@app.route("/datachat", methods=["POST"])
def dataChat() -> Response | tuple[Response, int]:
    request_id = secrets.token_hex(4)
    request_started = time.time()

    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")

    if not api_key:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f error_code=MISSING_API_KEY",
            request_id,
            (time.time() - request_started) * 1000,
        )
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f error_code=MISSING_USER_EMAIL",
            request_id,
            (time.time() - request_started) * 1000,
        )
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    if not request.json:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f error_code=MISSING_JSON_BODY user=%s",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
        )
        return jsonify({"error": "Missing JSON body"}), 400

    chat = request.json.get("chat")

    engine = getAgent(api_key)
    engine_name = engine.__class__.__name__ if engine is not None else "none"

    DATACHAT_RUNTIME_LOGGER.info(
        "datachat_request_start request_id=%s user=%s engine=%s message_len=%s",
        request_id,
        user_email,
        engine_name,
        len(str(chat or "")),
    )

    # Check if the Chat parameter is present
    if not chat:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f user=%s engine=%s error_code=MISSING_CHAT",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Missing Chat string"}), 400

    # Check if the Agent is active
    if not engine:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f user=%s engine=%s error_code=AGENT_NOT_ACTIVE",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Agent not active for this Api Key"}), 400

    # Checks if the User's tokens are enough for this operation

    user_tokens = database_pg.get_user_tokens(user_email)

    if user_tokens is None:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=500 duration_ms_total=%.2f user=%s engine=%s error_code=USER_TOKENS_NOT_FOUND",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    if int(config.datachat_token_cost) > user_tokens:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=500 duration_ms_total=%.2f user=%s engine=%s error_code=NOT_ENOUGH_TOKENS",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Perform the chat operation and get the response and explanation
    chat_started = time.time()

    response = engine.chat(chat, request_id=request_id)

    response_kind = response.get("kind") if isinstance(response, dict) else None

    DATACHAT_RUNTIME_LOGGER.info(
        "datachat_engine_done request_id=%s user=%s engine=%s duration_ms=%.2f response_kind=%s",
        request_id,
        user_email,
        engine_name,
        (time.time() - chat_started) * 1000,
        response_kind or "unknown",
    )

    trace = None
    log_id: Optional[int] = None
    structured_log_ok = False
    db_log_ok = False
    if hasattr(engine, "get_last_trace"):
        try:
            trace = engine.get_last_trace()  # type: ignore[attr-defined]
        except Exception as e:
            app.logger.warning(f"[datachat] Failed to read engine trace: {e}")

    trace_payload: Optional[dict[str, Any]] = None
    if isinstance(trace, dict) and trace.get("run_result") is not None:
        try:
            trace_payload = serialize_runresult(trace["run_result"])
            if isinstance(trace_payload.get("metrics"), dict):
                trace_payload["metrics"]["duration_ms"] = trace.get("duration_ms")
        except Exception as e:
            app.logger.error(f"[datachat] Failed to serialize trace: {e}")

    if trace_payload is not None:
        try:
            log_runresult(
                trace["run_result"],
                user=user_email,
                namespace="datachat",
                language="N/A",
                question=str(chat),
                extra={
                    "channel": "datachat",
                    "response_kind": response_kind,
                    "request_id": request_id,
                },
            )
            structured_log_ok = True
        except Exception as e:
            app.logger.error(f"[datachat] Structured logging failed: {e}")

        try:
            user = database_pg.get_user_by_username(user_email)
            if not user:
                raise ValueError(f"User '{user_email}' not found in DB")

            user_id = user.get("id")
            if not isinstance(user_id, int):
                raise TypeError(f"Invalid user_id: {user_id}")

            token_metrics = trace_payload.get("metrics", {}).get("token_usage", {})
            token_input = token_metrics.get("input") or 0
            token_output = token_metrics.get("output") or 0

            log_id = log_token_usage(
                user_id=user_id,
                token_input=token_input,
                token_output=token_output,
                model=config.models.datachat_model,
                provider=config.models.datachat_provider,
            )
            db_log_ok = True
            app.logger.info(f"[datachat] token usage logged log_id={log_id}")
        except Exception as e:
            app.logger.error(f"[datachat] Failed to log token usage: {e}")

    DATACHAT_RUNTIME_LOGGER.info(
        "datachat_trace_status request_id=%s user=%s engine=%s trace_present=%s structured_log_ok=%s db_log_ok=%s log_id=%s",
        request_id,
        user_email,
        engine_name,
        bool(trace_payload is not None),
        structured_log_ok,
        db_log_ok,
        log_id if log_id is not None else "none",
    )

    response = adapt_engine_output(response)
    adapter_fallback_used = consume_adapter_fallback_used()
    DATACHAT_RUNTIME_LOGGER.info(
        "datachat_adapter_status request_id=%s user=%s engine=%s adapter_fallback_used=%s",
        request_id,
        user_email,
        engine_name,
        adapter_fallback_used,
    )

    try:
        response_dict = normalize_datachat_response(response)
    except RuntimeError as e:
        DATACHAT_RUNTIME_LOGGER.info(
            "datachat_request_end request_id=%s status=error http_status=500 duration_ms_total=%.2f user=%s engine=%s response_kind=%s error_code=NORMALIZE_FAILED",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
            response_kind or "unknown",
        )
        return jsonify({"error": str(e)}), 500

    # Spends User's tokens
    edit_tokens(user_email, -int(config.datachat_token_cost))

    response_payload: dict[str, Any] = {
        "response": response_dict,
        "explanation": None,
    }
    if log_id is not None:
        response_payload["log_id"] = log_id

    DATACHAT_RUNTIME_LOGGER.info(
        "datachat_request_end request_id=%s status=ok http_status=200 duration_ms_total=%.2f user=%s engine=%s response_kind=%s adapter_fallback_used=%s log_id=%s",
        request_id,
        (time.time() - request_started) * 1000,
        user_email,
        engine_name,
        response_kind or "unknown",
        adapter_fallback_used,
        log_id if log_id is not None else "none",
    )

    return jsonify(response_payload)


# Define a route for the /datachat endpoint
@app.route("/buyreport", methods=["POST"])
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


@app.route("/completion.json", methods=["POST"])
def completion_handler() -> Union[Response, tuple[Response, int]]:
    try:
        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required_keys = ["chat", "username"]
        missing_keys = [key for key in required_keys if key not in r]
        if missing_keys:
            return (
                jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}),
                400,
            )

        api_key = request.headers.get("X-API-KEY")
        if not api_key:
            return jsonify({"error": "Missing X-API-KEY header"}), 400

        assert_valid_api_key(api_key, r["username"])

        # Token check
        user_tokens = database_pg.get_user_tokens(r["username"])
        if user_tokens is None:
            return jsonify({"error": "Could not retrieve user tokens"}), 500

        token_cost = int(config.completion_token_cost or "1")
        if token_cost > user_tokens:
            return (
                jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}),
                500,
            )

        # Request assembly
        chat_request = CompletionRequest(
            username=r["username"],
            info=r.get("info", []),
            chat=r["chat"],
        )
        namespace = r.get("namespace", "")

        # Scelta modelli
        llm_type = config.models.completion_model_provider or "google"
        model = config.models.completion_model or "gemini-2.5-flash"
        provider_api_key = os.getenv(PROVIDER_API_KEY_MAP.get(llm_type, ""))
        emb_llm_type = config.models.completion_embedding_model_provider or "Deepinfra"
        emb_model = (
            config.models.completion_embedding_model
            or "intfloat/multilingual-e5-large-instruct"
        )
        emb_api_key = os.getenv(PROVIDER_API_KEY_MAP.get(emb_llm_type, ""))

        embeddings = choose_emb_model(emb_llm_type, emb_model, api_key=emb_api_key)

        store = MauiVectorStore(embeddings, namespace)
        language = r.get("language", "ENG")
        resp = complete_chat(
            chat_request, store, llm_type, model, language, api_key=provider_api_key
        )

        if resp["answer"] or resp["vectors"]:
            edit_tokens(r["username"], -token_cost)

            log_id = None

            user = get_user_by_username(r["username"])
            if user:
                user_id = user.get("id")
                token_in = resp["token_usage"]["input_tokens"]
                token_out = resp["token_usage"]["output_tokens"]
                if isinstance(user_id, int) and (token_in > 0 or token_out > 0):
                    log_id = log_token_usage(
                        user_id=user_id,
                        token_input=token_in,
                        token_output=token_out,
                        model=model,
                        provider=llm_type,
                    )

            if resp["vectors"]:
                for vec in resp["vectors"]:
                    vec["similarity"] += 0.3

            response_payload = {
                "answer": resp["answer"],
                "vectors": resp["vectors"],
            }

            if log_id is not None:
                response_payload["log_id"] = log_id

            return jsonify(response_payload), 200

        return jsonify({"error": "No response from chat completion"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in completion_handler: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


textContentType = {"Content-Type": "text/plain"}


@app.route("/agentchat", methods=["POST"])
def agentchat() -> Response | tuple[Response, int]:
    """
    AI endpoint based on Smolagents.
    The agent must always use DinoRetrieverTool to retrieve context
    before generating the response.
    """
    try:

        # === INPUT VALIDATION ===

        r = request.get_json()
        if not r:
            return jsonify({"error": "No JSON data provided"}), 400

        required = ["chat", "username"]
        missing = [k for k in required if k not in r]
        if missing:
            return (
                jsonify({"error": f"Missing required keys: {', '.join(missing)}"}),
                400,
            )

        api_key = request.headers.get("X-API-KEY")
        if not api_key:
            return jsonify({"error": "Missing X-API-KEY header"}), 400

        # === Validate the provided API key for the given user email ===

        assert_valid_api_key(api_key, r["username"])

        # === PARAMETERS WITH FALLBACK ===

        chat = r["chat"]
        if not isinstance(chat, list) or not chat:
            return jsonify({"error": "Invalid 'chat': expected non-empty list"}), 400

        namespace = r.get("namespace") or config.rag.default_namespace
        language = r.get("language") or "ITA"
        token_cost = config.completion_token_cost

        app.logger.info(
            f"[agentchat] user={r['username']} ns={namespace} lang={language}"
        )

        # === TOKEN CHECK ===
        user_tokens = database_pg.get_user_tokens(r["username"])
        if user_tokens is None:
            return jsonify({"error": "Could not retrieve user tokens"}), 500
        if token_cost > user_tokens:
            return (
                jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}),
                403,
            )

        result = run_agentchat(
            chat=chat,
            namespace=namespace,
            language=language,
            username=r["username"],
            config=config,
        )
        payload = result["payload"]
        model_clean = result["model"]
        provider = result["provider"]
        duration_ms = payload.get("metrics", {}).get("duration_ms", 0)

        log_id: Optional[int] = None

        # === DATABASE TOKEN USAGE LOGGING ===

        try:
            # Retrieve user_id from username
            user = database_pg.get_user_by_username(r["username"])
            if not user:
                raise ValueError(f"User '{r['username']}' not found in DB")

            user_id = user.get("id")
            if not isinstance(user_id, int):
                raise TypeError(f"Invalid user_id: {user_id}")

            # Extract token usage from payload
            token_metrics = payload.get("metrics", {}).get("token_usage", {})
            token_input = token_metrics.get("input", 0)
            token_output = token_metrics.get("output", 0)

            # Use clean model name from earlier normalization
            model = model_clean

            # Log into PostgreSQL
            log_id = log_token_usage(
                user_id=user_id,
                token_input=token_input,
                token_output=token_output,
                model=model,
                provider=provider,
            )

        except Exception as e:
            app.logger.error(f"[agentchat] Failed to log token usage: {e}")

        # === TOKEN MANAGEMENT ===

        answer_text = payload.get("answer", "")
        if answer_text:
            edit_tokens(r["username"], -token_cost)

        app.logger.info(
            f"[agentchat] done user={r['username']} duration={duration_ms}ms "
            f"tools={len(payload.get('tool_calls', []))} "
            f"vectors={len(payload.get('vectors', []))} "
            f"fu={len(payload.get('follow_ups', []))}"
        )

        if log_id is not None:
            payload["log_id"] = log_id

        return jsonify(payload), 200

    except RuntimeError as e:
        app.logger.error(f"[agentchat] Runtime error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        import traceback

        app.logger.error(f"[agentchat] Unexpected error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route("/compare_docs", methods=["POST"])
def compare_docs():
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    user_tokens = database_pg.get_user_tokens(user_email)
    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    token_cost = config.compare_docs_token_cost
    if token_cost > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 403

    prompt = request.form.get("prompt")
    additional_context = request.form.get("additional_context")
    language = request.form.get("language")
    files = request.files.getlist("files")
    text_documents_raw = request.form.get("text_documents")
    file_roles_raw = request.form.get("file_roles")

    llm_type = config.models.compare_docs_provider or "Google"
    model = config.models.compare_docs_model or "gemini-2.5-flash"
    provider_api_key = os.getenv(PROVIDER_API_KEY_MAP.get(llm_type, ""))

    if not prompt:
        return (
            jsonify(
                {"error": "Invalid request", "details": "The prompt field is required."}
            ),
            400,
        )

    try:
        normalized_documents = []
        text_documents = []
        file_roles = []

        if text_documents_raw:
            text_documents = json.loads(text_documents_raw)

            if not isinstance(text_documents, list):
                raise ValueError("text_documents must be a JSON array")

        if file_roles_raw:
            file_roles = json.loads(file_roles_raw)

            if not isinstance(file_roles, list):
                raise ValueError("file_roles must be a JSON array")

        if file_roles and len(file_roles) != len(files):
            raise ValueError("file_roles length must match number of files")

        for role in file_roles:
            if not isinstance(role, str):
                raise ValueError("Each file role must be a string")

        if len(files) + len(text_documents) < 2:
            return (
                jsonify(
                    {
                        "error": "Invalid request",
                        "details": "At least two documents are required.",
                    }
                ),
                400,
            )

        for item in text_documents:
            if not isinstance(item, dict):
                raise ValueError("Each text document must be an object")

            if "content" not in item or not isinstance(item["content"], str):
                raise ValueError("Each text document must have a string 'content'")

        for text_document in text_documents:
            doc_input: DocumentInput = {
                "content": text_document.get("content"),
                "filename": text_document.get("filename"),
                "source_type": "text",
                "role": text_document.get("role"),
            }

            normalized = extract_and_normalize_document(doc_input)
            normalized_documents.append(normalized)

        for index, file in enumerate(files):
            role = file_roles[index] if file_roles else None

            doc_input: DocumentInput = {
                "content": file,
                "filename": file.filename,
                "source_type": "file",
                "role": role,
            }

            normalized = extract_and_normalize_document(doc_input)
            normalized_documents.append(normalized)

        service_result = compare_documents(
            documents=normalized_documents,
            prompt=prompt,
            llm_type=llm_type,
            model=model,
            additional_context=additional_context,
            language=language,
            api_key=provider_api_key,
        )

        result = service_result["comparison"]
        token_usage = service_result["token_usage"]

        try:
            user = database_pg.get_user_by_username(user_email)
            if not user:
                raise ValueError(f"User '{user_email}' not found in DB")

            user_id = user.get("id")
            if not isinstance(user_id, int):
                raise TypeError(f"Invalid user_id: {user_id}")

            log_token_usage(
                user_id=user_id,
                token_input=token_usage.get("input_tokens", 0),
                token_output=token_usage.get("output_tokens", 0),
                model=model,
                provider=llm_type,
            )

        except Exception as error:
            app.logger.error(f"[compare_docs] Failed to log token usage: {error}")

        edit_tokens(user_email, -token_cost)

    except ValueError as error:
        return jsonify({"error": "Invalid request", "details": str(error)}), 400

    except NotImplementedError as error:
        return (
            jsonify({"error": "Unsupported document format", "details": str(error)}),
            415,
        )

    return jsonify(result), 200


@app.route("/prompt.txt", methods=["POST"])
def prompt_handler() -> Union[str, tuple[str, int, dict[str, str]]]:
    prompt = request.form.get("prompt")
    username = request.form.get("username")
    api_key = request.headers.get("X-API-KEY")


# Define a route for the '/admin/costs' endpoint
@app.route("/admin/costs", methods=["GET"])
@admin_required
def admin_costs() -> str:
    costs = get_all_costs()
    return render_template("admin/costs.html", costs=costs)


# Define a route for the '/admin/costs/add' endpoint
@app.route("/admin/costs/add", methods=["POST"])
@admin_required
def admin_add_cost() -> Response:
    model = request.form.get("model")
    provider = request.form.get("provider")
    token_input_cost = float(request.form.get("token_input_cost"))
    token_output_cost = float(request.form.get("token_output_cost"))
    start_date_valid = request.form.get("start_date_valid")
    end_date_valid = request.form.get("end_date_valid")

    error = add_cost(
        model,
        provider,
        token_input_cost,
        token_output_cost,
        start_date_valid,
        end_date_valid,
    )
    if error:
        flash(error, "danger")
    else:
        flash("Cost added successfully", "success")
    return redirect(url_for("admin_costs"))


# Define a route for the '/admin/costs/edit/<int:cost_id>' endpoint
@app.route("/admin/costs/edit/<int:cost_id>", methods=["GET", "POST"])
@admin_required
def admin_edit_cost(cost_id: int):
    if request.method == "POST":
        model = request.form.get("model")
        provider = request.form.get("provider")
        token_input_cost = float(request.form.get("token_input_cost"))
        token_output_cost = float(request.form.get("token_output_cost"))
        start_date_valid = request.form.get("start_date_valid")
        end_date_valid = request.form.get("end_date_valid")

        error = update_cost(
            cost_id,
            model,
            provider,
            token_input_cost,
            token_output_cost,
            start_date_valid,
            end_date_valid,
        )
        if error:
            flash(error, "danger")
        else:
            flash("Cost updated successfully", "success")
        return redirect(url_for("admin_costs"))

    cost = get_cost_by_id(cost_id)
    if not cost:
        flash("Cost not found", "danger")
        return redirect(url_for("admin_costs"))
    return render_template("admin/edit_cost.html", cost=cost)


# Define a route for the '/admin/costs/delete/<int:cost_id>' endpoint
@app.route("/admin/costs/delete/<int:cost_id>", methods=["POST"])
@admin_required
def admin_delete_cost(cost_id: int):
    error = delete_cost(cost_id)
    if error:
        flash(error, "danger")
    else:
        flash("Cost deleted successfully", "success")
    return redirect(url_for("admin_costs"))


@app.route("/storeragfile", methods=["POST"])
def store_rag_file() -> tuple[Response, int] | tuple[str, int, dict[str, str]]:
    graphql_url = request.form.get("graphqlUrl")
    auth_token = request.form.get("authToken")
    user_email = request.form.get("userEmail")
    client = request.form.get("client")

    # backward compatibility for Dino
    # TODO: remove this fallback once Dino sends client explicitly
    if not client:
        client = "dino"

    if not auth_token:
        return "Missing authToken", 400, textContentType
    if client != "dino" and not user_email:
        return "Missing userEmail", 400, textContentType
    if client == "dino" and not graphql_url:
        return "Missing graphqlUrl", 400, textContentType

    if client == "dino":
        err = dino_authenticate(graphql_url, auth_token)
    else:
        assert user_email is not None
        err = external_authenticate(user_email, auth_token, client, graphql_url)

    if err:
        return str(err), 403, textContentType

    file = request.files.get("file")
    url = request.form.get("url")
    namespace = request.form.get("namespace") or ""
    language = request.form.get("language")

    if not file:
        return "File not provided", 400, textContentType
    if not url:
        return "Url not provided", 400, textContentType

    try:
        result = process_rag_file(
            file,
            url,
            namespace,
            language,
            whisper_model=config.models.whisper_model,
            deepinfra_api_key=config.api_keys.deepinfra_api_key,
            vision_provider=config.models.vision_provider,
            vision_model=config.models.vision_model,
            embedding_provider=config.models.completion_embedding_model_provider,
            embedding_model=config.models.completion_embedding_model,
            vision_api_key=os.getenv(
                PROVIDER_API_KEY_MAP.get(config.models.vision_provider or "", "")
            ),
            embedding_api_key=os.getenv(
                PROVIDER_API_KEY_MAP.get(
                    config.models.completion_embedding_model_provider or "", ""
                )
            ),
        )

        return (
            jsonify(
                {
                    "status": "ok",
                    "file_id": result.file_id,
                    "file_name": result.file_name,
                    "namespace": result.namespace,
                    "chunk_count": result.chunk_count,
                    "language": result.language,
                    "tracking_saved": result.tracking_saved,
                }
            ),
            200,
        )

    except ValueError as e:
        return str(e), 400, textContentType
    except Exception as e:
        return str(e), 500, textContentType


# Define a route for the '/transcribe' endpoint
@app.route("/transcribe", methods=["POST"])
def whisper_parse() -> Union[Response, tuple[Response, int]]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name_header = request.headers.get("X-USER-NAME")

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    if not user_name_header:
        return jsonify({"error": "Missing X-USER-NAME header"}), 400

    user_name = user_name_header.replace(" ", "_").strip()
    assert_valid_api_key(api_key, user_email)

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Missing file"}), 400

    lang = request.form.get("lang") or "ENG"

    if file.mimetype.startswith("audio"):
        if not config.models.whisper_model or not config.api_keys.deepinfra_api_key:
            return jsonify({"error": "Missing Whisper configuration"}), 500

        response = whisper_response(
            file, config.models.whisper_model, config.api_keys.deepinfra_api_key
        )
        if response.status_code == 200:
            try:
                return jsonify(response.json()), 200
            except Exception as e:
                return jsonify({"error": f"Invalid JSON from whisper: {str(e)}"}), 500
        else:
            app.logger.error(
                f"Whisper failed: {response.status_code} - {response.text}"
            )
            return jsonify({"error": "Whisper transcription failed"}), 500

    if file.mimetype == "application/pdf":
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
                file.save(temp.name)
                text = pymupdf4llm.to_markdown(temp.name)
                return jsonify({"text": text}), 200
        except Exception as e:
            return jsonify({"error": f"Error extracting text from pdf: {str(e)}"}), 422

    if file.mimetype.startswith("image"):
        try:
            b64 = base64.b64encode(file.read()).decode()
            dataurl = f"data:{file.mimetype};base64,{b64}"
            text = describe_image(
                dataurl,
                config.models.vision_provider or "",
                config.models.vision_model or "",
                api_key=os.getenv(
                    PROVIDER_API_KEY_MAP.get(config.models.vision_provider or "", "")
                ),
            )
            return jsonify({"text": text}), 200
        except Exception as e:
            return (
                jsonify({"error": f"Error extracting text from image: {str(e)}"}),
                500,
            )

    return jsonify({"error": f"Unexpected file mimetype: {file.mimetype}"}), 400


# Define a route for the '/audioformcompilation' endpoint
@app.route("/audioformcompilation", methods=["POST"])
def audio_form_compile() -> Union[Response, tuple[Response, int]]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")

    if not api_key:
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400

    formSchemaName = request.json.get("name")
    formSchemaExampleData = request.json.get("exampledata")
    formSchemaChoices = request.json.get("choices")
    transcribedAudio = request.json.get("transcribedAudio")

    if not formSchemaExampleData:
        return jsonify({"error": "Missing Schema example empty data"}), 400
    if not formSchemaName:
        return jsonify({"error": "Missing Schema Name"}), 400
    if not transcribedAudio:
        return jsonify({"error": "Missing Transcribed Audio"}), 400

    user_tokens = database_pg.get_user_tokens(user_email)
    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    token_cost = int(config.audio_form_token_cost or "1")
    if token_cost > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    model_name = config.models.audio_model or "gpt-3.5-turbo"
    llm_type = config.models.audio_provider or "openai"
    provider_api_key = os.getenv(PROVIDER_API_KEY_MAP.get(llm_type, ""))

    prompts = audioFormPromptBuild(
        formSchemaExampleData,
        formSchemaName,
        formSchemaChoices,
        transcribedAudio,
    )

    if not prompts:
        return jsonify({"error": "Failed to build prompts"}), 500

    try:
        result = audioFormCompilation(
            prompts["userprompt"],
            prompts["systemprompt"],
            llm_type,
            model_name,
            api_key=provider_api_key,
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    token_usage = result["token_usage"]
    user = database_pg.get_user_by_username(user_email)
    if user and (token_usage["input_tokens"] > 0 or token_usage["output_tokens"] > 0):
        log_token_usage(
            user_id=user["id"],
            token_input=token_usage["input_tokens"],
            token_output=token_usage["output_tokens"],
            model=model_name,
            provider=llm_type,
        )

    edit_tokens(user_email, -token_cost)

    app.logger.debug(f"Audio form compilation result: {result['content']}")
    return jsonify(result["content"]), 200


# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route("/summarize", methods=["GET"])
def summarize():
    return "The /summarize endpoint is not yet implemented.", 501


# Define a route for the '/summarize' endpoint that returns a "not yet implemented" message
@app.route("/categorize", methods=["GET"])
def categorize():
    return "The /categorize endpoint is not yet implemented.", 501


# Define a route for the '/img-comparison' endpoint that returns a "not yet implemented" message
@app.route("/img-comparison", methods=["GET"])
def img_comparison():
    return "The /img-comparison endpoint is not yet implemented.", 501


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if (
            username == config.admin.username
            and password
            and bcrypt.checkpw(password.encode("utf-8"), config.admin.password_hash)
        ):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            flash("Successfully logged in!", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials", "danger")

    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    session.pop("admin_username", None)
    flash("Successfully logged out", "info")
    return redirect(url_for("admin_login"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    env_vars = {
        "DATACHAT_MODEL": config.models.datachat_model,
        "DATACHAT_PROVIDER": config.models.datachat_provider,
        "PROMPT_MODEL": config.models.prompt_model,
        "PROMPT_PROVIDER": config.models.prompt_provider,
        "AUDIO_MODEL": config.models.audio_model,
        "AUDIO_PROVIDER": config.models.audio_provider,
        "COMPLETION_MODEL": config.models.completion_model,
        "COMPLETION_MODEL_PROVIDER": config.models.completion_model_provider,
        "COMPLETION_MODEL_AGENT_CHAT": config.models.completion_model_agent_chat,
        "COMPLETION_EMBEDDING_MODEL": config.models.completion_embedding_model,
        "COMPLETION_EMBEDDING_MODEL_PROVIDER": config.models.completion_embedding_model_provider,
        "WHISPER_MODEL": config.models.whisper_model,
        "VISION_PROVIDER": config.models.vision_provider,
        "VISION_MODEL": config.models.vision_model,
        "DATACHAT_TOKEN_COST": config.datachat_token_cost,
        "DATACHAT_MAX_STEPS": config.datachat.max_steps,
        "DATACHAT_RATE_LIMIT_PER_MIN": config.datachat.rate_limit_per_min,
        "DATACHAT_SESSION_TTL_MIN": config.datachat.session_ttl_min,
        "DATACHAT_LOG_LEVEL": config.datachat.log_level,
        "COMPLETION_TOKEN_COST": config.completion_token_cost,
        "PROMPT_TOKEN_COST": config.prompt_token_cost,
        "AUDIO_FORM_TOKEN_COST": config.audio_form_token_cost,
    }

    try:
        stats_data = get_users_stats()

        today = datetime.now().strftime("%Y-%m-%d")
        daily_stats = get_daily_stats(today)
        recent_activity = get_recent_activity()

        stats = {
            "total_users": stats_data["total_users"],
            "active_sessions": stats_data[
                "total_tokens"
            ],  # Keeping this for now, but will replace in template
            "daily_tokens": daily_stats["total_tokens"],
            "daily_cost": daily_stats["total_cost"],
            "total_orders": 0,
            "recent_activity": recent_activity,
            "db_connected": True,  # If we got here, DB is connected
            "cpu_percent": psutil.cpu_percent(interval=0.5),  # utilizzo CPU
            "memory": {
                "total": psutil.virtual_memory().total,
                "used": psutil.virtual_memory().used,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
        }
        return render_template("admin/dashboard.html", stats=stats, env_vars=env_vars)

    except Exception as e:
        flash(f"Errore nel caricamento dashboard: {str(e)}", "danger")
        stats = {
            "total_users": 0,
            "active_sessions": 0,
            "total_orders": 0,
            "db_connected": False,  # DB connection failed
            "recent_activity": [],
            "daily_tokens": 0,
            "daily_cost": 0.0,
            "cpu_percent": 0,
            "memory": {"total": 0, "used": 0, "available": 0, "percent": 0},
        }

        return render_template("admin/dashboard.html", stats=stats, env_vars=env_vars)


@app.route("/admin/users")
@admin_required
def admin_users():
    try:
        page = request.args.get("page", 1, type=int)
        search = request.args.get("search", "").strip() or None
        users_data = get_users_for_admin(page=page, limit=50, search=search)
        users = users_data["users"]
        pagination = {
            "page": users_data["page"],
            "total_pages": users_data["total_pages"],
            "total_count": users_data["total_count"],
        }
        return render_template(
            "admin/users.html",
            users=users,
            pagination=pagination,
            current_search=search or "",
        )

    except Exception as e:
        flash(f"Errore nel recupero utenti: {str(e)}", "danger")
        return render_template(
            "admin/users.html",
            users=[],
            pagination={"page": 1, "total_pages": 1},
            current_search="",
        )


@app.route("/admin/logs")
@admin_required
def admin_logs():
    try:
        # Get query parameters
        page = request.args.get("page", 1, type=int)

        # Date filter
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        search = request.args.get("search", "").strip() or None

        # Calculate default dates if not provided (for charts)
        if not start_date or not end_date:
            default_end = datetime.now()
            default_start = default_end - timedelta(days=7)
            chart_start = default_start.strftime("%Y-%m-%d")
            chart_end = default_end.strftime("%Y-%m-%d")
        else:
            chart_start = start_date
            chart_end = end_date

        logs_data = get_logs_for_admin(
            page=page,
            limit=50,
            start_date=chart_start,
            end_date=chart_end,
            search=search,
        )
        logs = logs_data["logs"]
        pagination = {
            "page": logs_data["page"],
            "total_pages": logs_data["total_pages"],
            "total_count": logs_data["total_count"],
        }

        stats = get_logs_stats(start_date=chart_start, end_date=chart_end)

        return render_template(
            "admin/logs.html",
            logs=logs,
            stats=stats,
            pagination=pagination,
            current_start_date=chart_start,
            current_end_date=chart_end,
            current_search=search or "",
        )

    except Exception as e:
        flash(f"Errore nel recupero logs: {str(e)}", "danger")
        return render_template(
            "admin/logs.html",
            logs=[],
            stats={},
            pagination={"page": 1, "total_pages": 1},
            current_start_date="",
            current_end_date="",
            current_search="",
        )


@app.route("/admin/feedback")
@admin_required
def admin_feedback():
    try:
        source_filter = request.args.get("source")
        if source_filter == "all":
            source_filter = None

        page = request.args.get("page", 1, type=int)

        # Date filter
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        # Calculate default dates if not provided
        if not start_date or not end_date:
            default_end = datetime.now()
            default_start = default_end - timedelta(
                days=30
            )  # Default to last 30 days for feedback
            chart_start = default_start.strftime("%Y-%m-%d")
            chart_end = default_end.strftime("%Y-%m-%d")
        else:
            chart_start = start_date
            chart_end = end_date

        feedback_data = get_feedback_for_admin(
            source_filter,
            page=page,
            limit=20,
            start_date=chart_start,
            end_date=chart_end,
        )
        feedbacks = feedback_data["feedbacks"]
        pagination = {
            "page": feedback_data["page"],
            "total_pages": feedback_data["total_pages"],
            "total_count": feedback_data["total_count"],
        }

        stats = get_feedback_stats(
            source_filter, start_date=chart_start, end_date=chart_end
        )

        return render_template(
            "admin/feedback.html",
            feedbacks=feedbacks,
            stats=stats,
            current_filter=source_filter,
            pagination=pagination,
            current_start_date=chart_start,
            current_end_date=chart_end,
        )
    except Exception as e:
        flash(f"Errore nel recupero feedback: {str(e)}", "danger")
        return render_template(
            "admin/feedback.html",
            feedbacks=[],
            stats={},
            pagination={"page": 1, "total_pages": 1},
            current_start_date="",
            current_end_date="",
        )


@app.route("/admin/users/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_user(user_id):
    if request.method == "POST":
        try:
            new_tokens = request.form.get("tokens", type=int)

            if new_tokens is None or new_tokens < 0:
                flash("Numero di token non valido", "danger")
                return redirect(url_for("admin_users"))

            success = update_user_tokens(user_id, new_tokens)

            if success:
                flash(f"Token aggiornati con successo a {new_tokens}", "success")
            else:
                flash("Utente non trovato", "danger")

        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", "danger")

        return redirect(url_for("admin_users"))

    # GET request - show edit form
    try:
        user = get_user_by_id(user_id)
        if user:
            return render_template("admin/edit_user.html", user=user)
        else:
            flash("Utente non trovato", "danger")
            return redirect(url_for("admin_users"))
    except Exception as e:
        flash(f"Errore: {str(e)}", "danger")
        return redirect(url_for("admin_users"))


@app.route("/admin/prompts")
@admin_required
def admin_prompts():
    try:
        prompts = get_all_prompts()
        return render_template("admin/prompts.html", prompts=prompts)
    except Exception as e:
        flash(f"Errore nel recupero prompt: {str(e)}", "danger")
        return render_template("admin/prompts.html", prompts=[])


@app.route("/admin/prompts/add", methods=["POST"])
@admin_required
def admin_add_prompt():
    try:
        title = request.form.get("title")
        version = request.form.get("version", type=int)
        message = request.form.get("message")

        if not title or not version or not message:
            flash("Tutti i campi sono obbligatori", "danger")
            return redirect(url_for("admin_prompts"))

        add_prompt(title, version, message)
        flash("Prompt aggiunto con successo", "success")
    except Exception as e:
        flash(f"Errore nell'aggiunta del prompt: {str(e)}", "danger")

    return redirect(url_for("admin_prompts"))


@app.route("/admin/prompts/<int:prompt_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_prompt(prompt_id):
    if request.method == "POST":
        try:
            title = request.form.get("title")
            version = request.form.get("version", type=int)
            message = request.form.get("message")

            if not title or not version or not message:
                flash("Tutti i campi sono obbligatori", "danger")
                return redirect(url_for("admin_edit_prompt", prompt_id=prompt_id))

            success = update_prompt(prompt_id, title, version, message)

            if success:
                flash("Prompt aggiornato con successo", "success")
            else:
                flash("Prompt non trovato", "danger")

        except Exception as e:
            flash(f"Errore nell'aggiornamento: {str(e)}", "danger")

        return redirect(url_for("admin_prompts"))

    # GET request - show edit form
    try:
        prompt = get_prompt_by_id(prompt_id)
        if prompt:
            return render_template("admin/edit_prompt.html", prompt=prompt)
        else:
            flash("Prompt non trovato", "danger")
            return redirect(url_for("admin_prompts"))
    except Exception as e:
        flash(f"Errore: {str(e)}", "danger")
        return redirect(url_for("admin_prompts"))


@app.route("/admin/prompts/<int:prompt_id>/delete", methods=["POST"])
@admin_required
def admin_delete_prompt(prompt_id):
    try:
        success = delete_prompt(prompt_id)
        if success:
            flash("Prompt eliminato con successo", "success")
        else:
            flash("Prompt non trovato", "danger")
    except Exception as e:
        flash(f"Errore nell'eliminazione: {str(e)}", "danger")

    return redirect(url_for("admin_prompts"))


@app.route("/admin/rag-files")
@admin_required
def admin_rag_files():
    try:
        rag_files = get_all_rag_files()
        return render_template("admin/rag_files.html", rag_files=rag_files)
    except Exception as e:
        flash(f"Error loading RAG files: {str(e)}", "danger")
        return render_template("admin/rag_files.html", rag_files=[])


@app.route("/admin/rag-files/upload", methods=["POST"])
@admin_required
def admin_upload_rag_file():
    file = request.files.get("file")
    namespace = request.form.get("namespace", "").strip()
    language = request.form.get("language", "").strip() or None

    if not file or not namespace:
        flash("File and namespace are required", "danger")
        return redirect(url_for("admin_rag_files"))

    url = file.filename or ""

    try:
        result = process_rag_file(
            file,
            url,
            namespace,
            language,
            whisper_model=config.models.whisper_model,
            deepinfra_api_key=config.api_keys.deepinfra_api_key,
            vision_provider=config.models.vision_provider,
            vision_model=config.models.vision_model,
            embedding_provider=config.models.completion_embedding_model_provider,
            embedding_model=config.models.completion_embedding_model,
            vision_api_key=os.getenv(
                PROVIDER_API_KEY_MAP.get(config.models.vision_provider or "", "")
            ),
            embedding_api_key=os.getenv(
                PROVIDER_API_KEY_MAP.get(
                    config.models.completion_embedding_model_provider or "", ""
                )
            ),
        )

        if result.chunk_count > 0:
            flash(
                f"File indexed successfully ({result.chunk_count} chunks)",
                "success",
            )
        else:
            flash("File was empty, nothing indexed", "warning")
    except Exception as e:
        flash(f"Error processing file: {str(e)}", "danger")

    return redirect(url_for("admin_rag_files"))


@app.route("/health")
def health():
    # Stato base
    status = {
        "status": "ok",
    }
    return jsonify(status)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
