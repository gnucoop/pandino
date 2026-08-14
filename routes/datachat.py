import os
import time
import logging
from typing import Any, Optional

from flask import Blueprint, Response, jsonify, request, current_app

from infrastructure.agent_manager import getAgent, createAgent, deleteAgent
from infrastructure.ai import choose_llm
from infrastructure.database_pg import edit_tokens, log_token_usage, get_user_by_username, get_user_tokens
from datachat.dataset_loader import load_csv_to_dataframe
from datachat.output_normalizer import normalize_datachat_response
from datachat.engine_output_adapter import adapt_engine_output, consume_adapter_fallback_used
from utils.agent_serialization import serialize_runresult
from utils.agent_logging import log_runresult
from utils.logging_config import get_request_id
from utils.usage_request_state import set_usage_log_id
from config import PROVIDER_API_KEY_MAP
from routes.utils import assert_valid_api_key

datachat_bp = Blueprint("datachat", __name__)

logger = logging.getLogger(__name__)


@datachat_bp.route("/enddatachat", methods=["POST"])
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


@datachat_bp.route("/startdatachat", methods=["POST"])
def startChat() -> Response | tuple[Response, int]:
    config = current_app.config["MAUI_CONFIG"]

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
    user_tokens = get_user_tokens(user_email)

    if user_tokens is None:
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    if int(config.datachat_token_cost) > user_tokens:
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Read the data from the provided CSV file
    data = load_csv_to_dataframe(request_file)

    provider_api_key = os.getenv(PROVIDER_API_KEY_MAP.get(llm_type, ""))

    # Initialize the language model based on the provided type
    llm = choose_llm(llm_type, model_name, api_key=provider_api_key)

    # Initialize the agent with the data and configuration
    try:
        engine = createAgent(api_key, data, llm, user_name, engine_type=current_app.config["MAUI_CONFIG"].datachat.engine)

        if engine is None:
            return jsonify({"error": "Agent creation failed"}), 500

        agentResponse: dict[str, Any] = {"Agent active": "active"}

        # Language-aware prompt generation
        logger.info(
            "event=datachat_engine_bootstrap_started language=%s",
            lang,
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


@datachat_bp.route("/datachat", methods=["POST"])
def dataChat() -> Response | tuple[Response, int]:
    config = current_app.config["MAUI_CONFIG"]
    _logger = current_app.config["DATACHAT_RUNTIME_LOGGER"]

    request_id = get_request_id()
    request_started = time.time()

    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")

    if not api_key:
        _logger.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f error_code=MISSING_API_KEY",
            request_id,
            (time.time() - request_started) * 1000,
        )
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        _logger.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f error_code=MISSING_USER_EMAIL",
            request_id,
            (time.time() - request_started) * 1000,
        )
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400

    assert_valid_api_key(api_key, user_email)

    if not request.json:
        _logger.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f error_code=MISSING_JSON_BODY user=%s",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
        )
        return jsonify({"error": "Missing JSON body"}), 400

    chat = request.json.get("chat")

    engine = getAgent(api_key)
    engine_name = engine.__class__.__name__ if engine is not None else "none"

    _logger.info(
        "datachat_request_start request_id=%s user=%s engine=%s message_len=%s",
        request_id,
        user_email,
        engine_name,
        len(str(chat or "")),
    )

    # Check if the Chat parameter is present
    if not chat:
        _logger.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f user=%s engine=%s error_code=MISSING_CHAT",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Missing Chat string"}), 400

    # Check if the Agent is active
    if not engine:
        _logger.info(
            "datachat_request_end request_id=%s status=error http_status=400 duration_ms_total=%.2f user=%s engine=%s error_code=AGENT_NOT_ACTIVE",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Agent not active for this Api Key"}), 400

    # Checks if the User's tokens are enough for this operation

    user_tokens = get_user_tokens(user_email)

    if user_tokens is None:
        _logger.info(
            "datachat_request_end request_id=%s status=error http_status=500 duration_ms_total=%.2f user=%s engine=%s error_code=USER_TOKENS_NOT_FOUND",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Could not retrieve user tokens"}), 500

    if int(config.datachat_token_cost) > user_tokens:
        _logger.info(
            "datachat_request_end request_id=%s status=error http_status=500 duration_ms_total=%.2f user=%s engine=%s error_code=NOT_ENOUGH_TOKENS",
            request_id,
            (time.time() - request_started) * 1000,
            user_email,
            engine_name,
        )
        return jsonify({"error": "Not enough tokens", "user_tokens": user_tokens}), 500

    # Perform the chat operation and get the response and explanation
    chat_started = time.time()

    response = engine.chat(chat)

    response_kind = response.get("kind") if isinstance(response, dict) else None

    _logger.info(
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
            logger.warning("event=datachat_trace_read_failed error=%s", e)

    trace_payload: Optional[dict[str, Any]] = None
    if isinstance(trace, dict) and trace.get("run_result") is not None:
        try:
            trace_payload = serialize_runresult(trace["run_result"])
            if isinstance(trace_payload.get("metrics"), dict):
                trace_payload["metrics"]["duration_ms"] = trace.get("duration_ms")
        except Exception as e:
            logger.error("event=datachat_trace_serialize_failed error=%s", e)

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
                },
            )
            structured_log_ok = True
        except Exception as e:
            logger.error("event=datachat_structured_log_failed error=%s", e)

        try:
            user = get_user_by_username(user_email)
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
                service="/datachat",
                request_id=request_id,
                source=user.get("client"),
            )
            set_usage_log_id(log_id)
            db_log_ok = True
            logger.info("event=datachat_token_usage_logged log_id=%s", log_id)
        except Exception as e:
            logger.error("event=datachat_token_usage_log_failed error=%s", e)

    _logger.info(
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
    _logger.info(
        "datachat_adapter_status request_id=%s user=%s engine=%s adapter_fallback_used=%s",
        request_id,
        user_email,
        engine_name,
        adapter_fallback_used,
    )

    try:
        response_dict = normalize_datachat_response(response)
    except RuntimeError as e:
        _logger.info(
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

    _logger.info(
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
