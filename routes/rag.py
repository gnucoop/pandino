import logging
import os
import traceback
from typing import Union, Optional

from flask import Blueprint, Response, current_app, jsonify, request

import infrastructure.database_pg as database_pg
from infrastructure.database_pg import edit_tokens, get_user_by_username, log_token_usage
from utils.logging_config import get_request_id
from utils.usage_request_state import set_usage_log_id
from infrastructure.ai import choose_emb_model
from infrastructure.vector_store import MauiVectorStore
from services.completion_service import complete_chat, CompletionRequest
from services.agentchat_service import run_agentchat
from config import PROVIDER_API_KEY_MAP
from routes.utils import assert_valid_api_key

rag_bp = Blueprint("rag", __name__)

logger = logging.getLogger(__name__)


@rag_bp.route("/completion.json", methods=["POST"])
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

        config = current_app.config["MAUI_CONFIG"]

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
        namespace = r.get("namespace") or config.rag.default_namespace

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
            chat_request,
            store,
            llm_type,
            model,
            language,
            api_key=provider_api_key,
            top_k=config.rag.top_k,
            min_sim=config.rag.min_sim,
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
                        service="/completion.json",
                        request_id=get_request_id(),
                        source=user.get("client"),
                    )
                    set_usage_log_id(log_id)

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
        logger.error("event=completion_request_failed error=%s", str(e))
        return jsonify({"error": "An unexpected error occurred"}), 500


@rag_bp.route("/agentchat", methods=["POST"])
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

        config = current_app.config["MAUI_CONFIG"]

        # === PARAMETERS WITH FALLBACK ===

        chat = r["chat"]
        if not isinstance(chat, list) or not chat:
            return jsonify({"error": "Invalid 'chat': expected non-empty list"}), 400

        namespace = r.get("namespace") or config.rag.default_namespace
        language = r.get("language") or "ITA"
        token_cost = config.completion_token_cost

        logger.info(
            "event=agentchat_request_started namespace=%s language=%s",
            namespace,
            language,
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
            user = get_user_by_username(r["username"])
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
                service="/agentchat",
                request_id=get_request_id(),
                source=user.get("client"),
            )
            set_usage_log_id(log_id)

        except Exception as e:
            logger.error("event=agentchat_token_usage_log_failed error=%s", e)

        # === TOKEN MANAGEMENT ===

        answer_text = payload.get("answer", "")
        if answer_text:
            edit_tokens(r["username"], -token_cost)

        logger.info(
            "event=agentchat_request_completed duration_ms=%s "
            "tools=%s vectors=%s follow_ups=%s",
            duration_ms,
            len(payload.get("tool_calls", [])),
            len(payload.get("vectors", [])),
            len(payload.get("follow_ups", [])),
        )

        if log_id is not None:
            payload["log_id"] = log_id

        return jsonify(payload), 200

    except RuntimeError as e:
        logger.error("event=agentchat_runtime_error error=%s", str(e))
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error("event=agentchat_unexpected_error error=%s", str(e))
        logger.error("event=agentchat_unexpected_error_trace trace=%s", traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred"}), 500
