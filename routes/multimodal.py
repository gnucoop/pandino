import base64
import logging
import os
from typing import Union

from flask import Blueprint, Response, current_app, jsonify, request

from config import PROVIDER_API_KEY_MAP
from services.document_text_service import extract_and_normalize_document, DocumentInput
import infrastructure.database_pg as database_pg
from infrastructure.ai import describe_image_with_usage, asr_response
from infrastructure.asr_accounting import resolve_asr_cost
from infrastructure.database_pg import edit_tokens, log_token_usage, log_usage_with_resolved_cost
from utils.logging_config import get_request_id
from utils.operational_event import build_operational_event
from utils.usage_request_state import set_usage_log_id
from services.audio_form_service import audioFormCompilation, audioFormPromptBuild
from routes.utils import assert_valid_api_key

multimodal_bp = Blueprint("multimodal", __name__)

logger = logging.getLogger(__name__)


# Define a route for the '/transcribe' endpoint
@multimodal_bp.route("/transcribe", methods=["POST"])
def asr_parse() -> Union[Response, tuple[Response, int]]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
    user_name_header = request.headers.get("X-USER-NAME")

    if not api_key:
        message, extra = build_operational_event(
            event="transcribe_request_rejected",
            details={"reason": "missing_required_header"},
        )
        logger.warning(message, extra=extra)
        return jsonify({"error": "Missing X-API-KEY header"}), 400
    if not user_email:
        message, extra = build_operational_event(
            event="transcribe_request_rejected",
            details={"reason": "missing_required_header"},
        )
        logger.warning(message, extra=extra)
        return jsonify({"error": "Missing X-USER-EMAIL header"}), 400
    if not user_name_header:
        message, extra = build_operational_event(
            event="transcribe_request_rejected",
            details={"reason": "missing_required_header"},
        )
        logger.warning(message, extra=extra)
        return jsonify({"error": "Missing X-USER-NAME header"}), 400

    user_name = user_name_header.replace(" ", "_").strip()
    assert_valid_api_key(api_key, user_email)

    file = request.files.get("file")
    if not file:
        message, extra = build_operational_event(
            event="transcribe_request_rejected",
            details={"reason": "missing_file"},
        )
        logger.warning(message, extra=extra)
        return jsonify({"error": "Missing file"}), 400

    lang = request.form.get("lang") or "ENG"

    config = current_app.config["MAUI_CONFIG"]

    if file.mimetype.startswith("audio"):
        message, extra = build_operational_event(
            event="transcribe_branch_selected",
            details={"branch": "audio"},
        )
        logger.info(message, extra=extra)

        asr_provider = config.models.asr_provider
        asr_base_url = config.models.asr_base_url
        asr_api_key = os.getenv(
            PROVIDER_API_KEY_MAP.get(asr_provider or "", "")
        ) or ""

        if not config.models.asr_model:
            message, extra = build_operational_event(
                event="transcribe_operation_blocked",
                details={"branch": "audio", "reason": "missing_model"},
            )
            logger.warning(message, extra=extra)
            return jsonify({"error": "Missing ASR configuration"}), 500

        if asr_provider in PROVIDER_API_KEY_MAP and not asr_api_key:
            message, extra = build_operational_event(
                event="transcribe_operation_blocked",
                provider=asr_provider,
                details={"branch": "audio", "reason": "missing_api_key"},
            )
            logger.warning(message, extra=extra)
            return jsonify({"error": "Missing ASR configuration"}), 500

        if (
            asr_provider == "Mistral"
            and config.models.asr_mistral_price_per_minute_usd is None
        ):
            message, extra = build_operational_event(
                event="transcribe_operation_blocked",
                provider=asr_provider,
                details={"branch": "audio", "reason": "missing_price"},
            )
            logger.warning(message, extra=extra)
            return jsonify({"error": "Missing ASR configuration"}), 500

        try:
            response = asr_response(
                file,
                asr_provider,
                config.models.asr_model,
                asr_api_key,
                asr_base_url,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 500

        if response.status_code == 200:
            try:
                payload = response.json()
            except Exception as e:
                message, extra = build_operational_event(
                    event="transcribe_operation_failed",
                    provider=asr_provider,
                    model=config.models.asr_model,
                    error_type=type(e).__name__,
                    details={"branch": "audio", "reason": "invalid_response"},
                )
                logger.error(message, extra=extra)
                return jsonify({"error": f"Invalid JSON from ASR: {str(e)}"}), 500

            text = payload.get("text")
            if text is None:
                message, extra = build_operational_event(
                    event="transcribe_operation_failed",
                    provider=asr_provider,
                    model=config.models.asr_model,
                    details={"branch": "audio", "reason": "missing_result"},
                )
                logger.error(message, extra=extra)
                return (
                    jsonify({"error": "ASR response missing 'text' field"}),
                    500,
                )

            message, extra = build_operational_event(
                event="transcribe_operation_completed",
                provider=asr_provider,
                model=config.models.asr_model,
                details={"branch": "audio"},
            )
            logger.info(message, extra=extra)

            if asr_provider in ("Deepinfra", "Mistral"):
                try:
                    cost = resolve_asr_cost(
                        asr_provider,
                        payload,
                        mistral_price_per_minute=config.models.asr_mistral_price_per_minute_usd,
                    )
                    user = database_pg.get_user_by_username(user_email)
                    if user is None:
                        raise RuntimeError(f"ASR usage user lookup failed for {user_email}")

                    log_id = log_usage_with_resolved_cost(
                        user_id=user["id"],
                        cost=cost,
                        model=config.models.asr_model,
                        provider=asr_provider,
                        service="/transcribe",
                        request_id=get_request_id(),
                        source=user.get("client"),
                    )
                    set_usage_log_id(log_id)
                except Exception:
                    logger.error(
                        "event=asr_usage_accounting_failed provider=%s model=%s",
                        asr_provider,
                        config.models.asr_model,
                        exc_info=True,
                    )

            return jsonify({"text": text}), 200
        else:
            message, extra = build_operational_event(
                event="transcribe_operation_failed",
                provider=asr_provider,
                model=config.models.asr_model,
                details={"branch": "audio", "reason": "http_error"},
            )
            logger.error(message, extra=extra)
            return jsonify({"error": "ASR transcription failed"}), 500

    filename = file.filename or ""
    ext = os.path.splitext(filename.lower())[1]

    if ext in (".pdf", ".docx", ".rtf"):
        message, extra = build_operational_event(
            event="transcribe_branch_selected",
            details={"branch": "document"},
        )
        logger.info(message, extra=extra)

        try:
            doc_input: DocumentInput = {
                "source_type": "file",
                "content": file,
                "filename": filename,
                "role": None,
            }
            result = extract_and_normalize_document(doc_input)
            return jsonify({"text": result["text"]}), 200
        except NotImplementedError:
            return jsonify({"error": f"Unsupported file format: {filename}"}), 415
        except ValueError as e:
            return jsonify({"error": str(e)}), 422
        except Exception as e:
            return jsonify({"error": f"Error extracting text from file: {str(e)}"}), 422

    if file.mimetype.startswith("image"):
        message, extra = build_operational_event(
            event="transcribe_branch_selected",
            details={"branch": "image"},
        )
        logger.info(message, extra=extra)

        vision_provider = config.models.vision_provider or ""
        vision_model = config.models.vision_model or ""

        try:
            b64 = base64.b64encode(file.read()).decode()
            dataurl = f"data:{file.mimetype};base64,{b64}"
            result = describe_image_with_usage(
                dataurl,
                vision_provider,
                vision_model,
                api_key=os.getenv(
                    PROVIDER_API_KEY_MAP.get(vision_provider, "")
                ),
            )
        except Exception as e:
            return (
                jsonify({"error": f"Error extracting text from image: {str(e)}"}),
                500,
            )

        text = result["description"]

        try:
            token_usage = result["token_usage"]
            user = database_pg.get_user_by_username(user_email)
            if user is None:
                raise RuntimeError(f"Vision usage user lookup failed for {user_email}")

            log_id = log_token_usage(
                user_id=user["id"],
                token_input=token_usage["input_tokens"],
                token_output=token_usage["output_tokens"],
                model=vision_model,
                provider=vision_provider,
                service="/transcribe",
                request_id=get_request_id(),
                source=user.get("client"),
            )
            set_usage_log_id(log_id)
        except Exception:
            logger.error(
                "event=vision_usage_accounting_failed provider=%s model=%s",
                vision_provider,
                vision_model,
                exc_info=True,
            )

        return jsonify({"text": text}), 200

    message, extra = build_operational_event(
        event="transcribe_branch_selected",
        details={"branch": "reject"},
    )
    logger.warning(message, extra=extra)

    return jsonify({"error": f"Unexpected file mimetype: {file.mimetype}"}), 400


# Define a route for the '/audioformcompilation' endpoint
@multimodal_bp.route("/audioformcompilation", methods=["POST"])
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

    config = current_app.config["MAUI_CONFIG"]

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
        log_id = log_token_usage(
            user_id=user["id"],
            token_input=token_usage["input_tokens"],
            token_output=token_usage["output_tokens"],
            model=model_name,
            provider=llm_type,
            service="/audioformcompilation",
            request_id=get_request_id(),
            source=user.get("client"),
        )
        set_usage_log_id(log_id)

    edit_tokens(user_email, -token_cost)

    logger.debug("event=audio_form_compile_result content=%s", result["content"])
    return jsonify(result["content"]), 200
