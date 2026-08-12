import json
import logging
import os
from typing import TypedDict
from flask import Blueprint, jsonify, request, current_app
from config import PROVIDER_API_KEY_MAP
from infrastructure.database_pg import edit_tokens, log_token_usage
import infrastructure.database_pg as database_pg
from services.document_comparison_service import (
    CONTEXT_WINDOW_ERROR_MESSAGE,
    DocumentComparisonPayloadTooLargeError,
    compare_documents,
)
from services.document_extraction_service import extract_document_text_with_metadata
from services.document_text_service import DocumentInput
from routes.utils import assert_valid_api_key

documents_bp = Blueprint("documents", __name__)

logger = logging.getLogger(__name__)


class TokenUsageDict(TypedDict):
    """
    Token usage counters shared by extraction and comparison steps.

    The route uses this shape to aggregate OCR usage with final comparison
    usage before writing the single compare_docs accounting log row.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int


def _zero_token_usage() -> TokenUsageDict:
    """
    Create an empty token usage accumulator for the current request.
    """
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _add_token_usage(total: TokenUsageDict, usage: TokenUsageDict) -> None:
    """
    Add one token usage record into the request-level accumulator.

    This mutates total in place and keeps usage read-only from the caller's
    perspective.
    """
    total["input_tokens"] += usage["input_tokens"]
    total["output_tokens"] += usage["output_tokens"]
    total["total_tokens"] += usage["total_tokens"]


@documents_bp.route("/compare_docs", methods=["POST"])
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

    config = current_app.config["MAUI_CONFIG"]
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
    ocr_provider = config.models.vision_provider
    ocr_model = config.models.vision_model
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
        ocr_token_usage = _zero_token_usage()
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

            extraction_result = extract_document_text_with_metadata(
                doc_input,
                ocr_provider=ocr_provider,
                ocr_model=ocr_model,
                ocr_api_key=None,
            )
            normalized_documents.append(extraction_result["document"])
            _add_token_usage(ocr_token_usage, extraction_result["ocr_token_usage"])

        for index, file in enumerate(files):
            role = file_roles[index] if file_roles else None

            doc_input: DocumentInput = {
                "content": file,
                "filename": file.filename,
                "source_type": "file",
                "role": role,
            }

            extraction_result = extract_document_text_with_metadata(
                doc_input,
                ocr_provider=ocr_provider,
                ocr_model=ocr_model,
                ocr_api_key=None,
            )
            normalized_documents.append(extraction_result["document"])
            _add_token_usage(ocr_token_usage, extraction_result["ocr_token_usage"])

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

            # COOPI release: aggregate OCR into the single compare_docs log row
            # because OCR and comparison share provider/model/cost basis there.
            # Usage stays separate internally so operation-level logging can
            # be introduced later if those models diverge.
            log_token_usage(
                user_id=user_id,
                token_input=token_usage.get("input_tokens", 0)
                + ocr_token_usage["input_tokens"],
                token_output=token_usage.get("output_tokens", 0)
                + ocr_token_usage["output_tokens"],
                model=model,
                provider=llm_type,
                service="/compare_docs",
            )

        except Exception as error:
            logger.error(
                "event=compare_docs_token_usage_log_failed error=%s", error
            )

        edit_tokens(user_email, -token_cost)

    except ValueError as error:
        return jsonify({"error": "Invalid request", "details": str(error)}), 400

    except DocumentComparisonPayloadTooLargeError:
        return (
            jsonify(
                {
                    "error": "Payload too large",
                    "details": CONTEXT_WINDOW_ERROR_MESSAGE,
                }
            ),
            413,
        )

    except NotImplementedError as error:
        return (
            jsonify({"error": "Unsupported document format", "details": str(error)}),
            415,
        )

    return jsonify(result), 200
