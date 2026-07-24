import os

from flask import Blueprint, Response, jsonify, request, current_app

from config import PROVIDER_API_KEY_MAP
from infrastructure.dino import dino_authenticate
from infrastructure.external_auth import external_authenticate
from services.rag_ingestion_service import process_rag_file

ingestion_bp = Blueprint("ingestion", __name__)

_TEXT_CONTENT_TYPE = {"Content-Type": "text/plain"}


@ingestion_bp.route("/storeragfile", methods=["POST"])
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
        return "Missing authToken", 400, _TEXT_CONTENT_TYPE
    if client != "dino" and not user_email:
        return "Missing userEmail", 400, _TEXT_CONTENT_TYPE
    if client == "dino" and not graphql_url:
        return "Missing graphqlUrl", 400, _TEXT_CONTENT_TYPE

    if client == "dino":
        err = dino_authenticate(graphql_url, auth_token)
    else:
        assert user_email is not None
        err = external_authenticate(user_email, auth_token, client, graphql_url)

    if err:
        return str(err), 403, _TEXT_CONTENT_TYPE

    file = request.files.get("file")
    url = request.form.get("url")
    namespace = request.form.get("namespace") or ""
    language = request.form.get("language")

    if not file:
        return "File not provided", 400, _TEXT_CONTENT_TYPE
    if not url:
        return "Url not provided", 400, _TEXT_CONTENT_TYPE

    config = current_app.config["MAUI_CONFIG"]

    asr_provider = config.models.asr_provider
    asr_api_key = os.getenv(
        PROVIDER_API_KEY_MAP.get(asr_provider or "", "")
    )
    asr_base_url = config.models.asr_base_url

    try:
        result = process_rag_file(
            file,
            url,
            namespace,
            language,
            asr_model=config.models.asr_model,
            asr_provider=asr_provider,
            asr_api_key=asr_api_key,
            asr_base_url=asr_base_url,
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
        return str(e), 400, _TEXT_CONTENT_TYPE
    except Exception as e:
        return str(e), 500, _TEXT_CONTENT_TYPE
