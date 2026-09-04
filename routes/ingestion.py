import os

from flask import Blueprint, Response, jsonify, request, current_app

from config import PROVIDER_API_KEY_MAP
from infrastructure.dino import dino_authenticate
from infrastructure.external_auth import external_authenticate
from services.rag_ingestion_service import process_rag_file
from usage.attribution import (
    USAGE_POLICY_LEGACY_DINO_INGESTION,
    attribute_usage_to_policy,
    attribute_usage_to_user,
    declare_usage_unattributed,
)

ingestion_bp = Blueprint("ingestion", __name__)

_TEXT_CONTENT_TYPE = {"Content-Type": "text/plain"}


def _attribute_ingestion_request(
    client: str, user_email: str | None, using_legacy_dino_fallback: bool
) -> None:
    """Declare which Usage attribution intent, if any, this request carries.

    Three shapes, per the approved design, and the distinction between the
    last two is the whole point: after the route's backward-compatibility
    rewrite, ``client == "dino"`` no longer tells a request that asked for
    Dino apart from one that said nothing.

    * explicit non-Dino client - the already-required, externally
      authenticated ``userEmail`` is the real identity and is attributed;
    * actual legacy Dino fallback (no ``client`` supplied at all) - and
      only this path - belongs to the technical accounting policy;
    * explicit ``client="dino"`` - the userEmail is NOT verified by Dino
      authentication and no technical identity is substituted; the request
      is deliberately unattributed, and silently so, until Dino provides a
      verifiable identity contract.

    The route owns the branch recognition, the authentication that makes a
    real identity usable, and the point in the flow at which the intent
    becomes valid. Everything behind the declaration - identity
    resolution, source policy, technical provisioning, service derivation,
    binding, fail-open diagnostics - belongs to the Usage boundary.
    """
    if using_legacy_dino_fallback:
        attribute_usage_to_policy(policy=USAGE_POLICY_LEGACY_DINO_INGESTION)
        return

    if client == "dino":
        declare_usage_unattributed()
        return

    if user_email:
        attribute_usage_to_user(username=user_email)


@ingestion_bp.route("/storeragfile", methods=["POST"])
def store_rag_file() -> tuple[Response, int] | tuple[str, int, dict[str, str]]:
    graphql_url = request.form.get("graphqlUrl")
    auth_token = request.form.get("authToken")
    user_email = request.form.get("userEmail")
    client = request.form.get("client")

    # Captured before the fallback below overwrites it: afterwards
    # client == "dino" no longer distinguishes a request that asked for
    # Dino from one that said nothing. Only the latter may use the
    # technical accounting identity.
    using_legacy_dino_fallback = not client

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

    # Observational only, and bound here so it precedes every embedding
    # contribution this route can produce (all of them originate inside
    # process_rag_file below). Never affects the HTTP contract.
    _attribute_ingestion_request(client, user_email, using_legacy_dino_fallback)

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
