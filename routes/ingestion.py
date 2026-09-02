import logging
import os

from flask import Blueprint, Response, jsonify, request, current_app

from config import PROVIDER_API_KEY_MAP
from infrastructure.database_pg import get_user_by_username
from infrastructure.dino import dino_authenticate
from infrastructure.external_auth import external_authenticate
from services.rag_ingestion_service import process_rag_file
from utils.logging_config import get_request_id
from utils.usage_attribution_state import bind_usage_attribution

ingestion_bp = Blueprint("ingestion", __name__)

logger = logging.getLogger(__name__)

_TEXT_CONTENT_TYPE = {"Content-Type": "text/plain"}

_SERVICE = "/storeragfile"


def _report_attribution_unavailable(reason: str, error_type: str | None = None) -> None:
    """Emit the one established, identity-safe attribution diagnostic.

    Carries only ``reason``, ``service``, ``request_id`` and ``error_type``:
    never the username, the userEmail, the configured technical username,
    the auth token, the API key or anything from the payload.
    """
    logger.warning(
        "event=embedding_usage_attribution_unavailable reason=%s "
        "service=%s request_id=%s error_type=%s",
        reason,
        _SERVICE,
        get_request_id(),
        error_type,
    )


def _bind_embedding_usage_attribution(username: str) -> None:
    """Best-effort bind of this request's Usage attribution metadata.

    Called once, after authentication has succeeded and before any
    embedding work can occur, so that embedding consumption accumulated
    later in the request can be attributed at request end.

    Purely observational: it resolves a user id and binds it, or it binds
    nothing and emits one safe diagnostic. It never raises and never
    changes the endpoint's control flow, status code or body. Deciding
    *which* username to pass - the authenticated real user, the legacy
    technical accounting identity, or none at all - is the caller's
    ingestion-specific responsibility and deliberately not part of this
    helper.

    ``source`` is always taken from the resolved row's ``client`` column,
    never hardcoded, so both the real and the technical identity are
    handled by the same tail.
    """
    reason = None
    error_type = None
    try:
        user = get_user_by_username(username)
        if not user:
            reason = "not_found"
        else:
            user_id = user.get("id")
            if isinstance(user_id, int):
                bind_usage_attribution(user_id, _SERVICE, user.get("client"))
            else:
                reason = "invalid_user_id"
    except Exception as exc:
        reason = "lookup_failed"
        error_type = type(exc).__name__

    if reason is not None:
        _report_attribution_unavailable(reason, error_type)


def _attribute_ingestion_request(
    client: str, user_email: str | None, using_legacy_dino_fallback: bool
) -> None:
    """Decide which identity, if any, this request may be attributed to.

    Three shapes, per the approved design:

    * explicit non-Dino client - the already-required, externally
      authenticated ``userEmail`` is the real identity and is used;
    * explicit ``client="dino"`` - the userEmail is NOT trusted and no
      technical identity is substituted; the request stays deliberately
      unattributed, silently, until Dino provides a verifiable identity
      contract;
    * actual legacy Dino fallback (no ``client`` supplied at all) - and
      only this path - may use the configured technical accounting
      identity. Absent configuration is the off-switch.
    """
    if using_legacy_dino_fallback:
        technical_username = current_app.config[
            "MAUI_CONFIG"
        ].dino_legacy_usage_username
        if not technical_username:
            _report_attribution_unavailable("not_configured")
            return
        _bind_embedding_usage_attribution(technical_username)
        return

    if client == "dino":
        # Explicitly declared Dino: the userEmail on this path is not
        # verified by Dino authentication, so it is not an identity we may
        # bill. No fallback identity either, and no diagnostic - this is
        # the expected steady state, not a failure.
        return

    if user_email:
        _bind_embedding_usage_attribution(user_email)


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
