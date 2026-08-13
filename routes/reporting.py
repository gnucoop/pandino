import os

from flask import Blueprint, request, current_app

from config import PROVIDER_API_KEY_MAP
from infrastructure.database_pg import (
    edit_tokens,
    log_token_usage,
    get_user_by_username,
)
from utils.logging_config import get_request_id
from utils.usage_request_state import set_usage_log_id
import infrastructure.database_pg as database_pg
from services.prompt_service import reply_to_prompt
from routes.utils import assert_valid_api_key

reporting_bp = Blueprint("reporting", __name__)

_TEXT_CONTENT_TYPE = {"Content-Type": "text/plain; charset=utf-8"}


@reporting_bp.route("/prompt.txt", methods=["POST"])
def prompt_handler():
    prompt = request.form.get("prompt")
    username = request.form.get("username")
    api_key = request.headers.get("X-API-KEY")

    if not api_key:
        return "Missing X-API-KEY header", 400, _TEXT_CONTENT_TYPE

    if not prompt:
        return "No prompt provided", 400, _TEXT_CONTENT_TYPE
    if not username:
        return "Username not provided", 400, _TEXT_CONTENT_TYPE

    assert_valid_api_key(api_key, username)

    user_tokens = database_pg.get_user_tokens(username)
    if user_tokens is None:
        return "Could not retrieve user tokens", 500, _TEXT_CONTENT_TYPE

    config = current_app.config["MAUI_CONFIG"]
    token_cost = config.prompt_token_cost
    if token_cost > user_tokens:
        return f"Not enough tokens, user_tokens: {user_tokens}", 400, _TEXT_CONTENT_TYPE

    llm_type = config.models.prompt_provider
    model_name = config.models.prompt_model
    provider_api_key = os.getenv(PROVIDER_API_KEY_MAP.get(llm_type, ""))

    language = request.form.get("language", "ITA")

    try:
        result = reply_to_prompt(
            prompt,
            llm_type,
            model_name,
            language=language,
            api_key=provider_api_key,
        )
    except RuntimeError as e:
        return str(e), 500, _TEXT_CONTENT_TYPE

    token_usage = result["token_usage"]
    user = database_pg.get_user_by_username(username)
    if user and (token_usage["input_tokens"] > 0 or token_usage["output_tokens"] > 0):
        log_id = log_token_usage(
            user_id=int(user["id"]),
            token_input=token_usage["input_tokens"],
            token_output=token_usage["output_tokens"],
            model=model_name,
            provider=llm_type,
            service="/prompt.txt",
            request_id=get_request_id(),
        )
        set_usage_log_id(log_id)

    edit_tokens(username, -token_cost)

    return result["content"], 200, _TEXT_CONTENT_TYPE
