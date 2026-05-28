"""Shared HTTP-layer utilities used across route Blueprints."""

from flask import abort
from infrastructure.database_pg import validate_api_key


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
