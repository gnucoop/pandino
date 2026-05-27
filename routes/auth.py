import secrets
from datetime import datetime
from typing import Union

from flask import Blueprint, Response, jsonify, request

import infrastructure.database_pg as database_pg
from infrastructure.database_pg import validate_api_key
from infrastructure.dino import dino_authenticate
from infrastructure.external_auth import external_authenticate

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/checkpandinouser", methods=["POST"])
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


@auth_bp.route("/validateapikey", methods=["POST"])
def validate() -> tuple[Response, int]:
    api_key = request.headers.get("X-API-KEY")
    user_email = request.headers.get("X-USER-EMAIL")
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
