import os
import requests


def external_authenticate(email: str, auth_token: str, client: str) -> str | None:
    gateway_url = os.environ.get("AUTH_GATEWAY_URL", "http://localhost:3000/validate")

    payload = {
        "email": email,
        "auth_token": auth_token,
        "client": client,
    }

    try:
        response = requests.post(gateway_url, json=payload, timeout=5)

        if response.status_code == 200:
            return None

        try:
            data = response.json()
            return data.get("error", "External authentication failed")
        except ValueError:
            return "External authentication failed"

    except requests.RequestException as e:
        return f"Gateway connection error: {e}"
