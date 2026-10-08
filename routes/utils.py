"""Shared HTTP-layer utilities used across route Blueprints."""

import logging
from typing import Any, Optional

from flask import abort, current_app
from infrastructure.database_pg import validate_api_key, get_user_by_username, log_token_usage
from utils.agent_serialization import serialize_runresult
from utils.agent_logging import log_runresult


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


def log_agent_run(
    engine: Any,
    *,
    namespace: str,
    user_email: str,
    question: str,
    request_id: str,
    response_kind: Optional[str],
    model: str,
    provider: str,
    runtime_logger: logging.Logger,
    engine_name: str,
) -> Optional[int]:
    """
    Log the engine's last run: structured trace to file, token usage to the DB.

    Never raises: a logging failure must not fail the user's request. Every
    outcome is reported on the runtime logger as one `<namespace>_trace_status`
    line.

    :param engine: the engine that just ran; its get_last_trace() is read if present.
    :param namespace: agent namespace, used for the structured log and the log prefix.
    :param user_email: requesting user, for the structured log and the token log.
    :param question: the user message that triggered the run.
    :param request_id: correlation id of the HTTP request.
    :param response_kind: kind of the engine's answer, for the structured log.
    :param model: model name recorded with the token usage.
    :param provider: provider name recorded with the token usage.
    :param runtime_logger: the agent's runtime logger.
    :param engine_name: engine label for the status line.
    :return: the token-usage log id, or None when nothing was logged to the DB.
    """
    trace = None
    log_id: Optional[int] = None
    structured_log_ok = False
    db_log_ok = False
    if hasattr(engine, "get_last_trace"):
        try:
            trace = engine.get_last_trace()  # type: ignore[attr-defined]
        except Exception as e:
            current_app.logger.warning(f"[{namespace}] Failed to read engine trace: {e}")

    trace_payload: Optional[dict[str, Any]] = None
    if isinstance(trace, dict) and trace.get("run_result") is not None:
        try:
            trace_payload = serialize_runresult(trace["run_result"])
            if isinstance(trace_payload.get("metrics"), dict):
                trace_payload["metrics"]["duration_ms"] = trace.get("duration_ms")
        except Exception as e:
            current_app.logger.error(f"[{namespace}] Failed to serialize trace: {e}")

    if trace_payload is not None:
        try:
            log_runresult(
                trace["run_result"],
                user=user_email,
                namespace=namespace,
                language="N/A",
                question=str(question),
                extra={
                    "channel": namespace,
                    "response_kind": response_kind,
                    "request_id": request_id,
                },
            )
            structured_log_ok = True
        except Exception as e:
            current_app.logger.error(f"[{namespace}] Structured logging failed: {e}")

        try:
            user = get_user_by_username(user_email)
            if not user:
                raise ValueError(f"User '{user_email}' not found in DB")

            user_id = user.get("id")
            if not isinstance(user_id, int):
                raise TypeError(f"Invalid user_id: {user_id}")

            token_metrics = trace_payload.get("metrics", {}).get("token_usage", {})
            token_input = token_metrics.get("input") or 0
            token_output = token_metrics.get("output") or 0

            log_id = log_token_usage(
                user_id=user_id,
                token_input=token_input,
                token_output=token_output,
                model=model,
                provider=provider,
            )
            db_log_ok = True
            current_app.logger.info(f"[{namespace}] token usage logged log_id={log_id}")
        except Exception as e:
            current_app.logger.error(f"[{namespace}] Failed to log token usage: {e}")

    runtime_logger.info(
        "%s_trace_status request_id=%s user=%s engine=%s trace_present=%s structured_log_ok=%s db_log_ok=%s log_id=%s",
        namespace,
        request_id,
        user_email,
        engine_name,
        bool(trace_payload is not None),
        structured_log_ok,
        db_log_ok,
        log_id if log_id is not None else "none",
    )
    return log_id
