"""
Emission contract for persistent Operational events.

This module builds the ``(message, extra)`` pair a call site passes to its
own module logger:

    message, extra = build_operational_event(event="...", ...)
    logger.info(message, extra=extra)

It owns no logger and performs no logging, routing or persistence of any
kind. ``logger.<level>()`` remains the emission boundary, so ``record.name``
still resolves to the emitting module. Stdlib-only.

``request_id``/``app_id`` are infrastructure-owned correlation metadata and
are deliberately absent from this API: there is no keyword for them, and no
``**kwargs`` exists to smuggle them in, so supplying them fails as a
``TypeError`` at the call site.
"""

import re
from typing import Optional

__all__ = ["build_operational_event"]

_EVENT_NAME_RE = re.compile(r"^[a-z0-9_]+$")

_SCALAR_DETAIL_TYPES = (str, int, float, bool, type(None))


def _validate_event(event: str) -> None:
    if not isinstance(event, str) or not event:
        raise ValueError("event must be a non-empty str")
    if not _EVENT_NAME_RE.match(event):
        raise ValueError(
            "event must be lowercase snake_case matching [a-z0-9_]+"
        )


def _validate_optional_str(name: str, value: Optional[str]) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a str or None")


def _validate_duration_ms(duration_ms: Optional[int]) -> None:
    if duration_ms is None:
        return
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
        raise ValueError("duration_ms must be an int or None")


def _validate_details(details: Optional[dict]) -> None:
    if details is None:
        return
    if not isinstance(details, dict):
        raise ValueError("details must be a dict or None")
    for key, value in details.items():
        if not isinstance(key, str):
            raise ValueError("details keys must be str")
        if isinstance(value, bool):
            continue
        if not isinstance(value, _SCALAR_DETAIL_TYPES):
            raise ValueError(
                "details values must be str, int, float, bool or None"
            )


def build_operational_event(
    *,
    event: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    duration_ms: Optional[int] = None,
    error_type: Optional[str] = None,
    details: Optional[dict] = None,
    message: Optional[str] = None,
) -> "tuple[str, dict]":
    """Build the (message, extra) pair for a persistent Operational event.

    Returns a fully rendered message string (no format args) and an extra
    mapping carrying only the maui_* metadata actually supplied, plus the
    always-present maui_persist and maui_event.
    """
    _validate_event(event)
    _validate_optional_str("provider", provider)
    _validate_optional_str("model", model)
    _validate_optional_str("error_type", error_type)
    _validate_optional_str("message", message)
    _validate_duration_ms(duration_ms)
    _validate_details(details)

    extra: dict = {
        "maui_persist": True,
        "maui_event": event,
    }
    if provider is not None:
        extra["maui_provider"] = provider
    if model is not None:
        extra["maui_model"] = model
    if duration_ms is not None:
        extra["maui_duration_ms"] = duration_ms
    if error_type is not None:
        extra["maui_error_type"] = error_type
    if details is not None:
        extra["maui_details"] = details
    if message is not None:
        extra["maui_message"] = message

    parts = [f"event={event}"]
    if provider is not None:
        parts.append(f"provider={provider}")
    if model is not None:
        parts.append(f"model={model}")
    if duration_ms is not None:
        parts.append(f"duration_ms={duration_ms}")
    if error_type is not None:
        parts.append(f"error_type={error_type}")
    if details is not None:
        for key in sorted(details):
            parts.append(f"{key}={details[key]}")
    if message is not None:
        parts.append(f"msg={message}")

    return " ".join(parts), extra
