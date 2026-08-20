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

# --- Q4 bounding policy (design §G) ----------------------------------------
# Two runtime normalization limits: oversized-but-valid content is truncated
# deterministically and NEVER raises, so logging cannot break the request it
# describes.
_MESSAGE_MAX_CHARS = 1000
_DETAILS_STR_VALUE_MAX_CHARS = 200

# Two programmer-contract limits: the shape of `details` is authored in a
# literal at the call site, so a violation is a programmer error and raises
# ValueError. Nothing is ever dropped, sorted-and-kept, or truncated here.
_DETAILS_MAX_KEYS = 20
_DETAILS_KEY_MAX_CHARS = 64

_TRUNCATION_MARKER = "...[truncated]"


def _truncate(value: str, limit: int) -> str:
    """Bound a valid runtime string to `limit` characters, marker included.

    Pure slicing: deterministic, silent, no I/O, no encoding, and it cannot
    raise. On the truncating path the result length is exactly `limit`, never
    `limit + len(_TRUNCATION_MARKER)`.
    """
    if len(value) <= limit:
        return value
    return value[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _normalize_message(message: Optional[str]) -> Optional[str]:
    if message is None:
        return None
    return _truncate(message, _MESSAGE_MAX_CHARS)


def _normalize_details(details: Optional[dict]) -> Optional[dict]:
    """Return a NEW dict with every key preserved and only oversized string
    values truncated. Runs only after structural validation has succeeded, so
    every value is already a permitted scalar. Never mutates the caller's dict.
    """
    if details is None:
        return None
    normalized: dict = {}
    for key, value in details.items():
        if isinstance(value, str):
            normalized[key] = _truncate(value, _DETAILS_STR_VALUE_MAX_CHARS)
        else:
            normalized[key] = value
    return normalized


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
    if len(details) > _DETAILS_MAX_KEYS:
        raise ValueError(
            f"details must declare at most {_DETAILS_MAX_KEYS} keys"
        )
    for key, value in details.items():
        if not isinstance(key, str):
            raise ValueError("details keys must be str")
        if len(key) > _DETAILS_KEY_MAX_CHARS:
            raise ValueError(
                f"details keys must be at most {_DETAILS_KEY_MAX_CHARS} "
                "characters"
            )
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

    # Normalization happens after validation and before both `extra` and the
    # rendered string are built, so stderr and the persistent metadata derive
    # from one and the same bounded value.
    message = _normalize_message(message)
    details = _normalize_details(details)

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
