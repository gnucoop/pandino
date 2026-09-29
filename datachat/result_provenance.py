"""Request-local trusted facts about tool-produced tables.

A tool records what the backend knows about a table it produced (for example
that the SQL cap cut off further rows, or a deterministic caveat). The route
later looks those facts up for the table the agent returned.

Trust is tied to the identity of the tool's ``data`` list and a snapshot of its
shape taken at recording time. A table that was rebuilt, copied or reshaped by
the agent has no trusted facts: lookup returns ``None`` and callers must treat
source completeness as unknown. Whatever the payload itself says in ``meta`` or
``note`` is never consulted.

State lives on ``flask.g``, so it ends with the request. Outside an application
context recording is a no-op and lookup returns ``None``.
"""

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "TrustedResult",
    "lookup_trusted_result",
    "record_trusted_result",
]

_G_REGISTRY_ATTR = "_maui_datachat_trusted_results"


@dataclass(frozen=True)
class TrustedResult:
    more_rows_available: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class _Entry:
    # Kept to pin the list alive, so its id() cannot be reused by another
    # object during the request.
    data: list[Any]
    row_count: int
    first_row_keys: tuple[str, ...]
    facts: TrustedResult


def _first_row_keys(data: list[Any]) -> tuple[str, ...]:
    first = data[0] if data else None
    return tuple(str(k) for k in first) if isinstance(first, dict) else ()


def _registry() -> Optional[dict[int, _Entry]]:
    from flask import g, has_app_context  # noqa: PLC0415

    if not has_app_context():
        return None
    registry = getattr(g, _G_REGISTRY_ATTR, None)
    if registry is None:
        registry = {}
        setattr(g, _G_REGISTRY_ATTR, registry)
    return registry


def record_trusted_result(
    payload: dict[str, Any],
    *,
    more_rows_available: bool = False,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Record trusted facts for ``payload["data"]`` and return ``payload``."""
    data = payload.get("data")
    registry = _registry()
    if registry is None or not isinstance(data, list):
        return payload
    if not more_rows_available and not note:
        return payload

    registry[id(data)] = _Entry(
        data=data,
        row_count=len(data),
        first_row_keys=_first_row_keys(data),
        facts=TrustedResult(more_rows_available=more_rows_available, note=note or None),
    )
    return payload


def lookup_trusted_result(payload: Any) -> Optional[TrustedResult]:
    """Return the trusted facts for a final table payload, or ``None`` if unknown."""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    registry = _registry()
    if registry is None or not isinstance(data, list):
        return None

    entry = registry.get(id(data))
    if entry is None or entry.data is not data:
        return None
    if len(data) != entry.row_count or _first_row_keys(data) != entry.first_row_keys:
        return None
    return entry.facts
