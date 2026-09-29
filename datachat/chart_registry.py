"""Request-local charts produced by the chart tool.

A successful chart tool call records the chart spec it built, together with the
trusted caveat of the table it was drawn from, if any. The route attaches the
recorded specs to the response as ``charts``. Nothing the agent writes into its
final answer can add a chart: only specs recorded here reach the client.

State lives on ``flask.g``, so it ends with the request. Outside an application
context recording is a no-op and lookup returns an empty list.
"""

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "RecordedChart",
    "get_recorded_charts",
    "record_chart",
]

_G_REGISTRY_ATTR = "_maui_datachat_charts"


@dataclass(frozen=True)
class RecordedChart:
    spec: dict[str, Any]
    # Backend-trusted caveat of the source table (see result_provenance).
    note: Optional[str] = None


def _registry(create: bool) -> Optional[list[RecordedChart]]:
    from flask import g, has_app_context  # noqa: PLC0415

    if not has_app_context():
        return None
    registry = getattr(g, _G_REGISTRY_ATTR, None)
    if registry is None and create:
        registry = []
        setattr(g, _G_REGISTRY_ATTR, registry)
    return registry


def record_chart(spec: dict[str, Any], *, note: Optional[str] = None) -> None:
    """Append ``spec`` to this request's charts, in production order."""
    registry = _registry(create=True)
    if registry is not None:
        registry.append(RecordedChart(spec=spec, note=note or None))


def get_recorded_charts() -> list[RecordedChart]:
    """The charts recorded during this request, oldest first."""
    return list(_registry(create=False) or [])
