"""Process-local temporary CSV exports of DataChat table results.

Each export is a CSV file owned by the API key that produced it and addressed
by an opaque token. Entries expire after a fixed TTL, checked lazily whenever
the store is used, and each API key keeps at most a bounded number of exports.
"""

import csv
import datetime
import logging
import math
import os
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

EXPORT_TTL_S = 60 * 60
MAX_EXPORTS_PER_API_KEY = 20
DEFAULT_FILENAME = "datachat-result.csv"

_export_dir = os.path.join(tempfile.gettempdir(), "datachat_exports")


@dataclass(frozen=True)
class Export:
    token: str
    api_key: str
    path: str
    filename: str
    created_at: float


_exports: dict[str, Export] = {}
_lock = threading.Lock()


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        # pandas NaT is a datetime subclass that cannot be formatted.
        if str(value) == "NaT":
            return ""
        return value.isoformat()
    if isinstance(value, (str, int, float)):
        return value
    try:
        import pandas as pd  # noqa: PLC0415

        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _csv_cell(value.item())
    return str(value)


def _write_csv(path: str, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(row.get(column)) for column in columns])


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(
            "event=datachat_export_file_remove_failed error_type=%s",
            type(exc).__name__,
        )


def _pop_expired(now: float) -> list[Export]:
    expired = [e for e in _exports.values() if now - e.created_at >= EXPORT_TTL_S]
    for entry in expired:
        del _exports[entry.token]
    return expired


def _pop_overflow(api_key: str) -> list[Export]:
    owned = [e for e in _exports.values() if e.api_key == api_key]
    overflow = owned[: max(0, len(owned) - MAX_EXPORTS_PER_API_KEY)]
    for entry in overflow:
        del _exports[entry.token]
    return overflow


def register_export(
    api_key: str,
    columns: list[str],
    rows: Iterable[dict[str, Any]],
    filename: str = DEFAULT_FILENAME,
) -> Export:
    """Write ``rows`` as CSV with ``columns`` in order and register it for ``api_key``."""
    token = secrets.token_urlsafe(32)
    os.makedirs(_export_dir, exist_ok=True)
    path = os.path.join(_export_dir, f"{token}.csv")
    try:
        _write_csv(path, columns, rows)
    except Exception:
        _remove_file(path)
        raise

    entry = Export(
        token=token,
        api_key=str(api_key),
        path=path,
        filename=filename,
        created_at=time.time(),
    )
    with _lock:
        _exports[token] = entry
        stale = _pop_expired(entry.created_at) + _pop_overflow(entry.api_key)
    for old in stale:
        _remove_file(old.path)
    return entry


def resolve_export(token: str) -> Optional[Export]:
    """Return the live export for ``token``, or ``None`` if unknown or expired."""
    with _lock:
        expired = _pop_expired(time.time())
        entry = _exports.get(str(token or ""))
    for old in expired:
        _remove_file(old.path)
    if entry is None or not os.path.isfile(entry.path):
        return None
    return entry


def purge_exports(api_key: str) -> int:
    """Remove every export owned by ``api_key``; return how many were removed."""
    key = str(api_key)
    with _lock:
        owned = [e for e in _exports.values() if e.api_key == key]
        for entry in owned:
            del _exports[entry.token]
    for entry in owned:
        _remove_file(entry.path)
    return len(owned)
