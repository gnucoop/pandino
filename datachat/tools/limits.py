"""Shared result caveats and limits for DataChat tools."""

from typing import Any, Optional

# Initial product threshold for flagging grouped results that rest on few rows.
# A factual cue for the reader, not a statistical definition of reliability.
MIN_RELIABLE_SAMPLE = 15


def sample_warning(smallest_n: int, count: int) -> Optional[str]:
    """Describe groups below MIN_RELIABLE_SAMPLE rows, or None if there are none."""
    if count <= 0 or smallest_n >= MIN_RELIABLE_SAMPLE:
        return None
    return (
        f"{count} group(s) are based on fewer than {MIN_RELIABLE_SAMPLE} rows "
        f"(the smallest has {smallest_n})."
    )


class InvalidLimit(ValueError):
    """Raised when an explicit result limit is not a positive integer."""


def optional_limit(n: Any) -> Optional[int]:
    """Return an explicit positive limit, or None (no limit) when n is omitted."""
    if n is None:
        return None
    try:
        limit = int(n)
    except (TypeError, ValueError):
        raise InvalidLimit(f"Invalid n: {n!r}. Expected a positive integer.") from None
    if limit < 1:
        raise InvalidLimit(f"Invalid n: {n!r}. Expected a positive integer.")
    return limit


def invalid_limit_error(exc: InvalidLimit) -> dict[str, Any]:
    return {"kind": "error", "message": str(exc), "code": "INVALID_LIMIT"}
