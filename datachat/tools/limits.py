"""Shared row/column limit handling for DataChat tools.

Tools used to cap their output at a hardcoded ceiling (50 rows, 20 rows, 10 columns) and
return the truncated result as if it were complete. Since table responses are now
previewed and exported to CSV by the transport layer, a cap inside a tool no longer keeps
the payload small -- it silently shrinks the user's download instead.

The rule these helpers encode:
  - no implicit cap; return everything and let the preview/export layer size the response
  - an explicit caller-supplied limit is honoured, but never silently
"""

from typing import Any, Optional

# Below this many observations a group mean is too noisy to rank or compare on.
MIN_RELIABLE_SAMPLE = 15


def resolve_limit(n: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Turn a caller-supplied `n` into a positive cap, or None meaning "no limit".

    `default` is for tools whose whole purpose is to return a handful of rows
    (top_rows, sample_rows); leave it None everywhere else.

    0 and None both mean "no limit" -- a request for zero rows is never what a caller
    meant, and treating it as unlimited matches the `n or default` idiom callers expect.
    """
    fallback = max(1, int(default)) if default else None

    if not n:
        return fallback
    try:
        return max(1, int(n))
    except (TypeError, ValueError):
        return fallback


def truncation_note(returned: int, total: int, unit: str = "rows") -> Optional[str]:
    """
    Describe a limit that dropped part of the result, or None if nothing was dropped.

    Surfaced to the user via the `note` field so a capped table is never mistaken for a
    complete one -- and so a short CSV export is explained rather than baffling.
    """
    if returned >= total:
        return None
    return (
        f"{total} {unit} available; only {returned} were returned because a limit "
        f"of {returned} was requested. Ask for more, or omit the limit, to get them all."
    )


def sample_warning(smallest_n: int, label: str = "group", count: int = 1) -> Optional[str]:
    """
    Warn when a result rests on too few observations to be meaningful, else None.

    A survey with 800 responses spread over 80 courses leaves ~10 each: ranking those means
    surfaces noise as signal, and the agent will report the top of the list as a finding.
    Saying so is the difference between an average and a conclusion.
    """
    if smallest_n >= MIN_RELIABLE_SAMPLE:
        return None

    if count > 1:
        return (
            f"Caution: {count} {label}s have fewer than {MIN_RELIABLE_SAMPLE} responses "
            f"(the smallest has {smallest_n}). Averages over so few answers are unreliable "
            f"and differences between them are probably noise."
        )
    return (
        f"Caution: this {label} has only {smallest_n} responses. "
        f"Below {MIN_RELIABLE_SAMPLE} an average is unreliable and should not be "
        f"presented as a firm result."
    )


def join_notes(*notes: Optional[str]) -> Optional[str]:
    """Combine several caveats into one `note`, dropping the empty ones."""
    parts = [n.strip() for n in notes if n and n.strip()]
    if not parts:
        return None
    return " ".join(parts)
