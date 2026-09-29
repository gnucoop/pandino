"""Shared result caveats for DataChat tools."""

from typing import Optional

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
