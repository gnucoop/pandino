"""Engine-local state for optional DataChat conversational memory.

Pure application state, owned (later) by one SmolagentsEngine: no Flask, no
smolagents, no Usage, no exports, no I/O. Independent from smolagents
AgentMemory; CodeAgent.run(reset=True) is untouched.

Lifecycle:
- begin(None) returns a *candidate*: it has an id but is not registered and
  occupies no slot. An abandoned candidate simply disappears.
- confirm() on a candidate registers it (activation) together with its first
  turn; confirm() on an active conversation appends a turn (sliding window).
- Only the store that created a conversation can confirm it.
- close() removes an active conversation from the map and marks the object
  closed, so a late confirm() holding that object has no effect.

Every transition runs under a container-owned lock, with no I/O inside.
Run serialization stays with the existing agent_manager run guard.
"""

from __future__ import annotations

import math
import threading
import uuid
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Literal, Optional

import pandas as pd

MAX_TURNS = 5
MAX_CONVERSATIONS = 10

# Initial compact-turn bounds. The rendered-history budget is applied later,
# at rendering time, not here.
MAX_QUESTION_CHARS = 500
MAX_TEXT_CHARS = 1000
MAX_TABLE_ROWS = 10
MAX_TABLE_COLUMNS = 10
MAX_CELL_CHARS = 100
MAX_NOTE_CHARS = 500

TRUNCATION_MARKER = "…"

# smolagents RunResult.state for a completed run. None means the state is
# unavailable (e.g. a test double without RunResult), not that the run failed.
_ACCEPTED_RUN_STATES = {"success", None}

TurnKind = Literal["text", "table"]


@dataclass(frozen=True)
class CompactTurn:
    """A bounded analytical turn, as built by build_compact_turn().

    result_rows/result_columns describe the result Maui observed, before the
    memory bounds; they say nothing about how much data the source holds.
    more_rows_available is None when source completeness is unknown.
    """

    question: str
    kind: TurnKind
    text: Optional[str] = None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[object, ...], ...] = ()
    result_rows: Optional[int] = None
    result_columns: Optional[int] = None
    more_rows_available: Optional[bool] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in ("text", "table"):
            raise ValueError(f"unsupported turn kind: {self.kind!r}")


def _clip(text: str, limit: int) -> str:
    """Deterministic visible truncation to at most ``limit`` characters."""
    if len(text) <= limit:
        return text
    return text[: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def _compact_cell(value: Any) -> Any:
    """Same scalar policy as output_normalizer, plus the cell bound."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _clip(value if isinstance(value, str) else str(value), MAX_CELL_CHARS)


def _list_table(data: list[Any]) -> Optional[tuple[list[str], list[tuple[Any, ...]], int]]:
    # Any non-record row makes the shape ambiguous: no turn rather than a
    # silently shrunk table.
    if not all(isinstance(row, dict) for row in data):
        return None
    columns: dict[str, None] = {}
    for row in data:
        columns.update(dict.fromkeys(str(k) for k in row))
    names = list(columns)
    kept = names[:MAX_TABLE_COLUMNS]
    rows = []
    for row in data[:MAX_TABLE_ROWS]:
        by_name = {str(k): v for k, v in row.items()}
        rows.append(tuple(by_name.get(c) for c in kept))
    return names, rows, len(data)


def _frame_table(data: pd.DataFrame) -> tuple[list[str], list[tuple[Any, ...]], int]:
    names = [str(c) for c in data.columns]
    head = data.iloc[:MAX_TABLE_ROWS, :MAX_TABLE_COLUMNS].to_numpy(dtype=object).tolist()
    return names, [tuple(r) for r in head], len(data)


def build_compact_turn(
    question: str,
    payload: Any,
    *,
    passed: bool,
    final_kind: Optional[str],
    run_state: Optional[str],
) -> Optional[CompactTurn]:
    """
    Build a bounded turn from an eligible DataChat result, or return None.

    Eligibility comes from the signals the engine already computed
    (contract check, final kind, run state), never from the payload's shape
    alone. Pure: the payload and its data are read, never modified.
    Trusted qualifiers are attached separately (with_trusted_qualifiers).
    """
    if not passed or run_state not in _ACCEPTED_RUN_STATES:
        return None
    if final_kind not in ("text", "table") or not isinstance(payload, dict):
        return None
    if str(payload.get("kind") or "").strip().lower() != final_kind:
        return None

    q = _clip(str(question or ""), MAX_QUESTION_CHARS)

    if final_kind == "text":
        text = payload.get("text")
        if text is None:
            text = payload.get("content")
        if text is None:
            text = payload.get("message")
        if text is None or not str(text).strip():
            return None
        return CompactTurn(question=q, kind="text", text=_clip(str(text), MAX_TEXT_CHARS))

    data = payload.get("data")
    if isinstance(data, pd.DataFrame):
        table = _frame_table(data)
    elif isinstance(data, list):
        table = _list_table(data)
    else:
        return None
    if table is None:
        return None
    names, rows, result_rows = table
    return CompactTurn(
        question=q,
        kind="table",
        columns=tuple(_clip(c, MAX_CELL_CHARS) for c in names[:MAX_TABLE_COLUMNS]),
        rows=tuple(tuple(_compact_cell(v) for v in row) for row in rows),
        result_rows=result_rows,
        result_columns=len(names),
    )


def with_trusted_qualifiers(
    turn: CompactTurn,
    *,
    more_rows_available: Optional[bool],
    note: Optional[str],
) -> CompactTurn:
    """
    Attach backend-trusted qualifiers supplied by the caller.

    Only a true more_rows_available is a fact; False or None leaves
    completeness unknown (None), as a missing TrustedResult does.
    """
    return replace(
        turn,
        more_rows_available=True if more_rows_available is True else None,
        note=_clip(note, MAX_NOTE_CHARS) if note else None,
    )


class ConversationNotFound(LookupError):
    """Unknown, closed or stale conversation id."""


class ConversationLimitReached(RuntimeError):
    """No free slot for a new conversation."""


class ConfirmOutcome(str, Enum):
    ACTIVATED = "activated"
    APPENDED = "appended"
    CLOSED = "closed"
    STALE = "stale"
    LIMIT_REACHED = "limit_reached"


class Conversation:
    """One conversation object. Mutated only through ConversationStore."""

    __slots__ = ("conversation_id", "_owner", "_turns", "_status")

    def __init__(self, conversation_id: str, owner: "ConversationStore") -> None:
        self.conversation_id = conversation_id
        # The store that created it; no other store may confirm it.
        self._owner = owner
        self._turns: deque[CompactTurn] = deque(maxlen=MAX_TURNS)
        self._status: Literal["candidate", "active", "closed"] = "candidate"

    @property
    def status(self) -> str:
        return self._status

    @property
    def turns(self) -> tuple[CompactTurn, ...]:
        return tuple(self._turns)


class ConversationStore:
    """Conversations of a single engine. Not a global registry."""

    def __init__(
        self,
        max_conversations: int = MAX_CONVERSATIONS,
    ) -> None:
        self._max_conversations = max_conversations
        self._active: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def begin(self, conversation_id: Optional[str]) -> Conversation:
        """Return the active conversation for an id, or a new candidate.

        Raises ConversationNotFound for an unknown/closed id and
        ConversationLimitReached when a candidate could never be activated.
        """
        with self._lock:
            if conversation_id is not None:
                conv = self._active.get(conversation_id)
                if conv is None:
                    raise ConversationNotFound(conversation_id)
                return conv
            if len(self._active) >= self._max_conversations:
                raise ConversationLimitReached()
        return Conversation(uuid.uuid4().hex, self)

    def get(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._active.get(conversation_id)

    def has_free_slot(self) -> bool:
        with self._lock:
            return len(self._active) < self._max_conversations

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def confirm(self, conv: Conversation, turn: CompactTurn) -> ConfirmOutcome:
        """Confirm an eligible turn on the conversation object held by a run."""
        with self._lock:
            if conv._owner is not self:
                return ConfirmOutcome.STALE
            if conv._status == "closed":
                return ConfirmOutcome.CLOSED
            registered = self._active.get(conv.conversation_id)
            if conv._status == "active":
                if registered is not conv:
                    return ConfirmOutcome.STALE
                conv._turns.append(turn)
                return ConfirmOutcome.APPENDED
            # Candidate: never overwrite an existing entry, recheck the slot.
            if registered is not None:
                return ConfirmOutcome.STALE
            if len(self._active) >= self._max_conversations:
                return ConfirmOutcome.LIMIT_REACHED
            conv._turns.append(turn)
            conv._status = "active"
            self._active[conv.conversation_id] = conv
            return ConfirmOutcome.ACTIVATED

    def close(self, conversation_id: str) -> Literal["closed", "not_active"]:
        """Idempotent explicit closure."""
        with self._lock:
            conv = self._active.pop(conversation_id, None)
            if conv is None:
                return "not_active"
            conv._status = "closed"
            return "closed"
