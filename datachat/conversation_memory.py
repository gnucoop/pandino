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

import threading
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

MAX_TURNS = 5
MAX_CONVERSATIONS = 10

TurnKind = Literal["text", "table"]


@dataclass(frozen=True)
class CompactTurn:
    """An already-prepared compact analytical turn.

    Producing it from a DataChat result (eligibility, truncation, trusted
    qualifiers) is the caller's job and is not part of this module.
    """

    question: str
    kind: TurnKind
    text: Optional[str] = None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[object, ...], ...] = ()
    total_rows: Optional[int] = None
    more_rows_available: Optional[bool] = None
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in ("text", "table"):
            raise ValueError(f"unsupported turn kind: {self.kind!r}")


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
