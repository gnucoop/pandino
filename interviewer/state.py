"""
state.py
--------

Everything the interviewer knows about an interview in progress.

The agent itself runs on a fresh memory every turn, exactly like DataChat: raw
query results would otherwise pile up in its context turn after turn. What must
survive between turns lives here instead and is rendered into each turn's task:
the questions (one per turn) and answers, the data findings the agent chose to
record, the drafts the user turned down and why. The interview opens with a
fixed goal question, asked by the bootstrap message rather than by the agent.

The state also enforces the two rules the agent cannot be trusted to follow on
its own, because the engine checks them rather than the prompt merely asking:

- the minimum interview: no draft before `min_questions` questions have been
  asked *and answered* (the opening question included), including one about the
  goal and one about the data;
- refinement after the user asks to keep working on a draft: no new draft
  before at least one follow-up question has been answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

QUESTION_TOPICS = ("goal", "data", "scope", "report")
REQUIRED_TOPICS = ("goal", "data")

# A rendered transcript entry is truncated to this many characters.
_MAX_ENTRY_CHARS = 2000
_MAX_NOTE_CHARS = 500


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class Question:
    text: str
    topic: str
    # Refinement round the question was asked in; 0 is the initial interview.
    round: int
    answered: bool = False


@dataclass
class InterviewState:
    max_turns: int
    max_notes: int
    min_questions: int

    transcript: list[dict[str, Any]] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # The draft awaiting the user's decision: {"summary", "brief", "prompt_text"}.
    pending_draft: Optional[dict[str, Any]] = None
    rejected_drafts: list[dict[str, Any]] = field(default_factory=list)

    # Incremented every time a draft is sent back for more work. Each round gets
    # its own turn budget and needs its own answered follow-up question.
    round: int = 0
    turns_in_round: int = 0

    # --- recording ---

    def record_opening_question(self, text: str) -> None:
        """The fixed goal question the bootstrap message asks."""
        self.record_question(text, "goal")

    def record_user_message(self, text: str) -> None:
        """A user answer: it answers the open question."""
        for question in self.questions:
            question.answered = True
        self.transcript.append({"role": "user", "kind": "answer", "text": str(text)})

    def record_question(self, text: str, topic: str) -> None:
        self.questions.append(Question(text=str(text), topic=topic, round=self.round))
        self.transcript.append(
            {"role": "interviewer", "kind": "question", "text": str(text), "topic": topic}
        )

    def record_draft(self, summary: str, brief: dict[str, Any], prompt_text: str) -> None:
        self.pending_draft = {"summary": summary, "brief": brief, "prompt_text": prompt_text}
        self.transcript.append(
            {"role": "interviewer", "kind": "brief_draft", "text": summary, "name": brief.get("name")}
        )

    def reject_draft(self, feedback: str) -> None:
        """The user wants more work on the pending draft: open a refinement round."""
        assert self.pending_draft is not None
        self.rejected_drafts.append(
            {"summary": self.pending_draft["summary"], "feedback": str(feedback)}
        )
        self.pending_draft = None
        self.transcript.append({"role": "user", "kind": "rejection", "text": str(feedback)})
        self._open_round()

    def approve_draft(self) -> dict[str, Any]:
        """The user approved the pending draft: it is final."""
        assert self.pending_draft is not None
        draft = self.pending_draft
        self.pending_draft = None
        self.transcript.append(
            {"role": "user", "kind": "approval", "text": "Approved", "name": draft["brief"].get("name")}
        )
        return draft

    def add_note(self, note: str) -> bool:
        """Record a data finding; False when the note budget is spent."""
        if len(self.notes) >= self.max_notes:
            return False
        self.notes.append(_clip(note, _MAX_NOTE_CHARS))
        return True

    def start_turn(self) -> None:
        self.turns_in_round += 1

    def _open_round(self) -> None:
        self.round += 1
        self.turns_in_round = 0

    # --- rules ---

    def answered(self) -> list[Question]:
        return [q for q in self.questions if q.answered]

    def draft_gate_reason(self) -> Optional[str]:
        """
        Why a brief draft may not be proposed now, or None when it may.

        Worded for the model, which receives it verbatim when its draft is
        turned down.
        """
        answered = self.answered()
        topics = {q.topic for q in answered}
        missing_topics = [t for t in REQUIRED_TOPICS if t not in topics]
        if len(answered) < self.min_questions or missing_topics:
            parts = [
                f"only {len(answered)} questions have been answered, and at "
                f"least {self.min_questions} are required"
            ]
            if missing_topics:
                parts.append(
                    "none of the answered questions is about: " + ", ".join(missing_topics)
                )
            return (
                "Ask more questions about the report goal and the relevant data before "
                "proposing a brief (" + "; ".join(parts) + ")."
            )

        if self.needs_follow_up():
            return (
                "The user rejected the previous draft: ask what to change, and wait for "
                "the answer, before proposing again."
            )
        return None

    def needs_follow_up(self) -> bool:
        """Whether a refinement round is open with no follow-up answered yet."""
        return self.round > 0 and not any(
            q.round == self.round for q in self.answered()
        )

    def turn_limit_reached(self) -> bool:
        return self.turns_in_round >= self.max_turns

    # --- rendering ---

    def render_for_task(self) -> str:
        """The interview so far, as the agent sees it at the start of a turn."""
        answered = self.answered()
        by_topic = {t: sum(1 for q in answered if q.topic == t) for t in QUESTION_TOPICS}
        lines = [
            f"Turn {self.turns_in_round} of {self.max_turns}"
            + (f" (refinement round {self.round})" if self.round else ""),
            f"Questions answered: {len(answered)} ("
            + ", ".join(f"{t}: {n}" for t, n in by_topic.items())
            + f"). Required before a draft: {self.min_questions} (the opening question "
            "included), including goal and data.",
            "",
            "TRANSCRIPT",
        ]

        if not self.transcript:
            lines.append("(the interview has not started)")
        for entry in self.transcript:
            lines.append(self._render_entry(entry))

        lines.extend(["", f"DATA FINDINGS ({len(self.notes)}/{self.max_notes}, recorded with record_finding)"])
        if self.notes:
            lines.extend(f"- {note}" for note in self.notes)
        else:
            lines.append("- (none yet)")

        if self.rejected_drafts:
            lines.extend(["", "DRAFTS THE USER DID NOT APPROVE"])
            for i, rejected in enumerate(self.rejected_drafts, start=1):
                lines.append(f"{i}. {_clip(rejected['summary'], _MAX_ENTRY_CHARS)}")
                lines.append(f"   User feedback: {_clip(rejected['feedback'], _MAX_ENTRY_CHARS)}")
        return "\n".join(lines)

    @staticmethod
    def _render_entry(entry: dict[str, Any]) -> str:
        kind = entry.get("kind")
        text = _clip(entry.get("text", ""), _MAX_ENTRY_CHARS)
        if kind == "question":
            return f"[interviewer] [{entry.get('topic')}] {text}"
        if kind == "brief_draft":
            return f"[interviewer] proposed brief draft {entry.get('name')!r}: {text}"
        if kind == "rejection":
            return f"[user] did NOT approve the draft. Feedback: {text}"
        if kind == "approval":
            return f"[user] APPROVED the brief {entry.get('name')!r}."
        return f"[user] {text}"
