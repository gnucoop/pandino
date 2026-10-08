"""
engine.py
---------

InterviewerEngine: interviews a user about the analysis they want and turns the
answers into an analysis brief for the data analyst.

It is a smolagents CodeAgent built on the same base as DataChat, reading the
same read-only SQL datasource through the same guarded sql_engine tool. What
differs is the shape of a turn. The interview opens with a fixed goal question,
asked by the bootstrap message and already in the state, then:

- chat(answer)     the user's answer (the first one states the goal) -> the
                   next question (exactly one per turn), or a brief draft. Refused while a draft is
                   pending: the user must decide on it first.
- reject(feedback) the user wants more work on the pending draft -> the agent
                   asks a follow-up question, then refines the draft.
- approve()        the user approves the pending draft -> the final brief.
                   No model call: the draft was validated when proposed.

Declining a draft needs nothing from the engine: the session is simply closed.

The model never finalizes a brief: it can only propose one, and the guard in
_check_final_answer turns a draft down — replaying the reason to the model —
until the minimum interview has taken place, after a rejection until a
follow-up question has been answered, and until validate_brief() passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Tuple

from config import InterviewerConfig
from datachat.engine_interface import EngineBootstrapResult
from datachat.final_answer_contract import (
    coerce_final_payload,
    validate_text_or_message,
)
from datachat.smolagents_base import SmolagentsEngineBase
from datachat.sql_datasource import SqlDatasource
from infrastructure.prompt_utils import load_prompt, render_prompt
from interviewer.bootstrap_static import get_opening_question, get_static_bootstrap_html
from interviewer.brief import render_brief_prompt, validate_brief
from interviewer.prompts import (
    INTERVIEWER_SQL_ADDENDUM_DEFAULT,
    INTERVIEWER_SQL_ADDENDUM_TITLE,
    INTERVIEWER_SYSTEM_DEFAULT,
    INTERVIEWER_SYSTEM_TITLE,
    INTERVIEWER_TURN_DEFAULT,
    INTERVIEWER_TURN_TITLE,
    LANGUAGE_NAMES,
    NEXT_STEP_AFTER_REJECTION,
    NEXT_STEP_GATE_NOT_MET,
    NEXT_STEP_OPEN,
    NEXT_STEP_TURN_LIMIT,
)
from interviewer.state import QUESTION_TOPICS, InterviewState
from interviewer.tools.record_finding_tool import RecordFindingTool

_DEFAULT_REJECTION_FEEDBACK = "The user did not approve the draft and gave no reason."


# ----------------------------
# Final answer contract
# ----------------------------

def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_question(payload: dict[str, Any]) -> Optional[str]:
    # A list, even of one, is the multi-question shape: one question per turn.
    if "questions" in payload:
        return "ONE_QUESTION_ONLY"
    if not _is_text(payload.get("text")):
        return "MISSING_TEXT"
    if str(payload.get("topic") or "").strip().lower() not in QUESTION_TOPICS:
        return "QUESTION_WITH_INVALID_TOPIC"
    return None


def _validate_brief_draft(payload: dict[str, Any]) -> Optional[str]:
    if not _is_text(payload.get("summary")):
        return "MISSING_SUMMARY"
    if not isinstance(payload.get("brief"), dict):
        return "MISSING_BRIEF"
    return None


_INTERVIEWER_KIND_VALIDATORS = {
    "question": _validate_question,
    "brief_draft": _validate_brief_draft,
    "error": validate_text_or_message,
}


def _contract_failure_message(reason: str) -> str:
    """Explain a rejected final answer to the model (see datachat's twin)."""
    return (
        f"The final answer does not satisfy the output contract ({reason}). "
        'Return exactly one JSON object carrying a "kind" field, with no wrapper, '
        "no extra nesting and no surrounding prose. Valid shapes: "
        '{"kind":"question","text":"<exactly one question>","topic":"goal|data|scope|report"}, '
        '{"kind":"brief_draft","summary":"...","brief":{...}}, '
        '{"kind":"error","message":"..."}.'
    )


def _brief_errors_message(errors: list[str]) -> str:
    return (
        "The brief draft is not valid yet. Fix every point below and propose it again "
        "(query the database if you need to check a name):\n"
        + "\n".join(f"- {error}" for error in errors)
    )


# ----------------------------
# Engine
# ----------------------------

@dataclass
class InterviewerEngine(SmolagentsEngineBase):
    config: InterviewerConfig
    lang: str = "ENG"

    ENGINE_NAME: ClassVar[str] = "interviewer"
    RUNTIME_LOGGER_NAME: ClassVar[str] = "interviewer.runtime"

    _state: InterviewState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The state exists before the agent: record_finding writes into it.
        self._state = InterviewState(
            max_turns=self.config.max_turns,
            max_notes=self.config.max_notes,
            min_questions=self.config.min_questions,
        )
        self._state.record_opening_question(get_opening_question(self.lang))
        super().__post_init__()

    # --- hooks ---

    def _init_config(self) -> None:
        self._provider = self.config.provider
        self._configured_model = self.config.model
        self._max_steps = self.config.max_steps

    def _build_instructions(self, datasource: Optional[SqlDatasource] = None) -> str:
        template = load_prompt(INTERVIEWER_SYSTEM_TITLE, default_text=INTERVIEWER_SYSTEM_DEFAULT)
        instructions = render_prompt(
            template,
            language=LANGUAGE_NAMES.get(str(self.lang or "").upper(), "English"),
            min_questions=self.config.min_questions,
        )

        addendum = self._sql_addendum(
            datasource,
            lambda: load_prompt(
                INTERVIEWER_SQL_ADDENDUM_TITLE, default_text=INTERVIEWER_SQL_ADDENDUM_DEFAULT
            ),
        )
        if addendum is None:
            return instructions
        return instructions + "\n\n" + addendum

    def _requirements_met(self) -> bool:
        # An interview about database data is pointless without the database.
        return self._sql_ready

    def _extra_tools(self) -> list[Any]:
        return [RecordFindingTool(self._state)]

    # --- final answer ---

    def _evaluate(self, output: Any) -> Tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
        """
        Check a final answer against the contract and the interview rules.

        :return: (payload, kind, problem). `problem` is None when the answer is
            acceptable, otherwise the explanation meant for the model.
        """
        payload, passed, final_kind, reason = coerce_final_payload(
            output, _INTERVIEWER_KIND_VALIDATORS
        )
        if not passed or payload is None:
            return payload, final_kind, _contract_failure_message(reason)

        if final_kind == "brief_draft":
            gate_reason = self._state.draft_gate_reason()
            if gate_reason is not None:
                return payload, final_kind, gate_reason
            errors = validate_brief(payload["brief"], self._sql_identifiers)
            if errors:
                return payload, final_kind, _brief_errors_message(errors)

        return payload, final_kind, None

    def _check_final_answer(self, *args: Any, **kwargs: Any) -> bool:
        """
        Guard the final answer before smolagents hands it back.

        Raising is the only way to tell the model why its answer was turned
        down (see SmolagentsEngine._check_final_answer).
        """
        candidate = args[0] if args else (
            kwargs.get("final_answer") or kwargs.get("answer") or kwargs.get("output")
        )
        _, final_kind, problem = self._evaluate(candidate)

        self._last_final_answer_check_passed = problem is None
        self._last_final_kind = final_kind

        self._log.info(
            "final_answer_check request_id=%s engine=%s user=%s passed=%s final_kind=%s",
            self._active_request_id or "n/a",
            self.ENGINE_NAME,
            self.user_name,
            problem is None,
            final_kind or "none",
        )

        if problem is not None:
            raise ValueError(problem)
        return True

    # --- turns ---

    def _next_step(self) -> str:
        state = self._state
        if state.needs_follow_up():
            return NEXT_STEP_AFTER_REJECTION
        gate_reason = state.draft_gate_reason()
        if gate_reason is not None:
            return render_prompt(NEXT_STEP_GATE_NOT_MET, reason=gate_reason)
        if state.turn_limit_reached():
            return NEXT_STEP_TURN_LIMIT
        return NEXT_STEP_OPEN

    def _render_task(self, latest_message: str) -> str:
        template = load_prompt(INTERVIEWER_TURN_TITLE, default_text=INTERVIEWER_TURN_DEFAULT)
        return render_prompt(
            template,
            interview_state=self._state.render_for_task(),
            latest_message=str(latest_message or "").strip() or "(no message)",
            next_step=self._next_step(),
        )

    def _missing_config_error(self) -> dict[str, Any]:
        return {
            "kind": "error",
            "message": (
                "The analysis interviewer is not configured: check INTERVIEWER_PROVIDER/"
                "INTERVIEWER_MODEL, the provider API key and the SQL datasource "
                "(DATACHAT_SQL_ENABLED)."
            ),
            "code": "MISSING_CONFIG",
        }

    def _begin(self, request_id: Optional[str], event: str, message: str) -> None:
        self._active_request_id = request_id or "n/a"
        self._last_final_answer_check_passed = None
        self._last_final_kind = None
        self._log.info(
            "chat_start request_id=%s engine=%s user=%s event=%s message_len=%s round=%s",
            self._active_request_id,
            self.ENGINE_NAME,
            self.user_name,
            event,
            len(str(message or "")),
            self._state.round,
        )

    def _end(self, result: dict[str, Any]) -> dict[str, Any]:
        self._log.info(
            "chat_end request_id=%s engine=%s user=%s duration_ms=%s response_kind=%s "
            "final_answer_check_passed=%s questions_answered=%s round=%s",
            self._active_request_id,
            self.ENGINE_NAME,
            self.user_name,
            self._last_run_duration_ms,
            result.get("kind"),
            bool(self._last_final_answer_check_passed),
            len(self._state.answered()),
            self._state.round,
        )
        self._active_request_id = None
        return result

    def _turn(self, latest_message: str) -> dict[str, Any]:
        """One agent run: the next question, or a brief draft."""
        self._state.start_turn()

        run_result, run_error = self._run_agent(self._render_task(latest_message))
        if run_result is None:
            return self._end(
                {"kind": "error", "message": f"The interviewer failed to run: {run_error}", "code": "RUN_FAILED"}
            )

        # Re-checked here, not only in the guard: when the agent runs out of
        # steps smolagents produces a final answer the guard never saw.
        payload, final_kind, problem = self._evaluate(getattr(run_result, "output", None))
        if problem is not None or payload is None:
            self._last_final_answer_check_passed = False
            return self._end(
                {
                    "kind": "error",
                    "message": "The interviewer could not produce a valid reply. Please answer again or rephrase.",
                    "code": "INVALID_OUTPUT",
                }
            )

        if final_kind == "question":
            text = str(payload["text"]).strip()
            topic = str(payload["topic"]).strip().lower()
            self._state.record_question(text, topic)
            return self._end({"kind": "question", "text": text, "topic": topic})

        if final_kind == "brief_draft":
            summary = str(payload["summary"]).strip()
            brief = payload["brief"]
            prompt_text = render_brief_prompt(brief)
            self._state.record_draft(summary, brief, prompt_text)
            return self._end(
                {"kind": "brief_draft", "summary": summary, "brief": brief, "prompt_text": prompt_text}
            )

        message = payload.get("message") or payload.get("text")
        return self._end({"kind": "error", "message": str(message), "code": "AGENT_ERROR"})

    # --- public API ---

    def bootstrap(self, lang: str) -> EngineBootstrapResult:
        return EngineBootstrapResult(suggested_questions_html=get_static_bootstrap_html(lang))

    def chat(self, message: str, request_id: Optional[str] = None) -> dict[str, Any]:
        """
        The user's answer to the open question; the first one states the goal.

        While a draft is pending there is no open question: the user must
        approve it, ask for more work on it (reject) or decline it.
        """
        self._begin(request_id, "answer", message)
        if self._agent is None:
            return self._end(self._missing_config_error())
        if self._state.pending_draft is not None:
            return self._end(
                {
                    "kind": "error",
                    "message": "A brief draft is awaiting your decision: approve, revise or decline it.",
                    "code": "DRAFT_PENDING",
                }
            )

        self._state.record_user_message(message)
        return self._turn(message)

    def reject(self, feedback: Optional[str], request_id: Optional[str] = None) -> dict[str, Any]:
        """The user wants more work on the pending draft: refine it."""
        self._begin(request_id, "reject", feedback or "")
        if self._agent is None:
            return self._end(self._missing_config_error())
        if self._state.pending_draft is None:
            return self._end(
                {"kind": "error", "message": "There is no brief draft to reject.", "code": "NO_PENDING_DRAFT"}
            )

        feedback_text = str(feedback or "").strip() or _DEFAULT_REJECTION_FEEDBACK
        self._state.reject_draft(feedback_text)
        return self._turn(feedback_text)

    def approve(self, request_id: Optional[str] = None) -> dict[str, Any]:
        """The user approves the pending draft: it becomes the final brief."""
        self._begin(request_id, "approve", "")
        if self._state.pending_draft is None:
            return self._end(
                {"kind": "error", "message": "There is no brief draft to approve.", "code": "NO_PENDING_DRAFT"}
            )

        draft = self._state.approve_draft()
        return self._end(
            {
                "kind": "brief",
                "summary": draft["summary"],
                "brief": draft["brief"],
                "prompt_text": draft["prompt_text"],
            }
        )

    @property
    def is_ready(self) -> bool:
        """Whether the engine could build its agent (model and database available)."""
        return self._agent is not None

    @property
    def has_pending_draft(self) -> bool:
        return self._state.pending_draft is not None

    @property
    def pending_draft(self) -> Optional[dict[str, Any]]:
        """The draft awaiting approval ({"summary", "brief", "prompt_text"}), or None."""
        draft = self._state.pending_draft
        return dict(draft) if draft is not None else None

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return list(self._state.transcript)

    @property
    def model(self) -> str:
        return self._configured_model

    @property
    def provider(self) -> str:
        return self._provider
