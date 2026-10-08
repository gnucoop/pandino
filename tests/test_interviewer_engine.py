"""
Tests for the analysis interviewer engine: the interview state carried across
turns, the fixed opening goal question, the minimum interview before a draft,
the approval flow, and refinement when the user asks for more work on a draft.

The engine is built through its real __post_init__ with the model, the SQL
datasource and CodeAgent patched, so the SQL gate and the tool wiring are
exercised too. A turn's output is scripted on the fake agent; a scripted run
passes its output through the engine's own final-answer guard first, as
smolagents does.
"""

from unittest.mock import MagicMock, patch

import pytest

import datachat.schema_snapshot_loader as loader
from config import InterviewerConfig
from interviewer.bootstrap_static import get_opening_question, get_static_bootstrap_html
from interviewer.engine import InterviewerEngine
from tests.fake_sql_datasource import FakeDatasource, operational_error

COLUMNS = [
    {"column": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
    {"column": "country", "type": "VARCHAR(2)", "nullable": True, "primary_key": False},
    {"column": "amount", "type": "NUMERIC", "nullable": True, "primary_key": False},
]


def make_config(**overrides):
    params = dict(
        provider="Deepinfra", model="some/model", max_steps=5,
        max_turns=12, max_notes=3, min_questions=4,
    )
    params.update(overrides)
    return InterviewerConfig(**params)


def valid_brief():
    return {
        "name": "Revenue by country",
        "description": "Revenue per country.",
        "category": "discovery",
        "tags": ["revenue", "country", "orders"],
        "language": "English",
        "audience": "Sales",
        "business_context": "Planning targets.",
        "data_scope": {
            "relations": [{"name": "orders", "columns": ["country", "amount"]}],
            "sql_hints": ['SELECT "country", SUM("amount") FROM "orders" GROUP BY 1'],
        },
        "data_notes": ["amount is in EUR."],
        "mission": "Sum amount per country.",
        "sub_tasks": [
            {"id": "data_preparation", "name": "Prep", "order": 1, "depends_on": [],
             "mission": "Check amount.", "outputs": {"required": ["row_count"]}},
            {"id": "by_country", "name": "By country", "order": 2, "depends_on": ["data_preparation"],
             "mission": "Totals.", "outputs": {"required": ["total_revenue"]}},
            {"id": "final_summary", "name": "Summary", "order": 3, "depends_on": ["by_country"],
             "mission": "Synthesize.", "outputs": {"required": ["key_findings"]}},
        ],
        "metrics_config": None,
        "report": {"format": "html", "sections": ["Summary"], "charts": ["Bar chart"],
                   "tone": "plain", "length": "short"},
        "open_questions": [],
    }


def question(topic):
    return {"kind": "question", "text": f"About {topic}?", "topic": topic}


def draft(brief=None):
    return {"kind": "brief_draft", "summary": "We will sum revenue.", "brief": brief or valid_brief()}


class _Run:
    def __init__(self, output):
        self.output = output


@pytest.fixture(autouse=True)
def prompts_from_code():
    with patch(
        "interviewer.engine.load_prompt",
        side_effect=lambda title, default_text="", **kwargs: default_text,
    ):
        yield


@pytest.fixture(autouse=True)
def reset_snapshot_cache():
    loader.invalidate_snapshot()
    yield
    loader.invalidate_snapshot()


def build_engine(datasource=None, config=None, model=object(), lang="ENG"):
    datasource = datasource if datasource is not None else FakeDatasource(tables=["orders"], columns=COLUMNS)
    with patch("datachat.smolagents_base.build_litellm_model", return_value=model), \
         patch("datachat.smolagents_base.sql_datasource.get_datasource", return_value=datasource), \
         patch("datachat.smolagents_base.CodeAgent") as agent_cls:
        engine = InterviewerEngine(
            api_key="k", user_name="tester", config=config or make_config(), lang=lang
        )
    return engine, agent_cls


@pytest.fixture
def built():
    return build_engine()


@pytest.fixture
def engine(built):
    return built[0]


def script(engine, *outputs):
    """
    Make each agent run end with the next output. Like smolagents, the run
    passes the output through the final-answer guard; the tasks it received
    are collected on engine.tasks.
    """
    queue = list(outputs)
    engine.tasks = []
    engine.guard_errors = []

    def run(task, **kwargs):
        engine.tasks.append(task)
        output = queue.pop(0)
        try:
            engine._check_final_answer(output, object(), agent=object())
        except ValueError as e:
            engine.guard_errors.append(str(e))
        return _Run(output)

    engine._agent.run.side_effect = run


def complete_the_minimum_interview(engine):
    """
    Answer the opening question, then goal and data, leaving scope open: the
    next answer is the fourth and meets the minimum.
    """
    script(engine, question("goal"), question("data"), question("scope"))
    engine.chat("I want revenue by country")
    engine.chat("For the sales targets")
    engine.chat("The orders table")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_the_agent_gets_sql_engine_and_record_finding(built):
    _, agent_cls = built

    tools = agent_cls.call_args.kwargs["tools"]
    assert sorted(t.name for t in tools) == ["record_finding", "sql_engine"]


def test_the_instructions_carry_the_schema_and_the_minimum(built):
    _, agent_cls = built

    instructions = agent_cls.call_args.kwargs["instructions"]
    assert "Analysis Interviewer" in instructions
    assert '"orders"(' in instructions
    assert "at least 4 questions (the opening one" in instructions
    assert "{sql_schema}" not in instructions


def test_the_final_answer_guard_is_installed(built):
    engine, agent_cls = built

    assert agent_cls.call_args.kwargs["final_answer_checks"] == [engine._check_final_answer]


def test_no_database_means_no_interviewer():
    engine, agent_cls = build_engine(
        datasource=FakeDatasource(tables=["orders"], columns=COLUMNS, raises=operational_error())
    )

    assert engine._agent is None
    agent_cls.assert_not_called()
    assert engine.chat("hi")["code"] == "MISSING_CONFIG"


def test_a_missing_model_is_reported_not_raised():
    engine, _ = build_engine(config=make_config(model=""))

    assert engine.chat("hi")["code"] == "MISSING_CONFIG"


# ---------------------------------------------------------------------------
# Questions and state
# ---------------------------------------------------------------------------


def test_the_interview_opens_with_the_goal_question(engine):
    script(engine, question("data"))

    engine.chat("I want revenue by country")

    task = engine.tasks[0]
    assert "[interviewer] [goal] Describe in a few words what you would like to analyse and why." in task
    assert "[user] I want revenue by country" in task


def test_the_goal_answer_counts_towards_the_minimum(engine):
    script(engine, question("data"))

    engine.chat("I want revenue by country")

    assert [(q.topic, q.answered) for q in engine._state.questions] == [
        ("goal", True),
        ("data", False),
    ]


def test_the_opening_question_follows_the_language():
    engine, _ = build_engine(lang="ITA")

    assert engine.transcript[0]["text"].startswith("Descrivi in poche parole")


def test_a_question_turn_returns_one_question(engine):
    script(engine, question("data"))

    result = engine.chat("revenue analysis")

    assert result == {"kind": "question", "text": "About data?", "topic": "data"}


def test_the_transcript_is_carried_into_the_next_turn(engine):
    script(engine, question("goal"), question("data"))

    engine.chat("I want revenue by country")
    engine.chat("To set targets")

    second_task = engine.tasks[1]
    assert "[user] I want revenue by country" in second_task
    assert "[interviewer] [goal] About goal?" in second_task
    assert "LATEST USER MESSAGE\nTo set targets" in second_task


def test_recorded_findings_reach_later_turns(engine):
    script(engine, question("goal"))
    engine._state.add_note('"orders"."country" holds ISO codes')

    engine.chat("hi")

    assert '- "orders"."country" holds ISO codes' in engine.tasks[0]


def test_the_record_finding_tool_stops_at_the_note_budget(built):
    engine, agent_cls = built
    tool = next(t for t in agent_cls.call_args.kwargs["tools"] if t.name == "record_finding")

    for i in range(3):
        assert tool.forward(f"fact {i}")["kind"] == "text"
    assert tool.forward("one too many")["code"] == "NOTES_FULL"
    assert tool.forward("  ")["code"] == "EMPTY_FINDING"
    assert len(engine._state.notes) == 3


@pytest.mark.parametrize(
    "output,reason",
    [
        ({"kind": "question", "text": "Hi", "questions": [{"text": "A?", "topic": "goal"},
                                                           {"text": "B?", "topic": "data"}]},
         "ONE_QUESTION_ONLY"),
        ({"kind": "question", "text": "Hi", "questions": [{"text": "A?", "topic": "goal"}]},
         "ONE_QUESTION_ONLY"),
        ({"kind": "question", "text": "  ", "topic": "goal"}, "MISSING_TEXT"),
        ({"kind": "question", "text": "Why?", "topic": "weather"}, "QUESTION_WITH_INVALID_TOPIC"),
        ({"kind": "question", "text": "Why?"}, "QUESTION_WITH_INVALID_TOPIC"),
        ({"kind": "text", "text": "hello"}, "INVALID_KIND"),
        ("plain prose", "NON_JSON_OR_NO_OBJECT"),
    ],
)
def test_a_broken_contract_is_explained_to_the_model(engine, output, reason):
    with pytest.raises(ValueError, match=reason):
        engine._check_final_answer(output)


def test_an_answer_the_guard_never_saw_is_still_checked(engine):
    """At max_steps smolagents returns a final answer without the guard."""
    engine._agent.run.side_effect = None
    engine._agent.run.return_value = _Run(draft())

    result = engine.chat("just write the brief")

    assert result["code"] == "INVALID_OUTPUT"
    assert engine._state.pending_draft is None


def test_a_failed_run_is_reported_not_raised(engine):
    engine._agent.run.side_effect = RuntimeError("provider down")

    result = engine.chat("hi")

    assert result["code"] == "RUN_FAILED"
    assert "provider down" in result["message"]


# ---------------------------------------------------------------------------
# Minimum interview
# ---------------------------------------------------------------------------


def test_a_draft_before_the_minimum_is_rejected(engine):
    script(engine, question("goal"), question("data"), draft())
    engine.chat("revenue")
    engine.chat("targets")

    result = engine.chat("orders table")

    assert "only 3 questions have been answered" in engine.guard_errors[0]
    assert result["code"] == "INVALID_OUTPUT"


def test_a_draft_without_a_data_question_is_rejected(engine):
    script(engine, question("goal"), question("scope"), question("report"), draft())
    for answer in ("revenue", "targets", "last year"):
        engine.chat(answer)

    engine.chat("a short report")

    assert "none of the answered questions is about: data" in engine.guard_errors[0]


def test_asked_but_unanswered_questions_do_not_count(engine):
    script(engine, question("goal"), question("data"), question("scope"))
    for answer in ("revenue", "targets", "orders table"):
        engine.chat(answer)

    # Four asked (opening included), the scope one still open: three answered.
    assert len(engine._state.questions) == 4
    assert "only 3 questions have been answered" in engine._state.draft_gate_reason()


def test_until_the_minimum_the_task_forbids_a_draft(engine):
    script(engine, question("goal"))

    engine.chat("revenue")

    assert "You may not propose a brief yet" in engine.tasks[0]


def test_a_draft_after_the_minimum_interview_is_proposed(engine):
    complete_the_minimum_interview(engine)
    script(engine, draft())

    result = engine.chat("A short HTML report")

    assert engine.guard_errors == []
    assert result["kind"] == "brief_draft"
    assert result["summary"] == "We will sum revenue."
    assert "# ANALYSIS BRIEF: Revenue by country" in result["prompt_text"]
    assert engine.has_pending_draft


def test_an_invalid_brief_is_sent_back_with_every_error(engine):
    complete_the_minimum_interview(engine)
    brief = valid_brief()
    brief["data_scope"]["relations"][0]["columns"].append("discount")
    brief["category"] = "astrology"
    script(engine, draft(brief))

    engine.chat("A short HTML report")

    message = engine.guard_errors[0]
    assert "column 'discount' does not exist in orders" in message
    assert "category must be one of" in message


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


@pytest.fixture
def drafted(engine):
    complete_the_minimum_interview(engine)
    script(engine, draft())
    engine.chat("A short HTML report")
    return engine


def test_approval_finalizes_the_draft_without_a_model_call(drafted):
    runs_before = drafted._agent.run.call_count

    result = drafted.approve()

    assert result["kind"] == "brief"
    assert result["brief"]["name"] == "Revenue by country"
    assert result["prompt_text"].startswith("# ANALYSIS BRIEF")
    assert drafted._agent.run.call_count == runs_before
    assert not drafted.has_pending_draft


def test_there_is_nothing_to_approve_before_a_draft(engine):
    assert engine.approve()["code"] == "NO_PENDING_DRAFT"
    assert engine.reject("no")["code"] == "NO_PENDING_DRAFT"


def test_the_transcript_records_the_approval(drafted):
    drafted.approve()

    assert drafted.transcript[-1]["kind"] == "approval"


# ---------------------------------------------------------------------------
# Refinement when the user does not approve
# ---------------------------------------------------------------------------


def test_rejecting_asks_follow_up_questions_before_a_new_draft(drafted):
    script(drafted, draft(), question("scope"))

    drafted.reject("Split it by month too")

    # The immediate resubmission was turned down with the reason...
    assert "The user rejected the previous draft" in drafted.guard_errors[0]
    # ...and the turn was told to refine, with the feedback in view.
    task = drafted.tasks[0]
    assert "did NOT approve the previous brief draft" in task
    assert "User feedback: Split it by month too" in task
    assert drafted._state.pending_draft is None


def test_a_refined_draft_is_accepted_once_a_follow_up_is_answered(drafted):
    script(drafted, question("scope"))
    drafted.reject("Split it by month too")

    script(drafted, draft())
    result = drafted.chat("Monthly, yes")

    assert drafted.guard_errors == []
    assert result["kind"] == "brief_draft"


def test_answering_while_a_draft_is_pending_is_refused(drafted):
    runs_before = drafted._agent.run.call_count

    result = drafted.chat("Not quite, add the region")

    assert result["code"] == "DRAFT_PENDING"
    assert drafted._agent.run.call_count == runs_before
    assert drafted.has_pending_draft


def test_rejection_without_a_reason_still_refines(drafted):
    script(drafted, question("goal"))

    drafted.reject(None)

    assert "did not approve the draft and gave no reason" in drafted.tasks[0]


def test_each_refinement_round_gets_a_fresh_turn_budget(drafted):
    script(drafted, question("scope"))

    drafted.reject("no")

    assert drafted._state.turns_in_round == 1


# ---------------------------------------------------------------------------
# Turn limit
# ---------------------------------------------------------------------------


def test_the_turn_limit_asks_for_a_draft_once_the_minimum_is_met():
    engine, _ = build_engine(config=make_config(max_turns=4))
    complete_the_minimum_interview(engine)
    script(engine, draft())

    engine.chat("A short HTML report")

    assert "reached its turn limit" in engine.tasks[0]


def test_the_turn_limit_never_overrides_the_minimum_interview():
    engine, _ = build_engine(config=make_config(max_turns=3))
    script(engine, question("goal"), question("goal"), question("goal"))

    for message in ("a", "b", "c"):
        engine.chat(message)

    assert "reached its turn limit" not in engine.tasks[-1]
    assert "You may not propose a brief yet" in engine.tasks[-1]


def test_bootstrap_greets_in_the_requested_language(engine):
    assert "Progettiamo" in engine.bootstrap("ITA").suggested_questions_html


@pytest.mark.parametrize("lang", ["ITA", "ENG", "FRA", "SPA", "XYZ", None])
def test_bootstrap_ends_with_the_opening_question(lang):
    html = get_static_bootstrap_html(lang)

    assert html.endswith(f"<p>{get_opening_question(lang)}</p>")


def test_an_unknown_language_opens_in_english():
    assert get_opening_question("XYZ") == get_opening_question("ENG")


def test_the_log_label_is_the_interviewer(engine):
    assert engine.ENGINE_NAME == "interviewer"
    assert isinstance(engine._agent, MagicMock)
