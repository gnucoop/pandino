"""
Tests that SQL reaches the agent only when the datasource is enabled, and that
the schema is rendered into the prompt rather than discovered through tools.

SmolagentsEngine is built via __new__ so that no LLM model is constructed.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import datachat.schema_snapshot_loader as loader
from datachat.smolagents_engine import SmolagentsEngine
from tests.fake_sql_datasource import FakeDatasource, operational_error

COLUMNS = [
    {"column": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
    {"column": "country", "type": "VARCHAR(2)", "nullable": True, "primary_key": False},
]


def make_datasource(**overrides):
    params = dict(
        tables=["orders"],
        views=["v_sales"],
        materialized_views=["mv_ltv"],
        columns=COLUMNS,
    )
    params.update(overrides)
    return FakeDatasource(**params)


@pytest.fixture
def engine():
    instance = SmolagentsEngine.__new__(SmolagentsEngine)
    instance.user_name = "tester"
    instance.data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    instance._sql_ready = False
    return instance


@pytest.fixture(autouse=True)
def prompts_from_code():
    """Skip the prompts table: use the in-code default, as load_prompt would."""
    with patch(
        "datachat.smolagents_engine.load_prompt",
        side_effect=lambda title, default_text="", **kwargs: default_text,
    ):
        yield


@pytest.fixture(autouse=True)
def reset_snapshot_cache():
    loader.invalidate_snapshot()
    yield
    loader.invalidate_snapshot()


# ---------------------------------------------------------------------------
# Gate: no datasource, nothing SQL
# ---------------------------------------------------------------------------


def test_no_sql_tools_when_datasource_is_disabled(engine):
    assert engine._sql_tools(None) == []


def test_instructions_omit_everything_sql_when_disabled(engine):
    instructions = engine._build_instructions(None)

    assert "SQL DATABASE" not in instructions
    assert "sql_engine" not in instructions


def test_disabled_datasource_never_reflects_the_schema(engine):
    """
    The gate must short-circuit before the loader is reached, not rely on the
    loader to refuse. Asserting on a spy, not just on the output text.
    """
    with patch.object(loader, "get_schema_snapshot") as get_snapshot:
        engine._build_instructions(None)

    get_snapshot.assert_not_called()


# ---------------------------------------------------------------------------
# Enabled: schema in the prompt, sql_engine as the only tool
# ---------------------------------------------------------------------------


def test_sql_engine_is_the_only_sql_tool(engine):
    datasource = make_datasource()
    engine._build_instructions(datasource)

    tools = engine._sql_tools(datasource)

    assert [t.name for t in tools] == ["sql_engine"]


def test_instructions_carry_the_rendered_schema(engine):
    instructions = engine._build_instructions(make_datasource())

    assert "SQL DATABASE" in instructions
    assert '"orders"("id":INTEGER PK, "country":VARCHAR(2))' in instructions
    assert '"v_sales"(' in instructions
    assert '"mv_ltv"(' in instructions
    assert "sql_engine" in instructions


def test_instructions_no_longer_describe_a_discovery_workflow(engine):
    instructions = engine._build_instructions(make_datasource())

    assert "SQL WORKFLOW" not in instructions
    assert "extract_tables" not in instructions
    assert "extract_table_info" not in instructions


def test_include_views_off_keeps_views_out_of_the_prompt(engine):
    instructions = engine._build_instructions(make_datasource(include_views=False))

    assert '"orders"(' in instructions
    assert "v_sales" not in instructions
    assert "mv_ltv" not in instructions


def test_denied_relations_never_reach_the_prompt(engine):
    datasource = make_datasource(
        tables=["orders", "secrets"], denied_tables=frozenset({"secrets"})
    )

    instructions = engine._build_instructions(datasource)

    assert '"orders"(' in instructions
    assert "secrets" not in instructions


def test_enabling_sql_only_appends_to_the_existing_instructions(engine):
    base = engine._build_instructions(None)
    extended = engine._build_instructions(make_datasource())

    assert extended.startswith(base)


def test_dataset_columns_are_actually_substituted(engine):
    """
    The literal JSON braces in the prompt used to make render_prompt give up and
    return the template with {columns} still in it.
    """
    instructions = engine._build_instructions(None)

    assert "{columns}" not in instructions
    assert "'a', 'b'" in instructions


# ---------------------------------------------------------------------------
# Reflection failure
# ---------------------------------------------------------------------------


def test_unreachable_database_drops_sql_entirely(engine):
    """
    An agent that can write SQL but was never told the schema is worse than one
    with no SQL at all, so the tool goes too.
    """
    datasource = make_datasource(raises=operational_error())

    instructions = engine._build_instructions(datasource)

    assert "SQL DATABASE" not in instructions
    assert engine._sql_tools(datasource) == []


def test_unreachable_database_does_not_raise_out_of_agent_construction(engine):
    datasource = make_datasource(raises=operational_error())

    # _build_instructions runs inside __post_init__; it must not propagate.
    assert isinstance(engine._build_instructions(datasource), str)


# ---------------------------------------------------------------------------
# Stored prompt overrides
# ---------------------------------------------------------------------------


def test_override_with_the_placeholder_receives_the_schema(engine):
    with patch(
        "datachat.smolagents_engine.load_prompt",
        side_effect=lambda title, default_text="", **kwargs: (
            "CUSTOM ADDENDUM\n{sql_schema}\nEND"
            if title == "data_chat_sql_addendum"
            else default_text
        ),
    ):
        instructions = engine._build_instructions(make_datasource())

    assert "CUSTOM ADDENDUM" in instructions
    assert '"orders"("id":INTEGER PK' in instructions
    assert "{sql_schema}" not in instructions


def test_override_predating_the_placeholder_still_gets_the_schema(engine):
    """
    A data_chat_sql_addendum row stored before this change has no {sql_schema}
    placeholder. The schema is appended rather than silently dropped.
    """
    with patch(
        "datachat.smolagents_engine.load_prompt",
        side_effect=lambda title, default_text="", **kwargs: (
            "OLD ADDENDUM WITHOUT PLACEHOLDER"
            if title == "data_chat_sql_addendum"
            else default_text
        ),
    ):
        instructions = engine._build_instructions(make_datasource())

    assert "OLD ADDENDUM WITHOUT PLACEHOLDER" in instructions
    assert '"orders"("id":INTEGER PK' in instructions


# ---------------------------------------------------------------------------
# Final-answer guard
# ---------------------------------------------------------------------------


@pytest.fixture
def guarded(engine):
    """The engine fields _check_final_answer touches, as __post_init__ would.

    The per-run budget is the only one left: the guard reads request identity
    from the ambient logging context and keeps no other state on the instance.
    """
    engine._empty_final_rejections = 0
    return engine


def test_a_valid_answer_passes(guarded):
    assert guarded._check_final_answer({"kind": "text", "text": "182,00"}) is True


def test_a_table_with_rows_passes_first_time(guarded):
    payload = {"kind": "table", "data": [{"primo_margine": 182.0}]}

    assert guarded._check_final_answer(payload) is True
    assert guarded._empty_final_rejections == 0


def test_a_broken_contract_is_rejected_with_a_usable_reason(guarded):
    """
    Returning False would reach the model as "failed with error:" and nothing
    more, so the reason has to travel inside the exception.
    """
    with pytest.raises(ValueError) as excinfo:
        guarded._check_final_answer("just some prose")

    message = str(excinfo.value)
    assert "NON_JSON_OR_NO_OBJECT" in message
    assert '{"kind":"table","data":[...]}' in message


def test_an_empty_table_is_rejected_the_first_time(guarded):
    with pytest.raises(ValueError) as excinfo:
        guarded._check_final_answer({"kind": "table", "data": []})

    message = str(excinfo.value)
    assert "SELECT DISTINCT" in message
    assert '{"kind":"text","text":"..."}' in message
    assert guarded._empty_final_rejections == 1


def test_an_empty_table_is_accepted_after_its_verification_round(guarded):
    """
    "Which clients lost money?" may genuinely have no answer. The guard buys one
    verification round; it must not trap the agent in a loop to max_steps.
    """
    with pytest.raises(ValueError):
        guarded._check_final_answer({"kind": "table", "data": []})

    assert guarded._check_final_answer({"kind": "table", "data": []}) is True
    assert guarded._empty_final_rejections == 1


def test_a_text_answer_is_never_treated_as_empty(guarded):
    """The escape hatch the rejection points at must not itself be rejected."""
    payload = {"kind": "text", "text": "Nessuna riga corrisponde a quel cliente."}

    assert guarded._check_final_answer(payload) is True
    assert guarded._empty_final_rejections == 0


def test_the_guard_accepts_the_smolagents_call_signature(guarded):
    """smolagents calls check(final_answer, memory, agent=agent)."""
    assert (
        guarded._check_final_answer(
            {"kind": "text", "text": "ok"}, object(), agent=object()
        )
        is True
    )


def test_chat_resets_the_empty_budget_between_runs(guarded):
    """The budget is per question, not per session."""
    guarded._agent = None  # short-circuits chat() before the model is needed
    guarded._empty_final_rejections = 1

    guarded.chat("qualsiasi domanda")

    assert guarded._empty_final_rejections == 0


def test_chat_returns_an_empty_table_as_an_empty_table(guarded):
    """
    The response kind is a contract. The guard has already bought the agent its
    verification round, which is where a wrong filter value gets caught; past
    that point an empty result is the answer, and rewriting it into text would
    change `kind` out from under a caller that asked for a table.
    """

    class _Run:
        output = {"kind": "table", "data": []}

    guarded._agent = MagicMock()
    guarded._agent.run.return_value = _Run()

    result = guarded.chat("clienti inesistenti")

    assert result["kind"] == "table"
    assert result["data"] == []


def test_chat_passes_a_table_with_rows_through(guarded):
    class _Run:
        output = {"kind": "table", "data": [{"primo_margine": 182.0}]}

    guarded._agent = MagicMock()
    guarded._agent.run.return_value = _Run()

    result = guarded.chat("primo margine CLO luglio 2026")

    assert result["kind"] == "table"
    assert result["data"] == [{"primo_margine": 182.0}]


def test_the_sql_tool_is_given_the_identifiers_it_may_quote(engine):
    """
    Built from the snapshot the prompt was rendered from, so a TTL expiry cannot
    leave the tool quoting names the agent was never shown.
    """
    datasource = make_datasource(tables=["Clienti"], views=[], materialized_views=[])
    engine._build_instructions(datasource)

    tool = engine._sql_tools(datasource)[0]

    assert tool._identifiers == {"Clienti": ("id", "country")}


def test_the_addendum_carries_the_case_folding_rule(engine):
    instructions = engine._build_instructions(make_datasource())

    assert "case-sensitive" in instructions
    assert 'relation "clienti" does not exist' in instructions
