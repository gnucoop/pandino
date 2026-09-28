"""
The CSV/DataFrame path, which is what DataChat did before the SQL datasource
existed and must keep doing unchanged when SQL is off.

SmolagentsEngine is built via __new__ so that no LLM model is constructed and no
database is contacted.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from datachat.smolagents_engine import SmolagentsEngine

EXPECTED_TOOLS = [
    "describe",
    "missing_values",
    "unique_values",
    "correlation",
    "sample_rows",
    "top_rows",
    "filter_rows",
    "row_count",
    "aggregate",
    "plot",
    "trend",
]


@pytest.fixture
def engine(tmp_path):
    instance = SmolagentsEngine.__new__(SmolagentsEngine)
    instance.user_name = "tester"
    instance.data = pd.DataFrame({"country": ["IT", "FR"], "sales": [10, 20]})
    instance._sql_ready = False
    instance._plots_dir = str(tmp_path)
    instance._empty_final_rejections = 0
    instance._last_run_result = None
    instance._last_run_duration_ms = None
    return instance


@pytest.fixture(autouse=True)
def prompts_from_code():
    """Skip the prompts table: use the in-code default, as load_prompt would."""
    with patch(
        "datachat.smolagents_engine.load_prompt",
        side_effect=lambda title, default_text="", **kwargs: default_text,
    ):
        yield


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def test_a_csv_session_builds_exactly_the_dataframe_tools(engine):
    tools = engine._data_tools()

    assert [t.name for t in tools] == EXPECTED_TOOLS


def test_no_sql_tool_without_a_datasource(engine):
    assert engine._sql_tools(None) == []


def test_without_data_there_are_no_dataframe_tools(engine):
    """A SQL-only session carries no dataframe, so it carries no pandas tools."""
    engine.data = None

    assert engine._data_tools() == []


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

def test_the_column_names_reach_the_prompt(engine):
    """
    The placeholder must be substituted, not shipped literally. It was shipped
    literally before render_prompt learned to leave the template's JSON examples
    alone, which cost the model the column names on every question.
    """
    instructions = engine._build_instructions(None)

    assert "{columns}" not in instructions
    assert "'country', 'sales'" in instructions


def test_the_json_contract_examples_survive_rendering(engine):
    """The literal braces the prompt needs must not be eaten by the renderer."""
    instructions = engine._build_instructions(None)

    assert '{"kind":"text","text":"..."}' in instructions
    assert '{"kind":"table","data":[...]}' in instructions


def test_a_csv_session_is_told_nothing_about_sql(engine):
    instructions = engine._build_instructions(None)

    assert "SQL" not in instructions
    assert "sql_engine" not in instructions


def test_an_empty_dataframe_renders_an_empty_column_list(engine):
    engine.data = None

    instructions = engine._build_instructions(None)

    assert "columns: []" in instructions
    assert "{columns}" not in instructions


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------

def test_an_empty_pandas_table_keeps_its_kind(engine):
    """
    Zero matching rows is a legitimate answer to a filter question, and the
    caller asked for a table. The verification round happens in
    _check_final_answer; it must not turn into a kind change here.
    """
    class _Run:
        output = {"kind": "table", "data": []}

    engine._agent = MagicMock()
    engine._agent.run.return_value = _Run()

    result = engine.chat("clienti che non esistono")

    assert result["kind"] == "table"
    assert result["data"] == []


def test_a_table_with_rows_passes_through(engine):
    class _Run:
        output = {"kind": "table", "data": [{"country": "IT", "sales": 10}]}

    engine._agent = MagicMock()
    engine._agent.run.return_value = _Run()

    result = engine.chat("vendite per paese")

    assert result["kind"] == "table"
    assert result["data"] == [{"country": "IT", "sales": 10}]


def test_the_empty_guard_buys_one_verification_round(engine):
    """The budget applies to a pandas answer exactly as it does to a SQL one."""
    payload = {"kind": "table", "data": []}

    with pytest.raises(ValueError) as exc_info:
        engine._check_final_answer(payload)

    assert "matched no rows" in str(exc_info.value)
    assert engine._empty_final_rejections == 1

    # Budget spent: the same answer is now taken at face value.
    assert engine._check_final_answer(payload) is True
