"""
I5: the agent is told how to use the analytical vocabulary honestly.

Presence/absence ratchets on behavioural markers, not prompt snapshots.
SmolagentsEngine is built via __new__ so that no LLM model is constructed.
"""

from unittest.mock import patch

import pandas as pd
import pytest

import datachat.schema_snapshot_loader as loader
from datachat.smolagents_engine import SmolagentsEngine
from datachat.tools.classify_match_tool import ClassifyMatchTool
from datachat.tools.plot_tool import PlotTool
from datachat.tools.sentiment_tool import SentimentAnalysisTool
from tests.fake_sql_datasource import FakeDatasource

COLUMNS = [
    {"column": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
    {"column": "amount", "type": "NUMERIC", "nullable": True, "primary_key": False},
]


def _engine(data):
    instance = SmolagentsEngine.__new__(SmolagentsEngine)
    instance.user_name = "tester"
    instance.data = data
    instance._sql_ready = False
    return instance


def _datasource():
    return FakeDatasource(tables=["orders"], views=[], materialized_views=[], columns=COLUMNS)


def _prompts(stored_system=None):
    def load(title, default_text="", **kwargs):
        if title == "data_chat_system" and stored_system is not None:
            return stored_system
        return default_text

    return patch("datachat.smolagents_engine.load_prompt", side_effect=load)


@pytest.fixture(autouse=True)
def reset_snapshot_cache():
    loader.invalidate_snapshot()
    yield
    loader.invalidate_snapshot()


@pytest.fixture
def df():
    return pd.DataFrame({"comment": ["ok"], "age": [30]})


def _sql_section(instructions):
    return instructions[instructions.index("SQL RULES"):]


# --- dataframe session ---------------------------------------------------------


def test_dataframe_session_carries_the_honesty_block(df):
    with _prompts():
        text = _engine(df)._build_instructions(None)

    assert "ANALYTICAL HONESTY" in text
    assert "trusted caveat" in text
    assert "all rows" in text and "do not present the result as complete" in text
    assert "classification_status" in text
    assert "Never infer" in text
    assert "not \"neutral\"" in text


def test_honesty_block_survives_a_stored_base_prompt_override(df):
    with _prompts(stored_system="STORED SYSTEM PROMPT {columns}"):
        text = _engine(df)._build_instructions(None)

    assert "STORED SYSTEM PROMPT" in text
    assert "You are DataChat" not in text
    assert "ANALYTICAL HONESTY" in text
    assert "CHARTS" in text


def test_taxonomy_workflow_is_a_proposal_not_a_clustering_tool(df):
    with _prompts():
        text = _engine(df)._build_instructions(None)

    assert "keywords" in text and "classify_match" in text
    assert "your proposal" in text
    assert "no automatic clustering tool" in text


# --- charts --------------------------------------------------------------------


def test_charts_route_distributions_by_column_kind(df):
    with _prompts():
        text = _engine(df)._build_instructions(None)
    charts = text[text.index("CHARTS"):text.index("ANALYTICAL HONESTY")]

    assert "Distribution of one column" not in charts
    assert 'Categorical or ordinal column' in charts and 'chart(kind="bar", x=' in charts
    assert 'continuous numeric column use plot(kind="hist"' in charts
    assert 'data=result["data"]' in charts
    assert "trend (or aggregate) first" in charts and 'chart(kind="line"' in charts
    for kind in ("Histogram", "box", "KDE", "hexbin"):
        assert kind in charts


# --- SQL-only session ------------------------------------------------------------


def test_sql_only_session_gets_sql_completeness_but_no_dataframe_guidance():
    with _prompts():
        text = _engine(None)._build_instructions(_datasource())

    assert "CHARTS" not in text
    assert "ANALYTICAL HONESTY" not in text
    sql = _sql_section(text)
    assert "GROUP BY" in sql and "COUNT/SUM/AVG/MIN/MAX" in sql
    assert "narrow the query and run it again" not in sql
    assert "Do not treat them as the whole source" in sql
    assert "part of what the user asked about" in sql


# --- hybrid session --------------------------------------------------------------


def test_hybrid_session_gets_both_guidance_sets(df):
    with _prompts():
        text = _engine(df)._build_instructions(_datasource())

    assert "ANALYTICAL HONESTY" in text
    assert "CHARTS" in text
    assert "GROUP BY" in _sql_section(text)


# --- tool descriptions -----------------------------------------------------------


def test_plot_description_is_positioned_as_the_statistical_fallback():
    description = PlotTool.description
    assert "histogram, box" in description and "hexbin" in description
    assert "prefer" in description and "'chart' tool" in description


def test_classify_match_description_points_coverage_at_classification_status():
    description = ClassifyMatchTool.description
    assert "A null category covers both 'unclassified' and 'unavailable'" in description
    assert "use classification_status, not category" in description


def test_sentiment_description_says_null_is_not_neutral():
    description = SentimentAnalysisTool.description
    assert "never 'neutral'" in description
    assert "prefer aggregate=True" in description
    assert "coverage note" in description


def test_honesty_block_explains_the_agent_visible_partial_flag(df):
    with _prompts(stored_system="Custom base prompt."):
        text = _engine(df)._build_instructions(None)

    honesty = text[text.index("ANALYTICAL HONESTY"):]
    assert '"more_rows_available": true' in honesty
    assert "incomplete source rows" in honesty
    assert "only the retrieved rows" in honesty
