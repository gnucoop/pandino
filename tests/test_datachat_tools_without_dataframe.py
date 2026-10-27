"""
The DataChat tools are built with df=None in a SQL-only session: they work on the
rows passed in data, and without data they return INVALID_DATA instead of crashing.
"""

import pytest

from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.chart_tool import ChartTool
from datachat.tools.classify_match_tool import ClassifyMatchTool
from datachat.tools.compare_groups_tool import CompareGroupsTool
from datachat.tools.correlation_tool import CorrelationTool
from datachat.tools.crosstab_tool import CrosstabTool
from datachat.tools.describe_tool import DescribeTool
from datachat.tools.filter_rows_tool import FilterRowsTool
from datachat.tools.keywords_tool import KeywordsTool
from datachat.tools.missing_values_tool import MissingValuesTool
from datachat.tools.plot_tool import PlotTool
from datachat.tools.row_count_tool import RowCountTool
from datachat.tools.sample_rows_tool import SampleRowsTool
from datachat.tools.sentiment_tool import SentimentAnalysisTool
from datachat.tools.top_rows_tool import TopRowsTool
from datachat.tools.trend_tool import TrendTool
from datachat.tools.unique_values_tool import UniqueValuesTool

LLM = {"model": None, "provider": "p", "model_name": "m"}

ROWS = [{"region": "north", "amount": 10.0}, {"region": "south", "amount": 20.0}, {"region": "north", "amount": 5.0}]

# (tool, required arguments of forward())
TOOLS_WITHOUT_DATAFRAME = [
    (AggregateTool(None), {"group_by": "region", "op": "sum", "metric": "amount"}),
    (ChartTool(None), {"kind": "bar", "x": "region", "y": "amount"}),
    (ClassifyMatchTool(None, **LLM), {"column": "region", "categories": ["a", "b"]}),
    (CompareGroupsTool(None), {"metric": "amount", "group_col": "region", "group_a": "north", "group_b": "south"}),
    (CorrelationTool(None), {}),
    (CrosstabTool(None), {"rows": "region", "columns": "amount"}),
    (DescribeTool(None), {}),
    (FilterRowsTool(None), {"where_col": "region", "value": "north"}),
    (KeywordsTool(None), {"column": "region", "language": "english"}),
    (MissingValuesTool(None), {}),
    (PlotTool(None, output_dir="/nonexistent"), {"kind": "bar", "x": "region"}),
    (RowCountTool(None), {}),
    (SampleRowsTool(None), {}),
    (SentimentAnalysisTool(None, **LLM), {"col": "region"}),
    (TopRowsTool(None), {"sort_by": "amount"}),
    (TrendTool(None), {"date_col": "day", "freq": "month", "op": "count"}),
    (UniqueValuesTool(None), {"column": "region"}),
]


@pytest.mark.parametrize(
    ("tool", "kwargs"), TOOLS_WITHOUT_DATAFRAME, ids=[tool.name for tool, _ in TOOLS_WITHOUT_DATAFRAME]
)
def test_without_a_dataframe_nor_data_the_tool_reports_there_is_no_data(tool, kwargs):
    out = tool.forward(**kwargs)

    assert out["kind"] == "error"
    assert out["code"] == "INVALID_DATA"
    assert "no dataset and no data was passed" in out["message"]


@pytest.mark.parametrize(
    ("tool", "kwargs", "expected"),
    [
        (RowCountTool(None), {}, [{"row_count": 3}]),
        (AggregateTool(None), {"group_by": "region", "op": "sum", "metric": "amount"}, None),
        (UniqueValuesTool(None), {"column": "region"}, None),
        (DescribeTool(None), {}, None),
    ],
    ids=["row_count", "aggregate", "unique_values", "describe"],
)
def test_without_a_dataframe_the_tool_works_on_the_passed_rows(tool, kwargs, expected):
    out = tool.forward(data=ROWS, **kwargs)

    assert out["kind"] == "table"
    assert out["data"]
    if expected is not None:
        assert out["data"] == expected
