"""Tests for the analyst's SQL tools (no PostgreSQL: FakeDatasource)."""

from decimal import Decimal

from analyst.tools.load_dataset_tool import LoadDatasetTool
from analyst.tools.sql_correlation_tool import SqlCorrelationTool
from analyst.tools.sql_engine_tool import AnalystSqlEngineTool
from analyst.tools.sql_group_stats_tool import SqlGroupStatsTool
from analyst.tools.sql_outliers_tool import SqlOutliersTool
from analyst.tools.sql_stats_tool import SqlStatsTool
from tests.fake_sql_datasource import FakeDatasource

IDENTIFIERS = {"Trasporti": ("idTrasporto", "Data", "KmTotali"), "report_mensile": ("mese", "ricavo")}


def datasource(**kwargs):
    return FakeDatasource(tables=list(IDENTIFIERS), **kwargs)


def last_sql(ds):
    return ds.run_select_calls[-1][0]


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_puts_a_dataframe_in_the_shared_dict():
    ds = datasource(
        result_columns=["mese", "ricavo"],
        rows=[("202604", Decimal("10.5")), ("202605", Decimal("12.25")), ("202606", None)],
    )
    datasets = {}
    tool = LoadDatasetTool(ds, datasets, max_rows=1000, identifiers=IDENTIFIERS)

    result = tool.forward("revenue", 'SELECT "mese", "ricavo" FROM "report_mensile"')

    assert result["kind"] == "dataset"
    assert result["rows"] == 3
    assert result["columns"] == {"mese": "object", "ricavo": "float64"}
    assert result["sample"][0] == {"mese": "202604", "ricavo": 10.5}
    assert datasets["revenue"]["ricavo"].sum() == 22.75
    assert ds.run_select_calls[-1][1] == 1000


def test_load_dataset_warns_when_rows_were_dropped():
    ds = datasource(result_columns=["a"], rows=[(1,)], truncated=True)
    result = LoadDatasetTool(ds, {}, max_rows=1).forward("x", "SELECT 1 AS a")
    assert result["truncated"] is True
    assert "computed on it are wrong" in result["warning"]
    assert "TABLESAMPLE SYSTEM" in result["warning"]


def test_load_dataset_refuses_writes_and_bad_names():
    ds = datasource()
    tool = LoadDatasetTool(ds, {}, max_rows=10)

    assert tool.forward("x", "DELETE FROM report_mensile")["code"] == "NON_SELECT_STATEMENT"
    assert tool.forward("Bad Name", "SELECT 1")["code"] == "INVALID_ARGUMENT"
    assert ds.run_select_calls == []


def test_load_dataset_refuses_hidden_tables():
    ds = datasource(allowed_tables=frozenset({"report_mensile"}))
    result = LoadDatasetTool(ds, {}, max_rows=10).forward("x", "SELECT * FROM secrets")
    assert result["code"] == "TABLE_NOT_ALLOWED"


# ---------------------------------------------------------------------------
# Statistics tools
# ---------------------------------------------------------------------------


def test_sql_stats_resolves_real_column_names():
    ds = datasource(
        result_columns=[f"c0_{s}" for s in ("count", "mean", "std", "min", "max", "p25", "median", "p75")],
        rows=[(10, Decimal("172.123456"), 5.0, 1, 400, 100.0, 170.0, 220.0)],
    )
    result = SqlStatsTool(ds, IDENTIFIERS).forward("trasporti", "kmtotali")

    sql = last_sql(ds)
    assert 'AVG("KmTotali") AS c0_mean' in sql
    assert 'FROM "Trasporti" AS source_data' in sql
    assert result["stats"]["kmtotali"]["mean"] == 172.1235
    assert result["stats"]["kmtotali"]["count"] == 10


def test_sql_stats_rejects_unknown_columns_and_relations():
    tool = SqlStatsTool(datasource(), IDENTIFIERS)
    assert tool.forward("Trasporti", "nope")["code"] == "INVALID_COLUMN"
    assert tool.forward("nowhere", "x")["code"] == "INVALID_TABLE"


def test_a_query_source_is_wrapped_and_guarded():
    ds = datasource(result_columns=["n", "r_0_1"], rows=[(5, 0.81234)])
    tool = SqlCorrelationTool(ds, IDENTIFIERS)

    result = tool.forward('SELECT "KmTotali" AS km, 2 AS b FROM "Trasporti"', "km, b")
    assert result["matrix"]["km"] == {"km": 1.0, "b": 0.8123}
    assert result["matrix"]["b"]["km"] == 0.8123
    assert '(SELECT "KmTotali" AS km, 2 AS b FROM "Trasporti") AS source_data' in last_sql(ds)

    # A write smuggled in as the source never reaches the database.
    calls = len(ds.run_select_calls)
    assert tool.forward("WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x", "a, b")["kind"] == "error"
    assert len(ds.run_select_calls) == calls


def test_sql_correlation_needs_two_columns():
    assert SqlCorrelationTool(datasource(), IDENTIFIERS).forward("Trasporti", "KmTotali")["code"] == "INVALID_ARGUMENT"


def test_sql_outliers_reports_total_bounds_and_sample():
    ds = datasource(
        result_columns=["count", "low", "high", "idTrasporto", "KmTotali"],
        rows=[(2, 10.0, 300.0, "T1", 900), (2, 10.0, 300.0, "T2", 5)],
    )
    tool = SqlOutliersTool(ds, max_rows=50, identifiers=IDENTIFIERS)

    result = tool.forward("Trasporti", "KmTotali", "iqr")

    assert result["total_outliers"] == 2
    assert result["threshold"] == 1.5
    assert result["bounds"] == {"low": 10.0, "high": 300.0}
    assert result["sample"] == [{"idTrasporto": "T1", "KmTotali": 900}, {"idTrasporto": "T2", "KmTotali": 5}]
    assert "PERCENTILE_CONT(0.25)" in last_sql(ds)
    assert ds.run_select_calls[-1][1] == 50


def test_sql_outliers_rejects_unknown_methods():
    tool = SqlOutliersTool(datasource(), max_rows=50, identifiers=IDENTIFIERS)
    assert tool.forward("Trasporti", "KmTotali", "magic")["code"] == "INVALID_ARGUMENT"


def test_sql_group_stats_builds_aliases_and_counts_groups():
    ds = datasource(
        result_columns=["total_groups", "mese", "ricavo_sum"],
        rows=[(4, "202606", Decimal("3500000.123456")), (4, "202605", Decimal("3100000"))],
        truncated=True,
    )
    tool = SqlGroupStatsTool(ds, max_rows=2, identifiers=IDENTIFIERS)

    result = tool.forward("report_mensile", "mese", "ricavo:sum")

    assert result["total_groups"] == 4
    assert result["truncated"] is True
    assert result["groups"][0] == {"mese": "202606", "ricavo_sum": 3500000.1235}
    sql = last_sql(ds)
    assert 'SUM("ricavo") AS "ricavo_sum"' in sql
    assert 'ORDER BY "ricavo_sum" DESC NULLS LAST' in sql


def test_sql_group_stats_rejects_unknown_aggregates():
    tool = SqlGroupStatsTool(datasource(), max_rows=10, identifiers=IDENTIFIERS)
    assert tool.forward("report_mensile", "mese", "ricavo:median")["code"] == "INVALID_ARGUMENT"


def test_database_errors_become_tool_errors():
    from tests.fake_sql_datasource import operational_error

    ds = datasource(raises=operational_error())
    result = SqlStatsTool(ds, IDENTIFIERS).forward("Trasporti", "KmTotali")
    assert result["kind"] == "error"
    assert "hunter2" not in result["message"]


# ---------------------------------------------------------------------------
# sql_engine
# ---------------------------------------------------------------------------


def test_sql_engine_is_capped_by_the_analyst_limit():
    ds = datasource(result_columns=["a"], rows=[(1,)])
    tool = AnalystSqlEngineTool(ds, max_rows=50, identifiers=IDENTIFIERS)

    tool.forward("SELECT 1 AS a")
    assert ds.run_select_calls[-1][1] == 50
    tool.forward("SELECT 1 AS a", max_rows=500)
    assert ds.run_select_calls[-1][1] == 50
    tool.forward("SELECT 1 AS a", max_rows=5)
    assert ds.run_select_calls[-1][1] == 5
