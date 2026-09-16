"""
Tests for sql_engine, the only remaining SQL tool. A duck-typed fake datasource
stands in for the database, so nothing here needs PostgreSQL.

Schema discovery is no longer a tool: it is reflected once and rendered into the
system prompt. Those tests live in tests/test_schema_snapshot_loader.py.
"""

import pytest

from datachat.tools.sql_engine_tool import SqlEngineTool
from tests.fake_sql_datasource import FakeDatasource, operational_error


# ---------------------------------------------------------------------------
# Tool construction
# ---------------------------------------------------------------------------


def test_sql_engine_declares_its_name():
    assert SqlEngineTool(FakeDatasource()).name == "sql_engine"


def test_sql_engine_instantiates_without_error():
    """Exercises smolagents' inputs/forward signature cross-check."""
    assert SqlEngineTool(FakeDatasource()).output_type == "object"



# ---------------------------------------------------------------------------
# sql_engine
# ---------------------------------------------------------------------------


def test_sql_engine_happy_path():
    datasource = FakeDatasource(
        result_columns=["country", "n"], rows=[("IT", 42), ("FR", 7)]
    )
    out = SqlEngineTool(datasource).forward("SELECT country, count(*) AS n FROM t")
    assert out["kind"] == "table"
    assert out["data"] == [{"country": "IT", "n": 42}, {"country": "FR", "n": 7}]
    assert out["meta"]["returned"] == 2
    assert out["meta"]["truncated"] is False
    assert out["meta"]["columns"] == ["country", "n"]


@pytest.mark.parametrize(
    "query,code",
    [
        ("", "MISSING_QUERY"),
        ("SELECT 1; DROP TABLE t", "MULTIPLE_STATEMENTS"),
        ("DROP TABLE t", "NON_SELECT_STATEMENT"),
        ("WITH x AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM x",
         "FORBIDDEN_STATEMENT"),
        ("SELECT pg_sleep(30)", "FORBIDDEN_FUNCTION"),
    ],
)
def test_sql_engine_never_queries_when_the_guard_rejects(query, code):
    datasource = FakeDatasource()
    out = SqlEngineTool(datasource).forward(query)
    assert out["kind"] == "error"
    assert out["code"] == code
    assert datasource.run_select_calls == []


def test_sql_engine_enforces_the_table_denylist():
    datasource = FakeDatasource(denied_tables=frozenset({"users"}))
    out = SqlEngineTool(datasource).forward("SELECT * FROM users")
    assert out["code"] == "TABLE_NOT_ALLOWED"
    assert datasource.run_select_calls == []


def test_sql_engine_max_rows_cannot_be_raised():
    datasource = FakeDatasource(max_rows=200)
    SqlEngineTool(datasource).forward("SELECT * FROM t", max_rows=99999)
    assert datasource.run_select_calls[0][1] == 200


def test_sql_engine_max_rows_can_be_lowered():
    datasource = FakeDatasource(max_rows=200)
    SqlEngineTool(datasource).forward("SELECT * FROM t", max_rows=5)
    assert datasource.run_select_calls[0][1] == 5


def test_sql_engine_reports_truncation():
    datasource = FakeDatasource(
        result_columns=["a"], rows=[(1,), (2,)], truncated=True, max_rows=2
    )
    out = SqlEngineTool(datasource).forward("SELECT a FROM t")
    assert out["meta"]["truncated"] is True
    assert len(out["data"]) == 2


def test_sql_engine_caps_columns():
    datasource = FakeDatasource(
        result_columns=["a", "b", "c"], rows=[(1, 2, 3)], max_columns=2
    )
    out = SqlEngineTool(datasource).forward("SELECT a, b, c FROM t")
    assert out["data"] == [{"a": 1, "b": 2}]
    assert out["meta"]["columns_truncated"] is True
    assert out["meta"]["column_count"] == 2


def test_sql_engine_truncates_wide_cells():
    datasource = FakeDatasource(
        result_columns=["a"], rows=[("x" * 500,)], max_cell_chars=10
    )
    out = SqlEngineTool(datasource).forward("SELECT a FROM t")
    assert out["data"][0]["a"] == "x" * 10 + "…"


def test_sql_engine_serialises_exotic_values():
    from datetime import date
    from decimal import Decimal
    from uuid import UUID

    datasource = FakeDatasource(
        result_columns=["amount", "day", "ref"],
        rows=[(Decimal("12.50"), date(2026, 9, 1), UUID(int=1))],
    )
    out = SqlEngineTool(datasource).forward("SELECT amount, day, ref FROM t")
    row = out["data"][0]
    assert row["amount"] == 12.5
    assert row["day"] == "2026-09-01"
    assert row["ref"] == "00000000-0000-0000-0000-000000000001"


@pytest.mark.parametrize(
    "sqlstate,code",
    [
        ("57014", "QUERY_TIMEOUT"),
        ("25006", "READ_ONLY_VIOLATION"),
        ("42501", "PERMISSION_DENIED"),
        ("42P01", "INVALID_TABLE"),
        ("42703", "INVALID_COLUMN"),
        ("42601", "INVALID_SQL"),
        (None, "DB_UNAVAILABLE"),
    ],
)
def test_sql_engine_maps_sqlstates(sqlstate, code):
    datasource = FakeDatasource(raises=operational_error(sqlstate))
    out = SqlEngineTool(datasource).forward("SELECT * FROM t")
    assert out["kind"] == "error"
    assert out["code"] == code


def test_connection_error_message_never_leaks_credentials():
    """libpq echoes the conninfo into connection failures, so it is suppressed."""
    datasource = FakeDatasource(raises=operational_error())
    out = SqlEngineTool(datasource).forward("SELECT * FROM t")
    assert "hunter2" not in out["message"]
    assert "secret" not in out["message"]
    assert len(out["message"]) <= 300


def test_statement_error_message_is_passed_through():
    """Statement errors are what let the agent repair its own query."""
    datasource = FakeDatasource(raises=operational_error("42703"))
    out = SqlEngineTool(datasource).forward("SELECT nope FROM t")
    assert out["code"] == "INVALID_COLUMN"
    assert "connection failed" in out["message"]


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


def test_sql_engine_hints_when_nothing_matched():
    """
    A valid query matching nothing is the one failure no other layer can see:
    the guard passed it, the database accepted it, and the answer is still wrong
    whenever the filter value was written the way the question phrased it rather
    than the way the data stores it.
    """
    datasource = FakeDatasource(result_columns=["primo_margine"], rows=[])

    out = SqlEngineTool(datasource).forward(
        "SELECT primo_margine FROM report_mensile WHERE mese = '07'"
    )

    assert out["kind"] == "table"
    assert out["data"] == []
    assert out["meta"]["returned"] == 0
    assert "SELECT DISTINCT" in out["meta"]["hint"]


def test_sql_engine_does_not_hint_when_rows_came_back():
    datasource = FakeDatasource(result_columns=["primo_margine"], rows=[(182.0,)])

    out = SqlEngineTool(datasource).forward("SELECT primo_margine FROM report_mensile")

    assert "hint" not in out["meta"]


# ---------------------------------------------------------------------------
# Identifier case repair
# ---------------------------------------------------------------------------

IDENTIFIERS = {"Clienti": ["Idcliente", "Descrizione"], "report_mensile": ["anno"]}


def test_sql_engine_executes_the_quoted_query():
    """
    What the guard validated must be exactly what runs, so the rewritten text is
    the text handed to run_select — not the agent's original.
    """
    datasource = FakeDatasource(result_columns=["Idcliente"], rows=[(1,)])

    SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        "SELECT Idcliente FROM Clienti"
    )

    executed, _ = datasource.run_select_calls[0]
    assert executed == 'SELECT "Idcliente" FROM "Clienti"'


def test_sql_engine_leaves_lowercase_relations_alone():
    datasource = FakeDatasource(result_columns=["anno"], rows=[("2026",)])

    SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        "SELECT anno FROM report_mensile"
    )

    assert datasource.run_select_calls[0][0] == "SELECT anno FROM report_mensile"


def test_sql_engine_without_identifiers_rewrites_nothing():
    """The default keeps the tool usable standalone, exactly as before."""
    datasource = FakeDatasource(result_columns=["Idcliente"], rows=[(1,)])

    SqlEngineTool(datasource).forward("SELECT Idcliente FROM Clienti")

    assert datasource.run_select_calls[0][0] == "SELECT Idcliente FROM Clienti"


def test_sql_engine_respects_the_kill_switch():
    datasource = FakeDatasource(
        result_columns=["Idcliente"], rows=[(1,)], quote_identifiers=False
    )

    SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        "SELECT Idcliente FROM Clienti"
    )

    assert datasource.run_select_calls[0][0] == "SELECT Idcliente FROM Clienti"


def test_the_guard_still_rejects_a_rewritten_query():
    """Rewriting happens before validation, so it cannot smuggle anything past it."""
    datasource = FakeDatasource()

    out = SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        "DELETE FROM Clienti"
    )

    assert out["kind"] == "error"
    assert datasource.run_select_calls == []


def statement_error(message, sqlstate):
    """A failed statement, as opposed to the failed connection above."""
    from sqlalchemy.exc import ProgrammingError

    class Orig(Exception):
        pass

    orig = Orig(message)
    orig.sqlstate = sqlstate
    return ProgrammingError("SELECT 1", {}, orig)


def test_an_unknown_table_error_suggests_the_real_spelling():
    datasource = FakeDatasource(
        raises=statement_error('relation "clienti" does not exist', "42P01")
    )

    out = SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        'SELECT 1 FROM "clienti"'
    )

    assert out["code"] == "INVALID_TABLE"
    assert '"Clienti"' in out["message"]


def test_an_unknown_column_error_suggests_the_real_spelling():
    datasource = FakeDatasource(
        raises=statement_error('column "descrizione" does not exist', "42703")
    )

    out = SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        'SELECT "descrizione" FROM "Clienti"'
    )

    assert out["code"] == "INVALID_COLUMN"
    assert '"Descrizione"' in out["message"]


def test_the_suggestion_survives_a_long_database_message():
    """It is the actionable half; truncation must never eat it."""

    datasource = FakeDatasource(
        raises=statement_error('relation "clienti" does not exist ' + "x" * 500, "42P01")
    )

    out = SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        'SELECT 1 FROM "clienti"'
    )

    assert out["message"].endswith("exactly as shown in your instructions.")
    assert '"Clienti"' in out["message"]


def test_an_unknown_name_gets_no_invented_suggestion():
    datasource = FakeDatasource(
        raises=statement_error('relation "sconosciuta" does not exist', "42P01")
    )

    out = SqlEngineTool(datasource, identifiers=IDENTIFIERS).forward(
        'SELECT 1 FROM "sconosciuta"'
    )

    assert "Did you mean" not in out["message"]
