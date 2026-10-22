"""Tests for the large relations note of the analyst's schema (no PostgreSQL: FakeDatasource)."""

from analyst.sql_prompt import large_relations_note
from tests.fake_sql_datasource import FakeDatasource

NAMES = ["Trasporti", "report_mensile", "Clienti"]


def test_large_relations_are_listed_largest_first_with_the_rules():
    ds = FakeDatasource(result_columns=["relname", "reltuples"],
                        rows=[("report_mensile", 850_000), ("Trasporti", 15_234_567)])

    note = large_relations_note(ds, NAMES, max_rows=100000)

    assert note.startswith('LARGE TABLES (rows estimated from PostgreSQL statistics): '
                           '"Trasporti" ~15.2M, "report_mensile" ~850k.')
    assert "only the first 100000 rows" in note
    assert "TABLESAMPLE SYSTEM" in note


def test_the_estimate_is_one_catalog_query_on_the_datasource_schema():
    ds = FakeDatasource()
    ds.schema = "it's"

    large_relations_note(ds, NAMES, max_rows=100000)

    [(sql, _)] = ds.run_select_calls
    assert "FROM pg_class" in sql
    assert "n.nspname = 'it''s'" in sql
    assert "c.reltuples > 100000" in sql


def test_relations_the_agent_was_not_shown_are_left_out():
    ds = FakeDatasource(rows=[("secret_table", 20_000_000), ("trasporti", 2_000_000)])

    note = large_relations_note(ds, NAMES, max_rows=100000)

    assert "secret_table" not in note
    assert '"Trasporti" ~2.0M' in note


def test_no_large_relation_or_no_estimate_means_no_note():
    assert large_relations_note(FakeDatasource(), NAMES, max_rows=100000) == ""
    failing = FakeDatasource(raises=RuntimeError("permission denied for pg_class"))
    assert large_relations_note(failing, NAMES, max_rows=100000) == ""
    odd_rows = FakeDatasource(rows=[("Trasporti",)])
    assert large_relations_note(odd_rows, NAMES, max_rows=100000) == ""
