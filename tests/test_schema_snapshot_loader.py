"""
Tests for datachat.schema_snapshot_loader.

These cover what the deleted extract_* tools used to guarantee — allow/deny
filtering, the include_views group gate, relation kinds never being confused,
connection failures degrading cleanly — plus the caching this module exists for.
"""

import threading
import time

import pytest

import datachat.schema_snapshot_loader as loader
from tests.fake_sql_datasource import FakeDatasource, operational_error

COLUMNS = [
    {"column": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
    {"column": "country", "type": "VARCHAR(2)", "nullable": True, "primary_key": False},
]


@pytest.fixture(autouse=True)
def reset_cache():
    """The snapshot cache is a module global; clear it around every test."""
    loader.invalidate_snapshot()
    yield
    loader.invalidate_snapshot()


# ---------------------------------------------------------------------------
# Gate: nothing happens without a datasource
# ---------------------------------------------------------------------------


def test_no_datasource_returns_none():
    assert loader.get_schema_snapshot(None) is None


def test_no_datasource_does_not_populate_the_cache():
    """
    The enabled check belongs to the caller. If one ever forgets it, a None
    datasource must still not leave anything cached behind.
    """
    loader.get_schema_snapshot(None)

    assert loader._snapshot is None
    assert loader._snapshot_at == 0.0


def test_invalidate_snapshot_is_safe_when_nothing_is_cached():
    loader.invalidate_snapshot()  # must not raise


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------


def test_loads_tables_with_columns_and_types():
    datasource = FakeDatasource(tables=["orders"], columns=COLUMNS)

    snapshot = loader.load_schema_snapshot(datasource)

    assert snapshot.schema == "public"
    assert snapshot.relation_names() == ("orders",)
    relation = snapshot.relations[0]
    assert relation.kind == "table"
    assert [(c.name, c.type, c.primary_key) for c in relation.columns] == [
        ("id", "INTEGER", True),
        ("country", "VARCHAR(2)", False),
    ]


def test_denied_relations_are_absent_from_the_snapshot():
    datasource = FakeDatasource(
        tables=["orders", "secrets"],
        columns=COLUMNS,
        denied_tables=frozenset({"secrets"}),
    )

    snapshot = loader.load_schema_snapshot(datasource)

    assert snapshot.relation_names() == ("orders",)


def test_allowlist_wins_over_everything_else():
    datasource = FakeDatasource(
        tables=["orders", "customers"],
        views=["v_sales"],
        columns=COLUMNS,
        allowed_tables=frozenset({"orders"}),
    )

    snapshot = loader.load_schema_snapshot(datasource)

    assert snapshot.relation_names() == ("orders",)


def test_include_views_off_omits_views_and_materialized_views():
    datasource = FakeDatasource(
        tables=["orders"],
        views=["v_sales"],
        materialized_views=["mv_ltv"],
        columns=COLUMNS,
        include_views=False,
    )

    snapshot = loader.load_schema_snapshot(datasource)

    assert snapshot.relation_names() == ("orders",)


def test_each_relation_keeps_its_own_kind():
    """A table must never surface as a view, or a view as a table."""
    datasource = FakeDatasource(
        tables=["orders"],
        views=["v_sales"],
        materialized_views=["mv_ltv"],
        columns=COLUMNS,
    )

    snapshot = loader.load_schema_snapshot(datasource)

    assert {r.name: r.kind for r in snapshot.relations} == {
        "orders": "table",
        "v_sales": "view",
        "mv_ltv": "materialized_view",
    }


def test_empty_database_is_not_an_error():
    snapshot = loader.load_schema_snapshot(FakeDatasource(tables=[]))

    assert snapshot.relations == ()


# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------


def test_foreign_keys_are_carried_into_the_snapshot():
    datasource = FakeDatasource(
        tables=["orders", "customers"],
        columns=COLUMNS,
        foreign_keys={
            "orders": [
                {
                    "columns": ["customer_id"],
                    "referred_table": "customers",
                    "referred_columns": ["id"],
                }
            ]
        },
    )

    snapshot = loader.load_schema_snapshot(datasource)

    orders = next(r for r in snapshot.relations if r.name == "orders")
    assert orders.foreign_keys[0].referred_table == "customers"
    assert orders.foreign_keys[0].columns == ("customer_id",)


def test_foreign_key_to_a_hidden_relation_is_dropped():
    """
    Rendering it would name a table the allow/deny rules exist to conceal.
    """
    datasource = FakeDatasource(
        tables=["orders", "secrets"],
        columns=COLUMNS,
        denied_tables=frozenset({"secrets"}),
        foreign_keys={
            "orders": [
                {
                    "columns": ["secret_id"],
                    "referred_table": "secrets",
                    "referred_columns": ["id"],
                }
            ]
        },
    )

    snapshot = loader.load_schema_snapshot(datasource)
    rendered = loader.render_schema(snapshot, datasource.schema_max_chars)

    orders = next(r for r in snapshot.relations if r.name == "orders")
    assert orders.foreign_keys == ()
    assert "secrets" not in rendered


# ---------------------------------------------------------------------------
# Caching — the reason this module exists
# ---------------------------------------------------------------------------


def test_snapshot_is_reflected_once_and_then_cached():
    datasource = FakeDatasource(tables=["orders"], columns=COLUMNS)

    first = loader.get_schema_snapshot(datasource)
    second = loader.get_schema_snapshot(datasource)
    third = loader.get_schema_snapshot(datasource)

    assert datasource.reflect_calls == 1
    assert first is second is third


def test_cache_is_shared_across_datasource_instances():
    """
    get_datasource() returns a fresh SqlDatasource per call, and the one-shot
    chat routes build an agent per request. The cache must survive that.
    """
    first_ds = FakeDatasource(tables=["orders"], columns=COLUMNS)
    second_ds = FakeDatasource(tables=["orders"], columns=COLUMNS)

    loader.get_schema_snapshot(first_ds)
    loader.get_schema_snapshot(second_ds)

    assert first_ds.reflect_calls == 1
    assert second_ds.reflect_calls == 0


def test_invalidate_snapshot_forces_a_rebuild():
    datasource = FakeDatasource(tables=["orders"], columns=COLUMNS)

    loader.get_schema_snapshot(datasource)
    loader.invalidate_snapshot()
    loader.get_schema_snapshot(datasource)

    assert datasource.reflect_calls == 2


def test_expired_ttl_rebuilds():
    datasource = FakeDatasource(tables=["orders"], columns=COLUMNS, schema_ttl_s=60)

    loader.get_schema_snapshot(datasource)
    loader._snapshot_at = time.time() - 61
    loader.get_schema_snapshot(datasource)

    assert datasource.reflect_calls == 2


def test_zero_ttl_rebuilds_every_time():
    datasource = FakeDatasource(tables=["orders"], columns=COLUMNS, schema_ttl_s=0)

    loader.get_schema_snapshot(datasource)
    loader.get_schema_snapshot(datasource)

    assert datasource.reflect_calls == 2


def test_concurrent_callers_reflect_only_once():
    """
    Agents are created per request, so several can be built at once. The
    double-checked lock must collapse that into a single reflection.
    """

    class SlowDatasource(FakeDatasource):
        def reflect_schema(self):
            time.sleep(0.05)  # widen the race window
            return super().reflect_schema()

    datasource = SlowDatasource(tables=["orders"], columns=COLUMNS)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(loader.get_schema_snapshot(datasource))
        )
        for _ in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert datasource.reflect_calls == 1
    assert all(r is results[0] for r in results)


def test_reflection_failure_returns_none_instead_of_raising():
    """
    This runs inside the engine's __post_init__, so an unreachable database
    must not raise out of agent construction.
    """
    datasource = FakeDatasource(raises=operational_error())

    assert loader.get_schema_snapshot(datasource) is None


def test_reflection_failure_is_not_cached():
    datasource = FakeDatasource(raises=operational_error())

    loader.get_schema_snapshot(datasource)

    assert loader._snapshot is None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_lists_every_visible_relation_under_its_heading():
    datasource = FakeDatasource(
        tables=["orders"],
        views=["v_sales"],
        materialized_views=["mv_ltv"],
        columns=COLUMNS,
    )

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "TABLES" in rendered
    assert "VIEWS" in rendered
    assert "MATERIALIZED VIEWS" in rendered
    assert '"orders"("id":INTEGER PK, "country":VARCHAR(2))' in rendered
    assert '"v_sales"(' in rendered
    assert '"mv_ltv"(' in rendered


def test_render_omits_hidden_relations():
    datasource = FakeDatasource(
        tables=["orders", "secrets"],
        columns=COLUMNS,
        denied_tables=frozenset({"secrets"}),
    )

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "orders" in rendered
    assert "secrets" not in rendered


def test_render_includes_the_schema_name():
    datasource = FakeDatasource(tables=["orders"], columns=COLUMNS)

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "schema: public" in rendered


def test_render_degrades_to_names_only_over_budget():
    datasource = FakeDatasource(tables=["orders", "customers"], columns=COLUMNS)
    snapshot = loader.load_schema_snapshot(datasource)

    rendered = loader.render_schema(snapshot, max_chars=80)

    # Names survive, column detail does not.
    assert "orders" in rendered
    assert "customers" in rendered
    assert "INTEGER" not in rendered


def test_render_of_an_empty_schema_says_so_plainly():
    snapshot = loader.load_schema_snapshot(FakeDatasource(tables=[]))

    rendered = loader.render_schema(snapshot, 20000)

    assert "no relations" in rendered
    # No dangling heading the model might try to fill in.
    assert "TABLES" not in rendered


# ---------------------------------------------------------------------------
# Value profiling
# ---------------------------------------------------------------------------

# The shape that produced the CLO/Luglio-2026 miss: a code column, a label
# column and a period column, none of which can be filtered correctly from the
# type alone.
PROFILE_COLUMNS = [
    {"column": "uuid", "type": "UUID", "nullable": False, "primary_key": True},
    {"column": "id_cliente", "type": "TEXT", "nullable": True, "primary_key": False},
    {
        "column": "descrizione_cliente",
        "type": "TEXT",
        "nullable": True,
        "primary_key": False,
    },
    {"column": "mese", "type": "TEXT", "nullable": True, "primary_key": False},
    {"column": "primo_margine", "type": "NUMERIC", "nullable": True, "primary_key": False},
]

PROFILE_SAMPLES = {
    "id_cliente": ["1053_3674", "1053_3019"],
    "descrizione_cliente": ["CLO", "ACME"],
    "mese": ["202604", "202607"],
}


def make_profiled(**overrides):
    params = dict(
        tables=["report_mensile"],
        columns=PROFILE_COLUMNS,
        samples=PROFILE_SAMPLES,
    )
    params.update(overrides)
    return FakeDatasource(**params)


def columns_by_name(relation):
    return {col.name: col for col in relation.columns}


def test_sampled_values_land_on_their_columns():
    snapshot = loader.load_schema_snapshot(make_profiled())

    columns = columns_by_name(snapshot.relations[0])
    assert columns["mese"].examples == ("202604", "202607")
    assert columns["descrizione_cliente"].examples == ("CLO", "ACME")


def test_only_profilable_types_are_sampled():
    """Measures and opaque ids are not worth prompt budget."""
    datasource = make_profiled()

    loader.load_schema_snapshot(datasource)

    relation, requested, sample_rows = datasource.sample_calls[0]
    assert relation == "report_mensile"
    assert requested == ["id_cliente", "descrizione_cliente", "mese"]
    assert sample_rows == datasource.schema_profile_sample_rows


def test_unsampled_columns_simply_have_no_examples():
    snapshot = loader.load_schema_snapshot(make_profiled())

    columns = columns_by_name(snapshot.relations[0])
    assert columns["primo_margine"].examples == ()
    assert columns["uuid"].examples == ()


def test_profiling_disabled_asks_the_database_for_nothing():
    datasource = make_profiled(schema_profile_values=False)

    snapshot = loader.load_schema_snapshot(datasource)

    assert datasource.sample_calls == []
    assert all(not c.examples for c in snapshot.relations[0].columns)


def test_a_relation_with_no_profilable_column_is_not_queried():
    datasource = FakeDatasource(
        tables=["metrics"],
        columns=[
            {"column": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
            {"column": "total", "type": "NUMERIC", "nullable": True, "primary_key": False},
        ],
        samples=PROFILE_SAMPLES,
    )

    loader.load_schema_snapshot(datasource)

    assert datasource.sample_calls == []


def test_hidden_relations_are_never_sampled():
    """A denied relation must not be queried even to describe it."""
    datasource = make_profiled(
        tables=["report_mensile", "secrets"],
        denied_tables=frozenset({"secrets"}),
    )

    loader.load_schema_snapshot(datasource)

    assert [call[0] for call in datasource.sample_calls] == ["report_mensile"]


def test_sampling_failure_still_produces_a_usable_snapshot():
    """Losing the hints must never cost the agent the schema itself."""
    datasource = make_profiled(samples_raise=operational_error())

    snapshot = loader.load_schema_snapshot(datasource)

    assert snapshot.relation_names() == ("report_mensile",)
    assert all(not c.examples for c in snapshot.relations[0].columns)


def test_examples_are_capped_in_count():
    datasource = make_profiled(
        samples={"mese": ["202601", "202602", "202603", "202604"]},
        schema_profile_max_values=2,
    )

    snapshot = loader.load_schema_snapshot(datasource)

    assert columns_by_name(snapshot.relations[0])["mese"].examples == ("202601", "202602")


def test_long_examples_are_truncated():
    datasource = make_profiled(
        samples={"descrizione_cliente": ["x" * 40]},
        schema_profile_max_value_chars=8,
    )

    snapshot = loader.load_schema_snapshot(datasource)

    value = columns_by_name(snapshot.relations[0])["descrizione_cliente"].examples[0]
    assert value == "x" * 8 + "…"


def test_render_shows_examples_as_sql_literals():
    datasource = make_profiled()

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "e.g." in rendered
    assert "\"mese\"='202604', '202607'" in rendered
    assert "\"descrizione_cliente\"='CLO', 'ACME'" in rendered
    # Columns with no samples do not appear on the examples line.
    assert '"primo_margine"=' not in rendered


def test_render_warns_that_examples_are_not_the_full_domain():
    """
    Without this the agent reads a sample as the column's domain and starts
    reporting data as missing because it was not in the handful of rows shown.
    """
    datasource = make_profiled()

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "NOT the complete set of values" in rendered


def test_render_escapes_a_quote_inside_an_example():
    datasource = make_profiled(samples={"descrizione_cliente": ["D'Angelo"]})

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "\"descrizione_cliente\"='D''Angelo'" in rendered


def test_render_sheds_examples_before_column_names():
    """Examples are the first tier to go: a column list is worth more."""
    datasource = make_profiled()
    snapshot = loader.load_schema_snapshot(datasource)

    full = loader.render_schema(snapshot, datasource.schema_max_chars)
    degraded = loader.render_schema(snapshot, max_chars=len(full) - 1)

    assert "e.g." not in degraded
    assert "NOT the complete set of values" not in degraded
    assert '"report_mensile"("uuid":UUID PK' in degraded


def test_render_falls_all_the_way_to_names_when_columns_do_not_fit():
    datasource = make_profiled()
    snapshot = loader.load_schema_snapshot(datasource)

    rendered = loader.render_schema(snapshot, max_chars=80)

    assert "report_mensile" in rendered
    assert "e.g." not in rendered
    assert "TEXT" not in rendered


def test_no_examples_means_no_caveat():
    """The sampling note only earns its tokens when something was sampled."""
    datasource = FakeDatasource(tables=["orders"], columns=COLUMNS, samples={})

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "e.g." not in rendered


# ---------------------------------------------------------------------------
# Identifier index
# ---------------------------------------------------------------------------


def test_identifier_index_maps_relations_to_their_columns():
    snapshot = loader.load_schema_snapshot(
        FakeDatasource(tables=["Clienti"], columns=COLUMNS)
    )

    assert loader.build_identifier_index(snapshot) == {"Clienti": ("id", "country")}


def test_identifier_index_excludes_hidden_relations():
    """
    The rewriter may only substitute names the agent was shown. Quoting a denied
    relation's name would disclose that it exists.
    """
    snapshot = loader.load_schema_snapshot(
        FakeDatasource(
            tables=["Clienti", "Segreti"],
            columns=COLUMNS,
            denied_tables=frozenset({"segreti"}),
        )
    )

    assert "Segreti" not in loader.build_identifier_index(snapshot)


def test_identifier_index_of_nothing_is_empty():
    assert loader.build_identifier_index(None) == {}


def test_render_quotes_names_in_the_names_only_tier():
    """The relation name is exactly what fails unquoted, so every tier quotes it."""
    datasource = FakeDatasource(tables=["Trasporti", "Clienti"], columns=COLUMNS)
    snapshot = loader.load_schema_snapshot(datasource)

    rendered = loader.render_schema(snapshot, max_chars=90)

    assert '"Trasporti"' in rendered
    assert "INTEGER" not in rendered


def test_render_explains_why_the_quotes_are_there():
    datasource = FakeDatasource(tables=["Trasporti"], columns=COLUMNS)

    rendered = loader.render_schema(
        loader.load_schema_snapshot(datasource), datasource.schema_max_chars
    )

    assert "folds an unquoted name to lowercase" in rendered
