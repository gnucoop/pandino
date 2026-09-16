"""
Tests for datachat.sql_datasource. No database is contacted: create_engine is
patched so the engine kwargs can be asserted directly.
"""

from unittest.mock import MagicMock, patch

import pytest

from sqlalchemy.engine.reflection import ObjectKind

import datachat.sql_datasource as sql_datasource
from config import DatachatSqlConfig


def make_config(**overrides) -> DatachatSqlConfig:
    base = dict(
        enabled=True,
        user="ro_user",
        password="p@ss word/1",
        host="sqlhost",
        db="analytics",
        port="5433",
        schema="reporting",
        max_rows=200,
        max_columns=25,
        max_cell_chars=300,
        statement_timeout_ms=10000,
        allowed_tables=(),
        denied_tables=(),
        include_views=True,
        schema_ttl_s=3600,
        schema_max_chars=20000,
        schema_include_fks=True,
        quote_identifiers=True,
        schema_profile_values=True,
        schema_profile_sample_rows=50,
        schema_profile_max_values=5,
        schema_profile_max_value_chars=32,
        pool_size=2,
        pool_max_overflow=3,
        pool_timeout_s=5,
        pool_recycle_s=1800,
    )
    base.update(overrides)
    return DatachatSqlConfig(**base)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Clear the module globals around every test."""
    sql_datasource._config = None
    sql_datasource._engine = None
    yield
    sql_datasource._config = None
    sql_datasource._engine = None


class FakeAppConfig:
    def __init__(self, datachat_sql):
        self.datachat_sql = datachat_sql


def test_init_performs_no_database_io():
    with patch("datachat.sql_datasource.create_engine") as create:
        sql_datasource.init(FakeAppConfig(make_config()))
        create.assert_not_called()


def test_get_datasource_returns_none_when_uninitialised():
    assert sql_datasource.get_datasource() is None


def test_get_datasource_returns_none_when_disabled():
    sql_datasource.init(FakeAppConfig(make_config(enabled=False)))
    assert sql_datasource.get_datasource() is None


def test_get_datasource_builds_no_engine():
    """Constructing the facade must stay free of database I/O."""
    with patch("datachat.sql_datasource.create_engine") as create:
        sql_datasource.init(FakeAppConfig(make_config()))
        datasource = sql_datasource.get_datasource()
        assert datasource is not None
        create.assert_not_called()


def _engine_kwargs(cfg: DatachatSqlConfig):
    with patch("datachat.sql_datasource.create_engine") as create:
        create.return_value = MagicMock()
        sql_datasource._get_engine(cfg)
        return create.call_args


def test_engine_is_read_only():
    args = _engine_kwargs(make_config())
    assert args.kwargs["execution_options"] == {"postgresql_readonly": True}


def test_engine_session_options():
    args = _engine_kwargs(make_config())
    options = args.kwargs["connect_args"]["options"]
    assert "default_transaction_read_only=on" in options
    assert "statement_timeout=10000" in options
    assert "idle_in_transaction_session_timeout=20000" in options
    assert "search_path=reporting" in options


def test_engine_pool_settings():
    args = _engine_kwargs(make_config())
    assert args.kwargs["pool_pre_ping"] is True
    assert args.kwargs["pool_size"] == 2
    assert args.kwargs["max_overflow"] == 3
    assert args.kwargs["pool_timeout"] == 5
    assert args.kwargs["pool_recycle"] == 1800


def test_engine_url_quotes_credentials():
    args = _engine_kwargs(make_config())
    url = args.args[0]
    assert url.startswith("postgresql+psycopg://")
    assert "p%40ss+word%2F1" in url
    assert "p@ss word/1" not in url
    assert "@sqlhost:5433/analytics" in url


def test_engine_is_built_once():
    cfg = make_config()
    with patch("datachat.sql_datasource.create_engine") as create:
        create.return_value = MagicMock()
        sql_datasource._get_engine(cfg)
        sql_datasource._get_engine(cfg)
        assert create.call_count == 1


def test_dispose_clears_the_engine():
    cfg = make_config()
    engine = MagicMock()
    with patch("datachat.sql_datasource.create_engine", return_value=engine):
        sql_datasource._get_engine(cfg)
    sql_datasource.dispose()
    engine.dispose.assert_called_once()
    assert sql_datasource._engine is None


def _listing_inspector(names) -> MagicMock:
    """Patch create_engine + inspect, and hand back the fake Inspector."""
    inspector = MagicMock()
    inspector.get_table_names.return_value = list(names)
    inspector.get_view_names.return_value = list(names)
    inspector.get_materialized_view_names.return_value = list(names)
    return inspector


@pytest.mark.parametrize(
    "method,inspector_method",
    [
        ("list_tables", "get_table_names"),
        ("list_views", "get_view_names"),
        ("list_materialized_views", "get_materialized_view_names"),
    ],
)
def test_listing_reflects_only_its_own_relation_kind(method, inspector_method):
    inspector = _listing_inspector(["b", "a", "a"])
    datasource = sql_datasource.SqlDatasource(make_config())

    with patch("datachat.sql_datasource.create_engine", return_value=MagicMock()):
        with patch("datachat.sql_datasource.inspect", return_value=inspector):
            names = getattr(datasource, method)()

    # Sorted and de-duplicated.
    assert names == ["a", "b"]

    # Exactly one reflection call, on the matching inspector method, scoped to
    # the configured schema.
    for candidate in (
        "get_table_names",
        "get_view_names",
        "get_materialized_view_names",
    ):
        called = getattr(inspector, candidate)
        if candidate == inspector_method:
            called.assert_called_once_with(schema="reporting")
        else:
            called.assert_not_called()


def test_list_tables_no_longer_folds_in_views():
    """list_tables reports base tables only; include_views must not affect it."""
    inspector = _listing_inspector([])
    inspector.get_table_names.return_value = ["orders"]
    inspector.get_view_names.return_value = ["v_sales"]
    datasource = sql_datasource.SqlDatasource(make_config(include_views=True))

    with patch("datachat.sql_datasource.create_engine", return_value=MagicMock()):
        with patch("datachat.sql_datasource.inspect", return_value=inspector):
            assert datasource.list_tables() == ["orders"]


def test_include_views_is_exposed():
    assert sql_datasource.SqlDatasource(make_config()).include_views is True
    assert (
        sql_datasource.SqlDatasource(make_config(include_views=False)).include_views
        is False
    )


def test_is_table_visible_with_denylist():
    datasource = sql_datasource.SqlDatasource(make_config(denied_tables=("users",)))
    assert datasource.is_table_visible("orders")
    assert not datasource.is_table_visible("users")
    assert not datasource.is_table_visible("USERS")
    assert not datasource.is_table_visible("")


def test_is_table_visible_with_allowlist_wins():
    datasource = sql_datasource.SqlDatasource(
        make_config(allowed_tables=("orders",), denied_tables=("orders",))
    )
    assert datasource.is_table_visible("orders")
    assert not datasource.is_table_visible("customers")


# ---------------------------------------------------------------------------
# reflect_schema
# ---------------------------------------------------------------------------


def _reflection_inspector():
    """
    A fake Inspector covering the get_multi_* API that reflect_schema uses.
    Keys are (schema, name) tuples, as SQLAlchemy returns them.
    """
    inspector = MagicMock()
    inspector.get_table_names.return_value = ["orders", "customers"]
    inspector.get_view_names.return_value = ["v_sales"]
    inspector.get_materialized_view_names.return_value = ["mv_ltv"]

    columns = [
        {"name": "id", "type": "INTEGER", "nullable": False},
        {"name": "customer_id", "type": "INTEGER", "nullable": True},
    ]
    inspector.get_multi_columns.return_value = {
        ("reporting", "orders"): columns,
        ("reporting", "customers"): columns,
        ("reporting", "v_sales"): columns,
        ("reporting", "mv_ltv"): columns,
    }
    inspector.get_multi_pk_constraint.return_value = {
        ("reporting", "orders"): {"constrained_columns": ["id"]},
    }
    inspector.get_multi_foreign_keys.return_value = {
        ("reporting", "orders"): [
            {
                "constrained_columns": ["customer_id"],
                "referred_table": "customers",
                "referred_columns": ["id"],
            }
        ],
    }
    return inspector


def _reflect(config, inspector):
    datasource = sql_datasource.SqlDatasource(config)
    with patch("datachat.sql_datasource.create_engine", return_value=MagicMock()):
        with patch("datachat.sql_datasource.inspect", return_value=inspector):
            return datasource.reflect_schema()


def test_reflect_schema_classifies_each_relation_kind():
    records = _reflect(make_config(), _reflection_inspector())

    assert {r["name"]: r["kind"] for r in records} == {
        "customers": "table",
        "orders": "table",
        "v_sales": "view",
        "mv_ltv": "materialized_view",
    }


def test_reflect_schema_marks_primary_keys():
    records = _reflect(make_config(), _reflection_inspector())

    orders = next(r for r in records if r["name"] == "orders")
    assert [(c["column"], c["primary_key"]) for c in orders["columns"]] == [
        ("id", True),
        ("customer_id", False),
    ]
    # A relation absent from the pk map simply has none.
    customers = next(r for r in records if r["name"] == "customers")
    assert all(not c["primary_key"] for c in customers["columns"])


def test_reflect_schema_collects_foreign_keys():
    records = _reflect(make_config(), _reflection_inspector())

    orders = next(r for r in records if r["name"] == "orders")
    assert orders["foreign_keys"] == [
        {
            "columns": ["customer_id"],
            "referred_table": "customers",
            "referred_columns": ["id"],
        }
    ]


def test_reflect_schema_skips_foreign_keys_when_disabled():
    inspector = _reflection_inspector()
    records = _reflect(make_config(schema_include_fks=False), inspector)

    orders = next(r for r in records if r["name"] == "orders")
    assert orders["foreign_keys"] == []
    inspector.get_multi_foreign_keys.assert_not_called()


def test_reflect_schema_omits_views_when_include_views_is_off():
    inspector = _reflection_inspector()
    records = _reflect(make_config(include_views=False), inspector)

    assert [r["name"] for r in records] == ["customers", "orders"]
    inspector.get_view_names.assert_not_called()
    inspector.get_materialized_view_names.assert_not_called()


def test_reflect_schema_asks_for_views_only_when_included():
    """
    get_multi_* defaults to ObjectKind.TABLE, so views and materialized views
    are reflected only when the kind is widened.
    """
    inspector = _reflection_inspector()
    _reflect(make_config(include_views=True), inspector)
    with_views = inspector.get_multi_columns.call_args.kwargs["kind"]

    inspector = _reflection_inspector()
    _reflect(make_config(include_views=False), inspector)
    without_views = inspector.get_multi_columns.call_args.kwargs["kind"]

    assert with_views == ObjectKind.ANY
    assert without_views == ObjectKind.TABLE


def test_reflect_schema_uses_one_connection_for_the_whole_schema():
    """
    The point of reflect_schema: a schema of any size costs a fixed number of
    round trips, unlike describe_table which opens a connection per relation.
    """
    engine = MagicMock()
    datasource = sql_datasource.SqlDatasource(make_config())

    with patch("datachat.sql_datasource.create_engine", return_value=engine):
        with patch(
            "datachat.sql_datasource.inspect", return_value=_reflection_inspector()
        ):
            datasource.reflect_schema()

    assert engine.connect.call_count == 1


def test_reflect_schema_skips_a_relation_with_no_reflected_columns():
    """A relation dropped mid-reflection must not produce a column-less entry."""
    inspector = _reflection_inspector()
    del inspector.get_multi_columns.return_value[("reporting", "v_sales")]

    records = _reflect(make_config(), inspector)

    assert "v_sales" not in [r["name"] for r in records]
    assert "orders" in [r["name"] for r in records]


def test_reflect_schema_applies_no_allow_deny_filtering():
    """Filtering is the snapshot loader's job; this reports what exists."""
    records = _reflect(make_config(denied_tables=("orders",)), _reflection_inspector())

    assert "orders" in [r["name"] for r in records]


# ---------------------------------------------------------------------------
# Value sampling
# ---------------------------------------------------------------------------


def _sample(config, rows, relation="report_mensile", columns=("mese",), sample_rows=50):
    """Run sample_column_values against a stubbed connection, return (sql, result)."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.fetchall.return_value = rows
    datasource = sql_datasource.SqlDatasource(config)

    with patch("datachat.sql_datasource.create_engine", return_value=engine):
        result = datasource.sample_column_values(relation, list(columns), sample_rows)

    sql = str(connection.execute.call_args[0][0])
    return sql, result


def test_sample_quotes_the_relation_and_schema():
    """Mixed-case names fold to lowercase and stop resolving if left bare."""
    sql, _ = _sample(make_config(), rows=[], relation="AnalisiAutomezzi")

    assert 'FROM "reporting"."AnalisiAutomezzi"' in sql


def test_sample_casts_every_column_to_text():
    """The caller must not have to know the column's type."""
    sql, _ = _sample(make_config(), rows=[], columns=("mese", "id_cliente"))

    assert 'CAST("mese" AS text), CAST("id_cliente" AS text)' in sql


def test_sample_bounds_the_scan_with_a_limit():
    sql, _ = _sample(make_config(), rows=[], sample_rows=25)

    assert sql.rstrip().endswith("LIMIT 25")


def test_sample_escapes_a_quote_in_an_identifier():
    sql, _ = _sample(make_config(), rows=[], columns=('we"ird',))

    assert 'CAST("we""ird" AS text)' in sql


def test_sample_deduplicates_in_first_seen_order():
    _, result = _sample(
        make_config(),
        rows=[("202607",), ("202604",), ("202607",)],
        columns=("mese",),
    )

    assert result == {"mese": ["202607", "202604"]}


def test_sample_skips_nulls():
    _, result = _sample(
        make_config(), rows=[(None,), ("202607",), (None,)], columns=("mese",)
    )

    assert result == {"mese": ["202607"]}


def test_sample_returns_an_entry_per_requested_column():
    _, result = _sample(
        make_config(),
        rows=[("202607", "CLO")],
        columns=("mese", "descrizione_cliente"),
    )

    assert result == {"mese": ["202607"], "descrizione_cliente": ["CLO"]}


def test_sample_without_columns_contacts_no_database():
    engine = MagicMock()
    datasource = sql_datasource.SqlDatasource(make_config())

    with patch("datachat.sql_datasource.create_engine", return_value=engine):
        assert datasource.sample_column_values("orders", [], 50) == {}

    assert engine.connect.call_count == 0
