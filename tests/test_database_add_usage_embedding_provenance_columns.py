import pytest

from infrastructure import database_pg
from infrastructure.database_pg import SchemaChangeResult

def test_add_usage_embedding_operation_kind_column_uses_fixed_maui_owned_intent(monkeypatch):
    calls = []

    def _fake_add_column_if_missing(table_schema, table_name, column_name, column_type):
        calls.append((table_schema, table_name, column_name, column_type))
        return SchemaChangeResult.CHANGED

    monkeypatch.setattr(database_pg, "schema", "maui_schema")
    monkeypatch.setattr(database_pg, "add_column_if_missing", _fake_add_column_if_missing)

    database_pg.add_usage_embedding_operation_kind_column()

    assert calls == [("maui_schema", "logs", "embedding_operation_kind", "TEXT")]


def test_add_usage_embedding_operation_kind_column_changed_is_success(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.CHANGED
    )

    database_pg.add_usage_embedding_operation_kind_column()


def test_add_usage_embedding_operation_kind_column_unchanged_is_success(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.UNCHANGED
    )

    database_pg.add_usage_embedding_operation_kind_column()


def test_add_usage_embedding_operation_kind_column_failed_raises(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.FAILED
    )

    with pytest.raises(RuntimeError):
        database_pg.add_usage_embedding_operation_kind_column()


def test_add_usage_embedding_operation_kind_column_takes_no_arguments():
    import inspect

    signature = inspect.signature(database_pg.add_usage_embedding_operation_kind_column)
    assert len(signature.parameters) == 0


def test_add_usage_quantity_origin_column_uses_fixed_maui_owned_intent(monkeypatch):
    calls = []

    def _fake_add_column_if_missing(table_schema, table_name, column_name, column_type):
        calls.append((table_schema, table_name, column_name, column_type))
        return SchemaChangeResult.CHANGED

    monkeypatch.setattr(database_pg, "schema", "maui_schema")
    monkeypatch.setattr(database_pg, "add_column_if_missing", _fake_add_column_if_missing)

    database_pg.add_usage_quantity_origin_column()

    assert calls == [("maui_schema", "logs", "quantity_origin", "TEXT")]


def test_add_usage_quantity_origin_column_changed_is_success(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.CHANGED
    )

    database_pg.add_usage_quantity_origin_column()


def test_add_usage_quantity_origin_column_unchanged_is_success(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.UNCHANGED
    )

    database_pg.add_usage_quantity_origin_column()


def test_add_usage_quantity_origin_column_failed_raises(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.FAILED
    )

    with pytest.raises(RuntimeError):
        database_pg.add_usage_quantity_origin_column()


def test_add_usage_quantity_origin_column_takes_no_arguments():
    import inspect

    signature = inspect.signature(database_pg.add_usage_quantity_origin_column)
    assert len(signature.parameters) == 0


def test_add_usage_cost_origin_column_uses_fixed_maui_owned_intent(monkeypatch):
    calls = []

    def _fake_add_column_if_missing(table_schema, table_name, column_name, column_type):
        calls.append((table_schema, table_name, column_name, column_type))
        return SchemaChangeResult.CHANGED

    monkeypatch.setattr(database_pg, "schema", "maui_schema")
    monkeypatch.setattr(database_pg, "add_column_if_missing", _fake_add_column_if_missing)

    database_pg.add_usage_cost_origin_column()

    assert calls == [("maui_schema", "logs", "cost_origin", "TEXT")]


def test_add_usage_cost_origin_column_changed_is_success(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.CHANGED
    )

    database_pg.add_usage_cost_origin_column()


def test_add_usage_cost_origin_column_unchanged_is_success(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.UNCHANGED
    )

    database_pg.add_usage_cost_origin_column()


def test_add_usage_cost_origin_column_failed_raises(monkeypatch):
    monkeypatch.setattr(
        database_pg, "add_column_if_missing", lambda *a, **k: SchemaChangeResult.FAILED
    )

    with pytest.raises(RuntimeError):
        database_pg.add_usage_cost_origin_column()


def test_add_usage_cost_origin_column_takes_no_arguments():
    import inspect

    signature = inspect.signature(database_pg.add_usage_cost_origin_column)
    assert len(signature.parameters) == 0


def test_embedding_usage_provenance_migrations_are_three_independent_commands():
    """
    The accepted Technical Design requires three independently runnable
    migrations, not one combined helper.
    """
    for name in (
        "add_usage_embedding_operation_kind_column",
        "add_usage_quantity_origin_column",
        "add_usage_cost_origin_column",
    ):
        assert callable(getattr(database_pg, name))
