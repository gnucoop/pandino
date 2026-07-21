import pytest

from infrastructure import database_methods, database_pg


class FakeCursor:
    def __init__(
        self,
        *,
        validate_row=None,
        table_exists_row=(1,),
        chunks_rowcount=0,
        row_rowcount=1,
        raise_on=None,
    ):
        self.calls = []
        self._next_fetchone = None
        self.validate_row = validate_row
        self.table_exists_row = table_exists_row
        self.chunks_rowcount = chunks_rowcount
        self.row_rowcount = row_rowcount
        self.raise_on = raise_on
        self.rowcount = -1

    def execute(self, query, params=()):
        self.calls.append((query, params))

        if query == "validate":
            self._next_fetchone = self.validate_row
            self.rowcount = -1
        elif query == "table_exists":
            self._next_fetchone = self.table_exists_row
            self.rowcount = -1
        elif query == "delete_chunks":
            if self.raise_on == "delete_chunks":
                raise RuntimeError("chunk delete failed")
            self.rowcount = self.chunks_rowcount
        elif query == "delete_row":
            if self.raise_on == "delete_row":
                raise RuntimeError("row delete failed")
            self.rowcount = self.row_rowcount
        else:
            raise AssertionError(f"Unexpected query: {query}")

    def fetchone(self):
        row = self._next_fetchone
        self._next_fetchone = None
        return row


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


@pytest.fixture
def patched_delete_builders(monkeypatch):
    monkeypatch.setattr(database_pg, "schema", "public")
    monkeypatch.setattr(
        database_pg,
        "build_get_rag_file_for_delete_query",
        lambda file_id, namespace: ("validate", (file_id, namespace)),
    )
    monkeypatch.setattr(
        database_pg,
        "build_check_table_exists_query",
        lambda table_schema, table_name: (
            "table_exists",
            (table_schema, table_name),
        ),
    )
    monkeypatch.setattr(
        database_pg,
        "build_delete_pgvector_by_file_id_query",
        lambda table_name, file_id: ("delete_chunks", (table_name, file_id)),
    )
    monkeypatch.setattr(
        database_pg,
        "build_delete_rag_file_query",
        lambda file_id, namespace: ("delete_row", (file_id, namespace)),
    )


def _patch_connection(monkeypatch, cursor):
    conn = FakeConnection(cursor)
    monkeypatch.setattr(database_pg, "connect", lambda: conn)
    return conn


def test_mismatched_namespace_does_not_delete_vector_chunks_or_rag_file_row(
    monkeypatch, patched_delete_builders
):
    cursor = FakeCursor(validate_row=None)
    conn = _patch_connection(monkeypatch, cursor)

    result = database_pg.delete_rag_file("file-1", " Customer-Docs ")

    assert result == {"row_deleted": False, "chunks_deleted": 0}
    assert cursor.calls == [("validate", ("file-1", "customer_docs"))]
    assert conn.rollbacks == 1
    assert conn.commits == 0
    assert conn.closed is True


def test_valid_namespace_validates_rag_file_before_deleting_chunks(
    monkeypatch, patched_delete_builders
):
    cursor = FakeCursor(
        validate_row=("file-1",),
        table_exists_row=(1,),
        chunks_rowcount=3,
        row_rowcount=1,
    )
    conn = _patch_connection(monkeypatch, cursor)

    result = database_pg.delete_rag_file("file-1", "Customer-Docs")

    assert result == {"row_deleted": True, "chunks_deleted": 3}
    assert [query for query, _params in cursor.calls] == [
        "validate",
        "table_exists",
        "delete_chunks",
        "delete_row",
    ]
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_rag_files_delete_uses_file_id_and_normalized_namespace(
    monkeypatch, patched_delete_builders
):
    cursor = FakeCursor(validate_row=("file-1",), table_exists_row=None)
    _patch_connection(monkeypatch, cursor)

    database_pg.delete_rag_file("file-1", "Customer-Docs")

    assert cursor.calls[-1] == ("delete_row", ("file-1", "customer_docs"))


def test_delete_rag_file_rolls_back_when_chunk_delete_raises(
    monkeypatch, patched_delete_builders
):
    cursor = FakeCursor(
        validate_row=("file-1",),
        table_exists_row=(1,),
        raise_on="delete_chunks",
    )
    conn = _patch_connection(monkeypatch, cursor)

    with pytest.raises(RuntimeError, match="chunk delete failed"):
        database_pg.delete_rag_file("file-1", "customer_docs")

    assert [query for query, _params in cursor.calls] == [
        "validate",
        "table_exists",
        "delete_chunks",
    ]
    assert conn.rollbacks == 1
    assert conn.commits == 0
    assert conn.closed is True


def test_delete_rag_file_rolls_back_if_locked_row_is_not_deleted(
    monkeypatch, patched_delete_builders
):
    cursor = FakeCursor(
        validate_row=("file-1",),
        table_exists_row=(1,),
        chunks_rowcount=3,
        row_rowcount=0,
    )
    conn = _patch_connection(monkeypatch, cursor)

    result = database_pg.delete_rag_file("file-1", "customer_docs")

    assert result == {"row_deleted": False, "chunks_deleted": 0}
    assert [query for query, _params in cursor.calls] == [
        "validate",
        "table_exists",
        "delete_chunks",
        "delete_row",
    ]
    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_rag_delete_sql_builders_validate_and_delete_with_namespace():
    validate_query, validate_params = (
        database_methods.build_get_rag_file_for_delete_query(
            "file-1", "customer_docs"
        )
    )
    delete_query, delete_params = database_methods.build_delete_rag_file_query(
        "file-1", "customer_docs"
    )

    assert validate_params == ("file-1", "customer_docs")
    assert "WHERE id = %s AND namespace = %s FOR UPDATE" in str(validate_query)
    assert delete_params == ("file-1", "customer_docs")
    assert "WHERE id = %s AND namespace = %s" in str(delete_query)
