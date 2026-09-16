"""
Security regression suite for datachat.sql_guard.

Runs without a database and without mocks. The three CTE/INTO/FOR-UPDATE cases
are the ones a keyword or regex check cannot catch, and are the reason the
guard parses with sqlglot; they must keep passing across sqlglot upgrades.
"""

import pytest

from datachat.sql_guard import validate_select

ACCEPTED = [
    "SELECT 1",
    "select * from t",
    "WITH t AS (SELECT 1) SELECT * FROM t",
    "SELECT 1 UNION ALL SELECT 2",
    "SELECT 1 EXCEPT SELECT 2",
    "SELECT 1 INTERSECT SELECT 2",
    "(SELECT 1)",
    "SELECT ';' AS a",
    "SELECT 1 -- ; DROP TABLE users",
    "SELECT * FROM t /* ; DROP TABLE users */",
    "SELECT * FROM t LIMIT 5",
    "SELECT 1;",
    "SELECT 1;;",
    "SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 1 ORDER BY a",
]

REJECTED = [
    ("", "MISSING_QUERY"),
    ("   ", "MISSING_QUERY"),
    ("SELECT 1; DROP TABLE users", "MULTIPLE_STATEMENTS"),
    ("SELECT 1; SELECT 2", "MULTIPLE_STATEMENTS"),
    ("INSERT INTO t VALUES (1)", "NON_SELECT_STATEMENT"),
    ("UPDATE t SET a = 1", "NON_SELECT_STATEMENT"),
    ("DELETE FROM t", "NON_SELECT_STATEMENT"),
    ("CREATE TABLE x (a int)", "NON_SELECT_STATEMENT"),
    ("DROP TABLE t", "NON_SELECT_STATEMENT"),
    ("ALTER TABLE t ADD COLUMN b int", "NON_SELECT_STATEMENT"),
    ("TRUNCATE TABLE t", "NON_SELECT_STATEMENT"),
    ("COPY t TO '/tmp/x'", "NON_SELECT_STATEMENT"),
    ("SET search_path TO evil", "NON_SELECT_STATEMENT"),
    ("VACUUM", "NON_SELECT_STATEMENT"),
    ("EXPLAIN ANALYZE SELECT 1", "NON_SELECT_STATEMENT"),
    ("CALL myproc()", "NON_SELECT_STATEMENT"),
    ("COMMIT", "NON_SELECT_STATEMENT"),
    # A write hidden in a CTE: the root is a Select, so only a full-tree walk
    # catches it.
    (
        "WITH x AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM x",
        "FORBIDDEN_STATEMENT",
    ),
    # SELECT ... INTO creates a table.
    ("SELECT * INTO newtab FROM t", "FORBIDDEN_STATEMENT"),
    # Row locking is a write-side effect.
    ("SELECT * FROM t FOR UPDATE", "FORBIDDEN_STATEMENT"),
    ("SELECT pg_sleep(10)", "FORBIDDEN_FUNCTION"),
    ("SELECT pg_read_file('/etc/passwd')", "FORBIDDEN_FUNCTION"),
    ("SELECT lo_import('/etc/passwd')", "FORBIDDEN_FUNCTION"),
    ("SELECT set_config('x', 'y', false)", "FORBIDDEN_FUNCTION"),
    ("SELECT FROM", "INVALID_SQL"),
    ("SELECT ((", "INVALID_SQL"),
    ("!!!", "INVALID_SQL"),
]


@pytest.mark.parametrize("query", ACCEPTED)
def test_accepted_queries(query):
    result = validate_select(query)
    assert result.ok, f"{query!r} rejected with {result.code}: {result.message}"
    assert result.code is None


@pytest.mark.parametrize("query,code", REJECTED)
def test_rejected_queries(query, code):
    result = validate_select(query)
    assert not result.ok, f"{query!r} was accepted"
    assert result.code == code


def test_rejection_message_is_actionable():
    result = validate_select("DROP TABLE t")
    assert "SELECT" in result.message


def test_tables_are_collected_lowercased():
    result = validate_select("SELECT * FROM Orders o JOIN Customers c ON o.id = c.id")
    assert result.ok
    assert result.tables == ("customers", "orders")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM users",
        "SELECT * FROM a JOIN users u ON a.id = u.id",
        "SELECT * FROM (SELECT * FROM users) s",
        "WITH c AS (SELECT * FROM users) SELECT * FROM c",
        "SELECT * FROM a WHERE id IN (SELECT id FROM users)",
    ],
)
def test_denylist_blocks_table_anywhere_in_the_tree(query):
    result = validate_select(query, denied_tables=frozenset({"users"}))
    assert result.code == "TABLE_NOT_ALLOWED"


def test_denylist_is_case_insensitive():
    result = validate_select("SELECT * FROM Users", denied_tables=frozenset({"users"}))
    assert result.code == "TABLE_NOT_ALLOWED"


def test_allowlist_rejects_unlisted_table():
    result = validate_select(
        "SELECT * FROM orders", allowed_tables=frozenset({"customers"})
    )
    assert result.code == "TABLE_NOT_ALLOWED"


def test_allowlist_accepts_listed_table():
    result = validate_select(
        "SELECT * FROM customers", allowed_tables=frozenset({"customers"})
    )
    assert result.ok


def test_allowlist_wins_over_denylist():
    result = validate_select(
        "SELECT * FROM users",
        allowed_tables=frozenset({"users"}),
        denied_tables=frozenset({"users"}),
    )
    assert result.ok


def test_no_scoping_configured_accepts_any_table():
    assert validate_select("SELECT * FROM anything").ok


# ---------------------------------------------------------------------------
# Seam with the identifier rewriter
# ---------------------------------------------------------------------------


def test_quoted_identifiers_still_lowercase_for_allow_deny():
    """
    sql_identifiers hands this module quoted names. The allow/deny lists are
    lowercased env values, so `tables` must keep folding — otherwise a denied
    relation would slip through the moment its name got quoted.
    """
    result = validate_select('SELECT "Idcliente" FROM "Clienti"')

    assert result.ok
    assert result.tables == ("clienti",)


def test_a_denied_relation_stays_denied_when_quoted():
    result = validate_select(
        'SELECT 1 FROM "Segreti"', denied_tables=frozenset({"segreti"})
    )

    assert not result.ok
    assert result.code == "TABLE_NOT_ALLOWED"
