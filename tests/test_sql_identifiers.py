"""
Tests for datachat.sql_identifiers.

No database and no schema snapshot: the rewriter takes a plain
{relation: [columns]} mapping, so every case here is a literal dict.

The bulk of these are safety tests. Rewriting SQL is only acceptable if it is
provably inert on everything that is not a relation or column reference, so each
construct that tokenizes as a bare word — function names, EXTRACT units, aliases,
CTE names — gets its own case, with the map deliberately loaded to collide.
"""

import pytest

from datachat.sql_identifiers import quote_ident, quote_identifiers, suggest_identifier

# Mirrors the shape of the live CAA schema: mixed-case base tables alongside
# all-lowercase materialized views, and `anno` present in both spellings.
RELATIONS = {
    "Clienti": ["Idcliente", "Descrizione", "Anno"],
    "Trasporti": ["IdTrasporto", "IdSede", "Descrizione", "Data"],
    "Sedi": ["IdSede", "Descrizione"],
    "Budgets": ["Anno", "Cliente"],
    "report_mensile": ["anno", "mese", "id_cliente", "primo_margine", "descrizione"],
}


def fixed(sql, relations=None):
    return quote_identifiers(sql, RELATIONS if relations is None else relations)[0]


def names(sql, relations=None):
    return quote_identifiers(sql, RELATIONS if relations is None else relations)[1]


# ---------------------------------------------------------------------------
# The reported failure
# ---------------------------------------------------------------------------


def test_the_observed_live_failure_is_repaired():
    """`relation "clienti" does not exist`, straight from logs/agent_runs.log."""
    out = fixed("SELECT Idcliente FROM Clienti WHERE Descrizione = 'CLO'")

    assert out == 'SELECT "Idcliente" FROM "Clienti" WHERE "Descrizione" = \'CLO\''


def test_relation_and_columns_are_quoted_across_a_join():
    out = fixed(
        "SELECT t.IdTrasporto, s.Descrizione FROM trasporti t "
        "JOIN sedi s ON s.IdSede = t.IdSede"
    )

    assert out == (
        'SELECT t."IdTrasporto", s."Descrizione" FROM "Trasporti" t '
        'JOIN "Sedi" s ON s."IdSede" = t."IdSede"'
    )


def test_a_relation_qualifier_is_quoted_but_an_alias_is_not():
    assert fixed("SELECT Clienti.Idcliente FROM Clienti") == (
        'SELECT "Clienti"."Idcliente" FROM "Clienti"'
    )
    assert fixed("SELECT c.Idcliente FROM Clienti c") == (
        'SELECT c."Idcliente" FROM "Clienti" c'
    )


def test_reports_the_names_it_substituted():
    assert set(names("SELECT Idcliente FROM Clienti")) == {"Clienti", "Idcliente"}


# ---------------------------------------------------------------------------
# Case ambiguity
# ---------------------------------------------------------------------------


def test_lowercase_relations_and_columns_are_left_alone():
    sql = "SELECT anno, mese, primo_margine FROM report_mensile WHERE anno = '2026'"

    assert fixed(sql) == sql
    assert names(sql) == ()


def test_an_ambiguous_column_resolves_by_the_relation_in_scope():
    """`anno` is Anno on Budgets and anno on report_mensile; scope decides."""
    assert fixed("SELECT Anno FROM Budgets") == 'SELECT "Anno" FROM "Budgets"'
    assert fixed("SELECT anno FROM report_mensile") == "SELECT anno FROM report_mensile"


def test_a_column_ambiguous_within_one_query_is_left_bare():
    """Both spellings in scope: guessing would break one of them."""
    out = fixed(
        "SELECT anno FROM Budgets JOIN report_mensile ON true"
    )

    assert '"Anno"' not in out
    assert "anno FROM" in out
    # The relations are still repaired; only the ambiguous column is withheld.
    assert '"Budgets"' in out


def test_relations_differing_only_by_case_are_left_alone():
    relations = {"Ordini": ["IdOrdine"], "ordini": ["id_ordine"]}

    assert fixed("SELECT 1 FROM ordini", relations) == "SELECT 1 FROM ordini"


# ---------------------------------------------------------------------------
# Safety: bare words that are not column references
# ---------------------------------------------------------------------------


def test_function_names_are_never_rewritten():
    """count/sum tokenize as bare words but are not identifiers."""
    relations = {"Trasporti": ["Count", "Sum", "IdTrasporto"]}
    sql = "SELECT count(*), sum(IdTrasporto) FROM Trasporti"

    assert fixed(sql, relations) == 'SELECT count(*), sum("IdTrasporto") FROM "Trasporti"'


def test_an_extract_unit_is_never_rewritten():
    """YEAR is an exp.Var, not a column, even with a Year column in the map."""
    relations = {"Trasporti": ["Year", "Data"]}
    out = fixed("SELECT EXTRACT(YEAR FROM Data) FROM Trasporti", relations)

    assert out == 'SELECT EXTRACT(YEAR FROM "Data") FROM "Trasporti"'


def test_a_table_alias_shadowing_a_real_column_is_not_rewritten():
    relations = {"Clienti": ["C", "Idcliente"]}
    out = fixed("SELECT c.Idcliente FROM Clienti c", relations)

    assert out == 'SELECT c."Idcliente" FROM "Clienti" c'


def test_a_column_alias_equal_to_a_real_column_is_withheld():
    """Suppression is by name, and fails towards today's behaviour."""
    out = fixed("SELECT count(*) AS Descrizione FROM Clienti")

    assert out == 'SELECT count(*) AS Descrizione FROM "Clienti"'


def test_a_cte_named_like_a_relation_is_left_alone():
    sql = "WITH Clienti AS (SELECT 1 AS n) SELECT n FROM Clienti"

    assert fixed(sql) == sql


def test_the_schema_qualifier_is_not_rewritten():
    out = fixed("SELECT Idcliente FROM public.Clienti")

    assert out == 'SELECT "Idcliente" FROM public."Clienti"'


# ---------------------------------------------------------------------------
# Safety: the text outside identifier spans
# ---------------------------------------------------------------------------


def test_string_literals_are_never_touched():
    sql = "SELECT Idcliente FROM Clienti WHERE Descrizione LIKE '%Clienti%'"

    assert fixed(sql).endswith("LIKE '%Clienti%'")


def test_an_escaped_quote_inside_a_literal_survives():
    sql = "SELECT 1 FROM Clienti WHERE Descrizione = 'L''Aquila'"

    assert fixed(sql) == 'SELECT 1 FROM "Clienti" WHERE "Descrizione" = \'L\'\'Aquila\''


def test_comments_survive_byte_for_byte():
    """Proves span patching rather than regeneration: sqlglot drops comments."""
    out = fixed("SELECT Idcliente FROM Clienti -- Descrizione and Clienti here\n")

    assert out.endswith("-- Descrizione and Clienti here\n")


def test_dollar_quoted_text_is_not_touched():
    out = fixed("SELECT $$Clienti$$, Idcliente FROM Clienti")

    assert "$$Clienti$$" in out
    assert '"Idcliente"' in out


def test_already_quoted_identifiers_are_left_as_they_are():
    sql = 'SELECT "Idcliente" FROM "Clienti"'

    assert fixed(sql) == sql


def test_the_rewrite_is_idempotent():
    once = fixed("SELECT Idcliente FROM Clienti WHERE Descrizione = 'CLO'")

    assert fixed(once) == once


def test_casts_and_stars_are_untouched():
    assert fixed("SELECT * FROM Clienti") == 'SELECT * FROM "Clienti"'
    assert fixed("SELECT Idcliente::text FROM Clienti") == (
        'SELECT "Idcliente"::text FROM "Clienti"'
    )


def test_offsets_survive_non_ascii_text_earlier_in_the_query():
    """
    The tokenizer's offsets must be character-based, not byte-based, or a
    multi-byte literal would shift every span after it.
    """
    out = fixed("SELECT 'città è àè' AS x, Idcliente FROM Clienti")

    assert out == "SELECT 'città è àè' AS x, \"Idcliente\" FROM \"Clienti\""


def test_a_non_ascii_identifier_is_quoted_correctly():
    relations = {"Comuni": ["Città"]}
    out = fixed("SELECT Città FROM Comuni", relations)

    assert out == 'SELECT "Città" FROM "Comuni"'


# ---------------------------------------------------------------------------
# Totality: never raise, never mangle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "not sql at all ((((",
        "SELECT 1; DROP TABLE Clienti",
        "SELECT Idcliente FROM Clienti WHERE",
    ],
)
def test_doubtful_input_is_returned_unchanged(sql):
    """The guard rejects these with a proper message; mangling them helps nobody."""
    assert quote_identifiers(sql, RELATIONS) == (sql, ())


def test_multiple_statements_are_left_for_the_guard():
    sql = "SELECT 1 FROM Clienti; SELECT 2 FROM Trasporti"

    assert quote_identifiers(sql, RELATIONS) == (sql, ())


def test_an_empty_relation_map_changes_nothing():
    sql = "SELECT Idcliente FROM Clienti"

    assert quote_identifiers(sql, {}) == (sql, ())


def test_unknown_names_are_left_for_the_database_to_reject():
    sql = "SELECT Sconosciuto FROM Inesistente"

    assert fixed(sql) == sql


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_quote_ident_doubles_an_embedded_quote():
    assert quote_ident('we"ird') == '"we""ird"'


def test_suggest_identifier_finds_the_real_spelling():
    assert suggest_identifier("trasporti", RELATIONS) == "Trasporti"
    assert suggest_identifier("TRASPORTI", RELATIONS) == "Trasporti"


def test_suggest_identifier_is_silent_when_there_is_nothing_to_add():
    assert suggest_identifier("Trasporti", RELATIONS) is None
    assert suggest_identifier("sconosciuto", RELATIONS) is None
    assert suggest_identifier("", RELATIONS) is None


# ---------------------------------------------------------------------------
# Qualified columns
# ---------------------------------------------------------------------------

# The real shape of Sedi/Trasporti: the same join key, spelled differently on
# each side. Query-wide the name is ambiguous; per-relation it is not.
JOIN_RELATIONS = {
    "Sedi": ["IdSede", "Descrizione"],
    "Trasporti": ["idSede", "idTrasporto", "Descrizione"],
}


def test_a_join_key_spelled_differently_on_each_side_is_resolved():
    out = fixed(
        "SELECT s.Descrizione, count(t.idTrasporto) FROM Sedi s "
        "LEFT JOIN Trasporti t ON s.IdSede = t.idSede",
        JOIN_RELATIONS,
    )

    assert 's."IdSede" = t."idSede"' in out
    assert 'FROM "Sedi" s LEFT JOIN "Trasporti" t' in out


def test_the_same_key_unqualified_is_still_left_alone():
    """Without a qualifier there is nothing to disambiguate with."""
    out = fixed(
        "SELECT IdSede FROM Sedi JOIN Trasporti ON true", JOIN_RELATIONS
    )

    assert '"IdSede"' not in out
    assert '"Sedi"' in out


def test_a_relation_qualifier_resolves_like_an_alias():
    out = fixed("SELECT Trasporti.idSede FROM Trasporti", JOIN_RELATIONS)

    assert out == 'SELECT "Trasporti"."idSede" FROM "Trasporti"'


def test_a_qualifier_does_not_borrow_another_relations_spelling():
    """t is Trasporti, so its idSede must not become Sedi's IdSede."""
    out = fixed("SELECT t.idSede FROM Trasporti t, Sedi s", JOIN_RELATIONS)

    assert 't."idSede"' in out
    assert 't."IdSede"' not in out
