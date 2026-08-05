"""Tests for the keywords tool, on Italian text because that is the real data.

Regression origin: sklearn ships English stopwords only, so Italian answers came back with
`il`, `gli`, `molto` and `erano` as their most prominent "themes".
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.keywords_tool import KeywordsTool

ITALIAN_FUNCTION_WORDS = {
    "il", "lo", "la", "gli", "le", "di", "del", "della", "per", "con", "non",
    "molto", "erano", "sono", "che", "una", "chi", "poco", "troppo", "abbastanza",
}


@pytest.fixture()
def comments_df():
    comments = [
        "il corso è stato molto interessante e utile per il mio lavoro",
        "la docente è stata molto chiara nella spiegazione dei contenuti",
        "troppo poco tempo per gli esercizi pratici, servirebbe più pratica",
        "gli orari non erano comodi, sarebbe meglio il pomeriggio",
        "il materiale didattico è di ottima qualità e molto completo",
        "poca pratica e troppa teoria, il tempo per esercitarsi è scarso",
        "gli orari del corso sono scomodi per chi lavora al mattino",
        "docente preparata e disponibile, spiegazioni molto chiare",
    ] * 6
    return pd.DataFrame({"commento": comments})


@pytest.fixture()
def tool(comments_df):
    return KeywordsTool(comments_df)


def test_italian_stopwords_never_appear(tool):
    """The regression: function words must not be reported as themes."""
    terms = {r["term"] for r in tool.forward(column="commento")["data"]}

    assert not (terms & ITALIAN_FUNCTION_WORDS)


def test_the_real_topics_surface_first(tool):
    top = [r["term"] for r in tool.forward(column="commento", top_n=5)["data"]]

    assert set(top) == {"corso", "docente", "orari", "pratica", "tempo"}


def test_counts_and_shares_are_consistent(tool, comments_df):
    result = tool.forward(column="commento")
    n_answers = len(comments_df)

    for row in result["data"]:
        assert row["answers"] <= row["count"]
        assert row["share_of_answers"] == pytest.approx(row["answers"] / n_answers, abs=1e-4)
        assert 0 < row["share_of_answers"] <= 1


def test_bigrams_are_two_words(tool):
    result = tool.forward(column="commento", ngram=2, top_n=5)

    assert result["data"]
    for row in result["data"]:
        assert len(row["term"].split()) == 2


def test_min_count_filters_rare_terms():
    df = pd.DataFrame({"c": ["parola frequente", "parola frequente", "raro unicorno"]})
    result = KeywordsTool(df).forward(column="c", min_count=2)
    terms = {r["term"] for r in result["data"]}

    assert "parola" in terms
    assert "unicorno" not in terms


def test_single_letter_tokens_and_digits_are_ignored():
    df = pd.DataFrame({"c": ["a b c 1 2 3 formazione utile", "a b c formazione utile"]})
    result = KeywordsTool(df).forward(column="c", min_count=1)
    terms = {r["term"] for r in result["data"]}

    assert terms == {"formazione", "utile"}


def test_all_terms_returned_when_no_limit(tool):
    result = tool.forward(column="commento")

    assert "note" not in result
    assert len(result["data"]) == result["meta"]["terms_found"]


def test_top_n_is_disclosed(tool):
    result = tool.forward(column="commento", top_n=3)

    assert len(result["data"]) == 3
    assert "terms available" in result["note"]


def test_explicit_language_is_honoured(tool):
    """Forcing English leaves the Italian function words in — proof the list is applied."""
    terms = {r["term"] for r in tool.forward(column="commento", language="english")["data"]}

    assert terms & ITALIAN_FUNCTION_WORDS


def test_blank_and_missing_rows_are_skipped():
    df = pd.DataFrame({"c": ["formazione utile", None, "   ", "formazione ottima"]})
    result = KeywordsTool(df).forward(column="c", min_count=1)

    assert result["meta"]["answers_analyzed"] == 2


def test_carries_export_name(tool):
    assert tool.forward(column="commento")["export_name"] == "keywords_commento"


def test_empty_column_is_an_error():
    df = pd.DataFrame({"c": [None, "", "  "]})
    result = KeywordsTool(df).forward(column="c")

    assert result["code"] == "EMPTY_COLUMN"


def test_unknown_column_is_an_error(tool):
    assert tool.forward(column="nope")["code"] == "INVALID_COLUMN"


def test_upstream_data_is_used(tool):
    result = tool.forward(column="x", data=[{"x": "formazione utile"}, {"x": "formazione ottima"}], min_count=1)

    assert result["meta"]["answers_analyzed"] == 2
    assert "formazione" in {r["term"] for r in result["data"]}
