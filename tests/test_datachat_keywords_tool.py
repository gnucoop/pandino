"""keywords: recurring words across free-text responses, ranked by response coverage."""

import datetime as dt
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from datachat.tools.keywords_stopwords import STOPWORDS
from datachat.tools.keywords_tool import KeywordsTool
from tests.test_datachat_result_preview import _ToolEngine, _chat
from tests.test_datachat_route_request_id import (  # noqa: F401  (autouse fixtures)
    restore_agent_runs_logger,
    restore_datachat_runtime_logger,
)

FIELDS = ["term", "answers", "count", "share_of_answers"]


def _run(values, **kwargs):
    kwargs.setdefault("column", "c")
    kwargs.setdefault("language", "all")
    return KeywordsTool(pd.DataFrame({"c": values})).forward(**kwargs)


def _terms(out):
    return [r["term"] for r in out["data"]]


def _by_term(out):
    return {r["term"]: r for r in out["data"]}


# --- core semantics -----------------------------------------------------------


def test_recurring_words_across_responses():
    out = _run(["corso utile", "corso lungo", "docente utile", "aula"])
    assert out["kind"] == "table"
    assert _terms(out) == ["corso", "utile"]
    for row in out["data"]:
        assert list(row) == FIELDS


def test_answers_rank_before_count():
    out = _run(["pratica pratica pratica pratica", "pratica", "teoria", "teoria", "teoria"])
    rows = _by_term(out)
    assert _terms(out) == ["teoria", "pratica"]
    assert (rows["teoria"]["answers"], rows["teoria"]["count"]) == (3, 3)
    assert (rows["pratica"]["answers"], rows["pratica"]["count"]) == (2, 5)


def test_count_breaks_ties_in_answers():
    out = _run(["alfa beta beta", "alfa beta"])
    assert _terms(out) == ["beta", "alfa"]


def test_repetition_within_one_response_counts_once_in_answers():
    row = _by_term(_run(["ciao ciao ciao", "ciao"]))["ciao"]
    assert (row["answers"], row["count"]) == (2, 4)


def test_duplicate_source_rows_are_separate_responses():
    row = _by_term(_run(["corso utile", "corso utile", "corso utile"]))["corso"]
    assert (row["answers"], row["count"], row["share_of_answers"]) == (3, 3, 1.0)


def test_lexical_tie_break():
    out = _run(["zeta alfa mela", "mela zeta alfa"])
    assert _terms(out) == ["alfa", "mela", "zeta"]


def test_share_of_answers():
    out = _run(["corso utile", "corso", "altro", "ancora altro"], language="english")
    rows = _by_term(out)
    assert rows["corso"]["share_of_answers"] == pytest.approx(0.5)
    assert isinstance(rows["corso"]["share_of_answers"], float)
    assert isinstance(rows["corso"]["answers"], int) and isinstance(rows["corso"]["count"], int)


def test_min_answers_applies_to_answers_not_count():
    out = _run(["unico unico unico unico", "comune", "comune"])
    assert _terms(out) == ["comune"]
    assert set(_terms(_run(["unico unico unico unico", "comune", "comune"], min_answers=1))) == {"unico", "comune"}
    assert _terms(_run(["aa bb", "aa bb", "aa"], min_answers=3)) == ["aa"]


def test_explicit_n_takes_the_top_of_the_full_ranking():
    values = ["aa bb cc", "aa bb", "aa"]
    out = _run(values, min_answers=1, n=2)
    assert _terms(out) == ["aa", "bb"]
    assert "note" not in out


def test_no_n_has_no_hidden_cap():
    words = [f"parola{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(300)]
    out = _run([" ".join(words)] * 2)
    assert len(out["data"]) == 300


# --- population ---------------------------------------------------------------


def test_missing_values_are_excluded_without_note():
    out = _run([None, np.nan, pd.NA, pd.NaT, "corso utile", "corso"])
    assert _by_term(out)["corso"]["share_of_answers"] == 1.0
    assert "note" not in out


def test_blank_strings_are_excluded_without_note():
    out = _run(["", "   ", "\t\n", "corso", "corso"])
    assert _by_term(out)["corso"]["share_of_answers"] == 1.0
    assert "note" not in out


def test_numbers_are_excluded_not_stringified():
    out = _run([1, 2.5, np.int64(3), "alfa beta", "alfa beta"], min_answers=1)
    assert set(_terms(out)) == {"alfa", "beta"}
    assert _by_term(out)["alfa"]["share_of_answers"] == 1.0
    assert out["note"] == "Excluded 3 non-text row(s) from keyword analysis."


def test_booleans_are_excluded_not_stringified():
    out = _run([True, False, np.bool_(True), "vero", "vero"], min_answers=1)
    assert _terms(out) == ["vero"]
    assert "true" not in _terms(out)
    assert out["note"] == "Excluded 3 non-text row(s) from keyword analysis."


def test_dates_times_and_durations_are_excluded():
    values = [
        dt.date(2024, 1, 2),
        dt.datetime(2024, 1, 2, 3, 4),
        dt.time(10, 30),
        dt.timedelta(days=1),
        pd.Timestamp("2024-05-01"),
        pd.Timedelta("1h"),
        "testo",
    ]
    out = _run(values, min_answers=1)
    assert _terms(out) == ["testo"]
    assert _by_term(out)["testo"]["share_of_answers"] == 1.0
    assert out["note"] == "Excluded 6 non-text row(s) from keyword analysis."


def test_mixed_type_column():
    out = _run(["corso", 7, None, "corso utile", {"a": 1}, "  "], min_answers=1)
    rows = _by_term(out)
    assert rows["corso"]["share_of_answers"] == 1.0
    assert rows["utile"]["share_of_answers"] == 0.5
    assert out["note"] == "Excluded 2 non-text row(s) from keyword analysis."


def test_literal_none_and_nan_strings_are_text():
    out = _run(["none", "null nan", "none null nan"], min_answers=1)
    assert set(_terms(out)) == {"none", "null", "nan"}
    assert "note" not in out


def test_text_without_surviving_terms_stays_in_the_denominator():
    out = _run(["corso", "il la di", "123", "!!!", "corso"])
    assert _by_term(out)["corso"]["share_of_answers"] == pytest.approx(2 / 5)
    assert "note" not in out


def test_markup_only_text_is_blank():
    out = _run(["<br>", "&nbsp;", "corso", "corso"])
    assert _by_term(out)["corso"]["share_of_answers"] == 1.0


# --- tokenizer ----------------------------------------------------------------


def _tokens_of(text, language="english"):
    return set(_terms(_run([text], min_answers=1, language=language)))


def test_italian_accents():
    assert _tokens_of("perché la città è università", "english") >= {"perché", "città", "università"}


def test_french_accents():
    assert _tokens_of("café résumé façade où") == {"café", "résumé", "façade", "où"}


def test_spanish_letters():
    assert _tokens_of("ñandú útil fácil camión país") == {"ñandú", "útil", "fácil", "camión", "país"}


def test_other_unicode_alphabetic_words():
    assert _tokens_of("über straße naïve ålborg") == {"über", "straße", "naïve", "ålborg"}


def test_decomposed_accents_are_one_token():
    assert _tokens_of("café") == {"café"}


def test_apostrophes_separate_tokens():
    assert _tokens_of("dell'aula can't", "french") == {"dell", "aula", "can"}
    assert _tokens_of("dell’aula", "french") == {"dell", "aula"}


def test_hyphens_and_underscores_separate_tokens():
    assert _tokens_of("hello-world foo_bar") == {"hello", "world", "foo", "bar"}


def test_single_letters_are_excluded():
    assert _tokens_of("x y z ab") == {"ab"}


def test_numbers_are_excluded():
    assert _tokens_of("123 4.5 2024 1,000 ok") == {"ok"}


def test_alphanumeric_tokens_are_excluded():
    assert _tokens_of("covid19 abc123 h2o 3d ok") == {"ok"}


def test_emoji_are_ignored():
    assert _tokens_of("😀great😀 👍 ok") == {"great", "ok"}


def test_case_folding():
    row = _by_term(_run(["Ciao CIAO ciao", "cIaO"]))["ciao"]
    assert (row["answers"], row["count"]) == (2, 4)


def test_urls_and_emails_split_on_punctuation():
    assert _tokens_of("https://www.example.com mario.rossi@gnucoop.com") == {
        "https", "www", "example", "com", "mario", "rossi", "gnucoop",
    }


# --- cleaning -----------------------------------------------------------------


def test_html_tags_are_removed():
    assert _tokens_of("<p class='x'>corso</p><br/>utile") == {"corso", "utile"}


def test_html_entities_are_unescaped():
    assert _tokens_of("caff&egrave; &amp; t&eacute;") == {"caffè", "té"}


def test_nbsp_and_whitespace_are_normalized():
    assert _tokens_of("corso&nbsp;utile\xa0docente\n\tbravo") == {"corso", "utile", "docente", "bravo"}


# --- stopwords ----------------------------------------------------------------

SAMPLES = {
    "italian": "il corso era molto utile per la pratica",
    "english": "the course was very useful for the practice",
    "french": "le cours était très utile pour la pratique",
    "spanish": "el curso fue muy útil para la práctica",
}
CONTENT = {
    "italian": {"corso", "utile", "pratica"},
    "english": {"course", "useful", "practice"},
    "french": {"cours", "utile", "pratique"},
    "spanish": {"curso", "útil", "práctica"},
}


@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_language_stopwords(language):
    assert _tokens_of(SAMPLES[language], language) == CONTENT[language]


def test_language_is_applied_not_guessed():
    # English stopwords leave the Italian function words in place: no detection.
    assert {"il", "era", "molto", "per", "la"} <= _tokens_of(SAMPLES["italian"], "english")


def test_explicit_all_is_the_union():
    mixed = " ".join(SAMPLES.values())
    expected = set().union(*CONTENT.values())
    assert _tokens_of(mixed, "all") == expected
    assert STOPWORDS["all"] == STOPWORDS["italian"] | STOPWORDS["english"] | STOPWORDS["french"] | STOPWORDS["spanish"]


def test_stopword_sets_are_lowercase_and_tokenizable():
    for words in STOPWORDS.values():
        for word in words:
            assert word == word.lower() and len(word) >= 2 and word.isalpha()


def test_meaningful_words_survive_the_all_list():
    assert _tokens_of("none null nan grazie grande bene fine", "all") == {
        "none", "null", "nan", "grazie", "grande", "bene", "fine",
    }


@pytest.mark.parametrize("language", ["klingon", "it", "Italian", "both", " all", "none"])
def test_unknown_language_is_rejected(language):
    out = _run(["corso"], language=language)
    assert out["kind"] == "error"
    assert out["code"] == "INVALID_LANGUAGE"


@pytest.mark.parametrize("kwargs", [{}, {"language": None}, {"language": ""}, {"language": "   "}])
def test_language_is_required(kwargs):
    out = KeywordsTool(pd.DataFrame({"c": ["car car", "car"]})).forward(column="c", **kwargs)
    assert out["kind"] == "error"
    assert out["code"] == "MISSING_LANGUAGE"


def test_language_has_no_default():
    assert KeywordsTool.inputs["language"]["description"].startswith("Required")
    assert "default" not in KeywordsTool.inputs["language"]["description"].replace("no default", "")


def test_english_content_word_survives_english():
    values = ["my car is old", "the car and the son", "car", "son"]
    assert _terms(_run(values, language="english")) == ["car", "son"]


def test_the_explicit_union_may_remove_another_language_word():
    # "car" is a French stopword and "son" a French/Spanish one: explicit "all" drops them.
    values = ["my car is old", "the car and the son", "car", "son"]
    assert _terms(_run(values, language="all")) == []


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize("column", [None, "", "   "])
def test_missing_column(column):
    assert _run(["corso"], column=column)["code"] == "MISSING_COLUMN"


def test_unknown_column():
    assert _run(["corso"], column="nope")["code"] == "INVALID_COLUMN"


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, 2.0, "2"])
def test_invalid_min_answers(value):
    assert _run(["corso"], min_answers=value)["code"] == "INVALID_MIN_ANSWERS"


@pytest.mark.parametrize("value", [0, -3, True, 1.5, 2.0, "2"])
def test_invalid_n(value):
    assert _run(["corso", "corso"], n=value)["code"] == "INVALID_LIMIT"


def test_numpy_integers_are_valid_parameters():
    out = _run(["aa bb", "aa bb"], n=np.int64(1), min_answers=np.int64(2))
    assert _terms(out) == ["aa"]


# --- empty semantics ----------------------------------------------------------


def test_empty_records_is_an_empty_table():
    assert KeywordsTool(pd.DataFrame({"c": ["x"]})).forward(column="c", language="all", data=[]) == {"kind": "table", "data": []}


def test_column_without_text_is_an_empty_table():
    assert _run([None, "", "  "]) == {"kind": "table", "data": []}
    assert _run([]) == {"kind": "table", "data": []}


def test_only_stopwords_is_an_empty_table():
    assert _run(["il la di", "the and of"]) == {"kind": "table", "data": []}


def test_nothing_meets_min_answers_is_an_empty_table():
    assert _run(["aa", "bb", "cc"]) == {"kind": "table", "data": []}


def test_only_non_text_is_an_empty_table_with_note():
    out = _run([1, 2, True])
    assert out["data"] == []
    assert out["note"] == "Excluded 3 non-text row(s) from keyword analysis."


# --- data= --------------------------------------------------------------------


def test_records_input():
    tool = KeywordsTool(pd.DataFrame({"other": [1]}))
    out = tool.forward(column="x", language="italian", data=[{"x": "corso utile"}, {"x": "corso"}])
    assert _terms(out) == ["corso"]


def test_wrapped_records_input():
    tool = KeywordsTool(pd.DataFrame({"other": [1]}))
    out = tool.forward(column="x", language="italian", data={"kind": "table", "data": [{"x": "corso"}, {"x": "corso"}]})
    assert _terms(out) == ["corso"]


def test_invalid_records():
    assert KeywordsTool(pd.DataFrame({"c": ["x"]})).forward(column="c", language="all", data="bad")["code"] == "INVALID_DATA"


# --- contract regressions -----------------------------------------------------


def test_no_meta_export_name_or_scores():
    out = _run(["corso utile", "corso utile", 5])
    assert set(out) == {"kind", "data", "note"}
    for row in out["data"]:
        assert set(row) == set(FIELDS)


def test_public_inputs():
    assert set(KeywordsTool.inputs) == {"column", "n", "min_answers", "language", "data"}


def test_description_claims_no_semantics():
    text = KeywordsTool.description.lower()
    for phrase in ("tf-idf", "tfidf", "classif", "detect", "semantic"):
        assert phrase not in text


def test_no_sklearn_import():
    code = (
        "import sys; import datachat.tools.keywords_tool as k; import pandas as pd; "
        "k.KeywordsTool(pd.DataFrame({'c': ['aa bb', 'aa']})).forward(column='c', language='all'); "
        "assert not any(m == 'sklearn' or m.startswith('sklearn.') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_ngram_path():
    with pytest.raises(TypeError):
        KeywordsTool(pd.DataFrame({"c": ["aa bb"]})).forward(column="c", ngram=2)
    out = _run(["corso utile", "corso utile"])
    assert all(" " not in t for t in _terms(out))


def test_no_provider_or_usage():
    tool = KeywordsTool(pd.DataFrame({"c": ["aa"]}))
    assert not hasattr(tool, "_model") and not hasattr(tool, "_provider")


# --- through POST /datachat ---------------------------------------------------


def test_exclusion_note_reaches_the_client(monkeypatch):
    df = pd.DataFrame({"c": ["corso utile", "corso", 3, True, None]})
    body = _chat(monkeypatch, _ToolEngine(lambda: KeywordsTool(df).forward(column="c", language="italian")))
    assert body["note"] == "Excluded 2 non-text row(s) from keyword analysis."
    assert body["result_rows"] == 1


def test_text_only_column_has_no_client_note(monkeypatch):
    df = pd.DataFrame({"c": ["corso utile", "corso", None, "  "]})
    body = _chat(monkeypatch, _ToolEngine(lambda: KeywordsTool(df).forward(column="c", language="italian")))
    assert "note" not in body
