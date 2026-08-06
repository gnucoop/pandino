"""Tool output must not presume the dataset's language.

Regression origin: the tools were built against an Italian test dataset and grew Italian
strings in their *output* — every count chart was labelled "numero di risposte", every blank
category "(vuoto)", every export "Export pronto: N righe." An English or French dataset got
Italian axis labels.

The rule, matching the one already applied to colours and Chart.js options: the backend emits
stable semantic tokens and the client decides presentation, language included. Reserved tokens
are declared in DINO_CLIENT_SPEC.md.
"""

import pandas as pd
import pytest

from datachat.tools.chart_tool import ChartTool
from datachat.tools.crosstab_tool import CrosstabTool
from datachat.tools.keywords_tool import KeywordsTool
from datachat.tools.stopwords import detect_language, get_stopwords

# Words that must never appear in a tool's output, whatever the dataset.
ITALIAN_LEAKS = (
    "numero di risposte",
    "densità",
    "vuoto",
    "pronto",
    "righe",
    "colonna",
    "grafico",
)


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "category": ["A", "A", "B", None],
            "other": ["X", "Y", "X", "Y"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _assert_neutral(blob: str):
    lowered = blob.lower()
    for word in ITALIAN_LEAKS:
        assert word not in lowered, f"Italian leaked into tool output: {word!r}"


# ---------------------------------------------------------------------------
# Chart specs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("bar", {"x": "category"}),
        ("bar", {"x": "category", "y": "value", "agg": "mean"}),
        ("line", {"x": "category", "y": "value"}),
        ("area", {"x": "category", "y": "value"}),
        ("pie", {"x": "category"}),
        ("doughnut", {"x": "category"}),
        ("scatter", {"x": "value", "y": "value"}),
        ("hist", {"x": "value"}),
        ("kde", {"x": "value"}),
    ],
)
def test_no_chart_kind_emits_italian(df, kind, kwargs):
    result = ChartTool(df).forward(kind=kind, **kwargs)

    assert result["kind"] == "chart", result
    _assert_neutral(str(result))


def test_count_charts_are_labelled_with_a_neutral_token(df):
    spec = ChartTool(df).forward(kind="bar", x="category")["chart"]

    assert spec["y_label"] == "count"
    assert spec["datasets"][0]["label"] == "count"


def test_aggregate_labels_use_a_function_shape(df):
    """"mean(value)" embeds a user column name, so it is not translatable wholesale."""
    spec = ChartTool(df).forward(kind="bar", x="category", y="value", agg="mean")["chart"]

    assert spec["y_label"] == "mean(value)"


def test_density_labels_are_neutral(df):
    spec = ChartTool(df).forward(kind="kde", x="value")["chart"]

    assert spec["y_label"] == "density"
    assert spec["datasets"][0]["label"] == "density(value)"


def test_blank_category_sentinel_is_neutral(df):
    spec = ChartTool(df).forward(kind="bar", x="category")["chart"]

    assert "(empty)" in spec["labels"]


def test_crosstab_blank_sentinel_is_neutral(df):
    result = CrosstabTool(df).forward(rows="category", columns="other")

    assert any(r["category"] == "(empty)" for r in result["data"])
    _assert_neutral(str(result))


# ---------------------------------------------------------------------------
# Four languages
# ---------------------------------------------------------------------------

SAMPLES = {
    "italian": [
        "il corso è stato molto interessante e utile per il mio lavoro",
        "la docente è stata molto chiara nella spiegazione dei contenuti",
    ],
    "english": [
        "the course was very interesting and useful for my job",
        "the teacher was clear in her explanation of the topics",
    ],
    "french": [
        "le cours était très intéressant et utile pour mon travail",
        "la formatrice était très claire dans la présentation des contenus",
    ],
    "spanish": [
        "el curso fue muy interesante y útil para mi trabajo",
        "la formadora fue muy clara en la presentación de los contenidos",
    ],
}


@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_each_supported_language_is_detected(language):
    assert detect_language(SAMPLES[language]) == language


def test_mixed_language_text_excludes_everything():
    """Guessing wrong on a mixed column is worse than excluding all four sets."""
    mixed = SAMPLES["italian"] + SAMPLES["french"] + SAMPLES["english"]

    assert detect_language(mixed) == "all"


def test_empty_text_falls_back_to_everything():
    assert detect_language([]) == "all"


@pytest.mark.parametrize(
    "alias,expected_language",
    [
        ("it", "italian"), ("ITA", "italian"), ("italiano", "italian"),
        ("en", "english"), ("English", "english"),
        ("fr", "french"), ("fra", "french"), ("français", "french"), ("francais", "french"),
        ("es", "spanish"), ("spa", "spanish"), ("español", "spanish"),
    ],
)
def test_language_aliases_resolve(alias, expected_language):
    assert get_stopwords(alias) == get_stopwords(expected_language)


def test_unknown_language_falls_back_to_everything():
    assert get_stopwords("klingon") == get_stopwords("all")


FUNCTION_WORDS = {
    "french": {"le", "la", "les", "des", "est", "était", "pour", "de", "et", "très", "qui"},
    "spanish": {"el", "la", "los", "de", "que", "una", "muy", "para", "en", "fue", "son"},
    "italian": {"il", "la", "che", "molto", "per", "di", "e", "più"},
    "english": {"the", "was", "for", "of", "and", "in", "her"},
}


@pytest.mark.parametrize("language", sorted(FUNCTION_WORDS))
def test_keywords_removes_function_words_in_every_language(language):
    texts = SAMPLES[language] * 6
    result = KeywordsTool(pd.DataFrame({"c": texts})).forward(column="c")
    terms = {row["term"] for row in result["data"]}

    leaked = terms & FUNCTION_WORDS[language]
    assert not leaked, f"{language}: function words survived -> {sorted(leaked)}"


def test_keywords_still_finds_the_content_words():
    """Stripping function words must not strip the signal with them."""
    result = KeywordsTool(pd.DataFrame({"c": SAMPLES["french"] * 6})).forward(column="c")
    terms = {row["term"] for row in result["data"]}

    assert {"cours", "formatrice", "contenus"} <= terms
