"""Stopword lists for text analysis on DataChat columns.

scikit-learn ships an English list only, so `stop_words="english"` applied to Italian
survey answers leaves the function words in place and they dominate the result -- cluster
labels came back as "il, molto, poca, troppa, teoria", where only "teoria" carries meaning.

Kept as a plain frozenset rather than pulling in nltk or spacy: a word list is not worth a
dependency, and neither package is currently installed.
"""

from typing import Iterable, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Italian function words: articles, prepositions (simple and articulated), conjunctions,
# pronouns, common auxiliary/verb forms, and the intensifiers that flood survey answers
# ("molto", "poco", "abbastanza") without indicating a topic.
ITALIAN_STOP_WORDS = frozenset(
    """
    a ad affinche agli ai al all alla alle allo altri altro anche ancora avendo avere avete
    aveva avevano avrebbe avuto
    abbastanza altra altre anzi appunto be bene c che chi ci cio cioe cioè ciò circa co
    col coi come comunque con contro cosa cosi così cui
    da dagli dai dal dall dalla dalle dallo degli dei del dell della delle dello dentro di
    dopo dove due dunque durante

    e ebbe ed egli entrambi era erano esse essendo essere essi
    fa facendo fanno fare fatto fin fine finche fino forse fosse fra fu fui furono
    gia già gli grande grazie
    ha hai hanno ho
    i il in infatti inoltre insieme intanto intorno invece io
    la le lei li lo loro lui lungo
    ma magari mai me meno mentre mi mia mie miei mio moltissimo molta molte molti molto
    ne nei nel nell nella nelle nello nessuno no noi non nostra nostre nostri nostro nulla
    o oltre ora ossia ovvero
    parecchi per percio perche perchè perché però piu più piuttosto po poca poche pochi
    poco poi possa potrebbe presso propria proprio puo può pure
    qua qual quale quali qualche qualcosa quando quanta quante quanti quanto quasi quella
    quelle quelli quello quest questa queste questi questo qui quindi
    sara sarebbe sarà se sei sembra sempre senza si sia siamo siete solo sono sopra sotto
    sta stata state stati stato stesso su sua sue sugli sui sul sull sulla sulle sullo suo
    suoi
    tale tali tanta tante tanti tanto te tra tranne tre troppa troppe troppi troppo tu tua
    tue tuo tuoi tutta tutte tutti tutto
    un una uno
    va vai verso vi via voi vostra vostre vostri vostro
    """.split()
)

# Neutral filler that appears in Italian *and* English free-text answers, plus the names
# of HTML entities: survey exports carry markup (this dataset has 795 "&nbsp;") and an
# unescaped entity name otherwise ranks as the most common "word" in the column.
_GENERIC_STOP_WORDS = frozenset(
    {
        "n", "na", "nan", "none", "null", "nd", "vari", "varie", "ecc",
        "nbsp", "amp", "quot", "apos", "lt", "gt", "ndash", "mdash",
    }
)

_ALL_STOP_WORDS = frozenset(ITALIAN_STOP_WORDS | set(ENGLISH_STOP_WORDS) | _GENERIC_STOP_WORDS)


def detect_language(texts: Iterable[str], threshold: float = 0.06) -> str:
    """
    Return "italian" or "english" from the share of tokens that are Italian stopwords.

    Deliberately crude -- this only picks a stopword list, so a wrong guess costs a few
    noisy terms, never a wrong number. `threshold` is the share of Italian function words
    above which we treat the text as Italian; ordinary Italian prose sits well above it.
    """
    total = 0
    italian_hits = 0
    for text in texts:
        for token in str(text).lower().split():
            token = token.strip(".,;:!?()[]\"'`-–—")
            if not token:
                continue
            total += 1
            if token in ITALIAN_STOP_WORDS:
                italian_hits += 1

    if total == 0:
        return "english"
    return "italian" if (italian_hits / total) >= threshold else "english"


def get_stopwords(
    lang: Optional[str] = None,
    texts: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Stopwords for a vectorizer, as the list sklearn expects.

    - lang="italian"/"english": that language plus generic filler
    - lang="both" or None with no texts: everything (safe default -- a mixed dataset keeps
      both languages' function words out)
    - lang=None with texts: autodetect via detect_language

    Returned as a list, not a set: sklearn accepts a list and it keeps behaviour stable
    across runs.
    """
    normalized = (lang or "").strip().lower()

    if normalized in {"it", "ita", "italian", "italiano"}:
        return sorted(ITALIAN_STOP_WORDS | _GENERIC_STOP_WORDS)
    if normalized in {"en", "eng", "english", "inglese"}:
        return sorted(set(ENGLISH_STOP_WORDS) | _GENERIC_STOP_WORDS)
    if normalized in {"both", "all", "mixed"}:
        return sorted(_ALL_STOP_WORDS)

    if not normalized and texts is not None:
        detected = detect_language(texts)
        return get_stopwords(detected)

    return sorted(_ALL_STOP_WORDS)
