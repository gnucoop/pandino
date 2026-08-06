"""Stopword lists for text analysis on DataChat columns.

scikit-learn ships an English list only, so `stop_words="english"` applied to non-English
answers leaves the function words in place and they dominate the result -- Italian cluster
labels came back as "il, molto, poca, troppa, teoria", where only "teoria" carried meaning.

The four languages here match what the app declares in `datachat/bootstrap_static.py`
(ITA/ENG/FRA/SPA). Kept as plain frozensets rather than pulling in nltk or spacy: a word list
is not worth a dependency, and neither package is installed.
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

# French: articles, elided forms, prepositions, pronouns, auxiliaries, and the common
# intensifiers ("tres", "beaucoup", "assez").
FRENCH_STOP_WORDS = frozenset(
    """
    a à afin ai aie ainsi ait alors apres après as assez au aucun aucune aujourd auquel
    aura aurait aussi autant autre autres aux avaient avais avait avant avec avoir ayant
    beaucoup bien bon
    c ca ça car ce ceci cela celle celles celui cependant certain certaine ces cet cette
    ceux chacun chaque chez ci comme comment
    d dans de dedans dehors deja déjà depuis des desquels deux devant devrait doit donc
    dont du duquel durant
    elle elles en encore entre es est et etaient étaient etais étais etait était etant
    étant ete été etre être eu eux
    fait faire fois font
    grace grâce
    hors
    ici il ils
    j jamais je jusqu jusque
    l la laquelle le lequel les lesquels leur leurs lors lorsque lui
    m ma mais malgre malgré me meme même memes mêmes mes mien moi moins mon
    n ne ni non nos notre nous
    on ont ou où oui
    par parce parmi pas pendant peu peut peuvent plus plutot plutôt pour pourquoi pourtant
    pu puis
    qu quand que quel quelle quelles quels quelque quelques qui quoi
    s sa sans se selon sera serait ses si sien soit son sont sous souvent sur
    t ta tandis tel telle tes toi ton toujours tous tout toute toutes tres très trop tu
    un une
    va vers veut vos votre vous
    y
    """.split()
)

# Spanish: articles, prepositions, pronouns, auxiliaries ("ser"/"estar"/"haber"), and the
# common intensifiers ("muy", "bastante", "poco").
SPANISH_STOP_WORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes aqui aquí asi así aun aún
    aunque
    bastante bien
    cada casi como cómo con contra cual cuales cuando cuanto cuánto cuyo
    de del demas demás desde donde dónde dos durante
    e el ella ellas ello ellos en entre era eran eres es esa esas ese eso esos esta estaba
    estaban estamos estan están estar estas este esto estos estoy
    fue fueron fui
    ha habia había han has hasta hay haya he hemos hizo hubo
    igual incluso
    ja
    la las le les lo los luego
    mas más me mediante mejor menos mi mientras mis mismo misma mucho muchos muy
    nada ni ningun ninguna no nos nosotros nuestra nuestro nunca
    o otra otras otro otros
    para pero poco por porque pues
    que qué quien quién quienes
    se sea segun según ser si sí siempre sido siendo sin sino sobre solo sólo son soy su
    sus
    tal tambien también tampoco tan tanto te tener tengo ti tiene tienen todo toda todos
    todas tras tu tus tuvo
    un una uno unos usted ustedes
    ya yo
    """.split()
)

# Neutral filler that appears in any language's free-text answers, plus the names of HTML
# entities: survey exports carry markup and an unescaped entity name otherwise ranks as the
# most common "word" in the column.
_GENERIC_STOP_WORDS = frozenset(
    {
        "n", "na", "nan", "none", "null", "nd", "ecc", "etc",
        "nbsp", "amp", "quot", "apos", "lt", "gt", "ndash", "mdash",
    }
)

_BY_LANGUAGE: dict[str, frozenset[str]] = {
    "italian": ITALIAN_STOP_WORDS,
    "english": frozenset(ENGLISH_STOP_WORDS),
    "french": FRENCH_STOP_WORDS,
    "spanish": SPANISH_STOP_WORDS,
}

_ALIASES: dict[str, str] = {
    "it": "italian", "ita": "italian", "italian": "italian", "italiano": "italian",
    "en": "english", "eng": "english", "english": "english", "inglese": "english",
    "fr": "french", "fra": "french", "french": "french",
    "français": "french", "francais": "french", "francese": "french",
    "es": "spanish", "spa": "spanish", "spanish": "spanish",
    "español": "spanish", "espanol": "spanish", "spagnolo": "spanish",
}

_ALL_STOP_WORDS = frozenset(
    set().union(*_BY_LANGUAGE.values()) | _GENERIC_STOP_WORDS
)


def _tokenize(texts: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for token in str(text).lower().split():
            token = token.strip(".,;:!?()[]\"'`-–—")
            if token:
                tokens.append(token)
    return tokens


def detect_language(texts: Iterable[str], threshold: float = 0.06) -> str:
    """
    Guess the language from the share of tokens that are its function words.

    Deliberately crude -- this only selects a stopword list, so a wrong guess costs a few noisy
    terms, never a wrong number. Returns "all" when no language is clearly ahead, which keeps
    every language's function words out of a mixed-language column instead of guessing.

    `threshold` is the share of function words a language must reach; ordinary prose sits well
    above it in any of these four.
    """
    tokens = _tokenize(texts)
    if not tokens:
        return "all"

    total = len(tokens)
    scores = {
        language: sum(1 for token in tokens if token in words) / total
        for language, words in _BY_LANGUAGE.items()
    }

    best = max(scores, key=lambda language: scores[language])
    if scores[best] < threshold:
        return "all"

    # Italian, French and Spanish share many short words ("a", "la", "e", "no"), so require a
    # clear margin before committing; otherwise exclude everything.
    runner_up = max((s for lang, s in scores.items() if lang != best), default=0.0)
    if runner_up > 0 and scores[best] < runner_up * 1.25:
        return "all"

    return best


def get_stopwords(
    lang: Optional[str] = None,
    texts: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Stopwords for a vectorizer, as the list sklearn expects.

    - an explicit `lang` (any alias in _ALIASES): that language plus generic filler
    - `lang="all"`/"both"/"mixed", or nothing to go on: every language -- the safe default,
      since a mixed-language column keeps all of their function words out
    - `lang=None` with `texts`: autodetect via detect_language

    Returned sorted as a list, not a set: sklearn accepts a list and it keeps behaviour stable
    across runs.
    """
    normalized = (lang or "").strip().lower()

    if normalized in _ALIASES:
        return sorted(_BY_LANGUAGE[_ALIASES[normalized]] | _GENERIC_STOP_WORDS)
    if normalized in {"all", "both", "mixed"}:
        return sorted(_ALL_STOP_WORDS)

    if not normalized and texts is not None:
        return get_stopwords(detect_language(texts))

    return sorted(_ALL_STOP_WORDS)
