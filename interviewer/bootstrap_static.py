from typing import Final

from datachat.bootstrap_static import normalize_lang_code

_LANG_TO_TEMPLATE_KEY: Final[dict[str, str]] = {
    "ITA": "it",
    "ENG": "en",
    "FRA": "fr",
    "SPA": "es",
}


# The fixed first question of every interview, as plain text. The bootstrap
# message ends with it and the user's first message answers it.
_OPENING_QUESTIONS: Final[dict[str, str]] = {
    "it": "Descrivi in poche parole cosa vorresti analizzare e perché.",
    "en": "Describe in a few words what you would like to analyse and why.",
    "fr": "Décrivez en quelques mots ce que vous souhaitez analyser et pourquoi.",
    "es": "Describe en pocas palabras que te gustaria analizar y por que.",
}


# The bootstrap message without its closing question, appended by
# get_static_bootstrap_html().
_BOOTSTRAP_TEMPLATES: Final[dict[str, str]] = {
    "it": (
        "<h2>Progettiamo la tua analisi</h2>"
        "<p>Ti farò alcune domande sull'obiettivo dell'analisi e sui dati che ti interessano. "
        "Nel frattempo esplorerò il database per proporti domande concrete.</p>"
        "<p>Alla fine ti proporrò un brief di analisi da approvare: se non ti convince, "
        "lo perfezioneremo insieme.</p>"
        "<h3>Per iniziare</h3>"
    ),
    "en": (
        "<h2>Let's design your analysis</h2>"
        "<p>I will ask you a few questions about the goal of the analysis and the data you care about. "
        "Meanwhile I will explore the database so that my questions are concrete.</p>"
        "<p>At the end I will propose an analysis brief for your approval: if it is not right, "
        "we will refine it together.</p>"
        "<h3>To start</h3>"
    ),
    "fr": (
        "<h2>Concevons votre analyse</h2>"
        "<p>Je vais vous poser quelques questions sur l'objectif de l'analyse et sur les données qui vous intéressent. "
        "Entre-temps, j'explorerai la base de données pour vous poser des questions concrètes.</p>"
        "<p>À la fin, je vous proposerai un brief d'analyse à approuver : s'il ne vous convient pas, "
        "nous l'affinerons ensemble.</p>"
        "<h3>Pour commencer</h3>"
    ),
    "es": (
        "<h2>Disenemos tu analisis</h2>"
        "<p>Te hare algunas preguntas sobre el objetivo del analisis y los datos que te interesan. "
        "Mientras tanto explorare la base de datos para que mis preguntas sean concretas.</p>"
        "<p>Al final te propondre un brief de analisis para aprobar: si no te convence, "
        "lo perfeccionaremos juntos.</p>"
        "<h3>Para empezar</h3>"
    ),
}


def _template_key(lang: str | None) -> str:
    return _LANG_TO_TEMPLATE_KEY.get(normalize_lang_code(lang), "en")


def get_static_bootstrap_html(lang: str | None) -> str:
    key = _template_key(lang)
    return f"{_BOOTSTRAP_TEMPLATES[key]}<p>{_OPENING_QUESTIONS[key]}</p>"


def get_opening_question(lang: str | None) -> str:
    return _OPENING_QUESTIONS[_template_key(lang)]
