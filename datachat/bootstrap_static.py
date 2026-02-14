from typing import Final


_LANG_TO_TEMPLATE_KEY: Final[dict[str, str]] = {
    "ITA": "it",
    "ENG": "en",
    "FRA": "fr",
    "SPA": "es",
}


_BOOTSTRAP_TEMPLATES: Final[dict[str, str]] = {
    "it": (
        "<h2>Benvenuto in Chat with your data</h2>"
        "<p>Qui puoi interrogare il tuo file in linguaggio naturale e ottenere risposte utili, tabelle e grafici.</p>"
        "<h3>Cosa puoi fare</h3>"
        "<ul>"
        "<li>Contare record, filtrare righe e confrontare gruppi.</li>"
        "<li>Calcolare totali, medie, minimi, massimi e altre aggregazioni.</li>"
        "<li>Analizzare valori mancanti, categorie e relazioni tra colonne.</li>"
        "<li>Richiedere visualizzazioni e analisi di trend temporali.</li>"
        "</ul>"
        "<h3>Esempi di domande</h3>"
        "<ul>"
        "<li>Quanti record ci sono in totale?</li>"
        "<li>Mostrami i dati dove la colonna X è uguale a Y.</li>"
        "<li>Qual è la media di vendite per regione?</li>"
        "<li>Crea un grafico dell'andamento mensile.</li>"
        "</ul>"
        "<h3>Buone pratiche</h3>"
        "<ul>"
        "<li>Indica sempre colonne, filtri e periodo temporale quando possibile.</li>"
        "<li>Fai una domanda alla volta per ottenere risposte più precise.</li>"
        "</ul>"
    ),
    "en": (
        "<h2>Welcome to Chat with your data</h2>"
        "<p>You can explore your file in natural language and get useful answers, tables, and charts.</p>"
        "<h3>What you can do</h3>"
        "<ul>"
        "<li>Count records, filter rows, and compare groups.</li>"
        "<li>Compute totals, averages, minimums, maximums, and other aggregations.</li>"
        "<li>Inspect missing values, categories, and relationships across columns.</li>"
        "<li>Request visualizations and time-based trend analysis.</li>"
        "</ul>"
        "<h3>Example questions</h3>"
        "<ul>"
        "<li>How many records are in this dataset?</li>"
        "<li>Show rows where column X equals Y.</li>"
        "<li>What is the average sales value by region?</li>"
        "<li>Create a chart of the monthly trend.</li>"
        "</ul>"
        "<h3>Good practices</h3>"
        "<ul>"
        "<li>Whenever possible, specify columns, filters, and time range.</li>"
        "<li>Ask one question at a time for more precise answers.</li>"
        "</ul>"
    ),
    "fr": (
        "<h2>Bienvenue dans Chat with your data</h2>"
        "<p>Vous pouvez explorer votre fichier en langage naturel et obtenir des réponses utiles, des tableaux et des graphiques.</p>"
        "<h3>Ce que vous pouvez faire</h3>"
        "<ul>"
        "<li>Compter les lignes, filtrer les données et comparer des groupes.</li>"
        "<li>Calculer des totaux, moyennes, minimums, maximums et autres agrégations.</li>"
        "<li>Analyser les valeurs manquantes, les catégories et les relations entre colonnes.</li>"
        "<li>Demander des visualisations et des analyses de tendance dans le temps.</li>"
        "</ul>"
        "<h3>Exemples de questions</h3>"
        "<ul>"
        "<li>Combien de lignes contient ce jeu de données ?</li>"
        "<li>Montre les lignes où la colonne X est égale à Y.</li>"
        "<li>Quelle est la moyenne des ventes par région ?</li>"
        "<li>Crée un graphique de la tendance mensuelle.</li>"
        "</ul>"
        "<h3>Bonnes pratiques</h3>"
        "<ul>"
        "<li>Si possible, précisez les colonnes, filtres et période temporelle.</li>"
        "<li>Posez une seule question à la fois pour des réponses plus précises.</li>"
        "</ul>"
    ),
    "es": (
        "<h2>Bienvenido a Chat with your data</h2>"
        "<p>Aqui puedes explorar tu archivo en lenguaje natural y obtener respuestas utiles, tablas y graficos.</p>"
        "<h3>Que puedes hacer</h3>"
        "<ul>"
        "<li>Contar registros, filtrar filas y comparar grupos.</li>"
        "<li>Calcular totales, promedios, minimos, maximos y otras agregaciones.</li>"
        "<li>Revisar valores faltantes, categorias y relaciones entre columnas.</li>"
        "<li>Solicitar visualizaciones y analisis de tendencias en el tiempo.</li>"
        "</ul>"
        "<h3>Ejemplos de preguntas</h3>"
        "<ul>"
        "<li>Cuantos registros hay en este dataset?</li>"
        "<li>Muestrame las filas donde la columna X es igual a Y.</li>"
        "<li>Cual es el promedio de ventas por region?</li>"
        "<li>Crea un grafico de la tendencia mensual.</li>"
        "</ul>"
        "<h3>Buenas practicas</h3>"
        "<ul>"
        "<li>Cuando sea posible, especifica columnas, filtros y periodo de tiempo.</li>"
        "<li>Haz una pregunta a la vez para obtener respuestas mas precisas.</li>"
        "</ul>"
    ),
}


def normalize_lang_code(lang: str | None) -> str:
    code = str(lang or "").strip().upper()
    if code in _LANG_TO_TEMPLATE_KEY:
        return code
    return "ENG"


def get_static_bootstrap_html(lang: str | None) -> str:
    normalized_lang = normalize_lang_code(lang)
    template_key = _LANG_TO_TEMPLATE_KEY.get(normalized_lang, "en")
    return _BOOTSTRAP_TEMPLATES[template_key]
