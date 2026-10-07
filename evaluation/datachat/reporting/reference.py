"""Evaluation Reference of the operator report (reporting design §6.5).

``load_evaluation_reference(repo_root, run_contract)`` reads the pinned
``ground_truth.json`` and projects, per scenario, the reference facts a
human reviewer needs next to the agent's answer:

    pinned ground_truth.json (hash-verified) → {scenario_id: EvaluationReference}

Binding is explicit and deterministic: one small projection per scenario
below selects values from the structured ground truth and frames them in
Italian; the expectation lines restate the pinned design (§9, FE-01…FE-14).
No prose is parsed and nothing is compared with the agent's answer: a
reference is material for later semantic review, never a verdict.

The ground truth on disk must equal the one recorded in ``run_grade.json``
(``contract.ground_truth_sha256`` and ``contract.artifacts``), and the
recorded design document must equal the pinned one the expectations were
written from; otherwise ``EvaluationReferenceError`` refuses the report.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evaluation.datachat import contract
from evaluation.datachat.evidence import sha256_file


class EvaluationReferenceError(Exception):
    """The pinned evaluation reference cannot be trusted for this run."""


@dataclass(frozen=True)
class ReferenceTable:
    header: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class EvaluationReference:
    """Reference material for one scenario. Values are verbatim ground-truth values."""

    scenario_id: str
    guarantees: tuple[tuple[str, str, str], ...]  # (id, title, ground_truth_type), verbatim
    facts: tuple[str, ...]
    tables: tuple[ReferenceTable, ...]
    expectations: tuple[str, ...]  # from the pinned design, for the reviewer


def _v(value: Any) -> str:
    """A ground-truth value exactly as recorded (JSON scalar form)."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _table(header, rows) -> ReferenceTable:
    return ReferenceTable(tuple(header), tuple(tuple(r) for r in rows))


def _ref(gt, sid, guarantees, facts=(), tables=(), expectations=()) -> EvaluationReference:
    g = gt["guarantees"]
    return EvaluationReference(sid, tuple((k, g[k]["title"], g[k]["ground_truth_type"]) for k in guarantees),
                               tuple(facts), tuple(tables), tuple(expectations))


def _top(mapping: dict) -> str:
    return max(mapping, key=mapping.get)


def _fe01(gt):
    counts = gt["categorical_counts"]["region"]
    return _ref(gt, "FE-01", ("G2",),
                [f"Regione con più richieste: {_top(counts)} ({_v(max(counts.values()))}); "
                 f"regione con meno richieste: {min(counts, key=counts.get)} ({_v(min(counts.values()))})."],
                [_table(("region", "richieste"), counts.items())],
                ["Conteggi esatti per tutte e cinque le regioni.",
                 "Un grafico categoriale (preferibilmente a barre; torta/ciambella ammesse)."])


def _fe02(gt):
    ev = gt["guarantees"]["G4"]["evidence"]
    top, second, overall = ev["south_top"], ev["south_runner_up"], ev["overall_top"]
    return _ref(gt, "FE-02", ("G4",),
                [f"Nel Sud il tempo medio di attesa più alto è di {top['service']} ({_v(top['mean'])} giorni, "
                 f"n={_v(top['n'])}), seguito da {second['service']} ({_v(second['mean'])}).",
                 f"Senza il filtro sul Sud il primo servizio sarebbe {overall['service']} ({_v(overall['mean'])}).",
                 f"Il gruppo {top['service']} nel Sud ha {_v(top['n'])} righe (sotto la soglia di 15 per un "
                 "campione affidabile)."],
                [_table(("service", "mean_wait_days (Sud)", "n"),
                        ((r["service"], r["mean"], r["n"]) for r in gt["aggregates"]["mean_wait_days_by_service_in_south"]))],
                ["Analisi limitata al Sud.", "Confronto che copre tutti e quattro i servizi (preferibilmente un grafico).",
                 "Riconoscimento della cautela sul gruppo piccolo."])


def _fe03(gt):
    ev = gt["guarantees"]["G5"]["evidence"]
    return _ref(gt, "FE-03", ("G5",),
                [f"Costo totale più alto: {ev['highest_sum']}.", f"Costo medio più alto: {ev['highest_mean']}."],
                [_table(("service", "n", "sum_case_cost_eur", "mean_case_cost_eur"),
                        ((r["service"], r["n"], r["sum_case_cost_eur"], r["mean_case_cost_eur"])
                         for r in gt["aggregates"]["service"]))],
                ["Distinzione esplicita tra costo totale e costo medio.", "Valori approssimativi (facoltativi)."])


def _fe04(gt):
    m = gt["missingness"]
    c = m["comment"]
    return _ref(gt, "FE-04", ("G10",),
                [f"Commenti: {_v(c['missing'])} mancanti (NA), {_v(c['blank'])} composti solo da spazi "
                 f"(non NA per pandas), {_v(c['text'])} con testo."],
                [_table(("colonna", "valori NA"), m["per_column_na"].items())],
                ["wait_days e comment con i conteggi NA indicati; tutte le altre colonne a 0.",
                 "Segnalare separatamente i commenti vuoti è facoltativo."])


def _correlation(gt, sid, key, guarantee, expectations, extra=()):
    c = gt["correlations"][key]
    return _ref(gt, sid, (guarantee,),
                [f"{c['x']} vs {c['y']}: Spearman {_v(c['spearman'])}, Pearson {_v(c['pearson'])}, n={_v(c['n'])}.",
                 *extra], (), expectations)


def _fe05(gt):
    lo, hi = gt["guarantees"]["G6"]["evidence"]["range"]
    return _correlation(gt, "FE-05", "wait_days_vs_satisfaction", "G6",
                        ["Associazione negativa moderata: attese più lunghe, soddisfazione più bassa.",
                         "Nessuna affermazione causale.",
                         "Spearman preferibile (soddisfazione ordinale 1–5); n delle osservazioni valide come contesto."],
                        [f"Intervallo di riferimento per Spearman: [{_v(lo)}, {_v(hi)}]."])


def _fe06(gt):
    bound = gt["guarantees"]["G7"]["evidence"]["bound"]
    return _correlation(gt, "FE-06", "case_cost_eur_vs_satisfaction", "G7",
                        ["Coefficiente vicino a zero; relazione descritta come trascurabile / assente.",
                         "Nessuna affermazione causale; citare il coefficiente è preferibile."],
                        [f"Limite di riferimento: |Pearson| e |Spearman| < {_v(bound)}."])


def _comparison(gt, sid, key, expectations):
    c = gt["group_comparisons"][key]
    fields = ("metric", "group_a", "group_b", "n_a", "n_b", "mean_a", "mean_b", "difference", "t_statistic", "p_value")
    return _ref(gt, sid, ("G8",), (), [_table(("campo", "valore"), ((f, c[f]) for f in fields))], expectations)


def _fe07(gt):
    return _comparison(gt, "FE-07", "online_vs_phone",
                       ["Soddisfazione Online più alta di Phone, con differenza vicina a quella di riferimento.",
                        "Nessuna affermazione di significatività senza un test; p < 0.001 se si usa linguaggio inferenziale."])


def _fe08(gt):
    return _comparison(gt, "FE-08", "in_person_vs_phone",
                       ["Piccola differenza osservata; nessuna evidenza di una differenza statisticamente rilevante.",
                        "Non concludere che i gruppi siano uguali o che sia dimostrato che non differiscono."])


def _monthly(gt, column):
    return _table(("month", column), ((r["month"], r[column]) for r in gt["temporal"]["monthly"]))


def _fe09(gt):
    ev = gt["guarantees"]["G9"]["evidence"]
    return _ref(gt, "FE-09", ("G9",),
                [f"Richieste mensili da {_v(ev['first_month'])} a {_v(ev['last_month'])}; "
                 f"{_v(ev['non_decreasing_transitions'])} transizioni mensili non decrescenti su 11."],
                [_monthly(gt, "requests")],
                ["Andamento mensile crescente.", "Grafico temporale (preferibilmente a linee; barre mensili ammesse)."])


def _fe10(gt):
    t = gt["temporal"]
    return _ref(gt, "FE-10", ("G9",),
                [f"Attesa media gen–giu {_v(t['mean_wait_days_jan_jun'])} giorni, lug–dic "
                 f"{_v(t['mean_wait_days_jul_dec'])} giorni (salto {_v(t['step_jul_dec_minus_jan_jun'])})."],
                [_monthly(gt, "mean_wait_days")],
                ["Salto verso l'alto intorno a luglio, di circa +4 / +6 giorni.",
                 "Nessun grafico richiesto."])


def _fe11(gt):
    w = gt["distribution"]["wait_days"]
    return _ref(gt, "FE-11", ("G12",),
                [f"wait_days asimmetrica a destra: skewness {_v(w['skewness'])} ({w['skewness_method']}).",
                 f"{_v(w['distinct_non_null'])} valori distinti non nulli; min {_v(w['min'])}, mediana "
                 f"{_v(w['median'])}, max {_v(w['max'])}."],
                (),
                ["La risposta deve consegnare un'immagine della distribuzione (requisito di consegna).",
                 "Preferibilmente un istogramma (box plot / KDE ammessi); non un grafico a barre per singolo valore.",
                 "L'adeguatezza del grafico si valuta sull'immagine."])


def _fe12(gt):
    o = gt["outliers"]
    services = sorted({r["service"] for r in o["rows"]})
    return _ref(gt, "FE-12", ("G11",),
                [f"Regola: {o['column']} {o['rule']} (mediana {_v(o['median'])}, soglia {_v(o['threshold'])}); "
                 f"valore massimo non anomalo {_v(o['max_non_outlier'])}.",
                 f"{len(o['rows'])} casi anomali nei servizi: {', '.join(services)}."],
                [_table(("record_id", "service", "region", "case_cost_eur"),
                        ((r["record_id"], r["service"], r["region"], r["case_cost_eur"]) for r in o["rows"]))],
                ["Riconoscere un piccolo numero di casi estremi (circa 1.060–1.200 €) e i servizi coinvolti.",
                 "Preferibilmente esattamente quattro casi, con record_id o valori."])


def _fe13(gt):
    shares = gt["semantic"]["negative_share_by_region"]
    c = gt["missingness"]["comment"]
    return _ref(gt, "FE-13", ("G13",),
                [f"Quota di commenti negativi più alta: {_top(shares)} ({_v(shares[_top(shares)])}).",
                 f"Margine sulla seconda regione: {_v(gt['guarantees']['G13']['evidence']['margin'])}.",
                 f"Denominatore: commenti con testo; {_v(c['missing'])} commenti mancanti e {_v(c['blank'])} vuoti "
                 "non hanno sentiment (mai neutri).",
                 "Le etichette 'mixed' sono metadati di progettazione, non output atteso del provider."],
                [_table(("region", "comments_with_text", "negative (progettati)", "negative_share"),
                        ((r["region"], r["comments_with_text"], r["sentiment_counts"].get("negative", 0),
                          r["negative_share"]) for r in gt["aggregates"]["region"]))],
                ["Sentiment calcolato con il tool, non stimato.",
                 "Sud chiaramente più alto (circa 45–65% dei commenti analizzati); dichiarare il denominatore.",
                 "Commenti nulli/vuoti non trattati come neutri; confronto tra regioni (preferibilmente un grafico).",
                 "Central (pochi commenti) non presentata senza cautela come la regione migliore."])


def _fe14(gt):
    s, c = gt["semantic"], gt["missingness"]["comment"]
    ev = gt["guarantees"]["G14"]["evidence"]
    return _ref(gt, "FE-14", ("G14",),
                [f"Temi progettati: {', '.join(s['taxonomy'])} (più 'none', quota {_v(ev['theme_none_share'])}).",
                 f"Prime 5 parole chiave: {', '.join(ev['keywords_top5'])}.",
                 f"Commenti senza testo: {_v(c['missing'])} mancanti, {_v(c['blank'])} vuoti."],
                [_table(("tema progettato", "commenti"), s["row_theme_counts"].items())],
                ["Categorie proposte dall'agente e presentate come proposta, non come tassonomia data.",
                 "Classificazione con classify_match e conteggi per categoria; keywords preferibilmente prima.",
                 "Commenti non classificati e non disponibili tenuti distinti dai temi.",
                 "Copertura semantica di almeno 4 dei 5 temi progettati; attesa tra i più numerosi."])


_PROJECTIONS: dict[str, Callable[[dict], EvaluationReference]] = {
    "FE-01": _fe01, "FE-02": _fe02, "FE-03": _fe03, "FE-04": _fe04, "FE-05": _fe05, "FE-06": _fe06,
    "FE-07": _fe07, "FE-08": _fe08, "FE-09": _fe09, "FE-10": _fe10, "FE-11": _fe11, "FE-12": _fe12,
    "FE-13": _fe13, "FE-14": _fe14,
}


def project_reference(ground_truth: dict, scenario_ids) -> dict[str, EvaluationReference]:
    """Reference for each known scenario id (unknown ids get none). Pure."""
    try:
        return {sid: _PROJECTIONS[sid](ground_truth) for sid in scenario_ids if sid in _PROJECTIONS}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise EvaluationReferenceError(f"ground_truth.json does not fit the reference projection: {exc!r}") from exc


def load_evaluation_reference(repo_root: Path, run_contract: dict, scenario_ids) -> dict[str, EvaluationReference]:
    """Verify the pinned reference against the run's recorded contract, then project it."""
    directory = contract.evaluation_dir(repo_root)
    recorded = run_contract.get("artifacts") or {}
    path = directory / contract.GROUND_TRUTH_JSON_NAME
    if not path.is_file():
        raise EvaluationReferenceError("ground_truth.json missing")
    actual = sha256_file(path)
    expected = {run_contract.get("ground_truth_sha256"), recorded.get(contract.GROUND_TRUTH_JSON_NAME)}
    if expected != {actual}:
        raise EvaluationReferenceError("ground_truth.json differs from the hash recorded in run_grade.json")
    design = contract.pinned_artifact(contract.DESIGN_DOC_NAME).sha256
    if recorded.get(contract.DESIGN_DOC_NAME) != design:
        raise EvaluationReferenceError("the run was graded against a design document other than the pinned one")
    try:
        ground_truth = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationReferenceError(f"ground_truth.json unreadable: {type(exc).__name__}") from exc
    return project_reference(ground_truth, scenario_ids)
