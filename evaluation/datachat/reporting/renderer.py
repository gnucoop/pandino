"""Italian Markdown renderer of the run-centric operator report (reporting design §3, §5–§6).

``render_markdown(report, drawn_charts)`` turns an ``OperatorRunReport`` into
the text of ``<run>/reports/report.md``. ``drawn_charts`` names the chart
files under ``reports/charts/`` that were drawn (a spec that cannot be drawn
faithfully is shown as data only).

Pure: no I/O, no clock, no environment. Every label is a one-to-one Italian
display of a persisted value, shown next to the machine value. The agent's
response is shown verbatim; the Evaluation Reference is shown next to it and
never compared with it. Links are relative to ``reports/report.md``.
"""

import json
from typing import Any, Iterable

from evaluation.datachat.grading.model import (
    ActivitySource,
    CheckResult,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ObservedActivity,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.model import Availability
from evaluation.datachat.reporting.charts import CHART_DIR
from evaluation.datachat.reporting.operator_report import (
    OperatorRunReport,
    ResultShape,
    ScenarioReport,
)
from evaluation.datachat.reporting.reference import EvaluationReference

REPORT_DIR = "reports"
REPORT_NAME = "report.md"

SECTIONS = ("Domanda", "Risposta dell'agente", "Attività osservata dell'agente", "Artefatti",
            "Validazione tecnica", "Riferimento di valutazione", "Provenienza tecnica")

# Run-relative path → path relative to reports/report.md.
_UP = "../"

# --- One-to-one display labels (machine values stay visible next to them) ----------------

OUTCOME_LABELS = {
    TechnicalOutcome.TECHNICAL_PASS: "SUPERATO",
    TechnicalOutcome.FAIL: "FALLITO",
    TechnicalOutcome.INDETERMINATE: "INDETERMINATO",
}
FAILURE_LABELS = {
    TechnicalFailure.EXECUTION: "ESECUZIONE",
    TechnicalFailure.RESPONSE_CONTRACT: "CONTRATTO DI RISPOSTA",
    TechnicalFailure.TOOL_EXECUTION: "ESECUZIONE TOOL",
    TechnicalFailure.ARTIFACT: "ARTEFATTO",
}
GATE_LABELS = {
    Gate.GRADEABLE: "valutabile",
    Gate.NOT_RUN: "non eseguito",
    Gate.NOT_SCOREABLE_INFRA: "non valutabile per problemi di infrastruttura",
    Gate.NOT_SCOREABLE: "non valutabile",
}
ASSESSMENT_LABELS = {
    ExecutionAssessment.OK: "esito utilizzabile",
    ExecutionAssessment.FAILED: "esecuzione fallita",
    ExecutionAssessment.RESPONSE_CONTRACT_VIOLATION: "violazione del contratto di risposta",
    ExecutionAssessment.INDETERMINATE: "indeterminata",
}
_R = ExecutionAssessmentReason
REASON_LABELS = {
    _R.USABLE_OUTCOME: "la risposta finale è utilizzabile",
    _R.ERROR_OUTCOME: "la risposta finale è un errore",
    _R.HTTP_5XX: "/datachat ha restituito un errore HTTP 5xx",
    _R.EMPTY_OUTCOME: "la risposta finale è vuota",
    _R.MISSING_RESPONSE: "nessuna risposta registrata",
    _R.CONFLICTING_KINDS: "tipi di risposta in conflitto tra le fonti",
    _R.HTTP_NON_2XX: "/datachat ha restituito uno stato HTTP non 2xx",
    _R.UNPARSEABLE_RESPONSE: "corpo della risposta non interpretabile",
    _R.RUNTIME_STATUS_CONFLICT: "stato runtime in conflitto con la risposta",
    _R.ADAPTER_FALLBACK: "risposta prodotta dal fallback dell'adattatore",
}
CHECK_STATUS_LABELS = {
    CheckStatus.MET: "soddisfatto",
    CheckStatus.NOT_MET: "non soddisfatto",
    CheckStatus.INDETERMINATE: "indeterminato",
    CheckStatus.NOT_APPLICABLE: "non applicabile",
}
EVIDENCE_CLASS_LABELS = {EvidenceClass.DIRECT: "diretta", EvidenceClass.INFERRED: "inferita"}
_TABLE_META_LABELS = {
    "result_rows": "righe del risultato", "result_columns": "colonne del risultato",
    "preview_rows": "righe in anteprima", "preview_columns": "colonne in anteprima",
    "truncated": "anteprima troncata",
}
_ACTIVITY_GROUPS = {
    ActivitySource.RUNTIME: "Eventi runtime (esecuzione osservata)",
    ActivitySource.GENERATED_CODE: "Codice generato (evidenza inferita, non prova di esecuzione)",
    ActivitySource.USAGE: "Dati Usage (lavoro aggiuntivo presso il provider)",
}


# --- Primitives --------------------------------------------------------------------------


def _value(value: Any) -> str:
    """A persisted value as text: strings as-is, enums by value, everything else as compact JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return json.dumps(value, ensure_ascii=False)


def _cell(value: Any) -> str:
    return _value(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def _table(header: Iterable[Any], rows: Iterable[Iterable[Any]]) -> list[str]:
    header = list(header)
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return lines + [""]


def _fence(text: str, info: str = "text") -> list[str]:
    """Verbatim block: the fence is longer than any backtick run inside ``text``."""
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{info}", text, fence, ""]


def _code(value: Any) -> str:
    text = _value(value)
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _yes(value: bool | None) -> str:
    return "sconosciuto" if value is None else ("sì" if value else "no")


def _link(run_relative: str) -> str:
    return f"[`{run_relative}`]({_UP}{run_relative})"


def _field(label: str, value: Any) -> str:
    return f"- **{label}:** {_value(value)}"


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def outcome_display(outcome: TechnicalOutcome | None, failures: tuple[TechnicalFailure, ...]) -> str:
    """``SUPERATO (`TECHNICAL PASS`)`` / ``FALLITO (`FAIL`) — ESECUZIONE TOOL (`TOOL EXECUTION`), ...``."""
    if outcome is None:
        return "nessun esito tecnico"
    text = f"{OUTCOME_LABELS[outcome]} (`{outcome.value}`)"
    if failures:
        text += " — " + ", ".join(f"{FAILURE_LABELS[f]} (`{f.value}`)" for f in failures)
    return text


def _gate(gate: Gate) -> str:
    return f"{GATE_LABELS[gate]} (`{gate.value}`)"


# --- Run ---------------------------------------------------------------------------------


def _overview(r: OperatorRunReport) -> list[str]:
    maui, dc, grader = r.maui, r.datachat, r.grader
    out = ["## Esecuzione", "",
           _field("Run", f"`{r.run_id}`"),
           _field("Tipo di esecuzione", "ufficiale" if r.official else "esplorativa (non ufficiale)"),
           _field("Confrontabile", _yes(r.comparable)),
           _field("Stato dell'esecuzione", f"`{r.status.value}`")]
    if r.failure is not None:
        f = r.failure
        out.append(_field("Errore dell'esecuzione", f"`{f.code.value}` — {f.summary}"
                                                    + (f" (fase `{f.phase.value}`)" if f.phase else "")))
    out += [_field("Maui", f"git `{_value(maui.get('git_sha'))}`, branch `{_value(maui.get('branch'))}`"
                           + (" (working tree modificato)" if maui.get("dirty") else "")),
            _field("DataChat", f"provider `{_value(dc.get('DATACHAT_PROVIDER'))}`, modello "
                               f"`{_value(dc.get('DATACHAT_MODEL'))}`, engine `{_value(dc.get('DATACHAT_ENGINE'))}`"),
            _field("Grading", f"`{_value(grader.get('implementation'))}` versione `{_value(grader.get('version'))}`, "
                              f"schema `{r.grading_schema_version}`, git `{_value(grader.get('git_sha'))}`"
                              + (" (working tree modificato)" if grader.get("dirty") else "")),
            "",
            "Il grading automatico valida solo il comportamento tecnico e funzionale. Un esito SUPERATO non "
            "afferma la correttezza semantica della risposta; il riferimento di valutazione di ogni scenario "
            "serve alla revisione semantica successiva e non è confrontato automaticamente. Non esiste un esito "
            "o un punteggio complessivo dell'esecuzione.", ""]
    out += ["### Esiti tecnici", ""]
    out += _table(("Esito tecnico", "Valore", "Scenari"),
                  ((OUTCOME_LABELS[o], f"`{o.value}`", n) for o, n in r.outcome_counts))
    nonzero = [(f, n) for f, n in r.failure_counts if n]
    if nonzero:
        out += ["Fallimenti tecnici (uno scenario FALLITO può contarne più di uno):", ""]
        out += _table(("Fallimento", "Valore", "Scenari"), ((FAILURE_LABELS[f], f"`{f.value}`", n) for f, n in nonzero))
    out += ["Scenari non valutabili (gate; non sono fallimenti):", ""]
    out += _table(("Gate", "Valore", "Scenari"), ((GATE_LABELS[g], f"`{g.value}`", n) for g, n in r.gate_counts))
    rows = [(f"[{e.scenario_id}](#{e.scenario_id.lower()})", _gate(e.gate), outcome_display(e.outcome, e.failures))
            for e in r.inventory]
    out += ["### Scenari", ""] + _table(("Scenario", "Gate", "Esito tecnico"), rows)
    return out


def _run_provenance(r: OperatorRunReport) -> list[str]:
    contract = dict(r.contract)
    artifacts = contract.pop("artifacts", None)
    executions = contract.pop("scenario_execution_sha256", None)
    out = ["## Provenienza tecnica dell'esecuzione", "",
           _field("SHA-256 di `grading/results/run_grade.json`", f"`{r.run_grade_sha256}`"),
           _field("Inizio / fine dell'esecuzione", f"{_value(r.started_at)} / {_value(r.ended_at)}"), "",
           "**Grader**", ""]
    out += _table(("Campo", "Valore"), r.grader.items())
    out += ["**Maui**", ""] + _table(("Campo", "Valore"), r.maui.items())
    out += ["**Contratto di valutazione**", ""] + _table(("Campo", "Valore"), contract.items())
    if isinstance(artifacts, dict):
        out += _table(("Artefatto del contratto", "SHA-256"), artifacts.items())
    if isinstance(executions, dict):
        out += _table(("Scenario", "SHA-256 di execution.json"), executions.items())
    return out


# --- 1 Domanda ---------------------------------------------------------------------------


def _question(s: ScenarioReport) -> list[str]:
    out = [f"### {SECTIONS[0]}", ""]
    if s.question is None:
        return out + ["_Domanda non inviata: il pacchetto dello scenario non è stato scritto._", ""]
    if not s.question_sent:
        out += ["_Domanda non inviata a DataChat._", ""]
    return out + _fence(s.question)


# --- 2 Risposta dell'agente --------------------------------------------------------------


def _data_table(rows: Any) -> list[str]:
    if not (isinstance(rows, list) and all(isinstance(x, dict) for x in rows)):
        return _fence(json.dumps(rows, ensure_ascii=False, indent=2), "json")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    if not columns:
        return ["_La tabella restituita non ha colonne._", ""]
    return _table(columns, ([row.get(c) for c in columns] for row in rows))


def _response(s: ScenarioReport) -> list[str]:
    a = s.agent_result
    out = [f"### {SECTIONS[1]}", ""]
    if a.shape is ResultShape.ABSENT:
        return out + ["_Nessuna risposta: nessuna richiesta a `/datachat` registrata per questo scenario._", ""]
    src = a.source
    if a.shape is ResultShape.NO_RESPONSE:
        out += ["_Nessuna risposta di `/datachat` è stata registrata._", ""]
        if src.transport_error is not None:
            out.append(_field("Errore di trasporto", f"`{src.transport_error.type}`: {src.transport_error.message}"))
        if src.availability is not None:
            out.append(_field("Disponibilità dell'evidenza", f"`{src.availability.value}`"))
        if src.reason:
            out.append(_field("Motivo registrato", src.reason))
        return out + [_field("Tentativi registrati", src.attempts), ""]
    c = src.content
    if a.shape is ResultShape.TEXT:
        out += ["Testo restituito (verbatim):", ""] + _fence(c.value)
    elif a.shape is ResultShape.TABLE:
        out += ["Tabella restituita (verbatim):", ""] + _data_table(c.value)
        meta = [_field(_TABLE_META_LABELS.get(k, k) + f" (`{k}`)", _value(v)) for k, v in c.table_meta.items()]
        if c.more_rows_available is not None:
            meta.append(_field("altre righe disponibili (`more_rows_available`)", _value(c.more_rows_available)))
        out += meta + ([""] if meta else [])
    elif a.shape is ResultShape.IMAGE:
        out += ["_La risposta è un'immagine: vedi **Artefatti**._", ""]
    elif a.shape is ResultShape.STRUCTURED:
        out += ["Valore strutturato restituito (verbatim):", ""]
        out += _fence(json.dumps(c.value, ensure_ascii=False, indent=2), "json")
    else:  # ERROR_BODY / RAW: verbatim body, not interpreted
        kind = "Corpo di errore" if a.shape is ResultShape.ERROR_BODY else f"Corpo grezzo (`{c.parse_status.value}`)"
        out += [f"{kind} — HTTP {_value(src.http_status)}, content type `{_value(src.content_type)}`:", ""]
        out += _fence(src.body.decode("utf-8", errors="replace"))
    if c.note is not None:
        out += ["Nota restituita (`note`):", ""] + _fence(_value(c.note))
    if c.explanation is not None:
        out += ["Spiegazione restituita (`explanation`):", ""] + _fence(_value(c.explanation))
    return out


# --- 3 Attività osservata ----------------------------------------------------------------


def activity_line(a: ObservedActivity) -> str:
    """Source-aware wording: never claims more than the source supports."""
    n = a.count
    if a.source is ActivitySource.RUNTIME:
        return (f"`{a.name}` — {n} {_plural(n, 'chiamata osservata', 'chiamate osservate')}, "
                f"esito: `{a.outcome}`")
    if a.source is ActivitySource.GENERATED_CODE:
        return (f"`{a.name}` — {n} {_plural(n, 'chiamata individuata', 'chiamate individuate')} nel codice "
                "generato (evidenza inferita)")
    name = "" if a.name == "provider_backed_call" else f" (`{a.name}`)"
    return (f"{n} {_plural(n, 'chiamata aggiuntiva', 'chiamate aggiuntive')} al provider "
            f"{_plural(n, 'osservata', 'osservate')} nei dati Usage{name}; il tool specifico non è "
            "identificabile da questa fonte")


def _activity(s: ScenarioReport) -> list[str]:
    out = [f"### {SECTIONS[2]}", ""]
    if not s.activity:
        if s.validation.gate is not Gate.GRADEABLE:
            return out + ["_Nessuna attività registrata: lo scenario non è valutabile._", ""]
        return out + ["_Nessuna attività di tool o capability registrata per questo scenario._", ""]
    for source, title in _ACTIVITY_GROUPS.items():
        items = [a for a in s.activity if a.source is source]
        if items:
            out.append(f"- **{title}**")
            out += [f"  - {activity_line(a)}" for a in items]
    return out + [""]


# --- 4 Artefatti -------------------------------------------------------------------------


def _artifact_path(s: ScenarioReport, scenario_relative: str) -> str:
    return f"{s.provenance.dir}/{scenario_relative}"


def _chart(s: ScenarioReport, chart, drawn: frozenset[str]) -> list[str]:
    spec = chart.spec
    title = spec.get("title")
    head = f"**Grafico {chart.index}** — tipo `{_value(spec.get('type'))}`"
    if title is not None:
        head += f", titolo «{_value(title)}»"
    out = [head, ""]
    if chart.filename in drawn:
        alt = f"Grafico {chart.index} di {s.scenario_id}"
        out += [f"![{alt}]({CHART_DIR}/{chart.filename})", "",
                "_Disegno derivato dalla specifica registrata in `charts[]`; nessun valore aggiunto o corretto._", ""]
    else:
        out += ["_La specifica registrata non può essere disegnata fedelmente: valori riportati solo come dati._", ""]
    flags = [f"{k} `{_value(spec[k])}`" for k in ("x_label", "y_label", "horizontal", "stacked") if k in spec]
    if flags:
        out += ["Opzioni registrate: " + ", ".join(flags) + ".", ""]
    labels, datasets = spec.get("labels"), spec.get("datasets")
    tabular = (isinstance(labels, list) and isinstance(datasets, list) and datasets
               and all(isinstance(d, dict) and isinstance(d.get("data"), list)
                       and len(d["data"]) == len(labels) for d in datasets))
    if tabular and spec.get("type") != "scatter":
        header = ["label"] + [_value(d.get("label")) or f"serie {i}" for i, d in enumerate(datasets, 1)]
        out += ["Valori registrati:", ""]
        out += _table(header, ([lab, *(d["data"][i] for d in datasets)] for i, lab in enumerate(labels)))
        extra = [{k: v for k, v in d.items() if k not in ("label", "data")} for d in datasets]
        out += [f"- altri campi della serie {i}: `{_value(e)}`" for i, e in enumerate(extra, 1) if e]
        if any(extra):
            out.append("")
    else:
        rest = {k: v for k, v in spec.items() if k not in ("type", "title")}
        out += ["Valori registrati:", ""] + _fence(json.dumps(rest, ensure_ascii=False, indent=2), "json")
    return out


def _image(s: ScenarioReport) -> list[str]:
    image, shape = s.agent_result.image, s.agent_result.shape
    if image is None or (image.availability is Availability.NOT_APPLICABLE and shape is not ResultShape.IMAGE):
        return []
    status = image.metadata.decode_status if image.metadata else None
    if image.file is not None and status == "OK":
        path = _artifact_path(s, image.file.path)
        return ["**Immagine restituita da DataChat**", "", f"![Immagine di {s.scenario_id}]({_UP}{path})", "",
                f"File: {_link(path)}", ""]
    out = ["**Immagine restituita da DataChat** — nessuna immagine decodificabile disponibile.", "",
           _field("Disponibilità", f"`{image.availability.value}`")]
    if status is not None:
        out.append(_field("Stato di decodifica registrato", f"`{status}`"))
    if image.reason:
        out.append(_field("Motivo registrato", image.reason))
    return out + [""]


def _export(s: ScenarioReport) -> list[str]:
    export = s.agent_result.export
    if export is None or export.availability is Availability.NOT_APPLICABLE:
        return []
    out = ["**Esportazione completa consegnata** (tabella completa, separata dall'anteprima nella risposta)", ""]
    if export.file is not None:
        out.append(_field("File", _link(_artifact_path(s, export.file.path))))
    else:
        out.append(_field("File", "non disponibile"))
    out.append(_field("Disponibilità", f"`{export.availability.value}`"))
    if export.reason:
        out.append(_field("Motivo registrato", export.reason))
    out += [_field(f"`{k}`", _code(v)) for k, v in export.locator.items()]
    return out + [""]


def _artifacts(s: ScenarioReport, drawn: frozenset[str]) -> list[str]:
    out = [f"### {SECTIONS[3]}", ""]
    body = [line for chart in s.charts for line in _chart(s, chart, drawn)]
    body += _image(s) + _export(s)
    return out + (body or ["Nessun artefatto consegnato (nessun grafico, immagine o esportazione).", ""])


# --- 5 Validazione tecnica ---------------------------------------------------------------


def _check_detail(c: CheckResult) -> list[str]:
    out = [f"- `{c.check_id}` — **{CHECK_STATUS_LABELS[c.status]}** (`{c.status.value}`), evidenza "
           f"{EVIDENCE_CLASS_LABELS[c.evidence_class]} (`{c.evidence_class.value}`)",
           f"  - atteso: {_code(c.expected)}",
           f"  - osservato: {_code(c.observed)}"]
    if c.failure is not None:
        out.append(f"  - fallimento: {FAILURE_LABELS[c.failure]} (`{c.failure.value}`)")
    if c.note:
        out.append(f"  - nota: {c.note}")
    if c.evidence_refs:
        out.append(f"  - evidenze: {', '.join(_link(x) for x in c.evidence_refs)}")
    return out


def _validation(s: ScenarioReport) -> list[str]:
    v, p = s.validation, s.provenance
    out = [f"### {SECTIONS[4]}", ""]
    if v.outcome is None:
        out += [_field("Gate", _gate(v.gate)),
                "- **Esito tecnico:** nessuno — lo scenario non è valutabile; non è conteggiato come fallimento."]
        if p.execution_status is not None:
            out.append(_field("Stato di esecuzione", f"`{p.execution_status.value}`"))
        f = p.execution_failure
        if f is not None:
            out.append(_field("Errore registrato", f"`{f.code.value}` — {f.summary}"
                                                   + (f" (fase `{f.phase.value}`)" if f.phase else "")))
            if f.detail:
                out.append(_field("Dettaglio", _code(f.detail)))
        return out + [""]
    out.append(f"- **Esito tecnico:** **{outcome_display(v.outcome, v.failures)}**")
    if v.execution_assessment is not None:
        reason = v.execution_assessment_reason
        out.append(_field("Valutazione dell'esecuzione",
                          f"{ASSESSMENT_LABELS[v.execution_assessment]} (`{v.execution_assessment.value}`) — "
                          f"{REASON_LABELS[reason]} (`{reason.value}`)"))
    if p.teardown_verified is False:
        out.append("- **Attenzione:** teardown dello scenario non verificato.")
    by_status = {st: [c for c in v.checks if c.status is st] for st in CheckStatus}
    for st, label in ((CheckStatus.MET, "Controlli soddisfatti"), (CheckStatus.NOT_APPLICABLE, "Controlli non applicabili")):
        if by_status[st]:
            out.append(_field(f"{label} (`{st.value}`)", ", ".join(f"`{c.check_id}`" for c in by_status[st])))
    if not v.checks:
        out.append("- **Controlli:** nessun controllo specifico dello scenario.")
    out.append("")
    attention = by_status[CheckStatus.NOT_MET] + by_status[CheckStatus.INDETERMINATE]
    if attention:
        out += ["Controlli non soddisfatti o indeterminati:", ""]
        for c in attention:
            out += _check_detail(c)
        out.append("")
    return out


# --- 6 Riferimento di valutazione --------------------------------------------------------


def _reference(ref: EvaluationReference | None) -> list[str]:
    out = [f"### {SECTIONS[5]}", ""]
    if ref is None:
        return out + ["_Nessun riferimento di valutazione definito per questo scenario._", ""]
    out += ["_Materiale di riferimento dal contratto di valutazione fissato (`ground_truth.json`, "
            "`datachat_functional_evaluation_design.md`), per la revisione semantica successiva. "
            "Non è confrontato automaticamente con la risposta dell'agente._", ""]
    out += [f"- Garanzia `{gid}` ({kind}): {title}" for gid, title, kind in ref.guarantees]
    out += [f"- {fact}" for fact in ref.facts]
    out.append("")
    for table in ref.tables:
        out += _table(table.header, table.rows)
    if ref.expectations:
        out += ["Aspettative di progettazione:", ""] + [f"- {e}" for e in ref.expectations] + [""]
    return out


# --- 7 Provenienza tecnica ---------------------------------------------------------------


def _provenance(s: ScenarioReport) -> list[str]:
    p, src = s.provenance, s.agent_result.source
    out = [f"### {SECTIONS[6]}", "",
           _field("Gate", f"`{s.validation.gate.value}`"),
           _field("Stato di esecuzione", p.execution_status and f"`{p.execution_status.value}`"),
           _field("Directory dello scenario", _link(p.dir)),
           _field("request_id", ", ".join(f"`{x}`" for x in p.request_ids))]
    if src is not None and src.exchange is not None:
        x = src.exchange
        out += [_field("/datachat", f"HTTP {_value(src.http_status)}, `{_value(src.content_type)}`, "
                                    f"{_value(x.duration_ms)} ms, tentativi {src.attempts}")]
        if src.content is not None and src.content.log_id is not None:
            out.append(_field("log_id", src.content.log_id))
        if src.body_file:
            out.append(_field("Corpo della risposta", _link(_artifact_path(s, src.body_file))))
    out += [_field("SHA-256 di execution.json", p.execution_sha256 and f"`{p.execution_sha256}`"),
            _field("SHA-256 della domanda", p.question_sha256 and f"`{p.question_sha256}`"),
            _field("Teardown verificato", _yes(p.teardown_verified))]
    if p.evidence_refs:
        out.append(_field("Evidenze citate dai controlli", ", ".join(_link(x) for x in p.evidence_refs)))
    out.append("")
    if p.evidence:
        out += ["<details><summary>Fonti di evidenza registrate</summary>", "",
                "| Fonte | Disponibilità | Motivo | File (SHA-256) |", "|---|---|---|---|"]
        for e in p.evidence:
            files = "<br>".join(f"{_link(_artifact_path(s, f.path))} `{f.sha256}`" for f in e.files)
            out.append(f"| `{_cell(e.source)}` | `{_cell(e.availability)}` | {_cell(e.reason)} | {files} |")
        out += ["", "</details>", ""]
    return out


# --- Entry point -------------------------------------------------------------------------


def _scenario(s: ScenarioReport, drawn: frozenset[str]) -> list[str]:
    return ([f"## {s.scenario_id}", ""] + _question(s) + _response(s) + _activity(s) + _artifacts(s, drawn)
            + _validation(s) + _reference(s.reference) + _provenance(s))


def render_markdown(report: OperatorRunReport, drawn_charts: Iterable[str] = ()) -> str:
    """The Italian Markdown report for one run. Pure and deterministic for given inputs."""
    drawn = frozenset(drawn_charts)
    lines = ["# Report di valutazione funzionale DataChat", "",
             f"Derivato da `grading/results/run_grade.json` (SHA-256 `{report.run_grade_sha256}`), dal pacchetto "
             "dell'esecuzione e dal contratto di valutazione fissato. Rigenerabile; non è una fonte di verità.", ""]
    lines += _overview(report)
    for s in report.scenarios:
        lines += _scenario(s, drawn)
    lines += _run_provenance(report)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
