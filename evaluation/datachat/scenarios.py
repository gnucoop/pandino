"""Executable DataChat Functional Evaluation scenario catalogue (FE-01 ... FE-14).

Only what execution needs: id, ordinal and the frozen question sent to
``/datachat``. Ground truth, MUST/SHOULD/MAY and grading rules are not here.

The evaluation design document remains the authority for the questions.
``check_drift`` re-reads it (pinned by SHA-256) and compares the
``- **Question:** `...` `` line under each ``### FE-xx`` heading with this
tuple; any difference must stop preflight.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.datachat.contract import (
    DESIGN_DOC_NAME,
    REPO_ROOT,
    evaluation_dir,
    pinned_artifact,
)
from evaluation.datachat.evidence import sha256_file


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    ordinal: int
    question: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario('FE-01', 1, 'Quante richieste ci sono per regione? Mostramele anche in un grafico.'),
    Scenario('FE-02', 2, 'Nel Sud, quale servizio ha il tempo medio di attesa più alto? Mostrami anche il confronto tra i servizi.'),
    Scenario('FE-03', 3, 'Quale servizio genera il costo totale maggiore e quale invece ha il costo medio maggiore?'),
    Scenario('FE-04', 4, 'Ci sono dati mancanti nel dataset? In quali colonne e quanti?'),
    Scenario('FE-05', 5, "C'è una relazione tra tempo di attesa e soddisfazione? Tieni conto che la soddisfazione è un voto da 1 a 5."),
    Scenario('FE-06', 6, 'Il costo del caso sembra collegato alla soddisfazione?'),
    Scenario('FE-07', 7, 'La soddisfazione degli utenti Online è diversa da quella degli utenti Phone?'),
    Scenario('FE-08', 8, 'Gli utenti In person e Phone hanno lo stesso livello di soddisfazione?'),
    Scenario('FE-09', 9, "Come è cambiato il numero di richieste nel corso del 2025? Fammi vedere l'andamento."),
    Scenario('FE-10', 10, "I tempi medi di attesa sono cambiati durante l'anno?"),
    Scenario('FE-11', 11, 'Fammi vedere la distribuzione dei tempi di attesa.'),
    Scenario('FE-12', 12, 'Ci sono casi con costi anomali rispetto agli altri? In quali servizi si trovano?'),
    Scenario('FE-13', 13, 'Analizza il sentiment dei commenti e dimmi in quale regione la quota di commenti negativi è più alta. Mostrami anche il confronto.'),
    Scenario('FE-14', 14, 'Quali temi emergono dai commenti? Raggruppali in categorie e dimmi quanti commenti ricadono in ciascuna.'),
)

SCENARIO_ORDER: tuple[str, ...] = tuple(scenario.scenario_id for scenario in SCENARIOS)


def question_sha256(question: str) -> str:
    """UTF-8 SHA-256 of a question, as stored in ``ScenarioExecution.question_sha256``."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


_FE_HEADING = re.compile(r"^###\s+(FE-\d{2})\b")
_ANY_HEADING = re.compile(r"^#{1,6}\s")
_QUESTION = re.compile(r"^- \*\*Question:\*\* `(.*)`\s*$")


def extract_questions(markdown: str) -> list[tuple[str, str]]:
    """``[(scenario_id, question), ...]`` in document order.

    Raises ``ValueError`` on a duplicate FE heading, an FE section without
    exactly one question line, or a question line outside an FE section.
    """
    found: list[tuple[str, str]] = []
    current: str | None = None
    question: str | None = None

    def close_section() -> None:
        if current is not None:
            if question is None:
                raise ValueError(f"{current}: question missing")
            found.append((current, question))

    for line in markdown.splitlines():
        heading = _FE_HEADING.match(line)
        if heading or _ANY_HEADING.match(line):
            close_section()
            current, question = None, None
            if heading:
                current = heading.group(1)
                if any(scenario_id == current for scenario_id, _ in found):
                    raise ValueError(f"{current}: duplicate scenario heading")
            continue
        match = _QUESTION.match(line)
        if match:
            if current is None:
                raise ValueError(f"question outside an FE section: {match.group(1)!r}")
            if question is not None:
                raise ValueError(f"{current}: more than one question")
            question = match.group(1)
    close_section()
    return found


@dataclass(frozen=True)
class DriftResult:
    ok: bool
    design_path: str
    design_sha256: str | None
    errors: tuple[str, ...] = field(default_factory=tuple)


def check_drift(
    repo_root: Path = REPO_ROOT, scenarios: tuple[Scenario, ...] = SCENARIOS
) -> DriftResult:
    """Compare the executable catalogue with the pinned design authority."""
    path = evaluation_dir(repo_root) / DESIGN_DOC_NAME
    if not path.is_file():
        return DriftResult(False, str(path), None, (f"design document not found: {path}",))

    errors: list[str] = []
    actual_sha = sha256_file(path)
    expected_sha = pinned_artifact(DESIGN_DOC_NAME).sha256
    if actual_sha != expected_sha:
        errors.append(f"design document hash {actual_sha} != pinned {expected_sha}")

    try:
        extracted = extract_questions(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        errors.append(f"design document unparseable: {exc}")
        return DriftResult(False, str(path), actual_sha, tuple(errors))

    expected_ids = [f"FE-{n:02d}" for n in range(1, 15)]
    doc_ids = [scenario_id for scenario_id, _ in extracted]
    if doc_ids != expected_ids:
        errors.append(f"design scenarios {doc_ids} != {expected_ids}")
    catalogue_ids = [scenario.scenario_id for scenario in scenarios]
    if catalogue_ids != doc_ids:
        errors.append(f"catalogue scenarios {catalogue_ids} != design {doc_ids}")
    for position, scenario in enumerate(scenarios, start=1):
        if scenario.ordinal != position:
            errors.append(f"{scenario.scenario_id}: ordinal {scenario.ordinal} != {position}")

    doc_questions = dict(extracted)
    for scenario in scenarios:
        expected = doc_questions.get(scenario.scenario_id)
        if expected is not None and expected != scenario.question:
            errors.append(f"{scenario.scenario_id}: question differs from design document")

    return DriftResult(not errors, str(path), actual_sha, tuple(errors))
