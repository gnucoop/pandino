"""Tests for the Italian run-centric Markdown report and the reporting CLI.

Synthetic graded packages come from the reader/operator fixtures in a
repository whose ``ground_truth.json`` is the synthetic reference; other
persisted grading states are substituted as typed ``ScenarioGrade`` records.
Offline.
"""

import io
import json
import re
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.evidence import sha256_bytes
from evaluation.datachat.grading.model import (
    ActivitySource,
    CheckStatus,
    EvidenceClass,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ObservedActivity,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.model import Availability, EvidenceSourceRecord, FileRef, ScenarioStatus, SuiteStatus
from evaluation.datachat.reporting import cli
from evaluation.datachat.reporting.reader import load_run_for_reporting
from evaluation.datachat.reporting.renderer import SECTIONS, render_markdown
from tests.test_fe_grading_catalogue import table
from tests.test_fe_grading_grader import FE01, agent, grade, repo_root, snapshot  # noqa: F401
from tests.test_fe_grading_reader import RID, RUN_ID, build_run
from tests.test_fe_reporting_operator import REAL_RUN, check, loaded, project, regraded, regraded_real_copy
from tests.test_fe_reporting_reference import JUDGMENT, ref_repo  # noqa: F401

F, T = TechnicalFailure, TechnicalOutcome
A = ActivitySource

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def md(run, repo, drawn=()) -> str:
    return render_markdown(project(run, repo), drawn)


def section(text: str, heading: str) -> str:
    """Body of the first ``### heading`` section."""
    start = text.index(f"### {heading}")
    end = re.compile(r"\n##+ ").search(text, start + 4)
    return text[start:end.start() if end else None]


def with_artifacts(tmp_path, repo, body, files, evidence):
    """A graded FE-01 package carrying extra artifact files and evidence records."""
    spec = {**FE01, "body": body, **evidence}
    root = build_run(tmp_path, repo, {"FE-01": spec})
    for path, data in files.items():
        target = root / "scenarios/FE-01" / path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    grade(root, repo)
    return root, load_run_for_reporting(root)


def refs(files):
    return {p: FileRef(p, sha256_bytes(b), len(b)) for p, b in files.items()}


# --- Run level ---------------------------------------------------------------------------


def test_run_level_is_italian_without_verdict_or_score(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    text = md(run, ref_repo)
    assert text.startswith("# Report di valutazione funzionale DataChat\n")
    for line in (f"**Run:** `{RUN_ID}`", "**Tipo di esecuzione:** esplorativa (non ufficiale)",
                 "**Confrontabile:** no", "**Stato dell'esecuzione:** `COMPLETED`", "**Maui:** git `abc1234`",
                 "**Grading:** `evaluation.datachat.grading`", "schema `2`",
                 "| SUPERATO | `TECHNICAL PASS` | 1 |", "| FALLITO | `FAIL` | 0 |",
                 "| non eseguito | `NOT_RUN` | 0 |",
                 "| [FE-01](#fe-01) | valutabile (`GRADEABLE`) | SUPERATO (`TECHNICAL PASS`) |"):
        assert line in text, line
    assert "Fallimenti tecnici" not in text  # only non-zero failure counts are listed
    assert run.run_grade_sha256 in text
    for forbidden in ("Verdict", "verdetto", "punteggio complessivo:", "tasso", "NEEDS REVIEW", "revisione manuale",
                      "Attention", "Grading\n", "Question", "Agent Result"):
        assert forbidden not in text, forbidden


def test_failure_counts_listed_when_non_zero(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    run = regraded(run, "FE-01", outcome=T.FAIL, failures=(F.ARTIFACT,),
                   checks=(check("fe01.chart_present", CheckStatus.NOT_MET, F.ARTIFACT),))
    run = replace(run, run_grade=replace(run.run_grade, counts_by_failure={**run.run_grade.counts_by_failure,
                                                                            "ARTIFACT": 1}))
    assert "| ARTEFATTO | `ARTIFACT` | 1 |" in md(run, ref_repo)


# --- Seven sections ----------------------------------------------------------------------


def test_every_scenario_has_the_seven_sections_in_order(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01, "FE-05": {**FE01}})
    text = md(run, ref_repo)
    for sid in ("FE-01", "FE-05"):
        start = text.index(f"\n## {sid}\n")
        end = text.find("\n## ", start + 1)
        headings = re.findall(r"^### (.+)$", text[start:end], re.MULTILINE)
        assert tuple(headings) == SECTIONS, headings


# --- 1 Domanda / 2 Risposta --------------------------------------------------------------


def test_question_and_verbatim_text_response(tmp_path, ref_repo):
    answer = "Riga 1\n- North: 140\n\n```codice``` | pipe   "
    body = {"explanation": "perché sì", "log_id": 7, "response": {"type": "str", "value": answer, "note": "nota x"}}
    _, run = loaded(tmp_path, ref_repo, {"FE-01": {**FE01, "body": body}})
    text = md(run, ref_repo)
    assert "```text\ndomanda?\n```" in section(text, "Domanda")
    response = section(text, "Risposta dell'agente")
    assert f"````text\n{answer}\n````" in response
    assert "Nota restituita (`note`):\n\n```text\nnota x\n```" in response
    assert "Spiegazione restituita (`explanation`):\n\n```text\nperché sì\n```" in response


def test_response_is_http_answer_not_agent_runs(tmp_path, ref_repo):
    record = agent("chart(kind='bar')")
    record["tool_calls"].append({"tool": "final_answer", "args": "INTERNAL ANSWER"})
    record["answer_excerpt"] = "INTERNAL ANSWER"
    _, run = loaded(tmp_path, ref_repo, {"FE-01": {**FE01, "agent_record": record}})
    text = md(run, ref_repo)
    assert "Ecco i conteggi." in section(text, "Risposta dell'agente") and "INTERNAL ANSWER" not in text


def test_dataframe_rendered_as_table_with_metadata(tmp_path, ref_repo):
    files = {"artifacts/export.csv": b"region,count\nNorth,140\nSouth,110\n"}
    ref = refs(files)
    body = {"explanation": None, "log_id": 7, "response": {
        **table([{"region": "North", "count": 140}]), "more_rows_available": True, "download_url": "/download/x"}}
    _, run = with_artifacts(tmp_path, ref_repo, body, files, {"export": EvidenceSourceRecord(
        "export", Availability.PRESENT, None, (ref["artifacts/export.csv"],),
        {"download_url": "/download/x", "http_status": 200})})
    text = md(run, ref_repo)
    response = section(text, "Risposta dell'agente")
    assert "| region | count |" in response and "| North | 140 |" in response
    assert "South" not in response  # the export is not merged into the inline preview
    assert "**anteprima troncata (`truncated`):** false" in response
    assert "**altre righe disponibili (`more_rows_available`):** true" in response
    artifacts = section(text, "Artefatti")
    assert "**Esportazione completa consegnata**" in artifacts
    assert "(../scenarios/FE-01/artifacts/export.csv)" in artifacts and "South" not in artifacts
    assert str(tmp_path) not in text


def test_structured_response_rendered_as_returned(tmp_path, ref_repo):
    body = {"explanation": None, "log_id": 7, "response": {"type": "dict", "value": {"a": 1, "b": [2, 3]}}}
    _, run = loaded(tmp_path, ref_repo, {"FE-01": {**FE01, "body": body}})
    response = section(md(run, ref_repo), "Risposta dell'agente")
    assert "Valore strutturato restituito (verbatim):" in response and '"b": [\n    2,' in response


def test_error_body_rendered_verbatim_with_http_context(tmp_path, ref_repo):
    raw = b'{"error": "boom | <b>"}'
    _, run = loaded(tmp_path, ref_repo, {"FE-01": {**FE01, "body": raw, "http_status": 500}})
    response = section(md(run, ref_repo), "Risposta dell'agente")
    assert "Corpo di errore — HTTP 500" in response and raw.decode() in response


# --- 3 Attività osservata ----------------------------------------------------------------


def test_activity_wording_follows_the_source(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    activity = (ObservedActivity("classify_match", A.RUNTIME, 1, "rejected"),
                ObservedActivity("classify_match", A.RUNTIME, 1, "result"),
                ObservedActivity("classify_match", A.GENERATED_CODE, 2),
                ObservedActivity("keywords", A.GENERATED_CODE, 1),
                ObservedActivity("provider_backed_call", A.USAGE, 1))
    run = regraded(run, "FE-01", outcome=T.TECHNICAL_PASS, activity=activity)
    block = section(md(run, ref_repo), "Attività osservata dell'agente")
    assert "**Eventi runtime (esecuzione osservata)**" in block
    assert "  - `classify_match` — 1 chiamata osservata, esito: `rejected`" in block
    assert "  - `classify_match` — 1 chiamata osservata, esito: `result`" in block
    assert ("  - `classify_match` — 2 chiamate individuate nel codice generato (evidenza inferita)" in block)
    assert "  - `keywords` — 1 chiamata individuata nel codice generato (evidenza inferita)" in block
    assert ("  - 1 chiamata aggiuntiva al provider osservata nei dati Usage; il tool specifico non è "
            "identificabile da questa fonte") in block
    assert block.index("rejected") < block.index("`result`")  # persisted order


def test_inferred_and_usage_activity_never_claim_execution(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    run = regraded(run, "FE-01", outcome=T.TECHNICAL_PASS, activity=(
        ObservedActivity("plot", A.GENERATED_CODE, 1), ObservedActivity("provider_backed_call", A.USAGE, 2)))
    lines = [x for x in section(md(run, ref_repo), "Attività osservata dell'agente").splitlines() if x.startswith("  - ")]
    assert len(lines) == 2
    for line in lines:
        assert "esito" not in line and "eseguit" not in line and "riuscit" not in line
    assert "2 chiamate aggiuntive al provider osservate nei dati Usage" in lines[1] and "`provider_backed_call`" not in lines[1]


def test_no_activity_is_stated(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    run = regraded(run, "FE-01", outcome=T.TECHNICAL_PASS, activity=())
    block = section(md(run, ref_repo), "Attività osservata dell'agente")
    assert "Nessuna attività di tool o capability registrata" in block


# --- 4 Artefatti -------------------------------------------------------------------------


def test_chart_drawing_referenced_with_values_reachable(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    spec = run.scenario("FE-01").agent_result.content.charts[0]
    block = section(md(run, ref_repo, {"FE-01_chart_1.png"}), "Artefatti")
    assert "**Grafico 1** — tipo `bar`" in block
    assert "![Grafico 1 di FE-01](charts/FE-01_chart_1.png)" in block
    for label, value in zip(spec["labels"], spec["datasets"][0]["data"]):
        assert f"| {label} | {value} |" in block


def test_undrawable_chart_shows_values_only(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    block = section(md(run, ref_repo, ()), "Artefatti")
    assert "![" not in block and "non può essere disegnata fedelmente" in block and "| North |" in block


def test_no_artifact_stated_in_one_line(tmp_path, ref_repo):
    body = {"explanation": None, "log_id": 7, "response": {"type": "str", "value": "solo testo"}}
    _, run = loaded(tmp_path, ref_repo, {"FE-01": {**FE01, "body": body}})
    block = section(md(run, ref_repo), "Artefatti")
    assert block.strip().splitlines()[-1] == "Nessun artefatto consegnato (nessun grafico, immagine o esportazione)."


def _image_meta(**kw):
    return {"kind": "image", "source_request_id": RID, "status": "PRESENT", "path": None, "sha256": None,
            "size": None, "wrapper_stripped": True, "decode_status": "OK", "png_signature_ok": True,
            "detected_format": "png", "http_status": None, **kw}


def test_decoded_image_embedded(tmp_path, ref_repo):
    png_ref = FileRef("artifacts/image.png", sha256_bytes(PNG), len(PNG))
    meta = _image_meta(path=png_ref.path, sha256=png_ref.sha256, size=png_ref.size)
    files = {"artifacts/image.png": PNG, "artifacts/image.json": json.dumps(meta).encode()}
    ref = refs(files)
    body = {"explanation": None, "log_id": 7, "response": {"type": "image", "value": "b'iVBOR'"}}
    _, run = with_artifacts(tmp_path, ref_repo, body, files, {"image": EvidenceSourceRecord(
        "image", Availability.PRESENT, None, (ref["artifacts/image.png"], ref["artifacts/image.json"]))})
    text = md(run, ref_repo)
    assert "![Immagine di FE-01](../scenarios/FE-01/artifacts/image.png)" in section(text, "Artefatti")
    assert "vedi **Artefatti**" in section(text, "Risposta dell'agente") and "iVBOR" not in text


def test_failed_image_decode_states_recorded_status(tmp_path, ref_repo):
    meta = _image_meta(decode_status="NOT_PNG", png_signature_ok=False, detected_format=None)
    files = {"artifacts/image.json": json.dumps(meta).encode()}
    ref = refs(files)
    body = {"explanation": None, "log_id": 7, "response": {"type": "image", "value": "b'eA=='"}}
    _, run = with_artifacts(tmp_path, ref_repo, body, files, {"image": EvidenceSourceRecord(
        "image", Availability.PRESENT, None, (ref["artifacts/image.json"],))})
    block = section(md(run, ref_repo), "Artefatti")
    assert "nessuna immagine decodificabile disponibile" in block
    assert "**Stato di decodifica registrato:** `NOT_PNG`" in block and "![" not in block


# --- 5 Validazione tecnica ---------------------------------------------------------------


def test_technical_pass_is_compact_and_technical(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    block = section(md(run, ref_repo), "Validazione tecnica")
    assert "**Esito tecnico:** **SUPERATO (`TECHNICAL PASS`)**" in block
    assert ("**Valutazione dell'esecuzione:** esito utilizzabile (`OK`) — la risposta finale è utilizzabile "
            "(`USABLE_OUTCOME`)") in block
    assert "**Controlli soddisfatti (`MET`):** `fe01.chart_present`" in block
    assert "atteso:" not in block and "corrett" not in block


def test_fail_with_multiple_failures_expands_failed_checks(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    chart = check("fe01.chart_present", CheckStatus.NOT_MET, F.ARTIFACT, expected=">= 1 chart", observed=0,
                  note="no chart")
    provider = check("fe01.provider", CheckStatus.NOT_MET, F.TOOL_EXECUTION, expected=">= 1 extra row", observed=0)
    ok = check("fe01.other", CheckStatus.MET)
    na = check("fe01.na", CheckStatus.NOT_APPLICABLE)
    run = regraded(run, "FE-01", outcome=T.FAIL, failures=(F.TOOL_EXECUTION, F.ARTIFACT),
                   checks=(provider, chart, ok, na))
    block = section(md(run, ref_repo), "Validazione tecnica")
    assert ("**Esito tecnico:** **FALLITO (`FAIL`) — ESECUZIONE TOOL (`TOOL EXECUTION`), ARTEFATTO "
            "(`ARTIFACT`)**") in block
    assert "**Controlli soddisfatti (`MET`):** `fe01.other`" in block
    assert "**Controlli non applicabili (`NOT_APPLICABLE`):** `fe01.na`" in block
    detail = block[block.index("Controlli non soddisfatti o indeterminati"):]
    assert "- `fe01.chart_present` — **non soddisfatto** (`NOT_MET`), evidenza diretta (`DIRECT`)" in detail
    assert "  - atteso: `>= 1 chart`" in detail and "  - osservato: `0`" in detail
    assert "  - fallimento: ARTEFATTO (`ARTIFACT`)" in detail and "  - nota: no chart" in detail
    assert "(../scenarios/FE-01/http/datachat.response.bin)" in detail
    assert "fe01.other" not in detail


def test_indeterminate_outcome_and_check(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    unclear = check("fe01.tool", CheckStatus.INDETERMINATE, evidence_class=EvidenceClass.INFERRED)
    run = regraded(run, "FE-01", (ExecutionAssessment.INDETERMINATE, ExecutionAssessmentReason.ADAPTER_FALLBACK),
                   outcome=T.INDETERMINATE, checks=(unclear,))
    block = section(md(run, ref_repo), "Validazione tecnica")
    assert "**Esito tecnico:** **INDETERMINATO (`INDETERMINATE`)**" in block
    assert "indeterminata (`INDETERMINATE`)" in block and "(`ADAPTER_FALLBACK`)" in block
    assert "- `fe01.tool` — **indeterminato** (`INDETERMINATE`), evidenza inferita (`INFERRED`)" in block


def test_teardown_problem_foregrounded(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    run = regraded(run, "FE-01", outcome=T.TECHNICAL_PASS, teardown_verified=False)
    text = md(run, ref_repo)
    assert "teardown dello scenario non verificato" in section(text, "Validazione tecnica")
    assert "**Teardown verificato:** no" in section(text, "Provenienza tecnica")


# --- Non-gradeable -----------------------------------------------------------------------


def test_not_run_has_no_outcome_and_is_not_a_failure(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": {"status": ScenarioStatus.NOT_RUN}},
                    status=SuiteStatus.ABORTED_INFRA)
    text = md(run, ref_repo)
    assert "_Domanda non inviata a DataChat._" in section(text, "Domanda")
    assert "Nessuna risposta" in section(text, "Risposta dell'agente")
    assert "lo scenario non è valutabile" in section(text, "Attività osservata dell'agente")
    block = section(text, "Validazione tecnica")
    assert "**Gate:** non eseguito (`NOT_RUN`)" in block
    assert "nessuno — lo scenario non è valutabile; non è conteggiato come fallimento" in block
    assert "FAIL" not in block and "FALLITO" not in block
    assert "| non eseguito | `NOT_RUN` | 1 |" in text
    assert "| [FE-01](#fe-01) | non eseguito (`NOT_RUN`) | nessun esito tecnico |" in text


def test_not_scoreable_shows_gate_and_captured_response(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    run = regraded(run, "FE-01", (None, None), gate=Gate.NOT_SCOREABLE_INFRA)
    text = md(run, ref_repo)
    block = section(text, "Validazione tecnica")
    assert "**Gate:** non valutabile per problemi di infrastruttura (`NOT_SCOREABLE_INFRA`)" in block
    assert "FALLITO" not in block and "Ecco i conteggi." in section(text, "Risposta dell'agente")


# --- 6 Riferimento / 7 Provenienza -------------------------------------------------------


def test_reference_shown_without_comparison(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    block = section(md(run, ref_repo), "Riferimento di valutazione")
    assert "Non è confrontato automaticamente con la risposta dell'agente" in block
    assert "- Garanzia `G2` (exact): Exact region counts" in block and "| North | 40 |" in block
    assert "Aspettative di progettazione:" in block
    assert not JUDGMENT.search(block.replace("Non è confrontato automaticamente", ""))
    for marker in ("✓", "✗", "✅", "❌", "⚠"):
        assert marker not in block


def test_provenance_is_secondary_and_complete(tmp_path, ref_repo):
    _, run = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    s = run.scenario("FE-01")
    block = section(md(run, ref_repo), "Provenienza tecnica")
    for line in ("**Gate:** `GRADEABLE`", "**Stato di esecuzione:** `EXECUTED`",
                 "**Directory dello scenario:** [`scenarios/FE-01`](../scenarios/FE-01)",
                 f"**request_id:** `{RID}`", "**/datachat:** HTTP 200",
                 f"**SHA-256 di execution.json:** `{s.execution_sha256}`",
                 f"**SHA-256 della domanda:** `{s.question_sha256}`", "**Teardown verificato:** sì",
                 "<details><summary>Fonti di evidenza registrate</summary>"):
        assert line in block, line


# --- Persistence / determinism / CLI -----------------------------------------------------


def test_cli_writes_report_and_charts_deterministically(tmp_path, ref_repo):
    root, _ = loaded(ref_repo / "evaluation" / "datachat", ref_repo, {"FE-01": FE01})
    before = snapshot(root, exclude=())
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([RUN_ID], repo_root=_repo(ref_repo), out=out, err=err)
    assert code == cli.EXIT_OK, err.getvalue()
    report, chart = root / "reports" / "report.md", root / "reports" / "charts" / "FE-01_chart_1.png"
    first = report.read_bytes(), chart.read_bytes()
    assert chart.read_bytes().startswith(b"\x89PNG") and b"](charts/FE-01_chart_1.png)" in first[0]
    after = snapshot(root, exclude=())
    assert {k: v for k, v in after.items() if not k.startswith("reports/")} == before
    assert set(after) - set(before) == {"reports/report.md", "reports/charts/FE-01_chart_1.png"}
    (root / "reports" / "charts" / "FE-09_chart_1.png").write_bytes(b"stale")
    assert cli.main(["--run-dir", str(root)], repo_root=_repo(ref_repo), out=io.StringIO(),
                    err=io.StringIO()) == cli.EXIT_OK
    assert (report.read_bytes(), chart.read_bytes()) == first
    assert sorted(p.name for p in (root / "reports" / "charts").iterdir()) == ["FE-01_chart_1.png"]


def test_cli_refuses_ungraded_and_unknown_runs(tmp_path, ref_repo):
    root = build_run(tmp_path, ref_repo, {"FE-01": FE01})
    err = io.StringIO()
    assert cli.main(["--run-dir", str(root)], repo_root=_repo(ref_repo), out=io.StringIO(), err=err) == cli.EXIT_USAGE
    assert "not graded" in err.getvalue() and not (root / "reports").exists()
    assert cli.main(["NOPE"], repo_root=_repo(ref_repo), out=io.StringIO(), err=io.StringIO()) == cli.EXIT_USAGE


def test_cli_refuses_changed_reference(tmp_path, ref_repo):
    root, _ = loaded(tmp_path, ref_repo, {"FE-01": FE01})
    path = contract.evaluation_dir(ref_repo) / contract.GROUND_TRUTH_JSON_NAME
    path.write_bytes(path.read_bytes() + b" ")
    err = io.StringIO()
    assert cli.main(["--run-dir", str(root)], repo_root=_repo(ref_repo), out=io.StringIO(), err=err) == cli.EXIT_USAGE
    assert "ground_truth.json differs" in err.getvalue() and not (root / "reports").exists()


def _repo(repo_root: Path) -> Path:
    """Satisfy the ``contract.resolve_repo_root`` markers (main.py, .git) in the synthetic repo."""
    (repo_root / "main.py").touch()
    (repo_root / ".git").mkdir(exist_ok=True)
    return repo_root


# --- Real run (regraded scratch copy, render only) ---------------------------------------


@pytest.mark.skipif(not REAL_RUN.is_dir(), reason="real FE-01 run package not present locally")
def test_real_fe01_markdown(tmp_path):
    run = load_run_for_reporting(regraded_real_copy(tmp_path))
    text = render_markdown(project(run, contract.REPO_ROOT), {"FE-01_chart_1.png"})
    for line in ("Quante richieste ci sono per regione? Mostramele anche in un grafico.", "- North: 140 richieste",
                 "titolo «Numero di richieste per regione»", "| Central | 12 |",
                 "**Esito tecnico:** **SUPERATO (`TECHNICAL PASS`)**", "Regione con più richieste: North (140)"):
        assert line in text, line
