"""Tests for the Markdown report and operator CLI (Reporting v1 Slice 3).

Synthetic graded packages come from the Slice 1/2 fixtures; other persisted
grading states are substituted as typed ``ScenarioGrade`` records. Offline.
"""

import io
import json
import socket
from pathlib import Path

import pytest

from evaluation.datachat.evidence import sha256_bytes
from evaluation.datachat.grading.model import (
    CheckStatus,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    FailureCategory,
    Verdict,
)
from evaluation.datachat.model import Availability, EvidenceSourceRecord, FileRef, ScenarioStatus, SuiteStatus
from evaluation.datachat.reporting import cli
from evaluation.datachat.reporting.operator_report import build_operator_report
from evaluation.datachat.reporting.reader import load_run_for_reporting
from evaluation.datachat.reporting.renderer import render_markdown
from tests.test_fe_grading_catalogue import table
from tests.test_fe_grading_grader import FE01, FE05, agent, grade, repo_root, snapshot  # noqa: F401
from tests.test_fe_grading_reader import RID, RUN_ID, build_run
from tests.test_fe_reporting_operator import REAL_RUN, SHOULD, check, loaded, regraded, review

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)


def md(run) -> str:
    return render_markdown(build_operator_report(run))


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n### ", start + len(heading))
    return text[start:end if end != -1 else None]


def with_artifacts(tmp_path, repo_root, body, files, evidence):
    """A graded FE-01 package carrying extra artifact files and evidence records."""
    spec = {**FE01, "body": body, **evidence}
    root = build_run(tmp_path, repo_root, {"FE-01": spec})
    for path, data in files.items():
        target = root / "scenarios/FE-01" / path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    grade(root, repo_root)
    return root, load_run_for_reporting(root)


def refs(files):
    return {p: FileRef(p, sha256_bytes(b), len(b)) for p, b in files.items()}


# --- PASS / fidelity ---------------------------------------------------------------------


def test_pass_report_shows_triad_identity_and_no_run_verdict(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    text = md(run)
    assert f"**Run:** `{RUN_ID}`" in text
    assert "**Official:** no" in text and "**Comparable:** no" in text and "**Grading complete:** yes" in text
    assert "## FE-01" in text
    assert "```text\ndomanda?\n```" in section(text, "### Question")
    assert "```text\nEcco i conteggi.\n```" in section(text, "### Agent Result")
    grading = section(text, "### Grading")
    assert "**Verdict:** **PASS**" in grading and "**Execution assessment:** OK" in grading
    assert f"{len(run.scenario('FE-01').grade.checks)} MET" in grading
    assert "| fe01.region_counts |" in grading
    assert run.run_grade_sha256 in text
    body = text.replace("There is no run-level verdict or score.", "")
    for forbidden in ("overall", "quality score", "pass rate", "%", "severity", "priority"):
        assert forbidden not in body.lower()
    assert text.index("### Question") < text.index("### Agent Result") < text.index("### Grading")


def test_agent_result_is_http_answer_not_agent_runs(tmp_path, repo_root):
    record = agent("chart(kind='bar')")
    record["tool_calls"].append({"tool": "final_answer", "args": "INTERNAL ANSWER"})
    record["answer_excerpt"] = "INTERNAL ANSWER"
    _, run = loaded(tmp_path, repo_root, {"FE-01": {**FE01, "agent_record": record}})
    text = md(run)
    assert "Ecco i conteggi." in section(text, "### Agent Result")
    assert "INTERNAL ANSWER" not in text


def test_multiline_text_and_backticks_preserved_verbatim(tmp_path, repo_root):
    answer = "Riga 1\n- North: 140\n\n```codice``` | pipe"
    body = {"explanation": None, "log_id": 7, "response": {"type": "str", "value": answer}}
    _, run = loaded(tmp_path, repo_root, {"FE-01": {**FE01, "body": body}})
    assert f"````text\n{answer}\n````" in md(run)


def test_chart_spec_is_rendered_as_recorded(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    spec = run.scenario("FE-01").agent_result.content.charts[0]
    chart = section(md(run), "### Agent Result")
    assert "**type:** bar" in chart
    for label, value in zip(spec["labels"], spec["datasets"][0]["data"]):
        assert f"| {label} | {value} |" in chart


# --- Table / export / image --------------------------------------------------------------


def test_table_preview_and_export_kept_separate(tmp_path, repo_root):
    files = {"artifacts/export.csv": b"region,count\nNorth,140\nSouth,110\n"}
    ref = refs(files)
    body = {"explanation": "spiegazione", "log_id": 7, "response": {
        **table([{"region": "North", "count": 140}]), "note": "anteprima", "more_rows_available": True,
        "download_url": "/download/x"}}
    _, run = with_artifacts(tmp_path, repo_root, body, files, {"export": EvidenceSourceRecord(
        "export", Availability.PRESENT, None, (ref["artifacts/export.csv"],),
        {"download_url": "/download/x", "http_status": 200})})
    result = section(md(run), "### Agent Result")
    assert "| region | count |" in result and "| North | 140 |" in result
    assert "South" not in result  # export rows are not merged into the preview
    assert "**note:** anteprima" in result and "**more_rows_available:** true" in result
    assert "**truncated:** false" in result and "spiegazione" in result
    assert "**Export artifact**" in result
    assert "(../scenarios/FE-01/artifacts/export.csv)" in result
    assert str(tmp_path) not in md(run)


def test_decoded_image_referenced_relatively(tmp_path, repo_root):
    png_ref = FileRef("artifacts/image.png", sha256_bytes(PNG), len(PNG))
    meta = {"kind": "image", "source_request_id": RID, "status": "PRESENT", "path": png_ref.path,
            "sha256": png_ref.sha256, "size": png_ref.size, "wrapper_stripped": True, "decode_status": "OK",
            "png_signature_ok": True, "detected_format": "png", "http_status": None}
    files = {"artifacts/image.png": PNG, "artifacts/image.json": json.dumps(meta).encode()}
    ref = refs(files)
    body = {"explanation": None, "log_id": 7, "response": {"type": "image", "value": "b'iVBOR'"}}
    _, run = with_artifacts(tmp_path, repo_root, body, files, {"image": EvidenceSourceRecord(
        "image", Availability.PRESENT, None, (ref["artifacts/image.png"], ref["artifacts/image.json"]))})
    result = section(md(run), "### Agent Result")
    assert "![DataChat image](../scenarios/FE-01/artifacts/image.png)" in result
    assert "iVBOR" not in result


def test_failed_image_decode_shows_recorded_status(tmp_path, repo_root):
    meta = {"kind": "image", "source_request_id": RID, "status": "PRESENT", "path": None, "sha256": None,
            "size": None, "wrapper_stripped": True, "decode_status": "NOT_PNG", "png_signature_ok": False,
            "detected_format": None, "http_status": None}
    files = {"artifacts/image.json": json.dumps(meta).encode()}
    ref = refs(files)
    body = {"explanation": None, "log_id": 7, "response": {"type": "image", "value": "b'eA=='"}}
    _, run = with_artifacts(tmp_path, repo_root, body, files, {"image": EvidenceSourceRecord(
        "image", Availability.PRESENT, None, (ref["artifacts/image.json"],))})
    result = section(md(run), "### Agent Result")
    assert "**Recorded decode status:** NOT_PNG" in result
    assert "No decodable persisted image is available." in result and "![" not in result


def test_error_body_rendered_verbatim(tmp_path, repo_root):
    raw = b'{"error": "boom | <b>"}'
    _, run = loaded(tmp_path, repo_root, {"FE-01": {**FE01, "body": raw, "http_status": 500}})
    result = section(md(run), "### Agent Result")
    assert "**Error body** — HTTP 500" in result and raw.decode() in result


# --- Exceptional verdicts ----------------------------------------------------------------


def test_pass_with_caveat_surfaces_caveat_items(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    miss = check("fe01.chart_bar", CheckStatus.NOT_MET, SHOULD, expected=["bar"], observed="pie")
    run = regraded(run, "FE-01", verdict=Verdict.PASS_WITH_CAVEAT, caveats=("fe01.chart_bar",),
                   checks=(check("fe01.region_counts", CheckStatus.MET), miss))
    grading = section(md(run), "### Grading")
    assert "**Verdict:** **PASS WITH CAVEAT**" in grading
    caveats = grading[grading.index("#### Caveats"):]
    assert "`fe01.chart_bar` — status **NOT_MET**" in caveats and 'expected: `["bar"]`' in caveats
    assert "yes — PASS WITH CAVEAT" in grading


def test_fail_foregrounds_root_failure_without_causation(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    counts = check("fe01.region_counts", CheckStatus.NOT_MET, category=FailureCategory.ANALYTICAL,
                   expected={"north": 140}, observed={"north": 141})
    run = regraded(run, "FE-01", verdict=Verdict.FAIL_ANALYTICAL, root_failure=FailureCategory.ANALYTICAL,
                   other_failures=(FailureCategory.HONESTY,),
                   checks=(counts, check("fe01.chart_present", CheckStatus.MET)))
    text = md(run)
    grading = section(text, "### Grading")
    assert "**Verdict:** **FAIL — ANALYTICAL**" in grading
    assert f"**Root failure: {FailureCategory.ANALYTICAL.value}**" in grading
    assert "Checks carrying this failure category: `fe01.region_counts`" in grading
    assert "**Other failure: " in grading
    attention = grading[grading.index("#### Checks requiring attention"):]
    assert 'expected: `{"north": 140}`' in attention and 'observed: `{"north": 141}`' in attention
    assert "caused" not in text.lower() and "because" not in text.lower()


def test_pending_review_shown_and_never_written(tmp_path, repo_root):
    root, run = loaded(tmp_path, repo_root, {"FE-05": FE05})
    before = snapshot(root, exclude=("reports/",))
    text = md(run)
    pending = [r for r in run.scenario("FE-05").grade.manual_reviews if r.status.value == "PENDING"]
    assert "**Grading complete:** no" in text and "**Grading incomplete**" in text
    section_ = text[text.index("#### Pending manual reviews"):]
    assert "(../grading/reviews/FE-05.json)" in section_
    for r in pending:
        assert r.question in section_ and r.dimension.value in section_
        assert r.requirement_level.value in section_ and "status **PENDING**" in section_
    cli.write_report(root, text)
    assert snapshot(root, exclude=("reports/",)) == before
    assert not (root / "grading" / "reviews").exists()


def test_execution_needs_review_is_distinct_from_review_items(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", (ExecutionAssessment.NEEDS_REVIEW, ExecutionAssessmentReason.ADAPTER_FALLBACK),
                   verdict=Verdict.NEEDS_REVIEW, checks=(check("fe01.region_counts", CheckStatus.MET),))
    grading = section(md(run), "### Grading")
    assert "#### Execution issue" in grading and "`ADAPTER_FALLBACK`" in grading
    assert "not resolvable through review files" in grading
    assert "#### Pending manual reviews" not in grading


# --- Non-gradeable -----------------------------------------------------------------------


def test_not_run_has_no_verdict_and_absent_result(tmp_path, repo_root):
    _, run = loaded(tmp_path, repo_root, {"FE-01": {"status": ScenarioStatus.NOT_RUN}},
                    status=SuiteStatus.ABORTED_INFRA)
    text = md(run)
    assert "_Not sent to DataChat._" in section(text, "### Question")
    assert "No Agent Result" in section(text, "### Agent Result")
    grading = section(text, "### Grading")
    assert "**Gate:** NOT_RUN" in grading and "no analytical verdict" in grading
    assert "FAIL" not in grading
    assert "| NOT_RUN | 1 |" in text


def test_not_scoreable_shows_gate_and_captured_result(tmp_path, repo_root):
    from evaluation.datachat.grading.model import Gate
    _, run = loaded(tmp_path, repo_root, {"FE-01": FE01})
    run = regraded(run, "FE-01", (None, None), gate=Gate.NOT_SCOREABLE_INFRA)
    text = md(run)
    grading = section(text, "### Grading")
    assert "**Gate:** NOT_SCOREABLE_INFRA" in grading and "no analytical verdict" in grading
    assert "FAIL" not in grading and "Ecco i conteggi." in section(text, "### Agent Result")


# --- Persistence / determinism / CLI -----------------------------------------------------


def test_cli_run_id_writes_only_the_report_deterministically(tmp_path, repo_root):
    root, _ = loaded(repo_root / "evaluation" / "datachat", repo_root, {"FE-01": FE01})
    before = snapshot(root, exclude=())
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([RUN_ID], repo_root=_repo(repo_root), out=out, err=err)
    assert code == cli.EXIT_OK, err.getvalue()
    report = root / "reports" / "report.md"
    first = report.read_bytes()
    assert "DataChat FE report generated" in out.getvalue() and "report.md" in out.getvalue()
    after = snapshot(root, exclude=())
    assert {k: v for k, v in after.items() if k != "reports/report.md"} == before
    assert set(after) - set(before) == {"reports/report.md"}
    assert cli.main(["--run-dir", str(root)], repo_root=_repo(repo_root), out=io.StringIO(),
                    err=io.StringIO()) == cli.EXIT_OK
    assert report.read_bytes() == first


def test_cli_refuses_ungraded_and_unknown_runs(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    err = io.StringIO()
    assert cli.main(["--run-dir", str(root)], repo_root=_repo(repo_root), out=io.StringIO(), err=err) == cli.EXIT_USAGE
    assert "not graded" in err.getvalue() and not (root / "reports").exists()
    assert cli.main(["NOPE"], repo_root=_repo(repo_root), out=io.StringIO(), err=io.StringIO()) == cli.EXIT_USAGE


def _repo(repo_root: Path) -> Path:
    """Satisfy the ``contract.resolve_repo_root`` markers (main.py, .git) in the synthetic repo."""
    (repo_root / "main.py").touch()
    (repo_root / ".git").mkdir(exist_ok=True)
    return repo_root


# --- Real FE-01 (read-only render, no write) ---------------------------------------------


@pytest.mark.skipif(not REAL_RUN.is_dir(), reason="real FE-01 run package not present locally")
def test_real_fe01_markdown():
    text = md(load_run_for_reporting(REAL_RUN))
    assert "Quante richieste ci sono per regione? Mostramele anche in un grafico." in text
    for line in ("- North: 140 richieste", "- Central: 12 richieste", "**title:** Numero di richieste per regione",
                 "| Central | 12 |", "**Verdict:** **PASS**", "**Execution assessment:** OK", "4 MET"):
        assert line in text
    assert "#### Caveats" not in text and "#### Pending manual reviews" not in text
