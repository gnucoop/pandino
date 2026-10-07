"""Tests for evaluation.datachat.reporting.reader, the Reporting v1 input boundary.

Offline only: synthetic run packages (real hashes) are built with the
grading-test builders and graded by the real grader into ``grading/results/``;
the reporting reader then loads them. No Maui, no DB, no provider, no network.
"""

import json
import socket

import pytest

from evaluation.datachat.evidence import sha256_bytes, sha256_file
from evaluation.datachat.grading.model import ExecutionAssessment, Gate, ReviewStatus, Verdict
from evaluation.datachat.grading.reader import ParseStatus
from evaluation.datachat.model import Availability, EvidenceSourceRecord, FileRef, SuiteStatus
from evaluation.datachat.reporting.reader import (
    ReportingInputError,
    UnsupportedSchemaError,
    load_run_for_reporting,
)
from tests.test_fe_grading_catalogue import table
from tests.test_fe_grading_grader import FE01, FE05, agent, grade, repo_root, snapshot  # noqa: F401
from tests.test_fe_grading_reader import RID, RUN_ID, build_run


@pytest.fixture
def offline(monkeypatch):
    """Any socket use fails the test: the reader must work without network access."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def graded(tmp_path, repo_root, specs, **kwargs):
    root = build_run(tmp_path, repo_root, specs, **kwargs)
    grade(root, repo_root)
    return root


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def results(root, sid=None):
    base = root / "grading" / "results"
    return base / "run_grade.json" if sid is None else base / "scenarios" / f"{sid}.json"


# --- Positive path --------------------------------------------------------------------


def test_valid_package_exposes_question_agent_result_and_grading(tmp_path, repo_root, offline):
    root = graded(tmp_path, repo_root, {"FE-01": FE01})
    before = snapshot(root, exclude=())
    run = load_run_for_reporting(root)

    assert run.run_id == RUN_ID and run.manifest.status is SuiteStatus.COMPLETED
    assert (run.run_grade.official, run.run_grade.comparable, run.run_grade.grading_complete) == (False, False, True)
    assert run.run_grade_sha256 == sha256_file(results(root))
    assert [s.scenario_id for s in run.scenarios] == ["FE-01"]

    s = run.scenario("FE-01")
    assert s.question == "domanda?" and s.question_sha256 == s.execution.question_sha256
    assert s.execution_sha256 == sha256_file(root / "scenarios/FE-01/execution.json")
    assert s.execution_sha256 == run.run_grade.contract["scenario_execution_sha256"]["FE-01"]

    a = s.agent_result
    assert (a.availability, a.http_status, a.attempts, a.exchange.request_id) == (Availability.PRESENT, 200, 1, RID)
    assert a.body == (root / "scenarios/FE-01" / a.body_file).read_bytes()
    assert (a.content.parse_status, a.content.response_type, a.content.value) == (ParseStatus.OK, "str", "Ecco i conteggi.")
    assert a.content.charts[0]["type"] == "bar" and a.content.log_id == 7
    assert a.image.availability is Availability.NOT_APPLICABLE and a.export.reason == "no download_url"

    g = s.grade
    assert (g.gate, g.verdict, g.execution_assessment) == (Gate.GRADEABLE, Verdict.PASS, ExecutionAssessment.OK)
    assert g.to_dict() == json.loads(results(root, "FE-01").read_text())
    assert snapshot(root, exclude=()) == before
    assert not (root / "reports").exists()


def test_agent_result_comes_from_http_not_agent_runs(tmp_path, repo_root):
    record = agent("chart(kind='bar')")
    record["tool_calls"].append({"tool": "final_answer", "args": "INTERNAL ANSWER"})
    record["answer_excerpt"] = "INTERNAL ANSWER"
    root = graded(tmp_path, repo_root, {"FE-01": {**FE01, "agent_record": record}})
    content = load_run_for_reporting(root).scenario("FE-01").agent_result.content
    assert content.value == "Ecco i conteggi."
    assert "INTERNAL ANSWER" not in json.dumps(content.parsed)


def test_table_export_and_image_failure_are_preserved(tmp_path, repo_root):
    csv, meta = b"region,count\nNorth,140\n", {
        "kind": "image", "source_request_id": RID, "status": "PRESENT", "path": None, "sha256": None,
        "size": None, "wrapper_stripped": True, "decode_status": "NOT_PNG", "png_signature_ok": False,
        "detected_format": None, "http_status": None}
    meta_bytes = json.dumps(meta).encode()
    files = {"artifacts/export.csv": csv, "artifacts/image.json": meta_bytes}
    ref = {p: FileRef(p, sha256_bytes(b), len(b)) for p, b in files.items()}
    body = {"explanation": "spiegazione", "log_id": 7, "response": {
        **table([{"region": "North", "count": 140}]), "note": "anteprima", "more_rows_available": True,
        "download_url": "/download/x"}}
    spec = {**FE01, "body": body,
            "export": EvidenceSourceRecord("export", Availability.PRESENT, None, (ref["artifacts/export.csv"],),
                                           {"download_url": "/download/x", "http_status": 200}),
            "image": EvidenceSourceRecord("image", Availability.PRESENT, None, (ref["artifacts/image.json"],),
                                          {"decode_status": "NOT_PNG"})}
    root = build_run(tmp_path, repo_root, {"FE-01": spec})
    for path, data in files.items():
        target = root / "scenarios/FE-01" / path
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    grade(root, repo_root)

    a = load_run_for_reporting(root).scenario("FE-01").agent_result
    c = a.content
    assert (c.response_type, c.value, c.note, c.more_rows_available) == (
        "dataframe", [{"region": "North", "count": 140}], "anteprima", True)
    assert (c.download_url, c.explanation, c.table_meta) == ("/download/x", "spiegazione", {"truncated": False})
    assert a.export.file == ref["artifacts/export.csv"] and a.export.locator["http_status"] == 200
    assert a.image.file is None and a.image.metadata.decode_status == "NOT_PNG"


def test_unwritten_scenario_keeps_its_gate_and_has_no_verdict(tmp_path, repo_root):
    root = graded(tmp_path, repo_root, {}, status=SuiteStatus.ABORTED_INFRA, order=["FE-01"])
    s = load_run_for_reporting(root).scenario("FE-01")
    assert (s.grade.gate, s.grade.verdict, s.index.verdict) == (Gate.NOT_RUN, None, None)
    assert (s.execution, s.question, s.agent_result) == (None, None, None)


def test_pending_reviews_are_read_from_the_persisted_grade(tmp_path, repo_root):
    root = graded(tmp_path, repo_root, {"FE-05": FE05})
    (root / "grading/reviews").mkdir(parents=True)
    (root / "grading/reviews/FE-05.json").write_text("not read by reporting")
    s = load_run_for_reporting(root).scenario("FE-05")
    assert s.grade.verdict is Verdict.NEEDS_REVIEW
    assert s.index.pending_reviews == sum(r.status is ReviewStatus.PENDING for r in s.grade.manual_reviews) > 0


# --- Refusals --------------------------------------------------------------------------


def _set_verdict(d):
    d["verdict"] = "FAIL — ANALYTICAL"
    d["root_failure"] = "ANALYTICAL"


@pytest.mark.parametrize("tamper,match", [
    (lambda root: results(root, "FE-01").unlink(), "scenario grade missing"),
    (lambda root: edit_json(results(root, "FE-01"), _set_verdict), "does not match run_grade.json"),
    (lambda root: edit_json(results(root), lambda d: d["scenarios"][0].update(pending_reviews=1)), "pending_reviews"),
])
def test_inconsistent_scenario_grade_refused(tmp_path, repo_root, tamper, match):
    root = graded(tmp_path, repo_root, {"FE-01": FE01})
    tamper(root)
    with pytest.raises(ReportingInputError, match=match):
        load_run_for_reporting(root)


def test_execution_digest_mismatch_refused(tmp_path, repo_root):
    root = graded(tmp_path, repo_root, {"FE-01": FE01})
    edit_json(results(root), lambda d: d["contract"]["scenario_execution_sha256"].update({"FE-01": "0" * 64}))
    with pytest.raises(ReportingInputError, match="execution digest mismatch"):
        load_run_for_reporting(root)


def test_tampered_agent_result_body_refused(tmp_path, repo_root):
    root = graded(tmp_path, repo_root, {"FE-01": FE01})
    path = root / "scenarios/FE-01/http/datachat.response.bin"
    path.write_bytes(path.read_bytes().replace(b"conteggi", b"CONTEGGI"))
    with pytest.raises(ReportingInputError, match="datachat response .* sha256 mismatch"):
        load_run_for_reporting(root)


def test_unsupported_schema_refused(tmp_path, repo_root):
    root = graded(tmp_path, repo_root, {"FE-01": FE01})
    edit_json(results(root), lambda d: d.update(schema_version=2))
    with pytest.raises(UnsupportedSchemaError, match="schema_version 2"):
        load_run_for_reporting(root)


def test_ungraded_run_refused_without_grading(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    with pytest.raises(ReportingInputError, match="not graded"):
        load_run_for_reporting(root)
    assert not (root / "grading").exists()
