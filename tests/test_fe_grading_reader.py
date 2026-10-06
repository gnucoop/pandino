"""Tests for evaluation.datachat.grading.reader, the read-only grading loader.

Offline only: synthetic run packages are generated under ``tmp_path`` with
real hashes; no Maui, no DB, no provider. The pinned evaluation artifacts are
replaced by tiny synthetic files so the tests never depend on ``docs/``.
"""

import json
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.evidence import RunLayout, file_ref, sha256_bytes, sha256_file, write_json_atomic, write_raw
from evaluation.datachat.grading.model import Gate
from evaluation.datachat.grading.reader import (
    GradingInputError,
    ParseStatus,
    load_run_for_grading,
    normalize_response,
    parse_runtime_log,
)
from evaluation.datachat.model import (
    Availability,
    EvaluationRun,
    EvidenceSourceRecord,
    Failure,
    HttpExchange,
    HttpRequestRecord,
    HttpResponseRecord,
    InfraCode,
    ScenarioExecution,
    ScenarioFlags,
    ScenarioStatus,
    SuiteStatus,
)

RID = "abcdef0123456789"
RUN_ID = "20260101T000000Z_abc1234"

RUNTIME_LOG = "\n".join([
    f"2026-01-01T00:00:01+00:00 INFO datachat.runtime request_id={RID} app_id=dino chat_start request_id={RID} engine=smolagents",
    f"2026-01-01T00:00:02+00:00 INFO datachat.tools.aggregate_tool request_id={RID} app_id=dino event=tool_call_result group_by=region op=count metric=record_id",
    f"2026-01-01T00:00:03+00:00 INFO datachat.tools.chart_tool request_id={RID} app_id=dino event=tool_call_result kind=bar mode=records",
    f"2026-01-01T00:00:03+00:00 INFO datachat.tools.classify_match_tool request_id={RID} app_id=dino event=tool_call_failed tool=classify_match rows=3",
    f"2026-01-01T00:00:04+00:00 INFO datachat.runtime request_id={RID} app_id=dino final_answer_check request_id={RID} passed=True final_kind=text reason=OK",
    f"2026-01-01T00:00:04+00:00 INFO datachat.runtime request_id={RID} app_id=dino chat_end request_id={RID} response_kind=text",
    f"2026-01-01T00:00:05+00:00 INFO datachat.runtime request_id={RID} app_id=dino datachat_trace_status request_id={RID} trace_present=True",
    f"2026-01-01T00:00:05+00:00 INFO datachat.runtime request_id={RID} app_id=dino datachat_adapter_status request_id={RID} adapter_fallback_used=False",
    f"2026-01-01T00:00:06+00:00 INFO datachat.runtime request_id={RID} app_id=dino datachat_request_end request_id={RID} status=ok http_status=200",
    "garbage line without structure",
]) + "\n"

TEXT_BODY = {
    "explanation": None,
    "log_id": 7,
    "response": {
        "type": "str",
        "value": "Ecco i conteggi.",
        "charts": [{"type": "bar", "labels": ["North"], "datasets": [{"data": [140], "label": "count"}]}],
    },
}

AGENT_RECORD = {
    "namespace": "datachat",
    "request_id": RID,
    "state": "success",
    "steps_count": 3,
    "tool_calls": [
        {"tool": "python_interpreter", "args": "result = aggregate(group_by='region', op='count')"},
        {"tool": "python_interpreter", "args": "chart(kind='bar', data=result['data'])"},
    ],
    "extra": {"channel": "datachat", "response_kind": "text"},
}

USAGE = {
    "schema_version": 1, "request_id": RID, "query_sql": "SELECT", "schema": "public", "executed_at": "x",
    "rows": [{"id": 6, "service": "semantic"}, {"id": 7, "service": "/datachat"}],
    "primary_log_id": 7, "primary_row_present": True,
}


# --- Fixture builders ---------------------------------------------------------


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    """Synthetic evaluation artifacts, pinned by their real hashes."""
    root = tmp_path / "repo"
    directory = contract.evaluation_dir(root)
    directory.mkdir(parents=True)
    pins = []
    for artifact in contract.PINNED_ARTIFACTS:
        content = f"synthetic {artifact.name}\n".encode()
        (directory / artifact.name).write_bytes(content)
        pins.append(contract.PinnedArtifact(artifact.name, sha256_bytes(content), artifact.required))
    monkeypatch.setattr(contract, "PINNED_ARTIFACTS", tuple(pins))
    return root


def _write_exchange(layout, sid, key, status, body, content_type="application/json"):
    meta, req_path, resp_path = layout.http_paths(sid, key)
    sdir = layout.scenario_dir(sid)
    req_ref = write_raw(req_path, b'{"chat": "q"}', relative_to=sdir)
    resp_ref = write_raw(resp_path, body, relative_to=sdir)
    exchange = HttpExchange(
        key=key, attempt=1, method="POST", path=f"/{key}", sent_at="t", completed_at="t", duration_ms=1,
        request=HttpRequestRecord(headers_redacted={}, content_type="application/json",
                                  body_file=req_ref.path, body_sha256=req_ref.sha256),
        response=HttpResponseRecord(status=status, headers={}, content_type=content_type,
                                    body_file=resp_ref.path, body_sha256=resp_ref.sha256, size=resp_ref.size),
        request_id=RID,
    )
    write_json_atomic(meta, exchange.to_dict())
    return meta.relative_to(sdir).as_posix(), req_ref, resp_ref


def build_scenario(
    layout, sid, *, status=ScenarioStatus.EXECUTED, scoreable=True, teardown_verified=True,
    body=None, http_status=200, content_type="application/json", runtime_log=RUNTIME_LOG,
    agent_record=AGENT_RECORD, usage=USAGE, image=None, export=None, failure=None,
):
    execution = ScenarioExecution.new(RUN_ID, sid, int(sid[3:]), "domanda?")
    if status is ScenarioStatus.NOT_RUN:
        execution.mark_not_run()
    else:
        execution.status = status
        execution.failure = failure
        sdir = layout.scenario_dir(sid)
        sdir.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(TEXT_BODY if body is None else body).encode() if not isinstance(body, bytes) else body
        meta_rel, req_ref, resp_ref = _write_exchange(layout, sid, "datachat", http_status, raw, content_type)
        execution.requests = {"datachat": [meta_rel]}
        ev = execution.evidence
        ev["http_datachat"] = EvidenceSourceRecord("http_datachat", Availability.PRESENT, None,
                                                   (req_ref, resp_ref), {"request_id": RID})
        log = write_raw(layout.runtime_log(sid, "datachat"), runtime_log.encode(), relative_to=sdir)
        ev["runtime_datachat"] = EvidenceSourceRecord("runtime_datachat", Availability.PRESENT, None, (log,))
        if agent_record is None:
            ev["agent_runs"] = EvidenceSourceRecord("agent_runs", Availability.EXPECTED_ABSENT, "no trace")
        else:
            ref = write_raw(layout.agent_runs_record(sid), (json.dumps(agent_record) + "\n").encode(), relative_to=sdir)
            ev["agent_runs"] = EvidenceSourceRecord("agent_runs", Availability.PRESENT, None, (ref,))
        uref = write_raw(layout.usage_rows(sid), json.dumps(usage).encode(), relative_to=sdir)
        ev["usage"] = EvidenceSourceRecord("usage", Availability.PRESENT, None, (uref,))
        ev["image"] = image or EvidenceSourceRecord("image", Availability.NOT_APPLICABLE, "response is not an image")
        ev["export"] = export or EvidenceSourceRecord("export", Availability.NOT_APPLICABLE, "no download_url")
        execution.flags = ScenarioFlags(
            reached_datachat=True, fresh_engine_proven=True, execution_complete=True,
            evidence_complete=all(r.availability is not Availability.UNAVAILABLE for r in ev.values()),
            scoreable=scoreable, teardown_verified=teardown_verified,
        )
    path = layout.execution(sid)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, execution.to_dict())
    return {"scenario_id": sid, "status": execution.status.value, "dir": f"scenarios/{sid}",
            "execution_sha256": sha256_file(path)}


def build_run(tmp_path, repo_root, specs, *, status=SuiteStatus.COMPLETED, order=None):
    """``specs``: {scenario_id: build_scenario kwargs}. Returns the run directory."""
    root = tmp_path / "runs" / RUN_ID
    root.mkdir(parents=True)
    layout = RunLayout(root)
    entries = [build_scenario(layout, sid, **kwargs) for sid, kwargs in specs.items()]
    checks = contract.verify_artifacts(repo_root)
    run = EvaluationRun(
        run_id=RUN_ID, status=status, official=False, comparable=False,
        maui={"git_sha": "abc1234", "dirty": False},
        evaluation_contract={"artifacts": [c.to_dict() for c in checks], "all_match": True, "required_match": True},
        scenario_order=order or list(specs), scenarios=entries,
    )
    write_json_atomic(layout.run_manifest, run.to_dict())
    return root


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


# --- Loading and integrity ----------------------------------------------------


def test_valid_run_loads_with_provenance(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}})
    run = load_run_for_grading(root, repo_root)
    p = run.provenance
    assert (p.run_id, p.status, p.official, p.comparable) == (RUN_ID, SuiteStatus.COMPLETED, False, False)
    assert p.dataset_sha256 == contract.pinned_artifact(contract.DATASET_NAME).sha256
    assert p.ground_truth_sha256 == contract.pinned_artifact(contract.GROUND_TRUTH_JSON_NAME).sha256
    s = run.scenario("FE-01")
    assert s.gate is Gate.GRADEABLE
    assert s.execution_sha256 == sha256_file(root / "scenarios/FE-01/execution.json")
    assert s.refs["http/datachat.response.bin"] == "scenarios/FE-01/http/datachat.response.bin"


def test_loading_never_mutates_the_package(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}, "FE-02": {"status": ScenarioStatus.NOT_RUN}},
                     status=SuiteStatus.ABORTED_INFRA)
    before = _tree(root)
    load_run_for_grading(root, repo_root)
    assert _tree(root) == before
    assert not (root / "grading").exists()


def test_running_run_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}}, status=SuiteStatus.RUNNING)
    with pytest.raises(GradingInputError, match="RUNNING"):
        load_run_for_grading(root, repo_root)


def test_unfinished_scenario_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {"status": ScenarioStatus.RUNNING}},
                     status=SuiteStatus.ABORTED_INFRA)
    with pytest.raises(GradingInputError, match="not finished"):
        load_run_for_grading(root, repo_root)


def test_missing_manifest_refused(tmp_path, repo_root):
    (tmp_path / RUN_ID).mkdir()
    with pytest.raises(GradingInputError, match="manifest missing"):
        load_run_for_grading(tmp_path / RUN_ID, repo_root)


def test_missing_scenario_in_completed_run_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}}, order=["FE-01", "FE-02"])
    with pytest.raises(GradingInputError, match="missing"):
        load_run_for_grading(root, repo_root)


def test_missing_referenced_file_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}})
    (root / "scenarios/FE-01/usage/rows.json").unlink()
    with pytest.raises(GradingInputError, match="missing"):
        load_run_for_grading(root, repo_root)


def test_tampered_evidence_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}})
    target = root / "scenarios/FE-01/runtime/datachat.log"
    target.write_bytes(target.read_bytes().replace(b"North", b"Nord").replace(b"count", b"cnt"))
    with pytest.raises(GradingInputError, match="mismatch"):
        load_run_for_grading(root, repo_root)


def test_execution_hash_mismatch_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}})
    path = root / "scenarios/FE-01/execution.json"
    data = json.loads(path.read_text())
    data["notes"] = ["edited after closure"]
    path.write_text(json.dumps(data))
    with pytest.raises(GradingInputError, match="execution.json sha256"):
        load_run_for_grading(root, repo_root)


def test_path_traversal_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}})
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    path = root / "scenarios/FE-01/execution.json"
    data = json.loads(path.read_text())
    data["evidence"]["usage"]["files"] = [{"path": "../../../../outside.bin", "sha256": sha256_bytes(b"x"), "size": 1}]
    path.write_text(json.dumps(data))
    manifest = json.loads((root / "run_manifest.json").read_text())
    manifest["scenarios"][0]["execution_sha256"] = sha256_file(path)
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(GradingInputError, match="unsafe path"):
        load_run_for_grading(root, repo_root)


def test_changed_contract_artifact_refused(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {}})
    (contract.evaluation_dir(repo_root) / contract.GROUND_TRUTH_JSON_NAME).write_text("changed")
    with pytest.raises(GradingInputError, match="contract"):
        load_run_for_grading(root, repo_root)


def test_optional_unavailable_evidence_is_accepted(tmp_path, repo_root):
    export = EvidenceSourceRecord("export", Availability.UNAVAILABLE, "export fetch returned HTTP 404")
    root = build_run(tmp_path, repo_root, {"FE-01": {"export": export, "agent_record": None}})
    s = load_run_for_grading(root, repo_root).scenario("FE-01")
    assert s.gate is Gate.GRADEABLE
    assert s.evidence_complete is False
    assert s.artifacts.export_availability is Availability.UNAVAILABLE
    assert s.agent_run.availability is Availability.EXPECTED_ABSENT and s.agent_run.code_blocks == ()


# --- Gates --------------------------------------------------------------------


def test_gates_from_execution_state(tmp_path, repo_root):
    infra = Failure(code=InfraCode.INFRA_EVIDENCE_UNAVAILABLE, summary="x")
    root = build_run(tmp_path, repo_root, {
        "FE-01": {},
        "FE-02": {"scoreable": False},
        "FE-03": {"status": ScenarioStatus.NOT_SCOREABLE_INFRA, "scoreable": False, "failure": infra},
        "FE-04": {"status": ScenarioStatus.NOT_RUN},
    }, status=SuiteStatus.ABORTED_INFRA)
    run = load_run_for_grading(root, repo_root)
    gates = {s.scenario_id: s.gate for s in run.scenarios}
    assert gates == {"FE-01": Gate.GRADEABLE, "FE-02": Gate.NOT_SCOREABLE,
                     "FE-03": Gate.NOT_SCOREABLE_INFRA, "FE-04": Gate.NOT_RUN}
    assert run.scenario("FE-03").failure_code == "INFRA_EVIDENCE_UNAVAILABLE"
    assert run.scenario("FE-03").response is None


def test_scenarios_never_written_after_preflight_abort_are_not_run(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {}, status=SuiteStatus.ABORTED_INFRA, order=["FE-01"])
    s = load_run_for_grading(root, repo_root).scenario("FE-01")
    assert s.gate is Gate.NOT_RUN and s.execution_sha256 is None


def test_teardown_only_failure_stays_gradeable(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": {"teardown_verified": False}}, status=SuiteStatus.ABORTED_INFRA)
    s = load_run_for_grading(root, repo_root).scenario("FE-01")
    assert s.gate is Gate.GRADEABLE and s.teardown_verified is False


# --- Response normalization ---------------------------------------------------


def test_text_response_with_charts(tmp_path, repo_root):
    s = load_run_for_grading(build_run(tmp_path, repo_root, {"FE-01": {}}), repo_root).scenario("FE-01")
    r = s.response
    assert r.parse_status is ParseStatus.OK
    assert (r.type, r.answer_kind, r.text, r.log_id, r.http_status) == ("str", "text", "Ecco i conteggi.", 7, 200)
    assert r.charts[0]["type"] == "bar"
    assert r.raw == TEXT_BODY
    assert (r.final_kind, r.adapter_fallback_used) == ("text", False)
    assert s.datachat_request_id == RID


def test_dataframe_response_with_export_metadata():
    body = {"response": {"type": "dataframe", "value": [{"a": 1}], "result_rows": 30, "result_columns": 1,
                         "preview_rows": 1, "preview_columns": 1, "truncated": True, "note": "n",
                         "more_rows_available": True, "download_url": "/datachat/export/t"}}
    r = normalize_response(json.dumps(body).encode(), 200, "application/json")
    assert r.answer_kind == "table" and r.table_rows == ({"a": 1},)
    assert r.table_meta == {"result_rows": 30, "result_columns": 1, "preview_rows": 1,
                            "preview_columns": 1, "truncated": True}
    assert (r.note, r.more_rows_available, r.download_url) == ("n", True, "/datachat/export/t")


def test_image_response():
    body = {"response": {"type": "image", "value": "b'iVBOR'"}}
    r = normalize_response(json.dumps(body).encode(), 200, "application/json")
    assert r.answer_kind == "image" and r.text is None and r.raw["response"]["value"] == "b'iVBOR'"


@pytest.mark.parametrize("body,ctype,status", [
    (b"{not json", "application/json", ParseStatus.MALFORMED_JSON),
    (b"<html>", "text/html", ParseStatus.NON_JSON),
    (b'{"error": "Agent not active"}', "application/json", ParseStatus.ERROR_BODY),
    (b'{"response": {"type": "weird"}}', "application/json", ParseStatus.UNKNOWN_SHAPE),
    (b"[1, 2]", "application/json", ParseStatus.UNKNOWN_SHAPE),
])
def test_unparseable_responses_are_represented_not_raised(body, ctype, status):
    r = normalize_response(body, 500, ctype)
    assert r.parse_status is status
    assert r.answer_kind is None


def test_http_string_with_runtime_error_is_an_error():
    log = RUNTIME_LOG.replace("final_kind=text", "final_kind=error")
    body = {"response": {"type": "str", "value": "RUN_FAILED: provider error"}}
    r = normalize_response(json.dumps(body).encode(), 200, "application/json", parse_runtime_log(log))
    assert r.type == "str" and r.runtime_error and r.answer_kind == "error"


def test_image_artifact_reference(tmp_path, repo_root):
    run_root = build_run(tmp_path, repo_root, {"FE-11": {
        "body": {"response": {"type": "image", "value": "b'eA=='"}},
    }})
    sdir = run_root / "scenarios/FE-11"
    assert load_run_for_grading(run_root, repo_root).scenario("FE-11").artifacts.image_ref is None
    meta = {"kind": "image", "decode_status": "NOT_PNG"}
    meta_path = sdir / "artifacts/image.json"
    meta_path.parent.mkdir()
    write_json_atomic(meta_path, meta)
    path = sdir / "execution.json"
    data = json.loads(path.read_text())
    data["evidence"]["image"] = EvidenceSourceRecord(
        "image", Availability.PRESENT, None, (file_ref(meta_path, sdir),)).to_dict()
    path.write_text(json.dumps(data))
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    manifest["scenarios"][0]["execution_sha256"] = sha256_file(path)
    (run_root / "run_manifest.json").write_text(json.dumps(manifest))
    art = load_run_for_grading(run_root, repo_root).scenario("FE-11").artifacts
    assert art.image_meta_ref == "scenarios/FE-11/artifacts/image.json"
    assert art.image_meta["decode_status"] == "NOT_PNG" and art.image_ref is None


# --- Runtime, agent_runs, Usage ------------------------------------------------


def test_runtime_and_tool_evidence(tmp_path, repo_root):
    s = load_run_for_grading(build_run(tmp_path, repo_root, {"FE-01": {}}), repo_root).scenario("FE-01")
    rt = s.runtime
    assert (rt.final_kind, rt.final_answer_passed, rt.response_kind) == ("text", True, "text")
    assert (rt.request_end_status, rt.request_end_http_status, rt.trace_present) == ("ok", 200, True)
    assert rt.unparsed_lines == 1
    events = [(e.tool, e.outcome) for e in s.tools.runtime_events]
    assert events == [("aggregate", "result"), ("chart", "result"), ("classify_match", "failed")]
    assert s.tools.runtime_events[0].fields["group_by"] == "region"


def test_agent_run_code_exposed_without_observations(tmp_path, repo_root):
    s = load_run_for_grading(build_run(tmp_path, repo_root, {"FE-01": {}}), repo_root).scenario("FE-01")
    a = s.agent_run
    assert a.parse_ok and (a.state, a.steps_count, a.response_kind) == ("success", 3, "text")
    assert a.code_blocks[1].startswith("chart(") and s.tools.code_blocks == a.code_blocks
    assert a.ref == "scenarios/FE-01/agent_runs/record.jsonl"


def test_usage_rows_exposed(tmp_path, repo_root):
    s = load_run_for_grading(build_run(tmp_path, repo_root, {"FE-13": {}}), repo_root).scenario("FE-13")
    u = s.usage
    assert len(u.rows) == 2 and u.primary_log_id == 7 and u.primary_row_present
    assert [r["id"] for r in u.extra_rows] == [6]
    assert s.tools.usage_extra_rows == 1
