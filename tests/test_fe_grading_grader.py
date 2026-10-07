"""Tests for evaluation.datachat.grading.grader: run orchestration and result persistence.

Offline only: synthetic run packages (real hashes) built with the reader-test
builders, synthetic contract artifacts pinned in a temporary repo root. No
Maui, no DB, no provider.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.evidence import RunLayout, sha256_bytes, sha256_file
from evaluation.datachat.grading import catalogue as cat
from evaluation.datachat.grading import grader as gr
from evaluation.datachat.grading.model import (
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.grading.reader import load_run_for_grading
from evaluation.datachat.model import Failure, InfraCode, ScenarioStatus, SuiteStatus
from tests.test_fe_grading_catalogue import REGIONS, chart, table, text
from tests.test_fe_grading_reader import AGENT_RECORD, RID, RUN_ID, build_run

EA = ExecutionAssessment
RS = ExecutionAssessmentReason
IDENTITY = {
    "implementation": gr.GRADER_IMPLEMENTATION,
    "version": gr.GRADER_VERSION,
    "grading_schema_version": 2,
    "git_sha": "f" * 40,
    "dirty": False,
}


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    """Pinned synthetic evaluation artifacts. Grading never reads their content."""
    root = tmp_path / "repo"
    directory = contract.evaluation_dir(root)
    directory.mkdir(parents=True)
    pins = []
    for artifact in contract.PINNED_ARTIFACTS:
        content = f"synthetic {artifact.name}\n".encode()
        (directory / artifact.name).write_bytes(content)
        pins.append(
            contract.PinnedArtifact(
                artifact.name, sha256_bytes(content), artifact.required
            )
        )
    monkeypatch.setattr(contract, "PINNED_ARTIFACTS", tuple(pins))
    return root


def runtime(
    *tool_lines,
    final_kind="text",
    response_kind=None,
    fallback=False,
    end="ok",
    http=200,
):
    """A runtime log: optional ``(tool, "event=... k=v")`` lines, then the generic markers."""
    pre = (
        f"2026-01-01T00:00:01+00:00 INFO datachat.runtime request_id={RID} app_id=dino"
    )
    lines = [
        f"2026-01-01T00:00:01+00:00 INFO datachat.tools.{tool}_tool request_id={RID} app_id=dino {msg}"
        for tool, msg in tool_lines
    ]
    lines += [
        f"{pre} final_answer_check request_id={RID} passed=True final_kind={final_kind} reason=OK",
        f"{pre} chat_end request_id={RID} response_kind={response_kind or final_kind}",
        f"{pre} datachat_adapter_status request_id={RID} adapter_fallback_used={fallback}",
        f"{pre} datachat_request_end request_id={RID} status={end} http_status={http}",
    ]
    return "\n".join(lines) + "\n"


def agent(*code, response_kind="text"):
    return {
        **AGENT_RECORD,
        "tool_calls": [{"tool": "python_interpreter", "args": c} for c in code],
        "extra": {"channel": "datachat", "response_kind": response_kind},
    }


def body(response):
    return {"explanation": None, "log_id": 7, "response": response}


FE01 = dict(
    body=body(
        text("Ecco i conteggi.", [chart("bar", list(REGIONS), list(REGIONS.values()))])
    ),
    runtime_log=runtime(("chart", "event=tool_call_result kind=bar mode=records")),
    agent_record=agent(
        "r = aggregate(group_by='region', op='count')", "chart(kind='bar', data=r)"
    ),
)
FE05 = dict(
    body=body(text("Associazione negativa moderata (Spearman -0,58).")),
    runtime_log=runtime(),
    agent_record=agent(
        "correlation(col_x='wait_days', col_y='satisfaction', method='spearman')"
    ),
)


def grade(root, repo_root):
    return gr.grade_run(root, repo_root, grader_identity=lambda _: dict(IDENTITY))


def results(root):
    return root / "grading" / "results"


def scenario_json(root, sid):
    return json.loads((results(root) / "scenarios" / f"{sid}.json").read_text())


def snapshot(root: Path, *, exclude=("grading/results/",)) -> dict[str, str]:
    return {
        rel: sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        for rel in [p.relative_to(root).as_posix()]
        if not rel.startswith(exclude)
    }


def loaded(tmp_path, repo_root, sid, **spec):
    root = build_run(tmp_path, repo_root, {sid: spec})
    return load_run_for_grading(root, repo_root).scenario(sid)


# --- Execution assessment ---------------------------------------------------------------


# Contract for the execution reason vocabulary: one case per assess_execution branch.
@pytest.mark.parametrize(
    "spec,assessment,reason",
    [
        (FE01, EA.OK, RS.USABLE_OUTCOME),
        (
            dict(body=body(text("", [chart("bar", ["North"], [140])]))),
            EA.OK,
            RS.USABLE_OUTCOME,
        ),
        (
            dict(
                runtime_log=runtime(final_kind="error"),
                agent_record=agent("x", response_kind="error"),
            ),
            EA.FAILED,
            RS.ERROR_OUTCOME,
        ),
        (
            dict(runtime_log=runtime(final_kind="error"), agent_record=None),
            EA.FAILED,
            RS.ERROR_OUTCOME,
        ),
        (
            dict(
                http_status=500,
                body={"error": "NORMALIZE_FAILED"},
                runtime_log=runtime(end="error", http=500),
            ),
            EA.FAILED,
            RS.HTTP_5XX,
        ),
        (dict(body=body(text("   "))), EA.FAILED, RS.EMPTY_OUTCOME),
        (dict(body=body(table([]))), EA.FAILED, RS.EMPTY_OUTCOME),
        (dict(body=body({"type": "image", "value": ""})), EA.FAILED, RS.EMPTY_OUTCOME),
        (
            dict(
                runtime_log=runtime(final_kind="error"),
                agent_record=agent("x", response_kind="text"),
            ),
            EA.INDETERMINATE,
            RS.CONFLICTING_KINDS,
        ),
        (
            dict(http_status=400, body={"error": "bad"}),
            EA.RESPONSE_CONTRACT_VIOLATION,
            RS.HTTP_NON_2XX,
        ),
        (
            dict(body=b"{not json"),
            EA.RESPONSE_CONTRACT_VIOLATION,
            RS.UNPARSEABLE_RESPONSE,
        ),
        (
            dict(body={"something": "else"}),
            EA.RESPONSE_CONTRACT_VIOLATION,
            RS.UNPARSEABLE_RESPONSE,
        ),
        (
            dict(runtime_log=runtime(end="error")),
            EA.INDETERMINATE,
            RS.RUNTIME_STATUS_CONFLICT,
        ),
        (
            dict(runtime_log=runtime(fallback=True)),
            EA.INDETERMINATE,
            RS.ADAPTER_FALLBACK,
        ),
    ],
)
def test_execution_assessment_and_reason(tmp_path, repo_root, spec, assessment, reason):
    finding = gr.assess_execution(loaded(tmp_path, repo_root, "FE-01", **spec))
    assert (finding.assessment, finding.reason) == (assessment, reason)


def test_missing_response_is_indeterminate(tmp_path, repo_root):
    ev = loaded(tmp_path, repo_root, "FE-01", **FE01)
    finding = gr.assess_execution(dataclasses.replace(ev, response=None))
    assert (finding.assessment, finding.reason) == (
        EA.INDETERMINATE,
        RS.MISSING_RESPONSE,
    )


def test_assessment_refuses_non_gradeable(tmp_path, repo_root):
    with pytest.raises(gr.RunGradingError):
        gr.assess_execution(loaded(tmp_path, repo_root, "FE-01", scoreable=False))


def test_assessment_has_no_scenario_specific_policy():
    source = Path(gr.__file__).read_text()
    assert "FE-0" not in source and "FE-1" not in source


# --- End-to-end orchestration -----------------------------------------------------------


def test_fe01_pass_end_to_end(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    rg = grade(root, repo_root)
    assert scenario_json(root, "FE-01")["outcome"] == "TECHNICAL PASS"
    assert rg.counts_by_outcome["TECHNICAL PASS"] == 1
    assert rg.counts_by_failure == {
        "EXECUTION": 0,
        "RESPONSE CONTRACT": 0,
        "TOOL EXECUTION": 0,
        "ARTIFACT": 0,
    }


def test_fe05_technical_pass_and_stray_review_files_are_ignored(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-05": FE05})
    stray = root / "grading" / "reviews" / "FE-05.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("{not json")
    rg = grade(root, repo_root)
    g = scenario_json(root, "FE-05")
    assert (g["outcome"], g["failures"]) == ("TECHNICAL PASS", [])
    assert {c["status"] for c in g["checks"]} == {"NOT_APPLICABLE"}  # generic checks only
    assert (
        "manual_reviews" not in g
        and "pending_reviews" not in rg.to_dict()["scenarios"][0]
    )
    assert stray.read_text() == "{not json"


def test_execution_failure_even_when_checks_pass(tmp_path, repo_root):
    spec = {
        **FE01,
        "runtime_log": runtime(
            ("chart", "event=tool_call_result kind=bar"), final_kind="error"
        ),
        "agent_record": None,
    }
    root = build_run(tmp_path, repo_root, {"FE-01": spec})
    grade(root, repo_root)
    g = scenario_json(root, "FE-01")
    assert (g["outcome"], g["failures"]) == ("FAIL", ["EXECUTION"])
    assert all(c["status"] in ("MET", "NOT_APPLICABLE") for c in g["checks"])


def test_multiple_technical_failures_are_all_recorded(tmp_path, repo_root):
    spec = {
        **FE01,
        "body": body(text("   ")),
        "runtime_log": runtime(),
        "agent_record": agent("x = 1"),
    }
    root = build_run(tmp_path, repo_root, {"FE-01": spec})
    rg = grade(root, repo_root)
    g = scenario_json(root, "FE-01")
    assert (g["outcome"], g["failures"]) == ("FAIL", ["EXECUTION", "ARTIFACT"])
    assert rg.scenarios[0].failures == (
        TechnicalFailure.EXECUTION,
        TechnicalFailure.ARTIFACT,
    )
    assert (
        rg.counts_by_outcome["FAIL"],
        rg.counts_by_failure["EXECUTION"],
        rg.counts_by_failure["ARTIFACT"],
    ) == (1, 1, 1)


def test_fe11_text_answer_is_response_contract_failure(tmp_path, repo_root):
    spec = dict(
        body=body(text("La distribuzione è asimmetrica.")),
        runtime_log=runtime(),
        agent_record=agent("x"),
    )
    root = build_run(tmp_path, repo_root, {"FE-11": spec})
    grade(root, repo_root)
    g = scenario_json(root, "FE-11")
    assert (g["outcome"], g["failures"]) == ("FAIL", ["RESPONSE CONTRACT"])
    assert [(c["check_id"], c["status"]) for c in g["checks"]] == [
        ("generic.charts_valid", "NOT_APPLICABLE"),
        ("generic.dataframe_valid", "NOT_APPLICABLE"),
        ("generic.image_valid", "NOT_APPLICABLE"),
        ("generic.export_persisted", "NOT_APPLICABLE"),
        ("fe11.image_delivered", "NOT_MET"),
    ]


def test_non_gradeable_scenarios_get_null_outcomes(tmp_path, repo_root, monkeypatch):
    infra = Failure(InfraCode.INFRA_TOKEN_BUDGET, "not enough tokens")
    root = build_run(
        tmp_path,
        repo_root,
        {
            "FE-01": FE01,
            "FE-02": {"status": ScenarioStatus.NOT_SCOREABLE_INFRA, "failure": infra},
            "FE-03": {"scoreable": False},
            "FE-04": {"status": ScenarioStatus.NOT_RUN},
        },
        status=SuiteStatus.ABORTED_INFRA,
    )
    graded = []
    real = cat.grade_scenario_checks
    monkeypatch.setattr(
        cat,
        "grade_scenario_checks",
        lambda ev: graded.append(ev.scenario_id) or real(ev),
    )
    rg = grade(root, repo_root)
    assert graded == ["FE-01"]
    for sid, gate in (
        ("FE-02", "NOT_SCOREABLE_INFRA"),
        ("FE-03", "NOT_SCOREABLE"),
        ("FE-04", "NOT_RUN"),
    ):
        g = scenario_json(root, sid)
        assert (g["gate"], g["outcome"], g["failures"], g["checks"]) == (
            gate,
            None,
            [],
            [],
        )
    assert [(e.scenario_id, e.gate, e.outcome) for e in rg.scenarios] == [
        ("FE-01", Gate.GRADEABLE, TechnicalOutcome.TECHNICAL_PASS),
        ("FE-02", Gate.NOT_SCOREABLE_INFRA, None),
        ("FE-03", Gate.NOT_SCOREABLE, None),
        ("FE-04", Gate.NOT_RUN, None),
    ]
    assert rg.counts_by_outcome == {
        "TECHNICAL PASS": 1,
        "FAIL": 0,
        "INDETERMINATE": 0,
        "NOT_RUN": 1,
        "NOT_SCOREABLE_INFRA": 1,
        "NOT_SCOREABLE": 1,
    }


def test_scenario_order_follows_the_run(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-05": FE05, "FE-01": FE01})
    # Manifest order is the run's frozen order, here deliberately non-alphabetical.
    assert [e.scenario_id for e in grade(root, repo_root).scenarios] == [
        "FE-05",
        "FE-01",
    ]


# --- RunGrade --------------------------------------------------------------------------


def test_run_grade_provenance(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    grade(root, repo_root)
    data = json.loads((results(root) / "run_grade.json").read_text())
    assert data["run_id"] == RUN_ID and (data["official"], data["comparable"]) == (
        False,
        False,
    )
    assert data["grader"] == IDENTITY and data["schema_version"] == 2
    assert (
        data["contract"]["ground_truth_sha256"]
        == contract.pinned_artifact(contract.GROUND_TRUTH_JSON_NAME).sha256
    )
    assert (
        data["contract"]["dataset_sha256"]
        == contract.pinned_artifact(contract.DATASET_NAME).sha256
    )
    assert data["contract"]["scenario_execution_sha256"] == {
        "FE-01": sha256_file(RunLayout(root).execution("FE-01"))
    }
    assert "score" not in json.dumps(data) and "run_outcome" not in data
    assert set(data) == {
        "run_id",
        "official",
        "comparable",
        "grader",
        "contract",
        "scenarios",
        "counts_by_outcome",
        "counts_by_failure",
        "schema_version",
    }


def test_default_grader_identity_reads_git(tmp_path):
    identity = gr.read_grader_identity(contract.REPO_ROOT)
    assert len(identity["git_sha"]) == 40 and isinstance(identity["dirty"], bool)
    with pytest.raises(gr.RunGradingError):
        gr.read_grader_identity(tmp_path)


# --- Persistence and immutability ------------------------------------------------------


def test_layout_and_immutability(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01, "FE-05": FE05})
    before = snapshot(root)
    grade(root, repo_root)
    assert snapshot(root) == before
    assert sorted(
        p.relative_to(results(root)).as_posix()
        for p in results(root).rglob("*")
        if p.is_file()
    ) == ["run_grade.json", "scenarios/FE-01.json", "scenarios/FE-05.json"]
    assert not list(root.rglob("*.tmp"))


def test_regrading_is_byte_identical_and_replaces(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01, "FE-05": FE05})
    grade(root, repo_root)
    first = snapshot(results(root), exclude=())
    (results(root) / "scenarios" / "FE-01.json").write_text("stale derived output")
    grade(root, repo_root)
    assert snapshot(results(root), exclude=()) == first


def test_contract_mismatch_refused_without_writing(tmp_path, repo_root):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    (contract.evaluation_dir(repo_root) / contract.DATASET_NAME).write_text("changed\n")
    before = snapshot(root, exclude=())
    with pytest.raises(gr.RunGradingError, match="contract"):
        grade(root, repo_root)
    assert snapshot(root, exclude=()) == before


def test_persistence_failure_is_a_grading_error(tmp_path, repo_root, monkeypatch):
    root = build_run(tmp_path, repo_root, {"FE-01": FE01})
    before = snapshot(root)

    def boom(path, obj):
        raise OSError("disk full")

    monkeypatch.setattr(gr, "write_json_atomic", boom)
    with pytest.raises(gr.RunGradingError, match="persist"):
        grade(root, repo_root)
    assert snapshot(root) == before


def test_grader_imports_no_runtime_dependencies():
    source = Path(gr.__file__).read_text()
    for forbidden in (
        "psycopg",
        "runner",
        "maui_process",
        "client",
        "datachat.engine",
        "requests",
    ):
        assert (
            f"import {forbidden}" not in source and f".{forbidden} import" not in source
        )


# --- Execution provenance in persisted grades ------------------------------------------


def test_persisted_execution_provenance(tmp_path, repo_root):
    failed = {**FE01, "runtime_log": runtime(final_kind="error"), "agent_record": None}
    root = build_run(
        tmp_path,
        repo_root,
        {"FE-01": FE01, "FE-02": failed, "FE-03": {"status": ScenarioStatus.NOT_RUN}},
        status=SuiteStatus.ABORTED_INFRA,
    )
    grade(root, repo_root)
    picked = {
        sid: {
            k: scenario_json(root, sid)[k]
            for k in ("execution_assessment", "execution_assessment_reason", "outcome")
        }
        for sid in ("FE-01", "FE-02", "FE-03")
    }
    assert picked == {
        "FE-01": {
            "execution_assessment": "OK",
            "execution_assessment_reason": "USABLE_OUTCOME",
            "outcome": "TECHNICAL PASS",
        },
        "FE-02": {
            "execution_assessment": "FAILED",
            "execution_assessment_reason": "ERROR_OUTCOME",
            "outcome": "FAIL",
        },
        "FE-03": {
            "execution_assessment": None,
            "execution_assessment_reason": None,
            "outcome": None,
        },
    }
    first = snapshot(results(root), exclude=())
    grade(root, repo_root)
    assert snapshot(results(root), exclude=()) == first


def test_execution_ambiguity_with_and_without_technical_failure(tmp_path, repo_root):
    chartless_fe01 = {
        **FE01,
        "runtime_log": runtime(fallback=True),
        "agent_record": agent("r = 1"),
        "body": body(text("Ecco i conteggi.")),
    }
    ambiguous_fe01 = {
        **FE01,
        "runtime_log": runtime(
            ("chart", "event=tool_call_result kind=bar"), fallback=True
        ),
    }
    root = build_run(tmp_path, repo_root, {"FE-01": ambiguous_fe01})
    grade(root, repo_root)
    g = scenario_json(root, "FE-01")
    assert (
        g["execution_assessment"],
        g["execution_assessment_reason"],
        g["outcome"],
        g["failures"],
    ) == ("INDETERMINATE", "ADAPTER_FALLBACK", "INDETERMINATE", [])
    other = tmp_path / "second"
    root = build_run(other, repo_root, {"FE-01": chartless_fe01})
    grade(root, repo_root)
    g = scenario_json(root, "FE-01")
    assert (
        g["execution_assessment"],
        g["execution_assessment_reason"],
        g["outcome"],
        g["failures"],
    ) == ("INDETERMINATE", "ADAPTER_FALLBACK", "FAIL", ["ARTIFACT"])


@pytest.mark.parametrize(
    "spec,outcome,failures",
    [
        (dict(http_status=400, body={"error": "bad"}), "FAIL", ["RESPONSE CONTRACT"]),
        (dict(body=b"{not json"), "FAIL", ["RESPONSE CONTRACT"]),
        (dict(runtime_log=runtime(end="error")), "INDETERMINATE", []),
    ],
)
def test_execution_reason_outcomes_end_to_end(
    tmp_path, repo_root, spec, outcome, failures
):
    root = build_run(tmp_path, repo_root, {"FE-05": {**FE05, **spec}})
    grade(root, repo_root)
    g = scenario_json(root, "FE-05")
    assert (g["outcome"], g["failures"]) == (outcome, failures)
