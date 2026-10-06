"""Tests for evaluation.datachat.model, the Functional Evaluation execution model.

Pure unit tests: no Flask app, no Maui process, no DB, no network, no files.
They pin the frozen enum values and phase order, immutability of value
records, the defaults of mutable execution records, and JSON round-trips of
the records the harness reads back from disk (HttpExchange,
ScenarioExecution, EvaluationRun).
"""

import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from evaluation.datachat.model import (
    SCHEMA_VERSION,
    ArtifactRecord,
    Availability,
    EvaluationRun,
    EvidenceSourceRecord,
    Failure,
    FileRef,
    HttpExchange,
    HttpRequestRecord,
    HttpResponseRecord,
    InfraCode,
    Phase,
    PhaseStatus,
    ScenarioExecution,
    ScenarioFlags,
    ScenarioStatus,
    SuiteStatus,
    TransportError,
    UsageEvidence,
    utc_now,
)

PHASE_ORDER = ["PRE_CLEANUP", "START", "DATACHAT", "COLLECT", "TEARDOWN", "CLOSE"]


def _json_round_trip(data: dict) -> dict:
    return json.loads(json.dumps(data))


def _datachat_exchange() -> HttpExchange:
    return HttpExchange(
        key="datachat",
        attempt=1,
        method="POST",
        path="/datachat",
        sent_at="2026-10-06T08:15:30+00:00",
        completed_at="2026-10-06T08:15:42+00:00",
        duration_ms=12000,
        request=HttpRequestRecord(
            headers_redacted={"X-API-KEY": "<redacted>", "Content-Type": "application/json"},
            content_type="application/json",
            body_file="http/datachat.request.bin",
            body_sha256="a" * 64,
        ),
        response=HttpResponseRecord(
            status=200,
            headers={"X-Request-ID": "0123456789abcdef"},
            content_type="application/json",
            body_file="http/datachat.response.bin",
            body_sha256="b" * 64,
            size=512,
        ),
        request_id="0123456789abcdef",
    )


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


@pytest.mark.parametrize(
    "enum_cls, values",
    [
        (SuiteStatus, ["RUNNING", "COMPLETED", "ABORTED_INFRA"]),
        (ScenarioStatus, ["NOT_RUN", "RUNNING", "EXECUTED", "NOT_SCOREABLE_INFRA"]),
        (PhaseStatus, ["PENDING", "OK", "FAILED", "SKIPPED"]),
        (Availability, ["PRESENT", "EXPECTED_ABSENT", "UNAVAILABLE", "NOT_APPLICABLE"]),
        (Phase, PHASE_ORDER),
    ],
)
def test_enums_have_exactly_the_frozen_values(enum_cls, values):
    assert [member.value for member in enum_cls] == values
    assert all(isinstance(member, str) for member in enum_cls)


def test_phase_preserves_execution_order():
    assert [phase.value for phase in Phase] == PHASE_ORDER


def test_infra_code_has_exactly_the_seven_accepted_codes():
    assert {code.value for code in InfraCode} == {
        "INFRA_PREFLIGHT",
        "INFRA_MAUI_UNAVAILABLE",
        "INFRA_EVIDENCE_UNAVAILABLE",
        "INFRA_FRESH_ENGINE_UNPROVEN",
        "INFRA_TEARDOWN_UNVERIFIED",
        "INFRA_TOKEN_BUDGET",
        "INFRA_HTTP_TIMEOUT",
    }
    assert len(InfraCode) == 7


def test_utc_now_is_timezone_aware_utc():
    parsed = datetime.fromisoformat(utc_now())
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


@pytest.mark.parametrize(
    "record, attribute",
    [
        (Failure(code=InfraCode.INFRA_PREFLIGHT, summary="port in use"), "summary"),
        (FileRef(path="usage/rows.json", sha256="c" * 64, size=10), "size"),
        (TransportError(type="ConnectionError", message="refused"), "message"),
        (HttpRequestRecord(headers_redacted={}), "content_type"),
        (_datachat_exchange(), "request_id"),
        (_datachat_exchange().response, "status"),
        (EvidenceSourceRecord(source="usage", availability=Availability.PRESENT), "reason"),
        (
            ArtifactRecord(kind="export", source_request_id="r", status=Availability.PRESENT),
            "path",
        ),
        (
            UsageEvidence(request_id="r", query_sql="SELECT 1", schema="maui", executed_at="t"),
            "rows",
        ),
    ],
)
def test_frozen_records_reject_mutation(record, attribute):
    with pytest.raises(FrozenInstanceError):
        setattr(record, attribute, None)


def test_enum_values_serialize_to_plain_strings():
    failure = Failure(
        code=InfraCode.INFRA_HTTP_TIMEOUT, summary="timeout", phase=Phase.DATACHAT
    )
    data = failure.to_dict()
    assert data == {
        "code": "INFRA_HTTP_TIMEOUT",
        "summary": "timeout",
        "phase": "DATACHAT",
        "detail": None,
        "traceback_file": None,
    }
    assert type(data["code"]) is str
    assert type(data["phase"]) is str

    artifact = ArtifactRecord(
        kind="image",
        source_request_id="0123456789abcdef",
        status=Availability.UNAVAILABLE,
        decode_status="BASE64_INVALID",
    ).to_dict()
    assert type(artifact["status"]) is str
    assert artifact["status"] == "UNAVAILABLE"


def test_usage_evidence_carries_schema_version():
    usage = UsageEvidence(
        request_id="0123456789abcdef",
        query_sql="SELECT ... FROM logs WHERE request_id = %s ORDER BY id",
        schema="maui",
        executed_at="2026-10-06T08:15:43+00:00",
        rows=({"id": 7, "service": "datachat"},),
        primary_log_id=7,
        primary_row_present=True,
    ).to_dict()
    assert usage["schema_version"] == 1
    assert usage["rows"] == [{"id": 7, "service": "datachat"}]
    assert json.loads(json.dumps(usage)) == usage


def test_http_exchange_round_trips():
    exchange = _datachat_exchange()
    data = exchange.to_dict()
    assert data["schema_version"] == 1
    assert HttpExchange.from_dict(_json_round_trip(data)) == exchange


def test_http_exchange_without_response_round_trips_nulls():
    exchange = HttpExchange(
        key="teardown",
        attempt=2,
        method="POST",
        path="/enddatachat",
        sent_at="2026-10-06T08:16:00+00:00",
        completed_at="2026-10-06T08:16:01+00:00",
        duration_ms=1000,
        request=HttpRequestRecord(
            headers_redacted={"X-API-KEY": "<redacted>"},
            form={"lang": "ENG", "dataset_sha256": "f" * 64},
        ),
        transport_error=TransportError(type="ConnectionError", message="refused"),
    )
    data = _json_round_trip(exchange.to_dict())
    assert data["response"] is None
    assert data["request_id"] is None
    assert data["request"]["body_file"] is None
    assert HttpExchange.from_dict(data) == exchange


def test_scenario_execution_new_creates_six_pending_phases_in_order():
    execution = ScenarioExecution.new("run-1", "FE-01", 1, "How many rows?")
    assert execution.status is ScenarioStatus.RUNNING
    assert execution.schema_version == 1
    assert execution.question_sha256 == hashlib.sha256(b"How many rows?").hexdigest()
    assert execution.started_at is not None
    assert len(execution.phases) == 6
    assert [outcome.phase.value for outcome in execution.phases] == PHASE_ORDER
    assert all(outcome.status is PhaseStatus.PENDING for outcome in execution.phases)
    assert execution.flags == ScenarioFlags()
    assert not any(vars(execution.flags).values())


def test_mark_not_run_skips_every_phase():
    execution = ScenarioExecution.new("run-1", "FE-02", 2, "q")
    execution.mark_not_run()
    assert execution.status is ScenarioStatus.NOT_RUN
    assert len(execution.phases) == 6
    assert all(outcome.status is PhaseStatus.SKIPPED for outcome in execution.phases)


def test_scenario_execution_round_trips():
    execution = ScenarioExecution.new("run-1", "FE-03", 3, "Plot revenue by region")
    execution.status = ScenarioStatus.NOT_SCOREABLE_INFRA
    execution.failure = Failure(
        code=InfraCode.INFRA_EVIDENCE_UNAVAILABLE,
        summary="datachat_request_end missing",
        phase=Phase.COLLECT,
        detail="TimeoutError: marker deadline",
        traceback_file="failures/FE-03.txt",
    )
    execution.requests = {"datachat": ["http/datachat.json"], "teardown": ["http/teardown.json", "http/teardown.2.json"]}
    execution.phases[0].status = PhaseStatus.OK
    execution.phases[0].request_keys.append("pre_cleanup")
    execution.phases[3].status = PhaseStatus.FAILED
    execution.phases[3].failure = execution.failure
    execution.phases[3].notes.append("runtime marker missing")
    execution.evidence = {
        "usage": EvidenceSourceRecord(
            source="usage",
            availability=Availability.PRESENT,
            files=(FileRef(path="usage/rows.json", sha256="d" * 64, size=99),),
            locator={"request_id": "0123456789abcdef", "row_count": 2},
        ),
        "agent_runs": EvidenceSourceRecord(
            source="agent_runs",
            availability=Availability.EXPECTED_ABSENT,
            reason="no trace on RUN_FAILED",
        ),
    }
    execution.flags.reached_datachat = True
    execution.flags.fresh_engine_proven = True
    execution.notes.append("previous engine absent")
    execution.ended_at = utc_now()

    data = _json_round_trip(execution.to_dict())
    assert data["status"] == "NOT_SCOREABLE_INFRA"
    assert data["phases"][3]["failure"]["code"] == "INFRA_EVIDENCE_UNAVAILABLE"
    assert data["evidence"]["agent_runs"]["availability"] == "EXPECTED_ABSENT"
    assert ScenarioExecution.from_dict(data) == execution


def test_scenario_execution_null_fields_round_trip():
    execution = ScenarioExecution.new("run-1", "FE-04", 4, "q")
    data = _json_round_trip(execution.to_dict())
    assert data["failure"] is None
    assert data["ended_at"] is None
    assert data["phases"][0]["started_at"] is None
    assert data["phases"][0]["failure"] is None
    assert ScenarioExecution.from_dict(data) == execution


def test_evaluation_run_round_trips():
    run = EvaluationRun(
        run_id="20261006T081530Z_aa1ff33",
        started_at="2026-10-06T08:15:30+00:00",
        status=SuiteStatus.ABORTED_INFRA,
        official=True,
        failure=Failure(
            code=InfraCode.INFRA_TEARDOWN_UNVERIFIED,
            summary="cleanup_result missing",
            phase=Phase.TEARDOWN,
        ),
        maui={"git_sha": "aa1ff33", "branch": "feat/x", "dirty": False, "dirty_files": []},
        evaluation_contract={"hashes": {"ground_truth.json": "e" * 64}, "all_match": True},
        runtime={"python": "3.10.13", "gunicorn_argv": ["gunicorn", "main:app"], "port": 5099},
        datachat_config={"provider": "openai", "sql_enabled": False, "api_key_present": True},
        eval_identity={"label": "fe-eval-v1", "user_id": 42, "api_key_sha256": None},
        token_balance_at_preflight=1000.0,
        scenario_order=["FE-01", "FE-02"],
        scenarios=[
            {"scenario_id": "FE-01", "status": "EXECUTED", "dir": "scenarios/FE-01", "execution_sha256": "0" * 64},
            {"scenario_id": "FE-02", "status": "NOT_RUN", "dir": "scenarios/FE-02", "execution_sha256": None},
        ],
    )
    data = _json_round_trip(run.to_dict())
    assert data["schema_version"] == 1
    assert data["status"] == "ABORTED_INFRA"
    assert data["ended_at"] is None
    assert EvaluationRun.from_dict(data) == run


def test_evaluation_run_defaults_are_conservative_and_null_safe():
    run = EvaluationRun(run_id="r")
    assert run.status is SuiteStatus.RUNNING
    assert run.official is False
    assert run.comparable is False
    data = _json_round_trip(run.to_dict())
    assert data["failure"] is None
    assert data["token_balance_at_preflight"] is None
    assert EvaluationRun.from_dict(data) == run


def test_mutable_defaults_are_independent_between_instances():
    first = ScenarioExecution.new("run-1", "FE-01", 1, "q")
    second = ScenarioExecution.new("run-1", "FE-02", 2, "q")
    first.notes.append("x")
    first.requests["datachat"] = ["http/datachat.json"]
    first.flags.scoreable = True
    first.phases[0].notes.append("y")
    first.phases[0].request_keys.append("pre_cleanup")
    assert second.notes == []
    assert second.requests == {}
    assert second.flags.scoreable is False
    assert second.phases[0].notes == []
    assert second.phases[0].request_keys == []
    assert first.phases[0] is not second.phases[0]

    run_a, run_b = EvaluationRun(run_id="a"), EvaluationRun(run_id="b")
    run_a.scenarios.append({"scenario_id": "FE-01"})
    run_a.maui["dirty"] = True
    run_a.scenario_order.append("FE-01")
    assert run_b.scenarios == []
    assert run_b.maui == {}
    assert run_b.scenario_order == []

    ev_a = EvidenceSourceRecord(source="a", availability=Availability.PRESENT)
    ev_b = EvidenceSourceRecord(source="b", availability=Availability.PRESENT)
    assert ev_a.locator is not ev_b.locator
