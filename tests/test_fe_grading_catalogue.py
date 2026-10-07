"""Tests for evaluation.datachat.grading.catalogue, the FE-01…FE-14 grading contract.

Offline: synthetic ScenarioEvidence. The catalogue must produce technical
checks only: no outcome, no ground-truth or prose-driven result.
"""

import copy
import json
from dataclasses import fields

import pytest

from evaluation.datachat.grading import catalogue as cat
from evaluation.datachat.grading.model import (
    CheckStatus,
    EvidenceClass,
    Gate,
    TechnicalFailure,
)
from evaluation.datachat.grading.reader import (
    AgentRunEvidence,
    ArtifactEvidence,
    ScenarioEvidence,
    ToolEvidence,
    UsageRows,
    normalize_response,
    parse_runtime_log,
)
from evaluation.datachat.model import Availability, ScenarioStatus
from evaluation.datachat.scenarios import SCENARIO_ORDER

MET, NOT_MET, IND, NA = (CheckStatus.MET, CheckStatus.NOT_MET, CheckStatus.INDETERMINATE,
                         CheckStatus.NOT_APPLICABLE)
F = TechnicalFailure

REGIONS = {"North": 140, "South": 110, "East": 80, "West": 58, "Central": 12}


def _log(*events):
    """Runtime log lines: (logger_suffix, "event=... k=v")."""
    return "\n".join(
        f"2026-01-01T00:00:{i:02d} INFO datachat.tools.{tool}_tool request_id=r1 app_id=a {msg}"
        for i, (tool, msg) in enumerate(events)
    )


def evidence(sid, response=None, *, code=None, events=(), usage_rows=None, image_meta=None, image=False,
             gate=Gate.GRADEABLE, raw_body=None):
    base = f"scenarios/{sid}"
    runtime = parse_runtime_log(_log(*events))
    body = raw_body if raw_body is not None else json.dumps({"response": response}).encode()
    resp = normalize_response(body, 200, "application/json", runtime)
    agent = (AgentRunEvidence(Availability.PRESENT, ref=f"{base}/agent_runs/record.jsonl", parse_ok=True,
                              code_blocks=tuple(code))
             if code is not None else AgentRunEvidence(Availability.UNAVAILABLE, "no trace"))
    usage = (UsageRows(Availability.PRESENT, ref=f"{base}/usage/rows.json",
                       rows=tuple({"id": i} for i in range(1, usage_rows + 1)), primary_log_id=1,
                       primary_row_present=True)
             if usage_rows is not None else UsageRows(Availability.UNAVAILABLE, "no usage"))
    artifacts = ArtifactEvidence(
        image_meta=image_meta,
        image_meta_ref=f"{base}/artifacts/image.json" if image_meta else None,
        image_ref=f"{base}/artifacts/image.png" if image else None,
    )
    return ScenarioEvidence(
        scenario_id=sid, ordinal=int(sid[3:]), question="q", status=ScenarioStatus.EXECUTED, gate=gate,
        execution_sha256="e" * 64,
        refs={"http/datachat.response.bin": f"{base}/http/datachat.response.bin",
              "runtime/datachat.log": f"{base}/runtime/datachat.log"},
        response=resp, runtime=runtime, agent_run=agent, usage=usage, artifacts=artifacts,
        tools=ToolEvidence(runtime.tool_events),
    )


def text(value, charts=None, note=None):
    out = {"type": "str", "value": value, "charts": charts or []}
    if note:
        out["note"] = note
    return out


def table(rows, charts=None):
    return {"type": "dataframe", "value": rows, "charts": charts or [], "truncated": False}


def chart(kind, labels, data, label=None):
    dataset = {"data": data}
    if label:
        dataset["label"] = label
    return {"type": kind, "labels": labels, "datasets": [dataset]}


def grade(ev):
    return cat.grade_scenario_checks(ev)


def items(result):
    return {x.check_id: x for x in result.checks}


def status(result, check_id):
    item = items(result)[check_id]
    return item.status



# --- Catalogue authority ----------------------------------------------------------


def test_catalogue_covers_exactly_the_frozen_scenarios():
    assert tuple(cat.CATALOGUE) == SCENARIO_ORDER
    assert "FE-15" not in cat.CATALOGUE


def test_unknown_scenario_is_a_programmer_error():
    ev = evidence("FE-01", text("x"))
    with pytest.raises(cat.CatalogueError):
        cat.grade_scenario_checks(ScenarioEvidence(**{**{f.name: getattr(ev, f.name) for f in fields(ev)},
                                                     "scenario_id": "FE-15"}))


@pytest.mark.parametrize("gate", [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE])
def test_non_gradeable_evidence_is_refused(gate):
    with pytest.raises(cat.CatalogueError):
        grade(evidence("FE-01", text("x"), gate=gate))


def test_catalogue_takes_no_ground_truth():
    import inspect

    assert list(inspect.signature(cat.grade_scenario_checks).parameters) == ["evidence"]
    assert not hasattr(cat, "load_ground_truth")


# --- Cross-cutting properties -----------------------------------------------------

GENERIC = ["generic.charts_valid", "generic.dataframe_valid", "generic.image_valid", "generic.export_persisted"]
TECHNICAL = {
    "FE-01": ["fe01.chart_present"],
    "FE-09": ["fe09.chart_present"],
    "FE-11": ["fe11.image_delivered"],
    "FE-13": ["fe13.semantic_provider_call", "fe13.sentiment_analysis"],
    "FE-14": ["fe14.semantic_provider_call", "fe14.classify_match"],
}


@pytest.mark.parametrize("sid", SCENARIO_ORDER)
def test_only_technical_checks(sid):
    result = grade(evidence(sid, text("Una risposta qualsiasi."), code=["x = 1"], usage_rows=1))
    assert [x.check_id for x in result.checks] == GENERIC + TECHNICAL.get(sid, [])
    assert not hasattr(result, "outcome")
    for check in result.checks:
        assert (check.failure is not None) == (check.status is NOT_MET)
        assert check.evidence_class in (EvidenceClass.DIRECT, EvidenceClass.INFERRED)


@pytest.mark.parametrize("sid", SCENARIO_ORDER)
def test_prose_never_changes_the_checks(sid):
    plain = grade(evidence(sid, text("Risposta."), usage_rows=2))
    loaded = grade(evidence(sid, text("L'attesa causa insoddisfazione: differenza significativa, valori "
                                      "anomali, sono uguali, North 140."), usage_rows=2))
    assert [(x.check_id, x.status) for x in plain.checks] == [(x.check_id, x.status) for x in loaded.checks]


@pytest.mark.parametrize("sid", SCENARIO_ORDER)
def test_deterministic_and_non_mutating(sid):
    ev = evidence(sid, text("Testo."), code=["chart(data=x)"], usage_rows=2)
    before = (copy.deepcopy(ev.response.raw), ev.response.charts)
    first, second = grade(ev), grade(ev)
    assert first == second
    assert ev.response.raw == before[0] and ev.response.charts == before[1]


def test_catalogue_imports_nothing_from_maui():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cat))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert all(not m.split(".")[0] in {"routes", "services", "infrastructure", "main", "datachat", "flask"}
               for m in modules)


# --- FE-01 / FE-09: requested chart -------------------------------------------------


@pytest.mark.parametrize("sid", ["FE-01", "FE-09"])
def test_chart_present_regardless_of_kind_or_values(sid):
    for kind in ("bar", "pie", "line"):
        r = grade(evidence(sid, text("Ecco.", [chart(kind, ["a", "b"], [1, 999])])))
        assert status(r, f"{sid.lower().replace('-', '')}.chart_present") is MET


@pytest.mark.parametrize("sid", ["FE-01", "FE-09"])
def test_missing_chart_without_attempt_is_artifact_failure(sid):
    rows = [{"region": k, "count": v} for k, v in REGIONS.items()]
    check = items(grade(evidence(sid, table(rows), code=["aggregate(group_by='region', op='count')"])))[
        f"{sid.lower().replace('-', '')}.chart_present"]
    assert check.status is NOT_MET and check.failure is F.ARTIFACT
    assert check.evidence_class is EvidenceClass.DIRECT
    assert "no chart attempt" in check.note


@pytest.mark.parametrize("sid", ["FE-01", "FE-09"])
def test_chart_attempt_without_charts_is_still_artifact_failure(sid):
    check = items(grade(evidence(sid, text("Ecco il grafico."),
                                 events=[("chart", "event=tool_call_result kind=bar")])))[
        f"{sid.lower().replace('-', '')}.chart_present"]
    assert check.status is NOT_MET and check.failure is F.ARTIFACT
    assert "attempted" in check.note


def test_fe01_unparseable_response_is_indeterminate_not_failure():
    assert status(grade(evidence("FE-01", raw_body=b"not json")), "fe01.chart_present") is IND


# --- FE-11: image delivery ----------------------------------------------------------

IMAGE = {"type": "image", "value": "b'iVBORw0KGgo='", "charts": []}


def test_fe11_image_delivered():
    r = grade(evidence("FE-11", IMAGE, image_meta={"decode_status": "OK"}, image=True,
                       code=["chart(kind='bar', x='wait_days')"]))
    assert (status(r, "fe11.image_delivered"), status(r, "generic.image_valid")) == (MET, MET)
    assert items(r)["fe11.image_delivered"].evidence_refs == (
        "scenarios/FE-11/http/datachat.response.bin", "scenarios/FE-11/artifacts/image.json")


@pytest.mark.parametrize("answer", [text("La media è 10 giorni."), text("/tmp/plot_abc.png"),
                                    table([{"a": 1}])])
def test_fe11_image_not_delivered_is_response_contract(answer):
    r = grade(evidence("FE-11", answer))
    delivered = items(r)["fe11.image_delivered"]
    assert delivered.status is NOT_MET and delivered.failure is F.RESPONSE_CONTRACT
    assert delivered.evidence_class is EvidenceClass.DIRECT
    assert status(r, "generic.image_valid") is NA


def test_fe11_invalid_image_is_artifact():
    r = grade(evidence("FE-11", IMAGE, image_meta={"decode_status": "NOT_PNG"}))
    assert status(r, "fe11.image_delivered") is MET
    valid = items(r)["generic.image_valid"]
    assert valid.status is NOT_MET and valid.failure is F.ARTIFACT


def test_fe11_ambiguous_evidence_is_indeterminate():
    r = grade(evidence("FE-11", IMAGE))  # image answer, no metadata
    assert (status(r, "fe11.image_delivered"), status(r, "generic.image_valid")) == (MET, IND)
    r = grade(evidence("FE-11", raw_body=b"not json"))
    assert (status(r, "fe11.image_delivered"), status(r, "generic.image_valid")) == (IND, IND)


def test_fe11_reuses_generic_image_validity():
    """One source of truth: FE-11 adds only the RESPONSE CONTRACT check."""
    r = grade(evidence("FE-11", IMAGE, image_meta={"decode_status": "BASE64_INVALID"}))
    assert [x.check_id for x in r.checks if x.check_id.startswith("fe11.")] == ["fe11.image_delivered"]
    assert items(r)["generic.image_valid"].failure is F.ARTIFACT


# --- Generic technical checks (grading design §9.1) ---------------------------------


def _generic(response=None, sid="FE-03", **kw):
    return items(grade(evidence(sid, response, **kw)))


VALID_CHARTS = [chart("bar", ["a", "b"], [1, None]), chart("line", ["x"], [2.5]),
                {"type": "scatter", "labels": None, "datasets": [{"label": "x / y", "data": [{"x": 1, "y": 2}]}]}]


def test_valid_charts_are_met():
    check = _generic(text("x", VALID_CHARTS))["generic.charts_valid"]
    assert check.status is MET and check.observed == "3 valid"
    assert check.evidence_refs == ("scenarios/FE-03/http/datachat.response.bin",)


@pytest.mark.parametrize("bad", [
    "not a chart",                                              # non-dict entry, dropped by ``charts``
    chart("bar", ["a", "b"], [1]),                              # label/data length mismatch
    chart("radar", ["a"], [1]),                                 # unsupported type
    {"type": "bar", "labels": ["a"]},                           # no datasets
    {"type": "bar", "datasets": [{"data": [1]}]},               # no labels
    chart("bar", ["a"], ["12"]),                                # non-numeric value
    chart("bar", ["a"], [True]),
    {"type": "scatter", "datasets": [{"data": [{"x": 1}]}]},    # point without y
])
def test_malformed_chart_is_artifact_failure(bad):
    check = _generic(text("x", [chart("bar", ["a"], [1]), bad]))["generic.charts_valid"]
    assert check.status is NOT_MET and check.failure is F.ARTIFACT
    assert check.evidence_class is EvidenceClass.DIRECT
    assert check.observed.startswith("chart 2: ")


def test_non_dict_chart_entry_is_not_hidden_by_normalization():
    ev = evidence("FE-01", text("x", ["oops"]))
    assert ev.response.charts == () and ev.response.chart_entries == ["oops"]
    r = items(grade(ev))
    assert r["generic.charts_valid"].status is NOT_MET
    assert r["fe01.chart_present"].status is NOT_MET


def test_charts_not_a_list_is_artifact_failure():
    check = _generic({"type": "str", "value": "x", "charts": {"type": "bar"}})["generic.charts_valid"]
    assert check.status is NOT_MET and check.failure is F.ARTIFACT


def test_no_charts_is_not_applicable():
    assert _generic(text("x"))["generic.charts_valid"].status is NA
    assert _generic({"type": "str", "value": "x"})["generic.charts_valid"].status is NA


def _df(rows, **meta):
    return {"type": "dataframe", "value": rows, "charts": [], **meta}


ROWS = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_valid_dataframe_is_met():
    full = dict(result_rows=2, result_columns=2, preview_rows=2, preview_columns=2, truncated=False)
    assert _generic(_df(ROWS, **full))["generic.dataframe_valid"].status is MET
    cut = dict(result_rows=10, result_columns=2, preview_rows=2, preview_columns=2, truncated=True)
    assert _generic(_df(ROWS, **cut))["generic.dataframe_valid"].status is MET
    assert _generic(_df([], result_rows=0, result_columns=3, preview_rows=0, preview_columns=0,
                        truncated=False))["generic.dataframe_valid"].status is MET
    assert _generic(text("x"))["generic.dataframe_valid"].status is NA


@pytest.mark.parametrize("response", [
    _df("not rows"),
    _df([{"a": 1}, ["b"]]),
    _df(ROWS, preview_rows=3),
    _df(ROWS, preview_columns=1),
    _df(ROWS, preview_rows="2"),
    _df(ROWS, truncated="no"),
    _df(ROWS, preview_rows=2, result_rows=1),
    _df(ROWS, result_rows=2, result_columns=2, preview_rows=2, preview_columns=2, truncated=True),
])
def test_malformed_dataframe_is_response_contract_failure(response):
    check = _generic(response)["generic.dataframe_valid"]
    assert check.status is NOT_MET and check.failure is F.RESPONSE_CONTRACT


@pytest.mark.parametrize("sid", ["FE-03", "FE-11"])
def test_generic_image_validity(sid):
    ok = _generic(IMAGE, sid, image_meta={"decode_status": "OK"})["generic.image_valid"]
    assert ok.status is MET and "scenarios/%s/artifacts/image.json" % sid in ok.evidence_refs
    bad = _generic(IMAGE, sid, image_meta={"decode_status": "NOT_PNG"})["generic.image_valid"]
    assert bad.status is NOT_MET and bad.failure is F.ARTIFACT
    assert _generic(IMAGE, sid)["generic.image_valid"].status is IND
    assert _generic(IMAGE, sid, image_meta={"other": 1})["generic.image_valid"].status is IND
    assert _generic(text("x"), sid)["generic.image_valid"].status is NA


def _export_evidence(availability=None, reason=None, locator=None, ref=None, url="/datachat/export/abc"):
    ev = evidence("FE-03", {**text("x"), "download_url": url} if url else text("x"))
    from dataclasses import replace
    return replace(ev, artifacts=ArtifactEvidence(export_availability=availability, export_reason=reason,
                                                   export_ref=ref, export_locator=locator or {}))


def test_export_persisted_when_download_url():
    check = items(grade(_export_evidence(Availability.PRESENT, ref="scenarios/FE-03/artifacts/export.csv",
                                         locator={"http_status": 200})))["generic.export_persisted"]
    assert check.status is MET
    assert "scenarios/FE-03/artifacts/export.csv" in check.evidence_refs


def test_failed_export_is_artifact_failure():
    check = items(grade(_export_evidence(Availability.UNAVAILABLE, "export fetch returned HTTP 404",
                                         {"http_status": 404})))["generic.export_persisted"]
    assert check.status is NOT_MET and check.failure is F.ARTIFACT


@pytest.mark.parametrize("availability,reason,locator", [
    (None, None, None),                                                    # no export record
    (Availability.UNAVAILABLE, "download_url rejected: unsafe", None),     # never fetched
    (Availability.UNAVAILABLE, "no HTTP response: Timeout", {"request_id": "r"}),
    (Availability.NOT_APPLICABLE, "no download_url", None),                # contradicts the response
])
def test_unclear_export_evidence_is_indeterminate(availability, reason, locator):
    check = items(grade(_export_evidence(availability, reason, locator)))["generic.export_persisted"]
    assert check.status is IND


def test_no_download_url_is_not_applicable():
    assert items(grade(_export_evidence(url=None)))["generic.export_persisted"].status is NA


def test_generic_checks_on_unparsed_response_are_indeterminate():
    r = _generic(raw_body=b"not json")
    assert {r[i].status for i in GENERIC} == {IND}


# --- FE-13 / FE-14: provider-backed semantic capability ------------------------------

SENT_EVENT = ("sentiment", "event=tool_call_result tool=sentiment_analysis aggregate=False rows=400")
CM_EVENT = ("classify_match", "event=tool_call_result tool=classify_match rows=400 analyzable_rows=340")
KW_EVENT = ("keywords", "event=tool_call_result col=comment language=italian")


@pytest.mark.parametrize("sid,tool,event", [("FE-13", "sentiment_analysis", SENT_EVENT),
                                            ("FE-14", "classify_match", CM_EVENT)])
def test_semantic_capability_observed(sid, tool, event):
    r = grade(evidence(sid, text("x"), events=[event], usage_rows=2))
    prefix = sid.lower().replace("-", "")
    provider, used = items(r)[f"{prefix}.semantic_provider_call"], items(r)[f"{prefix}.{tool}"]
    assert provider.status is MET and provider.evidence_class is EvidenceClass.DIRECT
    assert used.status is MET and used.evidence_class is EvidenceClass.INFERRED


@pytest.mark.parametrize("sid,tool", [("FE-13", "sentiment_analysis"), ("FE-14", "classify_match")])
def test_semantic_capability_absent(sid, tool):
    prefix = sid.lower().replace("-", "")
    r = grade(evidence(sid, text("x"), code=["df['comment'].str.contains('attesa').sum()"], usage_rows=1))
    provider = items(r)[f"{prefix}.semantic_provider_call"]
    assert provider.status is NOT_MET and provider.failure is F.TOOL_EXECUTION
    used = items(r)[f"{prefix}.{tool}"]
    # An INFERRED negative: the outcome builder never fails on it alone.
    assert used.status is NOT_MET and used.evidence_class is EvidenceClass.INFERRED
    assert used.failure is F.TOOL_EXECUTION
    r = grade(evidence(sid, text("x")))
    assert status(r, f"{prefix}.semantic_provider_call") is IND


def test_fe14_tool_order_is_not_graded():
    r = grade(evidence("FE-14", text("x"), events=[CM_EVENT, KW_EVENT], usage_rows=2))
    assert all(x.status in (MET, NA) for x in r.checks)
