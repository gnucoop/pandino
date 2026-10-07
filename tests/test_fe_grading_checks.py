"""Tests for evaluation.datachat.grading.checks, the deterministic primitives.

Pure unit tests: no files, no Maui, no DB, no provider, no scenario IDs.
Central property: parser or extraction uncertainty yields INDETERMINATE,
never NOT_MET.
"""

import json

import pytest

from evaluation.datachat.grading import checks as c
from evaluation.datachat.grading.model import (
    CheckStatus,
    EvidenceClass,
    TechnicalFailure,
)
from evaluation.datachat.grading.reader import normalize_response, parse_runtime_log

MET, NOT_MET, IND = CheckStatus.MET, CheckStatus.NOT_MET, CheckStatus.INDETERMINATE


def _response(payload, status=200, ctype="application/json"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return normalize_response(body, status, ctype)


# --- Exact values and labels ---------------------------------------------------


def test_compare_exact():
    assert c.compare_exact(140, 140).status is MET
    assert c.compare_exact(140, 141).status is NOT_MET
    assert c.compare_exact(True, 1).status is NOT_MET
    assert c.compare_exact("South", " south ").status is NOT_MET  # no implicit normalization
    assert c.compare_exact("South", " south ", normalize=True).status is MET
    assert c.compare_exact("South", "Southern", normalize=True).status is NOT_MET
    assert c.compare_exact(140, None).status is IND


def test_normalize_label_is_conservative():
    assert c.normalize_label("  In   Person ") == "in person"
    assert c.normalize_label("Southern") != c.normalize_label("South")
    assert c.normalize_label(3) is None and c.normalize_label("   ") is None


# --- Charts --------------------------------------------------------------------------

BAR = {"type": "bar", "labels": ["North", "South"], "datasets": [{"label": "count", "data": [140, 110]}]}


def _with_charts(charts):
    return _response({"response": {"type": "str", "value": "ok", "charts": charts}})


def test_chart_count():
    assert c.check_chart_count(_with_charts([])).status is NOT_MET
    assert c.check_chart_count(_response({"response": {"type": "str", "value": "ok"}})).status is NOT_MET
    assert c.check_chart_count(_with_charts([BAR])).status is MET
    assert c.check_chart_count(_with_charts([BAR, BAR]), min_count=2).status is MET
    assert c.check_chart_count(_with_charts([BAR]), min_count=2).status is NOT_MET
    assert c.check_chart_count(_response(b"<html>", 500, "text/html")).status is IND


# --- CheckResult construction -----------------------------------------------------------


def _policy(**extra):
    return dict(check_id="x.counts", evidence_class=EvidenceClass.DIRECT, failure=TechnicalFailure.ARTIFACT, **extra)


def test_make_check_attaches_policy_and_keeps_refs():
    refs = ("scenarios/X/http/datachat.response.bin",)
    result = c.make_check(c.compare_exact(140, 141), evidence_refs=refs, **_policy())
    assert result.status is NOT_MET and result.failure is TechnicalFailure.ARTIFACT
    assert result.evidence_refs == refs
    assert json.loads(json.dumps(result.to_dict()))["observed"] == 141


@pytest.mark.parametrize("outcome", [
    c.compare_exact(1, None), c.not_applicable("caller says so"), c.compare_exact(1, 1),
])
def test_failure_category_only_on_not_met(outcome):
    assert c.make_check(outcome, **_policy()).failure is None


# --- INDETERMINATE safety and Slice 1 integration ----------------------------------------


@pytest.mark.parametrize("outcome", [
    c.compare_exact(140, None),                                         # missing field
    c.compare_exact("South", 3, normalize=True),                        # label not normalizable
    c.check_chart_count(_response(b'{"response": {"type": "weird"}}')),
])
def test_uncertainty_is_indeterminate_never_not_met(outcome):
    assert outcome.status is IND


def test_normalized_response_flow_end_to_end():
    log = ("2026-01-01T00:00:00+00:00 INFO datachat.runtime request_id=r app_id=a "
           "final_answer_check request_id=r passed=True final_kind=text\n")
    body = json.dumps({"log_id": 1, "response": {"type": "str", "value": "ok", "charts": [BAR]}}).encode()
    resp = normalize_response(body, 200, "application/json", parse_runtime_log(log))
    assert c.charts(resp).value[0] == BAR
    outcome = c.check_chart_count(resp)
    result = c.make_check(outcome, evidence_refs=("scenarios/X/http/datachat.response.bin",), **_policy())
    assert result.status is MET and result.failure is None
