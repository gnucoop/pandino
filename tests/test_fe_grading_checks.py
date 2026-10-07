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


# --- Numbers -------------------------------------------------------------------


@pytest.mark.parametrize("raw,value", [
    (12, 12.0), (12.5, 12.5), ("12", 12.0), ("12.5", 12.5), ("12,5", 12.5), (" 12,5 ", 12.5),
    ("€ 1.234,50", 1234.5), ("€1,234.50", 1234.5), ("€1234.50", 1234.5), ("1234.50", 1234.5),
    ("1.234.567", 1234567.0), ("1,234,567", 1234567.0), ("1234 €", 1234.0),
    ("-0.579", -0.579), ("−0,579", -0.579), ("-€12", -12.0), ("€ -12", -12.0), ("+3", 3.0),
    ("0.579", 0.579), (0, 0.0), ("0", 0.0),
])
def test_parse_number_supported_forms(raw, value):
    parsed = c.parse_number(raw)
    assert parsed.ok and parsed.value == pytest.approx(value) and not parsed.is_percent


def test_parse_percent_is_explicit():
    parsed = c.parse_number("45%")
    assert parsed.ok and parsed.is_percent and parsed.value == 45.0
    assert c.to_number("45 %", percent=True).value == pytest.approx(0.45)
    assert c.to_number("45,5%", percent=True).value == pytest.approx(0.455)
    assert not c.to_number("45%").ok  # percent not declared → refused, never 45 or 0.45
    assert c.to_number(0.45, percent=True).value == 0.45


@pytest.mark.parametrize("raw", [
    "1.234", "1,234", "12.500",           # ambiguous: decimal or thousands
    "abc", "", "  ", "12a", "1.2.3", "1,23,4", "1.234,5,6", ".5", "€", "%",
    "circa 12", "12 e 13", True, False, None, [1], float("nan"), float("inf"), "€45%",
])
def test_parse_number_rejects_malformed_or_ambiguous(raw):
    parsed = c.parse_number(raw)
    assert not parsed.ok and parsed.value is None and parsed.error


# --- Exact, tolerance, ranges, signs ------------------------------------------


def test_compare_exact():
    assert c.compare_exact(140, 140).status is MET
    assert c.compare_exact(140, 141).status is NOT_MET
    assert c.compare_exact(True, 1).status is NOT_MET
    assert c.compare_exact("South", " south ").status is NOT_MET  # no implicit normalization
    assert c.compare_exact("South", " south ", normalize=True).status is MET
    assert c.compare_exact("South", "Southern", normalize=True).status is NOT_MET
    assert c.compare_exact(140, None).status is IND


@pytest.mark.parametrize("observed,status", [
    (18.77, MET), ("18,8", MET), (18.82, MET), (18.72, MET),   # 18.77 ± 0.05, inclusive
    (18.83, NOT_MET), (18.71, NOT_MET), ("diciotto", IND), (None, IND),
])
def test_compare_number_tolerance(observed, status):
    assert c.compare_number(18.77, observed, tolerance=0.05).status is status


def test_tolerance_boundaries_are_inclusive_in_both_directions():
    assert c.compare_number(0.10, 0.15, tolerance=0.05).status is MET
    assert c.compare_number(0.10, 0.05, tolerance=0.05).status is MET
    assert c.compare_number(-0.579, -0.569, tolerance=0.01).status is MET
    assert c.compare_number(-0.579, -0.5689, tolerance=0.01).status is NOT_MET
    assert c.compare_number(1163.29, "€1.164,29", tolerance=1).status is MET
    assert c.compare_number(1163.29, "€1.164,30", tolerance=1).status is NOT_MET
    assert c.compare_number(140, 140).status is MET and c.compare_number(140, 140.5).status is NOT_MET
    with pytest.raises(ValueError):
        c.compare_number(1, 1, tolerance=-1)


def test_compare_number_percent_units():
    assert c.compare_number(0.52, "52%", tolerance=0.01, percent=True).status is MET
    assert c.compare_number(0.52, "52%", tolerance=0.01).status is IND


def test_ranges():
    assert c.check_range(-0.04, max_value=0.10, include_max=False, absolute=True).status is MET
    assert c.check_range(0.10, max_value=0.10, include_max=False, absolute=True).status is NOT_MET
    assert c.check_range(0.10, max_value=0.10, absolute=True).status is MET
    assert c.check_range(-0.65, min_value=-0.65, max_value=-0.45).status is MET
    assert c.check_range(-0.65, min_value=-0.65, include_min=False).status is NOT_MET
    assert c.check_range("x", min_value=0).status is IND
    with pytest.raises(ValueError):
        c.check_range(1)


def test_threshold_and_sign_and_order():
    assert c.check_threshold(2e-11, "<", 0.001).status is MET
    assert c.check_threshold(0.001, "<", 0.001).status is NOT_MET
    assert c.check_threshold(0.001, "<=", 0.001).status is MET
    assert c.check_threshold(0.56, ">", 0.2).status is MET
    assert c.check_threshold("n/a", ">", 0.2).status is IND
    assert c.check_sign(-0.58, "negative").status is MET
    assert c.check_sign(0, "negative").status is NOT_MET
    assert c.check_sign(0, "positive").status is NOT_MET
    assert c.check_sign("0,0", "zero").status is MET
    assert c.compare_order(4.13, ">", "3,36").status is MET
    assert c.compare_order(3.36, ">", 3.36).status is NOT_MET
    assert c.compare_order(None, ">", 1).status is IND
    with pytest.raises(ValueError):
        c.check_sign(1, "big")


# --- Labels, sets, sequences, mappings ------------------------------------------


def test_normalize_label_is_conservative():
    assert c.normalize_label("  In   Person ") == "in person"
    assert c.normalize_label("Southern") != c.normalize_label("South")
    assert c.normalize_label(3) is None and c.normalize_label("   ") is None


def test_sets_and_sequences():
    assert c.compare_set(["North", "South"], [" south", "NORTH"]).status is MET
    out = c.compare_set(["North", "South"], ["North", "East"])
    assert out.status is NOT_MET and "missing" in out.note
    assert c.compare_set(["North"], ["North", "north"]).status is IND
    assert c.compare_set(["North"], None).status is IND
    assert c.check_subset(["Health", "Education"], ["education", "health", "food"]).status is MET
    assert c.check_subset(["Health", "Education"], ["health"]).status is NOT_MET
    assert c.compare_sequence([22, 24, 30], [22, 24, 30]).status is MET
    assert c.compare_sequence([22, 24, 30], [24, 22, 30]).status is NOT_MET
    assert c.compare_sequence(["Jan", "Feb"], ["jan ", "FEB"], normalize=True).status is MET


EXPECTED = {"North": 140, "South": 110, "Central": 12}


def test_mapping_order_and_label_normalization():
    out = c.compare_mapping(EXPECTED, [("central", 12), (" NORTH ", "140"), ("South", 110)])
    assert out.status is MET
    assert out.observed == {"north": 140.0, "south": 110.0, "central": 12.0}


@pytest.mark.parametrize("observed,status,word", [
    ({"North": 140, "South": 110}, NOT_MET, "missing"),
    ({"North": 140, "South": 110, "Central": 12, "Total": 262}, NOT_MET, "unexpected"),
    ({"North": 141, "South": 110, "Central": 12}, NOT_MET, "wrong"),
    ({"North": 140, "north": 140, "South": 110, "Central": 12}, IND, "duplicate"),
    ({"North": "molti", "South": 110, "Central": 12}, IND, "unparseable"),
    ({"North": "molti", "South": 999, "Central": 12}, NOT_MET, "wrong"),  # a reliable defect wins
])
def test_mapping_defects(observed, status, word):
    out = c.compare_mapping(EXPECTED, observed)
    assert out.status is status and word in out.note


def test_mapping_tolerance_and_extras():
    means = {"Health": 18.7692, "Water": 14.5926}
    assert c.compare_mapping(means, {"Health": "18,77", "Water": 14.59}, tolerance=0.05).status is MET
    assert c.compare_mapping(means, {"Health": 18.9, "Water": 14.59}, tolerance=0.05).status is NOT_MET
    assert c.compare_mapping({"a": 1}, {"a": 1, "b": 2}, allow_extra=True).status is MET
    assert c.compare_mapping(EXPECTED, None).status is IND
    with pytest.raises(ValueError):
        c.compare_mapping({"A": 1, "a": 2}, {})


# --- Tables ------------------------------------------------------------------------


def _table(rows, **meta):
    return _response({"response": {"type": "dataframe", "value": rows, **meta}})


def test_table_mapping_reordered_rows():
    resp = _table([{"service": "Water", "n": 100}, {"service": "Food", "n": 150}])
    pairs = c.table_mapping(resp, "service", "n")
    assert pairs.ok
    assert c.compare_mapping({"Food": 150, "Water": 100}, pairs.value).status is MET
    assert c.table_column(resp, "n").value == (100, 150)


@pytest.mark.parametrize("resp,error", [
    (_table([{"service": "Water"}]), "column"),
    (_table(["not a row"]), "column"),
    (_table([{"service": "A", "n": 1}], truncated=True), "truncated"),
    (_response({"response": {"type": "str", "value": "x"}}), "not a table"),
    (_response(b"{bad", 200), "MALFORMED_JSON"),
    (None, "no response"),
])
def test_table_extraction_failures(resp, error):
    pairs = c.table_mapping(resp, "service", "n")
    assert not pairs.ok and error in pairs.error
    assert c.from_extracted(pairs, c.compare_mapping, {"A": 1}).status is IND


def test_table_duplicate_category_and_bad_number_are_indeterminate():
    dup = c.table_mapping(_table([{"s": "A", "n": 1}, {"s": "a", "n": 1}]), "s", "n")
    assert c.compare_mapping({"A": 1}, dup.value).status is IND
    bad = c.table_mapping(_table([{"s": "A", "n": "1.234"}]), "s", "n")
    assert c.compare_mapping({"A": 1234}, bad.value).status is IND


def test_truncated_table_allowed_only_on_request():
    resp = _table([{"s": "A", "n": 1}], truncated=True)
    assert c.table_rows(resp, allow_truncated=True).ok


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


def test_chart_type():
    assert c.check_chart_type(BAR, {"bar"}).status is MET
    assert c.check_chart_type(BAR, {"pie", "doughnut"}).status is NOT_MET
    assert c.check_chart_type({"type": "Bar"}, {"bar", "pie"}).status is MET
    assert c.check_chart_type({"labels": []}, {"bar"}).status is IND
    assert c.check_chart_type("chart", {"bar"}).status is IND


@pytest.mark.parametrize("chart,error", [
    ({"type": "bar", "labels": ["A", "B"], "datasets": [{"data": [1]}]}, "length"),
    ({"type": "bar", "labels": ["A"]}, "labels/datasets"),
    ({"type": "bar", "labels": ["A"], "datasets": []}, "dataset 0"),
    (["not", "a", "chart"], "object"),
])
def test_chart_mapping_only_when_structurally_safe(chart, error):
    found = c.chart_mapping(chart)
    assert not found.ok and error in found.error
    assert c.from_extracted(found, c.compare_mapping, {"A": 1}).status is IND


def test_chart_mapping_to_comparison():
    pairs = c.chart_mapping(BAR)
    assert pairs.value == (("North", 140), ("South", 110))
    assert c.compare_mapping({"South": 110, "North": 140}, pairs.value).status is MET


# --- CheckResult construction -----------------------------------------------------------


def _policy(**extra):
    return dict(check_id="x.counts", evidence_class=EvidenceClass.DIRECT, failure=TechnicalFailure.ARTIFACT, **extra)


def test_make_check_attaches_policy_and_keeps_refs():
    refs = ("scenarios/X/http/datachat.response.bin",)
    result = c.make_check(c.compare_mapping(EXPECTED, {"North": 1}), evidence_refs=refs, **_policy())
    assert result.status is NOT_MET and result.failure is TechnicalFailure.ARTIFACT
    assert result.evidence_refs == refs
    assert json.loads(json.dumps(result.to_dict()))["observed"] == {"north": 1.0}


@pytest.mark.parametrize("outcome", [
    c.compare_number(1, "?"), c.compare_exact(1, None), c.not_applicable("caller says so"), c.compare_number(1, 1),
])
def test_failure_category_only_on_not_met(outcome):
    assert c.make_check(outcome, **_policy()).failure is None


# --- INDETERMINATE safety and Slice 1 integration ----------------------------------------


@pytest.mark.parametrize("outcome", [
    c.compare_number(140, "centoquaranta"),
    c.compare_number(1234, "1.234"),                                    # ambiguous separator
    c.compare_exact(140, None),                                         # missing field
    c.from_extracted(c.table_mapping(_with_charts([]), "a", "b"), c.compare_mapping, {"A": 1}),
    c.from_extracted(c.chart_mapping({"labels": ["A"], "datasets": [{"data": []}]}), c.compare_mapping, {"A": 1}),
    c.check_chart_count(_response(b'{"response": {"type": "weird"}}')),
])
def test_uncertainty_is_indeterminate_never_not_met(outcome):
    assert outcome.status is IND


def test_normalized_response_flow_end_to_end():
    log = ("2026-01-01T00:00:00+00:00 INFO datachat.runtime request_id=r app_id=a "
           "final_answer_check request_id=r passed=True final_kind=text\n")
    body = json.dumps({"log_id": 1, "response": {"type": "str", "value": "ok", "charts": [BAR]}}).encode()
    resp = normalize_response(body, 200, "application/json", parse_runtime_log(log))
    chart = c.charts(resp).value[0]
    outcome = c.from_extracted(c.chart_mapping(chart), c.compare_mapping, {"North": 140, "South": 110})
    result = c.make_check(outcome, evidence_refs=("scenarios/X/http/datachat.response.bin",), **_policy())
    assert result.status is MET and result.failure is None
