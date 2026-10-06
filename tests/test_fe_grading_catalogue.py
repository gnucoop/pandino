"""Tests for evaluation.datachat.grading.catalogue, the FE-01…FE-14 grading contract.

Offline: synthetic ScenarioEvidence and synthetic ground truth. Only the
binding test reads the real ``ground_truth.json`` (skipped when ``docs/`` is
absent). The catalogue must produce checks and review items, never a verdict.
"""

import copy
import hashlib
import json
from dataclasses import fields

import pytest

from evaluation.datachat import contract
from evaluation.datachat.grading import catalogue as cat
from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    EvidenceClass,
    FailureCategory,
    Gate,
    ManualReviewItem,
    RequirementLevel,
    ReviewStatus,
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
MUST, SHOULD = RequirementLevel.MUST, RequirementLevel.SHOULD
F = FailureCategory

# Synthetic ground truth: same shape as ground_truth.json, deliberately small.
GT = {
    "categorical_counts": {"region": {"North": 140, "South": 110, "East": 80, "West": 58, "Central": 12}},
    "missingness": {
        "per_column_na": {"record_id": 0, "region": 0, "wait_days": 8, "comment": 50},
        "comment": {"text": 340, "missing": 50, "blank": 10},
    },
    "aggregates": {
        "service": [
            {"service": "Food", "sum_case_cost_eur": 15101.08, "mean_case_cost_eur": 100.67},
            {"service": "Water", "sum_case_cost_eur": 9000.0, "mean_case_cost_eur": 90.0},
            {"service": "Health", "sum_case_cost_eur": 10000.0, "mean_case_cost_eur": 166.67},
        ],
        "mean_wait_days_by_service_overall": [
            {"service": "Water", "mean": 14.567, "n": 97},
            {"service": "Health", "mean": 10.9483, "n": 58},
            {"service": "Food", "mean": 8.5, "n": 150},
            {"service": "Education", "mean": 9.0, "n": 90},
        ],
        "mean_wait_days_by_service_in_south": [
            {"service": "Health", "mean": 18.7692, "n": 13},
            {"service": "Water", "mean": 14.5926, "n": 27},
            {"service": "Food", "mean": 9.1, "n": 40},
            {"service": "Education", "mean": 8.2, "n": 30},
        ],
        "region": [
            {"region": "North", "comments_with_text": 117}, {"region": "South", "comments_with_text": 94},
            {"region": "East", "comments_with_text": 70}, {"region": "West", "comments_with_text": 54},
            {"region": "Central", "comments_with_text": 10},
        ],
    },
    "correlations": {
        "wait_days_vs_satisfaction": {"x": "wait_days", "y": "satisfaction", "n": 392,
                                      "pearson": -0.5942, "spearman": -0.5788},
        "case_cost_eur_vs_satisfaction": {"x": "case_cost_eur", "y": "satisfaction", "n": 400,
                                          "pearson": -0.0211, "spearman": -0.0398},
    },
    "group_comparisons": {
        "online_vs_phone": {"group_a": "Online", "group_b": "Phone", "mean_a": 4.1267, "mean_b": 3.3615,
                            "difference": 0.7651, "p_value": 2.00451e-11},
        "in_person_vs_phone": {"group_a": "In person", "group_b": "Phone", "mean_a": 3.4333, "mean_b": 3.3615,
                               "difference": 0.0718, "p_value": 0.560422},
    },
    "temporal": {
        "monthly": [{"month": f"2025-{m:02d}", "requests": 20 + 2 * m, "mean_wait_days": 7.0 if m < 7 else 12.5}
                    for m in range(1, 13)],
        "mean_wait_days_jan_jun": 7.2785,
        "mean_wait_days_jul_dec": 12.7137,
        "step_jul_dec_minus_jan_jun": 5.4352,
    },
    "outliers": {
        "threshold": 689.0,
        "max_non_outlier": 315.29,
        "rows": [
            {"record_id": 48, "service": "Education", "case_cost_eur": 1059.9},
            {"record_id": 191, "service": "Health", "case_cost_eur": 1200.0},
            {"record_id": 219, "service": "Health", "case_cost_eur": 1100.5},
            {"record_id": 354, "service": "Education", "case_cost_eur": 1163.29},
        ],
    },
    "distribution": {"wait_days": {"skewness": 1.5172, "min": 0, "max": 51}},
    "semantic": {
        "negative_share_by_region": {"North": 0.1795, "South": 0.5213, "East": 0.2, "West": 0.2407, "Central": 0.0},
        "taxonomy": ["waiting", "staff", "cost", "information", "access"],
    },
    "guarantees": {
        "G6": {"evidence": {"range": [-0.65, -0.45]}},
        "G7": {"evidence": {"bound": 0.1}},
    },
}


def digest(refs):
    return hashlib.sha256("|".join(refs).encode()).hexdigest()


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
        tools=ToolEvidence(agent.code_blocks, runtime.tool_events, max(0, (usage_rows or 1) - 1)),
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


def grade(ev, gt=GT):
    return cat.grade_scenario_checks(ev, gt, digest=digest)


def items(result):
    return {x.check_id: x for x in (*result.checks, *result.manual_reviews)}


def status(result, check_id):
    item = items(result)[check_id]
    return item.status


REGIONS = GT["categorical_counts"]["region"]


# --- Catalogue authority ----------------------------------------------------------


def test_catalogue_covers_exactly_the_frozen_scenarios():
    assert tuple(cat.CATALOGUE) == SCENARIO_ORDER
    assert tuple(cat.TRUTH_BINDINGS) == SCENARIO_ORDER
    assert "FE-15" not in cat.CATALOGUE


def test_unknown_scenario_is_a_programmer_error():
    ev = evidence("FE-01", text("x"))
    with pytest.raises(cat.CatalogueError):
        cat.grade_scenario_checks(ScenarioEvidence(**{**{f.name: getattr(ev, f.name) for f in fields(ev)},
                                                     "scenario_id": "FE-15"}), GT, digest=digest)


@pytest.mark.parametrize("gate", [Gate.NOT_RUN, Gate.NOT_SCOREABLE_INFRA, Gate.NOT_SCOREABLE])
def test_non_gradeable_evidence_is_refused(gate):
    with pytest.raises(cat.CatalogueError):
        grade(evidence("FE-01", text("x"), gate=gate))


def test_missing_ground_truth_key_is_a_grading_input_error():
    gt = copy.deepcopy(GT)
    del gt["categorical_counts"]["region"]
    with pytest.raises(cat.GroundTruthError):
        grade(evidence("FE-01", text("x")), gt)
    gt = copy.deepcopy(GT)
    gt["categorical_counts"]["region"]["North"] = "many"
    with pytest.raises(cat.GroundTruthError):
        grade(evidence("FE-01", text("x")), gt)


def test_real_ground_truth_satisfies_every_binding():
    path = contract.evaluation_dir() / contract.GROUND_TRUTH_JSON_NAME
    if not path.is_file():
        pytest.skip("docs/ ground truth not present (local-only artifact)")
    gt = cat.load_ground_truth()
    for sid in SCENARIO_ORDER:
        bound = cat.bind_truth(gt, sid)
        assert set(bound) == set(cat.TRUTH_BINDINGS[sid])
        # The real truth also runs through every scenario without a ground-truth error.
        grade(evidence(sid, text("risposta")), gt)


def test_load_ground_truth_missing_file(tmp_path):
    with pytest.raises(cat.GroundTruthError):
        cat.load_ground_truth(tmp_path)


# --- Cross-cutting properties -----------------------------------------------------


@pytest.mark.parametrize("sid", SCENARIO_ORDER)
def test_every_scenario_emits_unique_ids_and_no_verdict(sid):
    result = grade(evidence(sid, text("Una risposta qualsiasi."), code=["x = 1"], usage_rows=1))
    ids = [x.check_id for x in (*result.checks, *result.manual_reviews)]
    assert ids and len(ids) == len(set(ids))
    assert all(i.startswith(sid.lower().replace("-", "")) for i in ids)
    assert not hasattr(result, "verdict")
    for review in result.manual_reviews:
        assert review.status is ReviewStatus.PENDING
        assert review.evidence_sha256 == digest(review.evidence_refs)
        assert review.question.endswith(("?", "."))
        if review.requirement_level is MUST:
            assert review.failure_category_if_failed is not None
    for check in result.checks:
        if check.requirement_level is not MUST:
            assert check.failure_category is None


@pytest.mark.parametrize("sid", SCENARIO_ORDER)
def test_deterministic_and_non_mutating(sid):
    gt = copy.deepcopy(GT)
    ev = evidence(sid, text("Testo."), code=["chart(data=x)"], usage_rows=2)
    before = (copy.deepcopy(gt), ev.response.raw, ev.response.charts)
    first, second = grade(ev, gt), grade(ev, gt)
    assert first == second
    assert gt == before[0] and ev.response.raw == before[1] and ev.response.charts == before[2]


def test_catalogue_imports_nothing_from_maui():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cat))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert all(not m.split(".")[0] in {"routes", "services", "infrastructure", "main", "datachat", "flask"}
               for m in modules)


# --- FE-01 -------------------------------------------------------------------------


def test_fe01_exact_counts_and_bar_chart():
    ev = evidence("FE-01", text("Ecco.", [chart("bar", list(REGIONS), list(REGIONS.values()))]))
    r = grade(ev)
    assert [status(r, i) for i in ("fe01.region_counts", "fe01.chart_present", "fe01.chart_categorical",
                                   "fe01.chart_bar")] == [MET] * 4
    assert not r.manual_reviews


def test_fe01_wrong_count_is_analytical():
    ev = evidence("FE-01", text("Ecco.", [chart("bar", list(REGIONS), [140, 110, 80, 58, 13])]))
    check = items(grade(ev))["fe01.region_counts"]
    assert check.status is NOT_MET and check.failure_category is F.ANALYTICAL


def test_fe01_missing_chart_without_attempt_is_understanding():
    rows = [{"region": k, "count": v} for k, v in REGIONS.items()]
    r = grade(evidence("FE-01", table(rows), code=["aggregate(group_by='region', op='count')"]))
    assert status(r, "fe01.region_counts") is MET  # table fallback
    check = items(r)["fe01.chart_present"]
    assert check.status is NOT_MET and check.failure_category is F.UNDERSTANDING
    assert status(r, "fe01.chart_categorical") is NA


def test_fe01_chart_attempt_without_charts_is_backend_contract():
    r = grade(evidence("FE-01", text("Ecco il grafico."), events=[("chart", "event=tool_call_result kind=bar")]))
    check = items(r)["fe01.chart_present"]
    assert check.failure_category is F.BACKEND_CONTRACT and check.evidence_class is EvidenceClass.DIRECT


def test_fe01_pie_is_a_should_miss_only():
    r = grade(evidence("FE-01", text("Ecco.", [chart("pie", list(REGIONS), list(REGIONS.values()))])))
    assert status(r, "fe01.chart_categorical") is MET
    bar = items(r)["fe01.chart_bar"]
    assert bar.status is NOT_MET and bar.requirement_level is SHOULD and bar.failure_category is None


def test_fe01_text_only_counts_go_to_targeted_review():
    r = grade(evidence("FE-01", text("North 140, South 110 ...")))
    review = items(r)["fe01.region_counts"]
    assert isinstance(review, ManualReviewItem) and "North 140" in review.question


def test_fe01_unparseable_response_is_indeterminate_not_failure():
    r = grade(evidence("FE-01", raw_body=b"not json"))
    assert status(r, "fe01.region_counts") is IND and status(r, "fe01.chart_present") is IND


# --- FE-02 -------------------------------------------------------------------------

SOUTH = {r["service"]: r["mean"] for r in GT["aggregates"]["mean_wait_days_by_service_in_south"]}
OVERALL = {r["service"]: r["mean"] for r in GT["aggregates"]["mean_wait_days_by_service_overall"]}


def test_fe02_south_result_and_chart():
    rows = [{"service": k, "mean": round(v, 2)} for k, v in SOUTH.items()]
    r = grade(evidence("FE-02", table(rows, [chart("bar", list(SOUTH), list(SOUTH.values()))]),
                       code=["s = aggregate(where_col='region', value='South')", "chart(data=s, kind='bar')"]))
    for cid in ("fe02.south_top_service", "fe02.services_compared", "fe02.chart_south_values",
                "fe02.chart_from_aggregate"):
        assert status(r, cid) is MET, cid
    assert items(r)["fe02.small_group_caveat"].requirement_level is SHOULD


def test_fe02_global_result_is_analytical_or_understanding():
    rows = [{"service": k, "mean": v} for k, v in OVERALL.items()]
    r = grade(evidence("FE-02", table(rows), code=["aggregate(group_by='service', where_col='region', value='South')"]))
    assert items(r)["fe02.south_top_service"].failure_category is F.ANALYTICAL
    r = grade(evidence("FE-02", table(rows), code=["aggregate(group_by='service', op='mean')"]))
    assert items(r)["fe02.south_top_service"].failure_category is F.UNDERSTANDING


def test_fe02_global_means_in_chart_is_chaining():
    rows = [{"service": k, "mean": v} for k, v in SOUTH.items()]
    r = grade(evidence("FE-02", table(rows, [chart("bar", list(OVERALL), list(OVERALL.values()))])))
    check = items(r)["fe02.chart_south_values"]
    assert check.failure_category is F.CHAINING and "global" in check.note
    assert status(r, "fe02.south_top_service") is MET


def test_fe02_chaining_unobservable_without_code():
    rows = [{"service": k, "mean": v} for k, v in SOUTH.items()]
    r = grade(evidence("FE-02", table(rows, [chart("bar", list(SOUTH), list(SOUTH.values()))])))
    derived = items(r)["fe02.chart_from_aggregate"]
    assert derived.status is IND and derived.evidence_class is EvidenceClass.INFERRED


def test_fe02_note_is_cited_in_caveat_review():
    r = grade(evidence("FE-02", text("Health", note="1 group(s) ... smallest has 13")))
    assert "smallest has 13" in items(r)["fe02.small_group_caveat"].question


# --- FE-03 -------------------------------------------------------------------------

SERVICE = GT["aggregates"]["service"]


def test_fe03_total_and_mean_distinguished():
    rows = [{"service": r["service"], "total_cost": r["sum_case_cost_eur"], "mean_cost": r["mean_case_cost_eur"]}
            for r in SERVICE]
    r = grade(evidence("FE-03", table(rows)))
    assert [status(r, i) for i in ("fe03.top_total", "fe03.top_mean", "fe03.metrics_distinguished",
                                   "fe03.approximate_values")] == [MET] * 4


def test_fe03_swapped_is_analytical():
    rows = [{"service": r["service"], "total_cost": r["mean_case_cost_eur"], "mean_cost": r["sum_case_cost_eur"]}
            for r in SERVICE]
    r = grade(evidence("FE-03", table(rows)))
    assert items(r)["fe03.top_total"].failure_category is F.ANALYTICAL
    assert items(r)["fe03.top_mean"].failure_category is F.ANALYTICAL


def test_fe03_text_answer_needs_review_for_the_distinction():
    r = grade(evidence("FE-03", text("Food ha il totale maggiore, Health la media maggiore.")))
    review = items(r)["fe03.metrics_distinguished"]
    assert isinstance(review, ManualReviewItem) and review.failure_category_if_failed is F.UNDERSTANDING
    assert status(r, "fe03.approximate_values") is IND


# --- FE-04 -------------------------------------------------------------------------

NA_COUNTS = GT["missingness"]["per_column_na"]


def _na_table(**override):
    counts = {**NA_COUNTS, **override}
    return table([{"column": k, "missing": v, "total": 400, "missing_pct": v / 400} for k, v in counts.items()])


def test_fe04_exact_missing_counts():
    r = grade(evidence("FE-04", _na_table()))
    assert status(r, "fe04.na_counts") is MET and status(r, "fe04.other_columns_complete") is MET


def test_fe04_blank_conflation_is_analytical():
    check = items(grade(evidence("FE-04", _na_table(comment=60))))["fe04.na_counts"]
    assert check.failure_category is F.ANALYTICAL and "conflated" in check.note
    assert status(grade(evidence("FE-04", _na_table(region=3))), "fe04.other_columns_complete") is NOT_MET


def test_fe04_malformed_structure_is_indeterminate():
    r = grade(evidence("FE-04", table([{"column": "wait_days", "missing": "otto"},
                                       {"column": "comment", "missing": 50}])))
    assert status(r, "fe04.na_counts") is IND or isinstance(items(r)["fe04.na_counts"], ManualReviewItem)
    r = grade(evidence("FE-04", raw_body=b"{"))
    assert status(r, "fe04.na_counts") is IND


# --- FE-05 -------------------------------------------------------------------------


def _corr(x, y, value, method="spearman", n=392):
    return table([{"col_x": x, "col_y": y, "method": method, "correlation": value, "n": n}])


def test_fe05_correct_negative_correlation():
    r = grade(evidence("FE-05", _corr("wait_days", "satisfaction", -0.579)))
    for cid in ("fe05.coefficient", "fe05.direction", "fe05.spearman_method", "fe05.sample_size_reported"):
        assert status(r, cid) is MET, cid
    assert {"fe05.moderate_strength", "fe05.no_causal_claim"} <= {x.check_id for x in r.manual_reviews}


def test_fe05_wrong_sign_and_value():
    r = grade(evidence("FE-05", _corr("wait_days", "satisfaction", 0.579)))
    assert items(r)["fe05.coefficient"].failure_category is F.ANALYTICAL
    assert items(r)["fe05.direction"].failure_category is F.INTERPRETATION


def test_fe05_missing_n_is_a_should_miss_only():
    r = grade(evidence("FE-05", text("Correlazione di Spearman moderata negativa (rho = -0,58).")))
    sample = items(r)["fe05.sample_size_reported"]
    assert sample.status is NOT_MET and sample.requirement_level is SHOULD and sample.failure_category is None
    assert status(r, "fe05.spearman_method") is MET
    r = grade(evidence("FE-05", text("Spearman -0,58 su 392 osservazioni valide.")))
    assert status(r, "fe05.sample_size_reported") is MET


def test_fe05_causal_wording_creates_review_not_failure():
    r = grade(evidence("FE-05", text("L'attesa causa una soddisfazione più bassa.")))
    review = items(r)["fe05.no_causal_claim"]
    assert isinstance(review, ManualReviewItem) and review.status is ReviewStatus.PENDING
    assert "causa" in review.question
    assert all(c.status is not NOT_MET for c in r.checks if c.requirement_level is MUST)


def test_fe05_pearson_only_is_inferred_should():
    r = grade(evidence("FE-05", text("Correlazione moderata."), code=["correlation(col_x='wait_days')"]))
    check = items(r)["fe05.spearman_method"]
    assert check.status is NOT_MET and check.evidence_class is EvidenceClass.INFERRED
    assert check.requirement_level is SHOULD


# --- FE-06 -------------------------------------------------------------------------


def test_fe06_negligible_correlation():
    r = grade(evidence("FE-06", _corr("case_cost_eur", "satisfaction", -0.0211, "pearson", 400)))
    assert status(r, "fe06.coefficient_near_zero") is MET and status(r, "fe06.coefficient_quoted") is MET
    assert items(r)["fe06.negligible_wording"].failure_category_if_failed is F.INTERPRETATION


def test_fe06_wrong_magnitude():
    r = grade(evidence("FE-06", _corr("case_cost_eur", "satisfaction", -0.35, "pearson", 400)))
    assert status(r, "fe06.coefficient_near_zero") is NOT_MET
    assert status(r, "fe06.coefficient_quoted") is NOT_MET


# --- FE-07 / FE-08 -----------------------------------------------------------------


def _cg(a, b, ma, mb, p):
    return table([{"group_a": a, "group_b": b, "n_a": 150, "n_b": 130, "mean_a": ma, "mean_b": mb,
                   "difference": ma - mb, "test": "welch", "statistic": 7.0, "p_value": p}])


CG_EVENT = ("compare_groups", "event=tool_call_result metric=satisfaction group_col=channel")


def test_fe07_online_higher_with_test():
    r = grade(evidence("FE-07", _cg("Online", "Phone", 4.1267, 3.3615, 2e-11), events=[CG_EVENT]))
    for cid in ("fe07.direction", "fe07.difference", "fe07.significance_supported", "fe07.compare_groups",
                "fe07.p_value_reported"):
        assert status(r, cid) is MET, cid
    assert items(r)["fe07.significance_supported"].evidence_class is EvidenceClass.INFERRED


def test_fe07_reversed_groups_are_reoriented():
    r = grade(evidence("FE-07", _cg("Phone", "Online", 3.3615, 4.1267, 2e-11), events=[CG_EVENT]))
    assert status(r, "fe07.direction") is MET and status(r, "fe07.difference") is MET


def test_fe07_difference_tolerance():
    rows = table([{"channel": "Online", "mean": 4.0}, {"channel": "Phone", "mean": 3.36}])
    r = grade(evidence("FE-07", rows, code=["aggregate(op='mean')"]))
    assert status(r, "fe07.direction") is MET
    assert items(r)["fe07.difference"].failure_category is F.ANALYTICAL  # 0.64 vs 0.7651 ± 0.05
    rows = table([{"channel": "Online", "mean": 4.13}, {"channel": "Phone", "mean": 3.36}])
    assert status(grade(evidence("FE-07", rows)), "fe07.difference") is MET


def test_fe07_wrong_direction_is_analytical():
    rows = table([{"channel": "Online", "mean": 3.0}, {"channel": "Phone", "mean": 3.5}])
    assert items(grade(evidence("FE-07", rows)))["fe07.direction"].failure_category is F.ANALYTICAL


def test_fe07_significance_without_test_goes_to_review():
    r = grade(evidence("FE-07", text("La differenza è statisticamente significativa."), code=["x = df.mean()"]))
    review = items(r)["fe07.significance_supported"]
    assert isinstance(review, ManualReviewItem) and review.failure_category_if_failed is F.HONESTY
    assert "significativa" in review.question
    assert items(r)["fe07.compare_groups"].status is NOT_MET


def test_fe08_small_difference_and_p_value():
    r = grade(evidence("FE-08", _cg("In person", "Phone", 3.4333, 3.3615, 0.5604), events=[CG_EVENT]))
    assert status(r, "fe08.small_difference") is MET
    assert status(r, "fe08.p_value_reported") is MET
    assert {"fe08.no_evidence_of_difference", "fe08.no_equality_claim"} <= {x.check_id for x in r.manual_reviews}


def test_fe08_p_value_compared_to_truth_not_to_a_threshold():
    # 0.30 would pass a "> 0.2" rule; it must not pass here: no invented threshold exists.
    r = grade(evidence("FE-08", _cg("In person", "Phone", 3.4333, 3.3615, 0.30), events=[CG_EVENT]))
    assert status(r, "fe08.p_value_reported") is NOT_MET
    assert items(r)["fe08.p_value_reported"].requirement_level is SHOULD
    import inspect
    assert "0.2" not in inspect.getsource(cat._fe08)


def test_fe08_equality_wording_is_review_with_tool_selection_hint():
    r = grade(evidence("FE-08", text("I due gruppi sono uguali."), code=["df.groupby('channel').mean()"]))
    review = items(r)["fe08.no_equality_claim"]
    assert review.failure_category_if_failed is F.INTERPRETATION
    assert "uguali" in review.question and "TOOL SELECTION" in review.question
    assert all(c.status is not NOT_MET for c in r.checks if c.requirement_level is MUST)


# --- FE-09 -------------------------------------------------------------------------

MONTHLY = {r["month"]: r["requests"] for r in GT["temporal"]["monthly"]}
TREND_EVENT = ("trend", "event=tool_call_result date_col=submitted_on freq=M op=count metric=None")


def test_fe09_monthly_line_chart():
    r = grade(evidence("FE-09", text("Crescita.", [chart("line", list(MONTHLY), list(MONTHLY.values()))]),
                       events=[TREND_EVENT]))
    for cid in ("fe09.temporal_chart", "fe09.monthly_counts", "fe09.trend_tool", "fe09.line_chart"):
        assert status(r, cid) is MET, cid
    assert "fe09.growth_identified" in {x.check_id for x in r.manual_reviews}


def test_fe09_month_end_labels_are_normalized():
    labels = [f"{m}-28" for m in MONTHLY]
    r = grade(evidence("FE-09", text("x", [chart("bar", labels, list(MONTHLY.values()))])))
    assert status(r, "fe09.monthly_counts") is MET and status(r, "fe09.line_chart") is NOT_MET


def test_fe09_wrong_or_missing_month():
    values = list(MONTHLY.values())
    values[3] += 1
    r = grade(evidence("FE-09", text("x", [chart("line", list(MONTHLY), values)])))
    assert items(r)["fe09.monthly_counts"].failure_category is F.ANALYTICAL
    r = grade(evidence("FE-09", text("x", [chart("line", list(MONTHLY)[:-1], list(MONTHLY.values())[:-1])])))
    assert status(r, "fe09.monthly_counts") is NOT_MET


def test_fe09_no_chart_is_understanding():
    check = items(grade(evidence("FE-09", text("Le richieste crescono."))))["fe09.temporal_chart"]
    assert check.status is NOT_MET and check.failure_category is F.UNDERSTANDING


def test_fe09_per_day_chart_is_reviewed():
    labels = ["2025-01-01", "2025-01-02", "2025-02-01"]
    r = grade(evidence("FE-09", text("x", [chart("line", labels, [1, 2, 3])])))
    assert status(r, "fe09.temporal_chart") is MET
    assert items(r)["fe09.daily_chart_readable"].failure_category_if_failed is F.TOOL_SELECTION


# --- FE-10 -------------------------------------------------------------------------

MEANS = {r["month"]: r["mean_wait_days"] for r in GT["temporal"]["monthly"]}


def test_fe10_monthly_means_and_reviews():
    rows = [{"period": k, "mean_wait_days": v} for k, v in MEANS.items()]
    r = grade(evidence("FE-10", table(rows),
                       events=[("trend", "event=tool_call_result freq=M op=mean metric=wait_days")]))
    assert status(r, "fe10.monthly_means") is MET and status(r, "fe10.trend_mean") is MET
    reviews = {x.check_id: x for x in r.manual_reviews}
    assert reviews["fe10.increase_identified"].requirement_level is MUST
    assert reviews["fe10.step_identified"].requirement_level is SHOULD
    assert "+4" in reviews["fe10.step_magnitude"].question


def test_fe10_wrong_means_and_stable_text():
    rows = [{"period": k, "mean_wait_days": 9.0} for k in MEANS]
    assert items(grade(evidence("FE-10", table(rows))))["fe10.monthly_means"].failure_category is F.ANALYTICAL
    r = grade(evidence("FE-10", text("I tempi sono rimasti stabili.")))
    assert status(r, "fe10.monthly_means") is NA
    assert items(r)["fe10.increase_identified"].failure_category_if_failed is F.INTERPRETATION


def test_fe10_gradual_wording_only_focuses_the_should_review():
    r = grade(evidence("FE-10", text("I tempi sono aumentati in modo graduale.")))
    review = items(r)["fe10.step_identified"]
    assert "graduale" in review.question and review.failure_category_if_failed is None
    assert all(c.status is not NOT_MET for c in r.checks if c.requirement_level is MUST)


# --- FE-11 -------------------------------------------------------------------------

IMAGE = {"type": "image", "value": "b'iVBORw0KGgo='", "charts": []}


def test_fe11_image_delivered_and_reviewed():
    r = grade(evidence("FE-11", IMAGE, image_meta={"decode_status": "OK"}, image=True,
                       code=["plot(kind='hist', x='wait_days')"]))
    assert [status(r, i) for i in ("fe11.image_delivered", "fe11.no_raw_value_bar", "fe11.histogram")] == [MET] * 3
    review = items(r)["fe11.image_suitable"]
    assert review.evidence_refs == ("scenarios/FE-11/artifacts/image.png",)
    assert items(r)["fe11.histogram"].evidence_class is EvidenceClass.INFERRED


def test_fe11_no_image():
    check = items(grade(evidence("FE-11", text("La media è 10 giorni."))))["fe11.image_delivered"]
    assert check.failure_category is F.UNDERSTANDING
    check = items(grade(evidence("FE-11", text("/tmp/plot_abc.png"))))["fe11.image_delivered"]
    assert check.failure_category is F.BACKEND_CONTRACT
    check = items(grade(evidence("FE-11", IMAGE, image_meta={"decode_status": "NOT_PNG"})))["fe11.image_delivered"]
    assert check.failure_category is F.BACKEND_CONTRACT
    assert "fe11.image_suitable" not in items(grade(evidence("FE-11", IMAGE)))


def test_fe11_bar_over_raw_values():
    bars = [chart("bar", [str(v) for v in range(30)], [1] * 30)]
    check = items(grade(evidence("FE-11", text("x", bars))))["fe11.no_raw_value_bar"]
    assert check.failure_category is F.TOOL_SELECTION and check.evidence_class is EvidenceClass.DIRECT
    check = items(grade(evidence("FE-11", IMAGE, code=["chart(kind='bar', x='wait_days')"])))[
        "fe11.no_raw_value_bar"]
    assert check.status is NOT_MET and check.evidence_class is EvidenceClass.INFERRED
    assert status(grade(evidence("FE-11", IMAGE, code=["plot(kind='box', x='wait_days')"])),
                  "fe11.histogram") is NOT_MET


# --- FE-12 -------------------------------------------------------------------------

OUTLIERS = GT["outliers"]["rows"]


def test_fe12_extremes_found():
    rows = [{"record_id": r["record_id"], "service": r["service"], "case_cost_eur": r["case_cost_eur"]}
            for r in OUTLIERS]
    r = grade(evidence("FE-12", table(rows)))
    for cid in ("fe12.not_plot_only", "fe12.extremes_found", "fe12.services", "fe12.four_cases", "fe12.record_ids"):
        assert status(r, cid) is MET, cid
    assert items(r)["fe12.ordinary_not_anomalous"].failure_category_if_failed is F.INTERPRETATION


def test_fe12_missed_extremes_and_wrong_services():
    rows = [{"record_id": r["record_id"], "service": "Food", "case_cost_eur": r["case_cost_eur"]}
            for r in OUTLIERS[:3]] + [{"record_id": 5, "service": "Food", "case_cost_eur": 300.0}]
    r = grade(evidence("FE-12", table(rows)))
    assert items(r)["fe12.extremes_found"].failure_category is F.ANALYTICAL
    assert status(r, "fe12.four_cases") is NOT_MET
    rows = [{"record_id": r["record_id"], "service": "Food", "case_cost_eur": r["case_cost_eur"]} for r in OUTLIERS]
    assert items(grade(evidence("FE-12", table(rows))))["fe12.services"].failure_category is F.ANALYTICAL


def test_fe12_plot_only_is_understanding():
    check = items(grade(evidence("FE-12", IMAGE)))["fe12.not_plot_only"]
    assert check.failure_category is F.UNDERSTANDING


def test_fe12_anomaly_wording_review_focus():
    r = grade(evidence("FE-12", text("Ci sono valori anomali in Education e Health.")))
    assert "anomali" in items(r)["fe12.ordinary_not_anomalous"].question
    assert isinstance(items(r)["fe12.extremes_found"], ManualReviewItem)


# --- FE-13 -------------------------------------------------------------------------

SHARES = GT["semantic"]["negative_share_by_region"]
SENT_EVENT = ("sentiment", "event=tool_call_result tool=sentiment_analysis aggregate=False rows=400")


def test_fe13_semantic_contribution_and_regional_result():
    r = grade(evidence("FE-13", text("Sud più negativo.", [chart("bar", list(SHARES), list(SHARES.values()))]),
                       events=[SENT_EVENT], usage_rows=2))
    for cid in ("fe13.semantic_provider_call", "fe13.sentiment_tool", "fe13.top_region", "fe13.regions_compared",
                "fe13.top_share_band", "fe13.chart_by_region"):
        assert status(r, cid) is MET, cid
    assert items(r)["fe13.sentiment_tool"].evidence_class is EvidenceClass.INFERRED
    reviews = {x.check_id: x for x in r.manual_reviews}
    assert reviews["fe13.nulls_not_neutral"].failure_category_if_failed is F.HONESTY
    assert "Central" in reviews["fe13.small_region_qualified"].question


def test_fe13_no_provider_call_is_honesty():
    check = items(grade(evidence("FE-13", text("x"), usage_rows=1)))["fe13.semantic_provider_call"]
    assert check.status is NOT_MET and check.failure_category is F.HONESTY
    assert status(grade(evidence("FE-13", text("x"))), "fe13.semantic_provider_call") is IND


def test_fe13_lost_region_is_chaining_and_wrong_top_analytical():
    labels = ["positive", "negative", "neutral"]
    r = grade(evidence("FE-13", table([{"region": "North", "share": 0.6}, {"region": "South", "share": 0.3}]),
                       usage_rows=2))
    assert items(r)["fe13.top_region"].failure_category is F.ANALYTICAL
    assert items(r)["fe13.regions_compared"].failure_category is F.CHAINING
    r = grade(evidence("FE-13", text("x", [chart("bar", labels, [1, 2, 3])]), usage_rows=2))
    assert isinstance(items(r)["fe13.regions_compared"], ManualReviewItem)


def test_fe13_counts_are_not_mistaken_for_shares():
    r = grade(evidence("FE-13", text("x", [chart("bar", list(SHARES), [21, 49, 14, 13, 0])])))
    assert isinstance(items(r)["fe13.top_region"], ManualReviewItem)
    assert status(r, "fe13.top_share_band") is IND


# --- FE-14 -------------------------------------------------------------------------

CM_EVENT = ("classify_match", "event=tool_call_result tool=classify_match rows=400 analyzable_rows=340")
KW_EVENT = ("keywords", "event=tool_call_result col=comment language=italian")


def test_fe14_full_semantic_chain():
    counts = table([{"category": "Attesa", "count": 96}, {"category": "Personale", "count": 68},
                    {"category": "Non classificato", "count": 37}])
    r = grade(evidence("FE-14", counts, events=[KW_EVENT, CM_EVENT], usage_rows=2))
    for cid in ("fe14.classify_match", "fe14.semantic_provider_call", "fe14.counts_per_category",
                "fe14.keywords_first"):
        assert status(r, cid) is MET, cid
    assert {x.check_id for x in r.manual_reviews} == {
        "fe14.proposal_framing", "fe14.null_handling", "fe14.taxonomy_coherence", "fe14.theme_coverage"}
    assert items(r)["fe14.taxonomy_coherence"].question.startswith("Do the proposed categories")


def test_fe14_substring_counting_is_inferred_tool_selection():
    code = ["df['comment'].str.contains('attesa').sum()"]
    r = grade(evidence("FE-14", text("Attesa: 40"), code=code, usage_rows=1))
    check = items(r)["fe14.classify_match"]
    assert check.status is NOT_MET and check.evidence_class is EvidenceClass.INFERRED
    assert check.failure_category is F.TOOL_SELECTION
    assert items(r)["fe14.semantic_provider_call"].status is NOT_MET
    assert status(r, "fe14.keywords_first") is NOT_MET
    assert isinstance(items(r)["fe14.counts_per_category"], ManualReviewItem)


def test_fe14_keywords_after_classification_is_should_miss():
    r = grade(evidence("FE-14", text("x"), events=[CM_EVENT, KW_EVENT], usage_rows=2))
    check = items(r)["fe14.keywords_first"]
    assert check.status is NOT_MET and check.requirement_level is SHOULD


# --- Generic primitive ----------------------------------------------------------------


def test_table_label_values_locates_columns_by_content():
    from evaluation.datachat.grading import checks as c

    resp = normalize_response(json.dumps({"response": table([{"r": "North", "n": 1, "pct": "50%"},
                                                              {"r": "South", "n": 2, "pct": "50%"}])}).encode(),
                              200, "application/json")
    assert c.table_label_values(resp, ["North", "South"]).value == (("North", 1), ("South", 2))
    two = normalize_response(json.dumps({"response": table([{"r": "North", "a": 1, "b": 2},
                                                             {"r": "South", "a": 3, "b": 4}])}).encode(),
                             200, "application/json")
    assert not c.table_label_values(two, ["North", "South"]).ok  # ambiguous value column
    assert c.table_label_values(two, ["North", "South"], value_columns=("b",)).value == (("North", 2), ("South", 4))
    assert not c.table_label_values(two, ["East", "West"]).ok


def test_evidence_digest_binds_refs_and_content(tmp_path):
    from evaluation.datachat.grading.reader import GradingInputError, evidence_digest

    (tmp_path / "scenarios/FE-01/http").mkdir(parents=True)
    body = tmp_path / "scenarios/FE-01/http/datachat.response.bin"
    body.write_bytes(b"a")
    refs = ("scenarios/FE-01/http/datachat.response.bin",)
    first = evidence_digest(tmp_path, refs)
    assert first == evidence_digest(tmp_path, list(refs))
    body.write_bytes(b"b")
    assert evidence_digest(tmp_path, refs) != first
    with pytest.raises(GradingInputError):
        evidence_digest(tmp_path, ("../outside",))
    with pytest.raises(GradingInputError):
        evidence_digest(tmp_path, ("scenarios/FE-01/missing.bin",))
