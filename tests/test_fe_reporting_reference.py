"""Tests for the Evaluation Reference binding of the operator report.

A synthetic ``ground_truth.json`` with the pinned structure is written into
the synthetic repository and re-pinned, so grading records its hash exactly
as for the real contract. Offline.
"""

import json
import re
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.evidence import sha256_bytes, sha256_file
from evaluation.datachat.reporting.reference import (
    EvaluationReferenceError,
    load_evaluation_reference,
    project_reference,
)
from tests.test_fe_grading_grader import repo_root  # noqa: F401

SCENARIOS = tuple(f"FE-{i:02d}" for i in range(1, 15))

# Words that would turn reference material into a judgment of the answer.
JUDGMENT = re.compile(r"\b(PASS|FAIL|NEEDS REVIEW|corrett[oaie]|errat[oaie]|sbagliat[oaie]|match|mismatch|"
                      r"concord|discord|verdetto|punteggio)\b", re.IGNORECASE)


def _g(title, kind, evidence):
    return {"title": title, "status": "pass", "ground_truth_type": kind, "evidence": evidence}


def synthetic_ground_truth() -> dict:
    """The pinned ``ground_truth.json`` structure with small synthetic values."""
    monthly = [{"month": f"2025-{m:02d}", "requests": 10 + m, "mean_wait_days": 5.0 + (m > 6) * 5} for m in range(1, 13)]
    return {
        "categorical_counts": {"region": {"North": 40, "South": 30, "Central": 3}},
        "missingness": {"per_column_na": {"wait_days": 2, "comment": 5, "region": 0},
                        "comment": {"text": 60, "missing": 5, "blank": 1}},
        "aggregates": {
            "service": [{"service": "Food", "n": 10, "sum_case_cost_eur": 900.0, "mean_case_cost_eur": 90.0},
                        {"service": "Health", "n": 4, "sum_case_cost_eur": 800.0, "mean_case_cost_eur": 200.0}],
            "mean_wait_days_by_service_in_south": [{"service": "Health", "mean": 18.5, "n": 13},
                                                   {"service": "Water", "mean": 14.5, "n": 27}],
            "region": [{"region": "North", "comments_with_text": 30, "sentiment_counts": {"negative": 6},
                        "negative_share": 0.2},
                       {"region": "South", "comments_with_text": 20, "sentiment_counts": {"negative": 10},
                        "negative_share": 0.5}],
        },
        "correlations": {
            "wait_days_vs_satisfaction": {"x": "wait_days", "y": "satisfaction", "n": 98, "pearson": -0.6, "spearman": -0.58},
            "case_cost_eur_vs_satisfaction": {"x": "case_cost_eur", "y": "satisfaction", "n": 100, "pearson": -0.02,
                                              "spearman": -0.04},
        },
        "group_comparisons": {
            key: {"metric": "satisfaction", "column": "channel", "group_a": a, "group_b": "Phone", "n_a": 50, "n_b": 40,
                  "mean_a": 4.1, "mean_b": 3.4, "difference": diff, "t_statistic": 7.0, "p_value": p}
            for key, a, diff, p in (("online_vs_phone", "Online", 0.7, 1e-11), ("in_person_vs_phone", "In person", 0.07, 0.56))
        },
        "temporal": {"monthly": monthly, "mean_wait_days_jan_jun": 5.0, "mean_wait_days_jul_dec": 10.0,
                     "step_jul_dec_minus_jan_jun": 5.0},
        "outliers": {"column": "case_cost_eur", "median": 80.0, "rule": ">= 8 x median", "threshold": 640.0,
                     "max_non_outlier": 300.0,
                     "rows": [{"record_id": 7, "service": "Health", "region": "East", "case_cost_eur": 1100.0}]},
        "distribution": {"wait_days": {"skewness": 1.5, "skewness_method": "pandas Series.skew",
                                       "distinct_non_null": 30, "min": 0, "median": 10.0, "max": 50}},
        "semantic": {"negative_share_by_region": {"North": 0.2, "South": 0.5},
                     "taxonomy": ["waiting", "staff"], "row_theme_counts": {"waiting": 20, "staff": 10, "none": 3}},
        "guarantees": {
            "G2": _g("Exact region counts", "exact", {}),
            "G4": _g("Highest mean wait", "exact", {"south_top": {"service": "Health", "mean": 18.5, "n": 13},
                                                    "south_runner_up": {"service": "Water", "mean": 14.5, "n": 27},
                                                    "overall_top": {"service": "Water", "mean": 14.0, "n": 90}}),
            "G5": _g("Sum vs mean", "exact", {"highest_sum": "Food", "highest_mean": "Health"}),
            "G6": _g("Spearman range", "bounded", {"spearman": -0.58, "range": [-0.65, -0.45]}),
            "G7": _g("No relationship", "bounded", {"bound": 0.1}),
            "G8": _g("Group comparisons", "exact", {}),
            "G9": _g("Trend", "qualitative", {"non_decreasing_transitions": 11, "first_month": 11, "last_month": 22}),
            "G10": _g("Missingness", "exact", {}),
            "G11": _g("Outliers", "exact", {}),
            "G12": _g("Right-skewed", "exact", {}),
            "G13": _g("South highest", "exact", {"margin": 0.3}),
            "G14": _g("Themes", "exact", {"keywords_top5": ["attesa", "richiesta"], "theme_none_share": 0.1}),
        },
    }


@pytest.fixture
def ref_repo(repo_root, monkeypatch):  # noqa: F811
    """``repo_root`` whose ``ground_truth.json`` is the synthetic reference, re-pinned before any run is built."""
    content = json.dumps(synthetic_ground_truth(), indent=2).encode()
    (contract.evaluation_dir(repo_root) / contract.GROUND_TRUTH_JSON_NAME).write_bytes(content)
    pins = tuple(contract.PinnedArtifact(p.name, sha256_bytes(content), p.required)
                 if p.name == contract.GROUND_TRUTH_JSON_NAME else p for p in contract.PINNED_ARTIFACTS)
    monkeypatch.setattr(contract, "PINNED_ARTIFACTS", pins)
    return repo_root


def run_contract(repo: Path) -> dict:
    """The contract block grading records for ``repo``."""
    gt = sha256_file(contract.evaluation_dir(repo) / contract.GROUND_TRUTH_JSON_NAME)
    return {"ground_truth_sha256": gt,
            "artifacts": {p.name: sha256_file(contract.evaluation_dir(repo) / p.name) for p in contract.PINNED_ARTIFACTS}}


def test_every_scenario_has_a_reference(ref_repo):
    refs = load_evaluation_reference(ref_repo, run_contract(ref_repo), SCENARIOS)
    assert tuple(refs) == SCENARIOS
    for sid, ref in refs.items():
        assert ref.scenario_id == sid and ref.guarantees and (ref.facts or ref.tables) and ref.expectations


def test_reference_values_come_from_ground_truth(ref_repo):
    refs = load_evaluation_reference(ref_repo, run_contract(ref_repo), SCENARIOS)
    fe01 = refs["FE-01"]
    assert fe01.guarantees == (("G2", "Exact region counts", "exact"),)
    assert fe01.tables[0].rows == (("North", 40), ("South", 30), ("Central", 3))
    assert "North (40)" in fe01.facts[0] and "Central (3)" in fe01.facts[0]
    assert refs["FE-03"].facts == ("Costo totale più alto: Food.", "Costo medio più alto: Health.")
    assert "Spearman -0.58" in refs["FE-05"].facts[0]
    assert any("immagine" in e for e in refs["FE-11"].expectations)
    assert "South (0.5)" in refs["FE-13"].facts[0]
    assert "waiting, staff" in refs["FE-14"].facts[0]


def test_reference_never_judges(ref_repo):
    for ref in load_evaluation_reference(ref_repo, run_contract(ref_repo), SCENARIOS).values():
        text = " ".join([*ref.facts, *ref.expectations])
        assert not JUDGMENT.search(text), (ref.scenario_id, JUDGMENT.search(text))


def test_unknown_scenarios_get_no_reference(ref_repo):
    assert load_evaluation_reference(ref_repo, run_contract(ref_repo), ["FE-99"]) == {}


@pytest.mark.parametrize("field", ["ground_truth_sha256", "artifacts"])
def test_ground_truth_hash_mismatch_refused(ref_repo, field):
    recorded = run_contract(ref_repo)
    if field == "artifacts":
        recorded["artifacts"][contract.GROUND_TRUTH_JSON_NAME] = "0" * 64
    else:
        recorded[field] = "0" * 64
    with pytest.raises(EvaluationReferenceError, match="differs"):
        load_evaluation_reference(ref_repo, recorded, SCENARIOS)


def test_changed_ground_truth_on_disk_refused(ref_repo):
    recorded = run_contract(ref_repo)
    path = contract.evaluation_dir(ref_repo) / contract.GROUND_TRUTH_JSON_NAME
    path.write_text(path.read_text() + " ")
    with pytest.raises(EvaluationReferenceError, match="differs"):
        load_evaluation_reference(ref_repo, recorded, SCENARIOS)


def test_design_document_mismatch_refused(ref_repo):
    recorded = run_contract(ref_repo)
    recorded["artifacts"][contract.DESIGN_DOC_NAME] = "0" * 64
    with pytest.raises(EvaluationReferenceError, match="design document"):
        load_evaluation_reference(ref_repo, recorded, SCENARIOS)


def test_ground_truth_not_fitting_projection_refused():
    broken = synthetic_ground_truth()
    del broken["outliers"]
    with pytest.raises(EvaluationReferenceError, match="does not fit"):
        project_reference(broken, ["FE-12"])


REAL_GT = contract.evaluation_dir(contract.REPO_ROOT) / contract.GROUND_TRUTH_JSON_NAME


@pytest.mark.skipif(not REAL_GT.is_file(), reason="pinned ground_truth.json not present locally")
def test_real_pinned_ground_truth_projects_every_scenario():
    assert sha256_file(REAL_GT) == contract.pinned_artifact(contract.GROUND_TRUTH_JSON_NAME).sha256
    refs = project_reference(json.loads(REAL_GT.read_text(encoding="utf-8")), SCENARIOS)
    assert tuple(refs) == SCENARIOS
    assert "North (140)" in refs["FE-01"].facts[0] and "Central (12)" in refs["FE-01"].facts[0]
    assert refs["FE-03"].facts == ("Costo totale più alto: Food.", "Costo medio più alto: Health.")
    assert "Spearman -0.5788" in refs["FE-05"].facts[0]
    assert "skewness 1.5172" in refs["FE-11"].facts[0]
    assert "South (0.5213)" in refs["FE-13"].facts[0]
    assert "waiting, staff, cost, information, access" in refs["FE-14"].facts[0]
