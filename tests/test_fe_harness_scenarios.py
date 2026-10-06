"""Tests for evaluation.datachat.scenarios, the executable FE-01…FE-14 catalogue.

Pure tests run the question extractor and drift guard against tmp_path
fixture documents. One integration-style test reads the real (git-ignored)
design authority under docs/ and is skipped when it is absent. No Maui,
DB, network or provider access.
"""

import hashlib
from pathlib import Path

import pytest

from evaluation.datachat import contract
from evaluation.datachat.contract import DESIGN_DOC_NAME, EVALUATION_DIR_RELATIVE, REPO_ROOT
from evaluation.datachat.model import ScenarioExecution
from evaluation.datachat.scenarios import (
    SCENARIO_ORDER,
    SCENARIOS,
    Scenario,
    check_drift,
    extract_questions,
    question_sha256,
)

FE_IDS = [f"FE-{n:02d}" for n in range(1, 15)]


def _doc(entries):
    lines = ["# Design", "", "## 9. Frozen scenario suite", ""]
    for scenario_id, question in entries:
        lines += [f"### {scenario_id} — Title", "", f"- **Question:** `{question}`", "- **MUST:** x", ""]
    lines += ["## 10. Next", "", "text"]
    return "\n".join(lines) + "\n"


def _catalogue_entries():
    return [(s.scenario_id, s.question) for s in SCENARIOS]


def _write_repo(tmp_path, monkeypatch, text):
    directory = tmp_path / EVALUATION_DIR_RELATIVE
    directory.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    (directory / DESIGN_DOC_NAME).write_bytes(data)
    pinned = tuple(
        contract.PinnedArtifact(a.name, hashlib.sha256(data).hexdigest(), a.required)
        if a.name == DESIGN_DOC_NAME
        else a
        for a in contract.PINNED_ARTIFACTS
    )
    monkeypatch.setattr(contract, "PINNED_ARTIFACTS", pinned)
    return tmp_path


# --- Catalogue ---------------------------------------------------------------


def test_catalogue_ids_and_ordinals():
    assert len(SCENARIOS) == 14
    assert [s.scenario_id for s in SCENARIOS] == FE_IDS
    assert list(SCENARIO_ORDER) == FE_IDS
    assert [s.ordinal for s in SCENARIOS] == list(range(1, 15))


def test_questions_non_empty_and_unique():
    questions = [s.question for s in SCENARIOS]
    assert all(q.strip() == q and q for q in questions)
    assert len(set(questions)) == 14


def test_scenario_has_exactly_the_execution_fields():
    assert list(Scenario.__dataclass_fields__) == ["scenario_id", "ordinal", "question"]


def test_question_hash_matches_hashlib_and_model():
    question = SCENARIOS[1].question  # contains non-ASCII "più"
    expected = hashlib.sha256(question.encode("utf-8")).hexdigest()
    assert question_sha256(question) == expected
    assert ScenarioExecution.new("r", "FE-02", 2, question).question_sha256 == expected


# --- Extraction --------------------------------------------------------------


def test_extract_minimal_fixture():
    text = _doc([("FE-01", "Quante? `no`"), ("FE-02", "C'è più?")])
    assert extract_questions(text) == [("FE-01", "Quante? `no`"), ("FE-02", "C'è più?")]


def test_extract_rejects_duplicate_id():
    with pytest.raises(ValueError, match="duplicate"):
        extract_questions(_doc([("FE-01", "a"), ("FE-01", "b")]))


def test_extract_rejects_missing_question():
    text = "### FE-01 — T\n\n- **Intent:** x\n\n### FE-02 — T\n\n- **Question:** `b`\n"
    with pytest.raises(ValueError, match="FE-01: question missing"):
        extract_questions(text)


def test_extract_rejects_two_questions_in_one_section():
    text = "### FE-01 — T\n- **Question:** `a`\n- **Question:** `b`\n"
    with pytest.raises(ValueError, match="more than one"):
        extract_questions(text)


# --- Drift guard -------------------------------------------------------------


def test_drift_ok_on_matching_fixture(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path, monkeypatch, _doc(_catalogue_entries()))
    result = check_drift(repo)
    assert result.ok, result.errors


def test_drift_detects_changed_question(tmp_path, monkeypatch):
    entries = _catalogue_entries()
    entries[4] = (entries[4][0], entries[4][1] + " ")
    result = check_drift(_write_repo(tmp_path, monkeypatch, _doc(entries)))
    assert not result.ok
    assert result.errors == ("FE-05: question differs from design document",)


def test_drift_detects_missing_scenario(tmp_path, monkeypatch):
    result = check_drift(_write_repo(tmp_path, monkeypatch, _doc(_catalogue_entries()[:13])))
    assert not result.ok
    assert any("design scenarios" in e for e in result.errors)


def test_drift_detects_extra_scenario(tmp_path, monkeypatch):
    entries = _catalogue_entries() + [("FE-15", "extra")]
    result = check_drift(_write_repo(tmp_path, monkeypatch, _doc(entries)))
    assert not result.ok
    assert any("FE-15" in e for e in result.errors)


def test_drift_detects_reordered_scenarios(tmp_path, monkeypatch):
    entries = _catalogue_entries()
    entries[0], entries[1] = entries[1], entries[0]
    result = check_drift(_write_repo(tmp_path, monkeypatch, _doc(entries)))
    assert not result.ok
    assert any("design scenarios" in e for e in result.errors)


def test_drift_detects_catalogue_out_of_order(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path, monkeypatch, _doc(_catalogue_entries()))
    swapped = (SCENARIOS[1], SCENARIOS[0]) + SCENARIOS[2:]
    result = check_drift(repo, scenarios=swapped)
    assert not result.ok
    assert any("catalogue scenarios" in e for e in result.errors)


def test_drift_detects_wrong_design_hash(tmp_path):
    directory = tmp_path / EVALUATION_DIR_RELATIVE
    directory.mkdir(parents=True)
    (directory / DESIGN_DOC_NAME).write_text(_doc(_catalogue_entries()), encoding="utf-8")
    result = check_drift(tmp_path)  # real pin, fixture bytes
    assert not result.ok
    assert result.errors == (
        f"design document hash {result.design_sha256} != pinned {contract.pinned_artifact(DESIGN_DOC_NAME).sha256}",
    )


def test_drift_reports_unparseable_document(tmp_path, monkeypatch):
    repo = _write_repo(tmp_path, monkeypatch, "### FE-01 — T\n\nno question\n")
    result = check_drift(repo)
    assert not result.ok
    assert any("unparseable" in e for e in result.errors)


def test_drift_reports_missing_document(tmp_path):
    result = check_drift(tmp_path)
    assert not result.ok
    assert result.design_sha256 is None
    assert "not found" in result.errors[0]


# --- CURRENT authority -------------------------------------------------------

_REAL_DESIGN = Path(REPO_ROOT) / EVALUATION_DIR_RELATIVE / DESIGN_DOC_NAME


@pytest.mark.skipif(not _REAL_DESIGN.is_file(), reason="git-ignored design authority not present locally")
def test_catalogue_matches_current_design_authority():
    result = check_drift()
    assert result.ok, result.errors
    assert result.design_sha256 == contract.pinned_artifact(DESIGN_DOC_NAME).sha256
