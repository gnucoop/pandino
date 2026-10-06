"""FE-01 … FE-14 grading contract (grading design §15).

    ScenarioEvidence + ground_truth.json + frozen policy + checks.py primitives
        → ScenarioCheckSet(checks, manual_reviews)

The catalogue answers "which checks belong to FE-xx" and produces grading
ingredients only. It decides no verdict, selects no root failure, merges no
human review and persists nothing: that is the verdict builder's job.

Conventions:

- one requirement → one entry with a stable ``check_id``: either a
  ``CheckResult`` or, when a required fact cannot be recovered from
  structured evidence but a human can read the answer, a PENDING
  ``ManualReviewItem`` with the same ID. IDs are unique across both lists;
- expected values come from ``ground_truth.json`` through ``TRUTH_BINDINGS``;
  bounds stated only in the design (FE-10 +4…+6 days, FE-13 45–65 %,
  FE-07 p < 0.001) and the design tolerances are module constants;
- structured sources (``charts[]``, the answer table, tool-shaped records)
  come before prose. Prose is never parsed for numbers: a text-only fact
  goes to targeted review;
- lexical screens never decide a check (grading design D5). They only add a
  focus hint to the question of a review item that is emitted regardless;
- runtime ``tool_call_*`` events and generated code support INFERRED
  tool-use checks only (grading design §5.1).

Only GRADEABLE scenarios are accepted. Output is deterministic: same
evidence, same ground truth, same catalogue → same checks in the same order.
"""

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from evaluation.datachat import contract
from evaluation.datachat.grading import checks as c
from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    Dimension,
    EvidenceClass,
    FailureCategory,
    Gate,
    ManualReviewItem,
    RequirementLevel,
)
from evaluation.datachat.grading.reader import ParseStatus, ScenarioEvidence
from evaluation.datachat.model import Availability
from evaluation.datachat.scenarios import SCENARIO_ORDER

MUST, SHOULD = RequirementLevel.MUST, RequirementLevel.SHOULD
DIRECT, INFERRED = EvidenceClass.DIRECT, EvidenceClass.INFERRED
MET, NOT_MET, IND = CheckStatus.MET, CheckStatus.NOT_MET, CheckStatus.INDETERMINATE
D, F = Dimension, FailureCategory

# Design §3 / grading design §11 tolerances.
TOL_MEAN = 0.05
TOL_MONEY = 1.0
TOL_CORRELATION = 0.01

# Bounds frozen in design §9 (not in ground_truth.json).
FE07_P_THRESHOLD = 0.001
FE10_STEP_RANGE = (4.0, 6.0)
FE13_SOUTH_SHARE_RANGE = (0.45, 0.65)


class CatalogueError(Exception):
    """Programmer/configuration misuse: unknown scenario ID or non-gradeable evidence."""


class GroundTruthError(Exception):
    """``ground_truth.json`` is missing or lacks a value the catalogue needs. Never a scenario verdict."""


# --- Ground truth ---------------------------------------------------------------

TRUTH_BINDINGS: dict[str, tuple[str, ...]] = {
    "FE-01": ("categorical_counts.region",),
    "FE-02": ("aggregates.mean_wait_days_by_service_in_south", "aggregates.mean_wait_days_by_service_overall"),
    "FE-03": ("aggregates.service",),
    "FE-04": ("missingness.per_column_na", "missingness.comment"),
    "FE-05": ("correlations.wait_days_vs_satisfaction", "guarantees.G6.evidence.range"),
    "FE-06": ("correlations.case_cost_eur_vs_satisfaction", "guarantees.G7.evidence.bound"),
    "FE-07": ("group_comparisons.online_vs_phone",),
    "FE-08": ("group_comparisons.in_person_vs_phone",),
    "FE-09": ("temporal.monthly",),
    "FE-10": ("temporal.monthly", "temporal.mean_wait_days_jan_jun", "temporal.mean_wait_days_jul_dec",
              "temporal.step_jul_dec_minus_jan_jun"),
    "FE-11": ("distribution.wait_days",),
    "FE-12": ("outliers.rows", "outliers.threshold", "outliers.max_non_outlier"),
    "FE-13": ("semantic.negative_share_by_region", "aggregates.region", "missingness.comment"),
    "FE-14": ("semantic.taxonomy", "missingness.comment"),
}


def load_ground_truth(repo_root: Path = contract.REPO_ROOT) -> dict[str, Any]:
    """Read ``ground_truth.json`` from the evaluation directory. No defaults are substituted."""
    path = contract.evaluation_dir(repo_root) / contract.GROUND_TRUTH_JSON_NAME
    if not path.is_file():
        raise GroundTruthError(f"ground truth not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GroundTruthError(f"ground truth unreadable: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise GroundTruthError("ground truth is not a JSON object")
    return data


def bind_truth(ground_truth: Mapping[str, Any], scenario_id: str) -> dict[str, Any]:
    """The ground-truth values bound to one scenario, by dotted key. Missing keys raise."""
    if scenario_id not in TRUTH_BINDINGS:
        raise CatalogueError(f"unknown scenario: {scenario_id!r}")
    bound = {}
    for key in TRUTH_BINDINGS[scenario_id]:
        node: Any = ground_truth
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise GroundTruthError(f"{scenario_id}: ground truth key missing: {key}")
            node = node[part]
        bound[key] = node
    return bound


def _gt_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundTruthError(f"{where}: expected a number")
    return float(value)


def _gt_map(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise GroundTruthError(f"{where}: expected a non-empty label → number map")
    return {str(k): _gt_number(v, f"{where}.{k}") for k, v in value.items()}


def _gt_records(value: Any, where: str, *keys: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise GroundTruthError(f"{where}: expected a non-empty list")
    for item in value:
        if not isinstance(item, Mapping) or any(k not in item for k in keys):
            raise GroundTruthError(f"{where}: records need keys {keys}")
    return value


def _gt_column_map(value: Any, where: str, label: str, number: str) -> dict[str, float]:
    return {str(r[label]): _gt_number(r[number], f"{where}.{number}") for r in _gt_records(value, where, label, number)}


# --- Output ---------------------------------------------------------------------

Digest = Callable[[tuple[str, ...]], str]


@dataclass(frozen=True)
class ScenarioCheckSet:
    """Grading ingredients of one scenario. Not a verdict."""

    scenario_id: str
    checks: tuple[CheckResult, ...]
    manual_reviews: tuple[ManualReviewItem, ...]


def grade_scenario_checks(
    evidence: ScenarioEvidence, ground_truth: Mapping[str, Any], *, digest: Digest
) -> ScenarioCheckSet:
    """Run the FE contract of ``evidence.scenario_id`` on GRADEABLE evidence.

    ``digest(refs)`` returns the evidence SHA-256 a review answer is bound to
    (``reader.evidence_digest`` over the run directory in production).
    """
    sid = evidence.scenario_id
    if sid not in CATALOGUE:
        raise CatalogueError(f"unknown scenario: {sid!r}")
    if evidence.gate is not Gate.GRADEABLE:
        raise CatalogueError(f"{sid}: gate {evidence.gate.value} is not gradeable")
    sheet = _Sheet(evidence, digest)
    CATALOGUE[sid](sheet, bind_truth(ground_truth, sid))
    return sheet.result()


class _Sheet:
    """Collects one scenario's checks and review items in emission order."""

    def __init__(self, ev: ScenarioEvidence, digest: Digest) -> None:
        self.ev = ev
        self._digest = digest
        self._checks: list[CheckResult] = []
        self._reviews: list[ManualReviewItem] = []
        self._ids: set[str] = set()

    def _claim(self, check_id: str) -> None:
        if check_id in self._ids:
            raise CatalogueError(f"duplicate check id: {check_id}")
        self._ids.add(check_id)

    def check(
        self,
        outcome: c.Outcome,
        check_id: str,
        dimension: Dimension,
        level: RequirementLevel,
        evidence_class: EvidenceClass,
        refs: Iterable[str | None],
        failure: FailureCategory | None = None,
    ) -> CheckResult:
        self._claim(check_id)
        result = c.make_check(
            outcome,
            check_id=check_id,
            dimension=dimension,
            requirement_level=level,
            evidence_class=evidence_class,
            evidence_refs=_refs(refs),
            failure_category=failure if level is MUST else None,
        )
        self._checks.append(result)
        return result

    def review(
        self,
        check_id: str,
        dimension: Dimension,
        level: RequirementLevel,
        question: str,
        refs: Iterable[str | None],
        failure: FailureCategory | None,
        *,
        focus: Sequence[str] = (),
    ) -> ManualReviewItem | None:
        """A PENDING targeted question. Returns None (no item) when no evidence file can be cited."""
        evidence_refs = _refs(refs)
        if not evidence_refs:
            self.check(c.indeterminate("no evidence file to review"), check_id, dimension, level,
                       EvidenceClass.MANUAL_REVIEW_ONLY, (), failure)
            return None
        self._claim(check_id)
        if focus:
            question = f"{question} Lexical screen (focus only, not a verdict): {', '.join(focus)}."
        item = ManualReviewItem(
            check_id=check_id,
            dimension=dimension,
            requirement_level=level,
            question=question,
            evidence_refs=evidence_refs,
            failure_category_if_failed=failure if level is MUST else None,
            evidence_sha256=self._digest(evidence_refs),
        )
        self._reviews.append(item)
        return item

    def fact(
        self,
        outcome: c.Outcome,
        check_id: str,
        dimension: Dimension,
        level: RequirementLevel,
        refs: Iterable[str | None],
        failure: FailureCategory | None,
        question: str,
    ) -> None:
        """A DIRECT structured check; if not recoverable and the answer is readable, a targeted review."""
        if outcome.status is IND and _readable(self.ev):
            self.review(check_id, dimension, level, question, (_response_ref(self.ev),), failure)
        else:
            self.check(outcome, check_id, dimension, level, DIRECT, refs, failure)

    def result(self) -> ScenarioCheckSet:
        return ScenarioCheckSet(self.ev.scenario_id, tuple(self._checks), tuple(self._reviews))


# --- Evidence access --------------------------------------------------------------

_RESPONSE_REF = "http/datachat.response.bin"
_RUNTIME_REF = "runtime/datachat.log"


def _refs(refs: Iterable[str | None]) -> tuple[str, ...]:
    out: list[str] = []
    for ref in refs:
        if ref and ref not in out:
            out.append(ref)
    return tuple(out)


def _response_ref(ev: ScenarioEvidence) -> str | None:
    return ev.refs.get(_RESPONSE_REF)


def _runtime_ref(ev: ScenarioEvidence) -> str | None:
    return ev.refs.get(_RUNTIME_REF)


def _code_ref(ev: ScenarioEvidence) -> str | None:
    return ev.agent_run.ref if ev.agent_run else None


def _usage_ref(ev: ScenarioEvidence) -> str | None:
    return ev.usage.ref if ev.usage else None


def _tool_refs(ev: ScenarioEvidence) -> tuple[str | None, ...]:
    return (_runtime_ref(ev), _code_ref(ev))


def _parsed(ev: ScenarioEvidence) -> bool:
    return ev.response is not None and ev.response.parse_status is ParseStatus.OK


def _text(ev: ScenarioEvidence) -> str | None:
    return ev.response.text if _parsed(ev) and ev.response.text and ev.response.text.strip() else None


def _readable(ev: ScenarioEvidence) -> bool:
    """A human can read an answer (text or table) in the response body."""
    return _parsed(ev) and (_text(ev) is not None or bool(ev.response.table_rows))


def _code(ev: ScenarioEvidence) -> str | None:
    """Generated code, when the agent_runs record was parsed (INFERRED evidence)."""
    if ev.agent_run is None or not ev.agent_run.parse_ok:
        return None
    return "\n".join(ev.agent_run.code_blocks)


def _calls(code: str, tool: str) -> list[str]:
    """Argument text of ``tool(...)`` calls up to the first closing parenthesis (best effort)."""
    return re.findall(rf"\b{re.escape(tool)}\s*\(([^)]*)", code)


def _tool_events(ev: ScenarioEvidence, tool: str, outcome: str = "result") -> list[Any]:
    events = ev.tools.runtime_events if ev.tools else ()
    return [e for e in events if e.tool == tool and e.outcome == outcome]


def _tool_use(ev: ScenarioEvidence, tool: str) -> c.Outcome:
    """INFERRED: was ``tool`` used? Runtime result event or a call in generated code."""
    if _tool_events(ev, tool):
        return c.met(tool, "runtime tool_call_result")
    code = _code(ev)
    if code is not None and _calls(code, tool):
        return c.met(tool, "call in generated code")
    if code is not None or ev.runtime is not None:
        return c.not_met(tool, None, "no runtime event and no call in generated code")
    return c.indeterminate("no generated code and no runtime log", tool)


def _lexical(text: str | None, pattern: str) -> tuple[str, ...]:
    """Distinct matches of a wording screen, for review focus only."""
    if not text:
        return ()
    return tuple(sorted({m.group(0).casefold() for m in re.finditer(pattern, text, re.IGNORECASE)}))[:5]


_CAUSAL = r"\bcaus\w*|\bprovoc\w*|\bdetermin\w*|\bperch[eé]\b|\bporta(?:no)? a\b"
_STRENGTH = r"\bfort\w*|\bdebol\w*|\bmoderat\w*|\btrascurabil\w*|\bnessuna\b"
_SIGNIFICANCE = r"significativ\w*|p[- ]?value|valore p|\bp\s*[<=>≈]"
_EQUALITY = r"\buguali\b|stess[oa] livello|identic\w*|non differiscono|equivalent\w*"
_STEP = r"gradual\w*|lugli\w*|salto|gradino|brusc\w*|metà anno"
_ANOMALY = r"anomal\w*|outlier|estrem\w*|insolit\w*"
_NULLS = r"neutr\w*|mancant\w*|vuot\w*|null\w*|senza testo|altro"
_PROPOSAL = r"propost\w*|propongo|suggeri\w*|ipotes\w*|indicativ\w*"


# --- Structured extraction ----------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:g}"


def _fmt_map(values: Mapping[str, float]) -> str:
    return ", ".join(f"{k} {_fmt(v)}" for k, v in values.items())


def _norm(value: Any) -> str | None:
    return c.normalize_label(value)


def _chart_with_labels(ev: ScenarioEvidence, labels: Iterable[str]) -> Any:
    """First recorded chart whose labels include at least two expected labels (one if only one expected)."""
    wanted = {_norm(x) for x in labels}
    found = c.charts(ev.response)
    if not found.ok:
        return None
    for chart in found.value:
        pairs = c.chart_mapping(chart)
        if pairs.ok and sum(_norm(label) in wanted for label, _ in pairs.value) >= min(2, len(wanted)):
            return chart
    return None


def _labelled_values(
    ev: ScenarioEvidence,
    labels: Iterable[str],
    *,
    value_columns: Sequence[str] = (),
    order: Sequence[str] = ("charts", "table"),
) -> c.Extracted:
    """``(label, value)`` pairs for known labels: ``charts[]`` / answer table in ``order``."""
    labels = list(labels)
    errors = []
    for source in order:
        if source == "charts":
            chart = _chart_with_labels(ev, labels)
            if chart is not None:
                return c.chart_mapping(chart)
            errors.append("no chart with the expected labels")
        else:
            found = c.table_label_values(ev.response, labels, value_columns=value_columns)
            if found.ok:
                return found
            errors.append(f"table: {found.error}")
    return c.Extracted(error="; ".join(errors))


def _lookup(pairs: Sequence[tuple[Any, Any]], label: str) -> Any:
    for key, value in pairs:
        if _norm(key) == _norm(label):
            return value
    return None


def _numeric(pairs: Sequence[tuple[Any, Any]], *, percent: bool = False) -> dict[str, float] | None:
    """Normalized label → number, or None if any value is unparseable or a label repeats."""
    out: dict[str, float] = {}
    for label, value in pairs:
        key, parsed = _norm(label), c.to_number(value, percent=percent)
        if key is None or key in out or not parsed.ok:
            return None
        out[key] = parsed.value
    return out


def _top(values: Mapping[str, float]) -> str | None:
    """Label of the strict maximum; None on a tie or an empty map."""
    if not values:
        return None
    best = max(values.values())
    winners = [k for k, v in values.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def _top_outcome(pairs: c.Extracted, expected_top: str, *, percent: bool = False) -> c.Outcome:
    if not pairs.ok:
        return c.indeterminate(f"extraction failed: {pairs.error}", expected_top)
    values = _numeric(pairs.value, percent=percent)
    if values is None:
        return c.indeterminate("values not parseable or labels repeated", expected_top)
    top = _top(values)
    if top is None:
        return c.indeterminate("no unique maximum", expected_top)
    return c.compare_exact(expected_top, top, normalize=True)


def _all(*parts: tuple[str, c.Outcome]) -> c.Outcome:
    """Conjunction of named outcomes: any NOT_MET → NOT_MET, else any INDETERMINATE → INDETERMINATE."""
    statuses = [o.status for _, o in parts]
    expected = {name: o.expected if not isinstance(o.expected, (list, dict)) else str(o.expected) for name, o in parts}
    observed = {name: o.observed if not isinstance(o.observed, (list, dict)) else str(o.observed) for name, o in parts}
    notes = "; ".join(f"{name}: {o.note}" for name, o in parts if o.note) or None
    if NOT_MET in statuses:
        return c.not_met(expected, observed, notes)
    if IND in statuses:
        return c.indeterminate(notes or "indeterminate", expected, observed)
    return c.met(expected, observed, notes)


def _with_note(outcome: c.Outcome, note: str) -> c.Outcome:
    return replace(outcome, note=f"{outcome.note}; {note}" if outcome.note else note)


def _records(ev: ScenarioEvidence) -> list[dict[str, Any]] | None:
    rows = c.table_rows(ev.response)
    if not rows.ok or not rows.value or not all(isinstance(r, dict) for r in rows.value):
        return None
    return list(rows.value)


def _single_record(ev: ScenarioEvidence, *keys: str) -> dict[str, Any] | None:
    """The answer table's only row, if it carries all ``keys`` (tool-shaped result)."""
    rows = _records(ev)
    if rows is None or len(rows) != 1 or any(k not in rows[0] for k in keys):
        return None
    return rows[0]


_MONTH = re.compile(r"^(\d{4})-(\d{2})(?:-\d{2})?(?:[T ][\d:.]+)?$")


def _month_key(label: Any) -> str | None:
    match = _MONTH.match(label.strip()) if isinstance(label, str) else None
    return f"{match.group(1)}-{match.group(2)}" if match else None


def _monthly(pairs: Sequence[tuple[Any, Any]]) -> c.Extracted:
    """Pairs relabelled to ``YYYY-MM``; one point per month, else not a monthly series."""
    out, seen = [], set()
    for label, value in pairs:
        key = _month_key(label)
        if key is None:
            return c.Extracted(error=f"label is not a date: {str(label)[:40]!r}")
        if key in seen:
            return c.Extracted(error="several points per month (not bucketed by month)")
        seen.add(key)
        out.append((key, value))
    return c.Extracted(tuple(out))


def _temporal_chart(ev: ScenarioEvidence) -> tuple[Any, c.Extracted | None]:
    """First chart whose labels are all dates, and its monthly pairs (or why not monthly)."""
    found = c.charts(ev.response)
    if not found.ok:
        return None, None
    for chart in found.value:
        pairs = c.chart_mapping(chart)
        if pairs.ok and pairs.value and all(_month_key(label) for label, _ in pairs.value):
            return chart, _monthly(pairs.value)
    return None, None


def _monthly_table(ev: ScenarioEvidence, value_columns: Sequence[str]) -> c.Extracted:
    """``(YYYY-MM, value)`` from an answer table with one all-date column."""
    rows = _records(ev)
    if rows is None:
        return c.Extracted(error="no table")
    date_cols = [k for k in rows[0] if all(_month_key(r.get(k)) for r in rows)]
    if len(date_cols) != 1:
        return c.Extracted(error="no unique date column")
    others = [k for k in rows[0] if k != date_cols[0]]
    named = [k for k in value_columns if k in others]
    if named:
        value_col = named[0]
    else:
        numeric = [k for k in others if all(c.to_number(r.get(k)).ok for r in rows)]
        if len(numeric) != 1:
            return c.Extracted(error="no unique numeric column")
        value_col = numeric[0]
    return _monthly([(r[date_cols[0]], r[value_col]) for r in rows])


def _correlation(ev: ScenarioEvidence, x: str, y: str) -> dict[str, Any] | None:
    """The ``correlation`` tool record for the pair (x, y), when it is the answer table's only row."""
    record = _single_record(ev, "col_x", "col_y", "correlation")
    if record is None or {_norm(record["col_x"]), _norm(record["col_y"])} != {_norm(x), _norm(y)}:
        return None
    return record


def _usage_contribution(ev: ScenarioEvidence) -> c.Outcome:
    """DIRECT: Usage rows beyond the primary one (a semantic provider call happened)."""
    usage = ev.usage
    if usage is None or usage.availability is not Availability.PRESENT:
        reason = usage.reason if usage else "not loaded"
        return c.indeterminate(f"usage rows unavailable: {reason}", ">= 1 extra row")
    if usage.primary_log_id is None:
        return c.indeterminate("primary log id unknown", ">= 1 extra row", len(usage.rows))
    extra = len(usage.extra_rows)
    return (c.met if extra >= 1 else c.not_met)(">= 1 extra row", extra)


# --- Scenarios --------------------------------------------------------------------


def _fe01(s: _Sheet, t: dict[str, Any]) -> None:
    ev, resp = s.ev, s.ev.response
    counts = _gt_map(t["categorical_counts.region"], "categorical_counts.region")
    pairs = _labelled_values(ev, counts)
    s.fact(c.from_extracted(pairs, c.compare_mapping, counts), "fe01.region_counts", D.ANALYTICAL, MUST,
           (_response_ref(ev),), F.ANALYTICAL,
           f"Does the answer report exactly these request counts per region: {_fmt_map(counts)}?")

    present = c.check_chart_count(resp)
    dimension, failure = D.UNDERSTANDING, F.UNDERSTANDING
    if present.status is NOT_MET:
        attempt = _tool_use(ev, "chart")
        if attempt.status is MET:
            # Grading design §16 step 4: INFERRED call confirmed by the DIRECT empty charts[].
            dimension, failure = D.BACKEND_CONTRACT, F.BACKEND_CONTRACT
            present = _with_note(present, f"chart attempted ({attempt.observed}) but charts[] is empty")
        else:
            present = _with_note(present, "no chart attempt in generated code or runtime events")
    s.check(present, "fe01.chart_present", dimension, MUST, DIRECT, (_response_ref(ev), *_tool_refs(ev)), failure)

    found = c.charts(resp)
    chart = _chart_with_labels(ev, counts) or (found.value[0] if found.ok and found.value else None)
    if chart is None:
        categorical = bar = c.not_applicable("no chart recorded")
    else:
        categorical = c.check_chart_type(chart, ("bar", "pie", "doughnut"))
        bar = c.check_chart_type(chart, ("bar",))
    s.check(categorical, "fe01.chart_categorical", D.UNDERSTANDING, MUST, DIRECT, (_response_ref(ev),),
            F.UNDERSTANDING)
    s.check(bar, "fe01.chart_bar", D.UNDERSTANDING, SHOULD, DIRECT, (_response_ref(ev),))


_MEAN_COLUMNS = ("mean", "mean_wait_days", "wait_days", "media")


def _fe02(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    key_south = "aggregates.mean_wait_days_by_service_in_south"
    south = _gt_column_map(t[key_south], key_south, "service", "mean")
    overall = _gt_column_map(t["aggregates.mean_wait_days_by_service_overall"],
                             "aggregates.mean_wait_days_by_service_overall", "service", "mean")
    south_n = _gt_column_map(t[key_south], key_south, "service", "n")
    top = _top(south)
    if top is None:
        raise GroundTruthError(f"{key_south}: no unique top service")

    # The answer's own result: table first, then charts[] (charts are checked separately below).
    pairs = _labelled_values(ev, south, value_columns=_MEAN_COLUMNS, order=("table", "charts"))
    top_value = _lookup(pairs.value, top) if pairs.ok else None
    outcome = _all(("top", _top_outcome(pairs, top)),
                   (top, c.compare_number(south[top], top_value, tolerance=TOL_MEAN)
                    if top_value is not None else c.indeterminate("top service value not found")))
    failure = F.ANALYTICAL
    code = _code(ev)
    if outcome.status is NOT_MET and code is not None and not re.search(r"['\"](South|Sud)['\"]", code):
        # Design §9 FE-02: INFERRED absence of any South constraint, confirmed by the DIRECT wrong result.
        failure = F.UNDERSTANDING
        outcome = _with_note(outcome, "generated code shows no South constraint")
    s.fact(outcome, "fe02.south_top_service", D.ANALYTICAL, MUST, (_response_ref(ev), _code_ref(ev)), failure,
           f"Does the answer name {top} as the service with the highest mean wait in the South "
           f"(≈{south[top]:.2f} days, ±{TOL_MEAN})?")

    labels = [label for label, _ in pairs.value] if pairs.ok else None
    s.fact(c.check_subset(list(south), labels) if labels is not None else c.indeterminate("comparison not found"),
           "fe02.services_compared", D.UNDERSTANDING, MUST, (_response_ref(ev),), F.UNDERSTANDING,
           f"Does the answer compare all four services within the South ({', '.join(south)})?")

    chart = _chart_with_labels(ev, south)
    if chart is None:
        chart_values = c.not_applicable("no service chart recorded")
    else:
        chart_values = c.from_extracted(c.chart_mapping(chart), c.compare_mapping, south, tolerance=TOL_MEAN)
        if chart_values.status is NOT_MET:
            global_match = c.from_extracted(c.chart_mapping(chart), c.compare_mapping, overall, tolerance=TOL_MEAN)
            if global_match.status is MET:
                chart_values = _with_note(chart_values, "charts[] carry the global (unfiltered) means")
    # MUST "South only" applied to the recorded comparison; mismatch maps to CHAINING (design §9 FE-02).
    s.check(chart_values, "fe02.chart_south_values", D.CHAINING, MUST, DIRECT, (_response_ref(ev),), F.CHAINING)

    if chart is None:
        derived = c.not_applicable("no service chart recorded")
    elif code is None:
        derived = c.indeterminate("no generated code", "chart(data=<aggregate result>)")
    else:
        chart_calls = _calls(code, "chart")
        derived = (c.met if any("data" in args for args in chart_calls) else c.not_met)(
            "chart(data=<aggregate result>)", f"{len(chart_calls)} chart call(s)")
    s.check(derived, "fe02.chart_from_aggregate", D.CHAINING, SHOULD, INFERRED, (_code_ref(ev),))

    small = sorted(k for k, n in south_n.items() if n < 15)
    note = ev.response.note if _parsed(ev) else None
    lead = (f"The response carries the trusted note {note!r}. " if note
            else "The response carries no trusted note. ")
    s.review("fe02.small_group_caveat", D.HONESTY, SHOULD,
             lead + f"Does the answer acknowledge that the South result for {', '.join(small)} rests on a small "
             f"group (n={', '.join(_fmt(south_n[k]) for k in small)})?",
             (_response_ref(ev),), F.HONESTY)


def _fe03(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    rows = t["aggregates.service"]
    sums = _gt_column_map(rows, "aggregates.service", "service", "sum_case_cost_eur")
    means = _gt_column_map(rows, "aggregates.service", "service", "mean_case_cost_eur")
    top_sum, top_mean = _top(sums), _top(means)
    if top_sum is None or top_mean is None:
        raise GroundTruthError("aggregates.service: no unique top service")

    def column(keywords: tuple[str, ...]) -> c.Extracted:
        records = _records(ev)
        if records is None:
            return c.Extracted(error="no table")
        names = [k for k in records[0] if any(w in (_norm(k) or "") for w in keywords)]
        if len(names) != 1:
            return c.Extracted(error=f"no unique column named like {keywords}")
        return c.table_label_values(ev.response, sums, value_columns=names)

    sum_pairs = column(("sum", "total", "somma"))
    mean_pairs = column(("mean", "avg", "average", "media"))
    ref = (_response_ref(ev),)
    s.fact(_top_outcome(sum_pairs, top_sum), "fe03.top_total", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer name {top_sum} as the service with the highest TOTAL case cost?")
    s.fact(_top_outcome(mean_pairs, top_mean), "fe03.top_mean", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer name {top_mean} as the service with the highest MEAN case cost?")
    both = c.met("total and mean reported separately", "two metric columns") if sum_pairs.ok and mean_pairs.ok \
        else c.indeterminate("metric distinction not recoverable from structure")
    s.fact(both, "fe03.metrics_distinguished", D.UNDERSTANDING, MUST, ref, F.UNDERSTANDING,
           "Does the answer explicitly distinguish total cost from mean cost (two different metrics, "
           "possibly two different services)?")

    values = _all(
        (f"total {top_sum}", c.compare_number(sums[top_sum], _lookup(sum_pairs.value, top_sum), tolerance=TOL_MONEY)
         if sum_pairs.ok else c.indeterminate("total not structured")),
        (f"mean {top_mean}", c.compare_number(means[top_mean], _lookup(mean_pairs.value, top_mean),
                                              tolerance=TOL_MONEY)
         if mean_pairs.ok else c.indeterminate("mean not structured")),
    )
    s.check(values, "fe03.approximate_values", D.ANALYTICAL, SHOULD, DIRECT, ref)


def _fe04(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    na = _gt_map(t["missingness.per_column_na"], "missingness.per_column_na")
    comment = _gt_map(t["missingness.comment"], "missingness.comment")
    nonzero = {k: v for k, v in na.items() if v}
    zero = [k for k, v in na.items() if not v]
    pairs = c.table_label_values(ev.response, na, value_columns=("missing", "missing_count", "n_missing", "na",
                                                                 "na_count", "null_count", "nulls"))
    ref = (_response_ref(ev),)

    counts = c.from_extracted(pairs, c.compare_mapping, nonzero, allow_extra=True)
    observed_comment = _lookup(pairs.value, "comment") if pairs.ok else None
    conflated = comment.get("missing", 0) + comment.get("blank", 0)
    if counts.status is NOT_MET and c.to_number(observed_comment).value == conflated:
        counts = _with_note(counts, "comment count equals NA + whitespace-only blanks (conflated)")
    s.fact(counts, "fe04.na_counts", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer report exactly these missing (NA) counts: {_fmt_map(nonzero)} "
           f"(whitespace-only comments are not NA)?")

    if not pairs.ok:
        others = c.indeterminate(f"extraction failed: {pairs.error}", 0)
    else:
        listed = {_norm(k): v for k, v in pairs.value}
        present = {k: listed[_norm(k)] for k in zero if _norm(k) in listed}
        parsed = {k: c.to_number(v) for k, v in present.items()}
        wrong = {k: p.value for k, p in parsed.items() if p.ok and p.value != 0}
        if wrong:
            others = c.not_met(0, wrong)
        elif not all(p.ok for p in parsed.values()):
            others = c.indeterminate("unparseable count", 0)
        elif len(present) == len(zero):
            others = c.met(0, "all other columns listed with 0")
        else:
            others = c.indeterminate("not every other column is listed", 0, len(present))
    s.fact(others, "fe04.other_columns_complete", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer make clear that no other column has missing values ({', '.join(zero)})?")


def _correlation_value(truth: Mapping[str, Any], method: str | None, observed: Any) -> c.Outcome:
    """Against the method-matched reference; either reference when the method is not structured."""
    refs = {m: _gt_number(truth.get(m), f"correlation {m}") for m in ("pearson", "spearman")}
    candidates = [method] if method in refs else list(refs)
    outcomes = [c.compare_number(refs[m], observed, tolerance=TOL_CORRELATION) for m in candidates]
    for outcome in outcomes:
        if outcome.status is MET:
            return outcome
    return outcomes[0] if len(outcomes) == 1 else _all(*zip(candidates, outcomes))


def _fe05(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    truth = t["correlations.wait_days_vs_satisfaction"]
    low, high = (_gt_number(v, "G6.range") for v in t["guarantees.G6.evidence.range"])
    n = _gt_number(truth["n"], "correlations.n")
    record = _correlation(ev, truth["x"], truth["y"])
    ref = (_response_ref(ev),)
    text = _text(ev)

    if record is None:
        value = c.indeterminate("no structured correlation record")
    else:
        method = _norm(record.get("method"))
        value = _all(("reference", _correlation_value(truth, method, record["correlation"])),
                     ("G6", c.check_range(record["correlation"], min_value=low, max_value=high)))
    s.fact(value, "fe05.coefficient", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer report a coefficient for wait_days vs satisfaction within ±{TOL_CORRELATION} of the "
           f"method-matched reference (Spearman {_fmt(truth['spearman'])}, Pearson {_fmt(truth['pearson'])}) "
           "and inside G6 bounds? Answer NOT_APPLICABLE if no coefficient is quoted.")
    s.fact(c.check_sign(record["correlation"], "negative") if record else c.indeterminate("no structured sign"),
           "fe05.direction", D.INTERPRETATION, MUST, ref, F.INTERPRETATION,
           "Does the answer state a negative association: longer waits go with lower satisfaction?")
    s.review("fe05.moderate_strength", D.INTERPRETATION, MUST,
             "Does the answer describe the association as moderate (not strong, not weak or negligible)?",
             ref, F.INTERPRETATION, focus=_lexical(text, _STRENGTH))
    s.review("fe05.no_causal_claim", D.INTERPRETATION, MUST,
             "Does the answer avoid claiming that waiting time causes lower satisfaction?",
             ref, F.INTERPRETATION, focus=_lexical(text, _CAUSAL))

    code = _code(ev)
    if record is not None and _norm(record.get("method")):
        spearman = c.compare_exact("spearman", record.get("method"), normalize=True)
        spearman_class = DIRECT
    elif text and re.search(r"spearman", text, re.IGNORECASE):
        spearman, spearman_class = c.met("spearman", "named in the answer"), DIRECT
    elif code is not None and re.search(r"spearman", code, re.IGNORECASE):
        spearman, spearman_class = c.met("spearman", "in generated code"), INFERRED
    elif code is not None:
        spearman, spearman_class = c.not_met("spearman", "not in generated code"), INFERRED
    else:
        spearman, spearman_class = c.indeterminate("method not observable", "spearman"), INFERRED
    s.check(spearman, "fe05.spearman_method", D.TOOL_SELECTION, SHOULD, spearman_class, (*ref, _code_ref(ev)))

    # Supporting diagnostic quantity (grading design D7): SHOULD only, never a MUST failure.
    if record is not None and "n" in record:
        sample = c.compare_number(n, record["n"])
    elif text is not None:
        token = re.search(rf"(?<![\d.,]){int(n)}(?![\d.,])", text)
        sample = (c.met if token else c.not_met)(int(n), "reported" if token else "not reported")
    else:
        sample = c.indeterminate("no text or structured n", int(n))
    s.check(sample, "fe05.sample_size_reported", D.ANALYTICAL, SHOULD, DIRECT, ref)


def _fe06(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    truth = t["correlations.case_cost_eur_vs_satisfaction"]
    bound = _gt_number(t["guarantees.G7.evidence.bound"], "G7.bound")
    record = _correlation(ev, truth["x"], truth["y"])
    ref = (_response_ref(ev),)
    near_zero = (c.check_range(record["correlation"], max_value=bound, include_max=False, absolute=True)
                 if record else c.indeterminate("no structured correlation record"))
    s.fact(near_zero, "fe06.coefficient_near_zero", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer show a case_cost_eur vs satisfaction association near zero (|r| < {_fmt(bound)}; "
           f"reference Pearson {_fmt(truth['pearson'])}, Spearman {_fmt(truth['spearman'])})?")
    s.review("fe06.negligible_wording", D.INTERPRETATION, MUST,
             "Does the answer describe the relationship as negligible / no appreciable association, without "
             "reading a real (negative) relationship or a causal link into the small coefficient?",
             ref, F.INTERPRETATION, focus=_lexical(_text(ev), _CAUSAL + "|" + _STRENGTH))
    quoted = (_correlation_value(truth, _norm(record.get("method")), record["correlation"]) if record
              else c.indeterminate("coefficient not structured; prose is not parsed"))
    s.check(quoted, "fe06.coefficient_quoted", D.ANALYTICAL, SHOULD, DIRECT, ref)


def _comparison(ev: ScenarioEvidence, truth: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Structured group comparison: (``{group: mean}``, ``{"difference"|"p_value": raw}``) in ground-truth orientation."""
    a, b = str(truth["group_a"]), str(truth["group_b"])
    record = _single_record(ev, "group_a", "group_b", "mean_a", "mean_b")
    if record is not None and {_norm(record["group_a"]), _norm(record["group_b"])} == {_norm(a), _norm(b)}:
        flipped = _norm(record["group_a"]) != _norm(a)
        means = {a: record["mean_b" if flipped else "mean_a"], b: record["mean_a" if flipped else "mean_b"]}
        extra = {"p_value": record.get("p_value")}
        diff = c.to_number(record.get("difference"))
        if diff.ok:
            extra["difference"] = -diff.value if flipped else diff.value
        return means, extra
    pairs = c.table_label_values(ev.response, (a, b), value_columns=("mean", "mean_satisfaction", "satisfaction"))
    if not pairs.ok:
        chart = _chart_with_labels(ev, (a, b))
        pairs = c.chart_mapping(chart) if chart is not None else pairs
    if pairs.ok:
        ma, mb = c.to_number(_lookup(pairs.value, a)), c.to_number(_lookup(pairs.value, b))
        if ma.ok and mb.ok:
            return {a: ma.value, b: mb.value}, {"difference": ma.value - mb.value}
    return {}, {}


def _test_evidence(ev: ScenarioEvidence) -> c.Outcome:
    """INFERRED: an inferential test ran (compare_groups, or a scipy test in generated code)."""
    used = _tool_use(ev, "compare_groups")
    code = _code(ev)
    if used.status is not MET and code is not None and re.search(r"ttest|mannwhitney|scipy\.stats", code):
        return c.met("statistical test", "scipy test in generated code")
    return used


def _fe07(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    truth = t["group_comparisons.online_vs_phone"]
    a, b = str(truth["group_a"]), str(truth["group_b"])
    difference = _gt_number(truth["difference"], "online_vs_phone.difference")
    means, extra = _comparison(ev, truth)
    ref = (_response_ref(ev),)
    higher, lower = (a, b) if difference > 0 else (b, a)

    direction = (c.compare_order(means[higher], ">", means[lower]) if means
                 else c.indeterminate("group means not structured"))
    s.fact(direction, "fe07.direction", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer state that {higher} satisfaction is higher than {lower}?")
    diff = (c.compare_number(difference, extra["difference"], tolerance=TOL_MEAN) if "difference" in extra
            else c.indeterminate("difference not structured"))
    s.fact(diff, "fe07.difference", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer report a difference of ≈{difference:.2f} points (±{TOL_MEAN}) between {a} and {b}, "
           "directly or through both group means?")

    test = _test_evidence(ev)
    if test.status is MET:
        s.check(_with_note(test, "an inferential test ran, so a significance statement can be supported"),
                "fe07.significance_supported", D.HONESTY, MUST, INFERRED, _tool_refs(ev), F.HONESTY)
    else:
        s.review("fe07.significance_supported", D.HONESTY, MUST,
                 "No statistical test is visible in generated code or runtime events. Does the answer avoid "
                 "claiming statistical significance or quoting a p-value?",
                 (*ref, *_tool_refs(ev)), F.HONESTY, focus=_lexical(_text(ev), _SIGNIFICANCE))

    s.check(_tool_use(ev, "compare_groups"), "fe07.compare_groups", D.TOOL_SELECTION, SHOULD, INFERRED,
            _tool_refs(ev))
    p = (c.check_threshold(extra["p_value"], "<", FE07_P_THRESHOLD) if extra.get("p_value") is not None
         else c.indeterminate("p-value not structured; inferential wording is judged with the answer text"))
    s.check(p, "fe07.p_value_reported", D.HONESTY, SHOULD, DIRECT, ref)


def _reported_precision(raw: Any) -> float | None:
    """Half a unit of the last reported decimal (at most 6, the ground-truth precision)."""
    text = raw if isinstance(raw, str) else repr(raw) if isinstance(raw, float) else None
    if text is None:
        return None
    match = re.search(r"[.,](\d+)", text)
    decimals = min(len(match.group(1)), 6) if match else 0
    return 0.5 * 10 ** -decimals


def _fe08(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    truth = t["group_comparisons.in_person_vs_phone"]
    a, b = str(truth["group_a"]), str(truth["group_b"])
    difference = _gt_number(truth["difference"], "in_person_vs_phone.difference")
    p_truth = _gt_number(truth["p_value"], "in_person_vs_phone.p_value")
    means, extra = _comparison(ev, truth)
    ref = (_response_ref(ev),)
    text = _text(ev)

    diff = (c.compare_number(difference, extra["difference"], tolerance=TOL_MEAN) if "difference" in extra
            else c.indeterminate("difference not structured"))
    s.fact(diff, "fe08.small_difference", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer recognise a small observed difference between {a} (≈{_gt_number(truth['mean_a'], 'mean_a'):.2f}) "
           f"and {b} (≈{_gt_number(truth['mean_b'], 'mean_b'):.2f}), about {difference:.2f} points?")
    s.review("fe08.no_evidence_of_difference", D.INTERPRETATION, MUST,
             "Does the answer state that there is no evidence of a statistically meaningful difference "
             "(e.g. 'nessuna evidenza di una differenza', 'differenza non significativa / trascurabile')?",
             ref, F.INTERPRETATION, focus=_lexical(text, _SIGNIFICANCE))
    test = _test_evidence(ev)
    no_test = "" if test.status is MET else (
        " No statistical test is visible in generated code or runtime events: an equality claim would also "
        "be a TOOL SELECTION failure.")
    s.review("fe08.no_equality_claim", D.INTERPRETATION, MUST,
             "Does the answer avoid concluding that the groups are equal or proven not to differ "
             "('sono uguali', 'hanno lo stesso livello', 'è dimostrato che non differiscono')?" + no_test,
             (*ref, *_tool_refs(ev)), F.INTERPRETATION, focus=_lexical(text, _EQUALITY))

    s.check(_tool_use(ev, "compare_groups"), "fe08.compare_groups", D.TOOL_SELECTION, SHOULD, INFERRED,
            _tool_refs(ev))
    # Grading design §21: compared with the ground-truth value at reported precision; no normative threshold.
    raw_p = extra.get("p_value")
    precision = _reported_precision(raw_p)
    p = (c.compare_number(p_truth, raw_p, tolerance=precision + 1e-6) if raw_p is not None and precision is not None
         else c.indeterminate("p-value not structured"))
    s.check(p, "fe08.p_value_reported", D.HONESTY, SHOULD, DIRECT, ref)


def _fe09(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    monthly = _gt_column_map(t["temporal.monthly"], "temporal.monthly", "month", "requests")
    ref = (_response_ref(ev),)
    chart, chart_months = _temporal_chart(ev)

    found = c.charts(ev.response)
    if not found.ok:
        temporal = c.indeterminate(f"extraction failed: {found.error}", "temporal chart")
    elif not found.value:
        temporal = c.not_met("temporal chart", 0, "no chart recorded")
    elif chart is None:
        temporal = c.not_met("temporal chart", [c.chart_type(x).value for x in found.value],
                             "no recorded chart has date labels")
    else:
        temporal = c.met("temporal chart", c.chart_type(chart).value)
    s.check(temporal, "fe09.temporal_chart", D.UNDERSTANDING, MUST, DIRECT, ref, F.UNDERSTANDING)

    series = chart_months if chart_months is not None and chart_months.ok else _monthly_table(
        ev, ("count", "requests", "n"))
    s.fact(c.from_extracted(series, c.compare_mapping, monthly), "fe09.monthly_counts", D.ANALYTICAL, MUST, ref,
           F.ANALYTICAL, f"Does the answer report these monthly request counts for 2025: {_fmt_map(monthly)}?")
    first, last = next(iter(monthly)), list(monthly)[-1]
    s.review("fe09.growth_identified", D.INTERPRETATION, MUST,
             f"Does the answer describe the number of requests as increasing over 2025 "
             f"({_fmt(monthly[first])} in {first} to {_fmt(monthly[last])} in {last})?",
             ref, F.INTERPRETATION)
    if chart_months is not None and not chart_months.ok:
        s.review("fe09.daily_chart_readable", D.TOOL_SELECTION, MUST,
                 "The temporal chart is not bucketed by month (several points per month). Does it still show "
                 "the 2025 growth readably?",
                 ref, F.TOOL_SELECTION)

    runtime_trend = [e for e in _tool_events(ev, "trend")
                     if (e.fields.get("freq") or "").casefold().startswith("m")
                     and (e.fields.get("op") or "").casefold() == "count"]
    code = _code(ev)
    if runtime_trend:
        trend = c.met("trend(freq=month, op=count)", "runtime tool_call_result")
    elif code is not None and any("month" in args and "count" in args for args in _calls(code, "trend")):
        trend = c.met("trend(freq=month, op=count)", "call in generated code")
    elif code is not None or ev.runtime is not None:
        trend = c.not_met("trend(freq=month, op=count)", None)
    else:
        trend = c.indeterminate("no generated code and no runtime log", "trend(freq=month, op=count)")
    s.check(trend, "fe09.trend_tool", D.TOOL_SELECTION, SHOULD, INFERRED, _tool_refs(ev))
    s.check(c.check_chart_type(chart, ("line",)) if chart is not None else c.not_applicable("no temporal chart"),
            "fe09.line_chart", D.UNDERSTANDING, SHOULD, DIRECT, ref)


def _fe10(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    monthly = _gt_column_map(t["temporal.monthly"], "temporal.monthly", "month", "mean_wait_days")
    first_half = _gt_number(t["temporal.mean_wait_days_jan_jun"], "mean_wait_days_jan_jun")
    second_half = _gt_number(t["temporal.mean_wait_days_jul_dec"], "mean_wait_days_jul_dec")
    step = _gt_number(t["temporal.step_jul_dec_minus_jan_jun"], "step_jul_dec_minus_jan_jun")
    ref = (_response_ref(ev),)
    text = _text(ev)

    # Monthly means are graded only when the answer surfaces a series that names the waiting-time mean.
    chart, chart_months = _temporal_chart(ev)
    label = ""
    if chart is not None:
        datasets = chart.get("datasets") or [{}]
        label = _norm(datasets[0].get("label") if isinstance(datasets[0], dict) else None) or ""
    if chart_months is not None and chart_months.ok and re.search(r"wait|attesa|mean|media", label):
        series = chart_months
    else:
        series = _monthly_table(ev, ("mean_wait_days", "wait_days", "mean"))
        rows = _records(ev)
        if series.ok and not any(k in (rows[0] if rows else {}) for k in ("mean_wait_days", "wait_days", "mean")):
            series = c.Extracted(error="series not identified as mean wait_days")
    means = (c.compare_mapping(monthly, series.value, tolerance=TOL_MEAN) if series.ok
             else c.not_applicable("no monthly mean series surfaced"))
    s.check(means, "fe10.monthly_means", D.ANALYTICAL, MUST, DIRECT, ref, F.ANALYTICAL)

    s.review("fe10.increase_identified", D.INTERPRETATION, MUST,
             "Does the answer state that mean waiting time increased during 2025 (not stable, not decreasing)?",
             ref, F.INTERPRETATION)
    low, high = FE10_STEP_RANGE
    s.review("fe10.step_magnitude", D.ANALYTICAL, MUST,
             f"If the answer states the size of the change, is it within +{_fmt(low)}…+{_fmt(high)} days "
             f"(reference +{step:.2f}: Jan–Jun {first_half:.2f} vs Jul–Dec {second_half:.2f})? "
             "Answer NOT_APPLICABLE if no magnitude is stated.",
             ref, F.ANALYTICAL)
    # Design §9 FE-10 boundary: "gradual" with no step → PASS WITH CAVEAT, hence SHOULD.
    s.review("fe10.step_identified", D.INTERPRETATION, SHOULD,
             "Does the answer identify the change as a step around July / mid-year rather than a gradual increase?",
             ref, F.INTERPRETATION, focus=_lexical(text, _STEP))

    code = _code(ev)
    runtime_trend = [e for e in _tool_events(ev, "trend")
                     if (e.fields.get("op") or "").casefold() == "mean" and "wait" in (e.fields.get("metric") or "")]
    if runtime_trend:
        trend = c.met("trend(op=mean, metric=wait_days)", "runtime tool_call_result")
    elif code is not None and any("mean" in args and "wait_days" in args for args in _calls(code, "trend")):
        trend = c.met("trend(op=mean, metric=wait_days)", "call in generated code")
    elif code is not None or ev.runtime is not None:
        trend = c.not_met("trend(op=mean, metric=wait_days)", None)
    else:
        trend = c.indeterminate("no generated code and no runtime log", "trend(op=mean, metric=wait_days)")
    s.check(trend, "fe10.trend_mean", D.TOOL_SELECTION, SHOULD, INFERRED, _tool_refs(ev))


def _fe11(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    dist = t["distribution.wait_days"]
    if not isinstance(dist, Mapping) or any(k not in dist for k in ("skewness", "min", "max")):
        raise GroundTruthError("distribution.wait_days: skewness/min/max required")
    art = ev.artifacts
    ref = (_response_ref(ev), art.image_meta_ref if art else None)

    dimension, failure = D.BACKEND_CONTRACT, F.BACKEND_CONTRACT
    if not _parsed(ev):
        delivered = c.indeterminate("response not parsed", "image")
    elif ev.response.type == "image":
        meta = art.image_meta if art else None
        status = meta.get("decode_status") if meta else None
        delivered = (c.indeterminate("image metadata unavailable", "decode_status OK") if status is None
                     else c.compare_exact("OK", status))
    elif ev.response.type == "str" and ".png" in (ev.response.text or ""):
        delivered = c.not_met("image", "str", "a .png server path instead of a delivered image")
    else:
        dimension, failure = D.UNDERSTANDING, F.UNDERSTANDING
        delivered = c.not_met("image", ev.response.type, "no image answer")
    s.check(delivered, "fe11.image_delivered", dimension, MUST, DIRECT, ref, failure)

    found = c.charts(ev.response)
    def raw_value_bar(chart: Any) -> bool:
        pairs = c.chart_mapping(chart)
        return (c.check_chart_type(chart, ("bar",)).status is MET and pairs.ok and bool(pairs.value)
                and all(c.to_number(label).ok for label, _ in pairs.value))

    raw_bars = [x for x in (found.value if found.ok else ()) if raw_value_bar(x)]
    code = _code(ev)
    raw_bar_code = code is not None and any(re.search(r"['\"]bar['\"]", a) and "wait_days" in a
                                            for a in _calls(code, "chart"))
    if raw_bars:
        no_bar, bar_class = c.not_met("no per-value bar chart", "bar chart over raw values in charts[]"), DIRECT
    elif raw_bar_code:
        no_bar, bar_class = c.not_met("no per-value bar chart", "chart(kind='bar') over wait_days in code"), INFERRED
    elif found.ok:
        no_bar, bar_class = c.met("no per-value bar chart", "none in charts[]"), DIRECT
    else:
        no_bar, bar_class = c.indeterminate(f"extraction failed: {found.error}"), DIRECT
    s.check(no_bar, "fe11.no_raw_value_bar", D.TOOL_SELECTION, MUST, bar_class,
            (_response_ref(ev), _code_ref(ev)), F.TOOL_SELECTION)

    if code is None:
        hist = c.indeterminate("no generated code", "plot(kind='hist')")
    else:
        plots = _calls(code, "plot")
        hist = (c.met if any(re.search(r"kind\s*=\s*['\"]hist", a) for a in plots) else c.not_met)(
            "plot(kind='hist')", f"{len(plots)} plot call(s)")
    s.check(hist, "fe11.histogram", D.TOOL_SELECTION, SHOULD, INFERRED, (_code_ref(ev),))

    if art is not None and art.image_ref:
        s.review("fe11.image_suitable", D.TOOL_SELECTION, MUST,
                 f"Does the image show the distribution of wait_days (histogram, box plot or KDE; not a per-value "
                 f"categorical bar chart), with the right skew visible (skewness {_fmt(dist['skewness'])}, "
                 f"range {_fmt(dist['min'])}–{_fmt(dist['max'])} days)?",
                 (art.image_ref,), F.TOOL_SELECTION)


def _fe12(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    rows = _gt_records(t["outliers.rows"], "outliers.rows", "record_id", "service", "case_cost_eur")
    threshold = _gt_number(t["outliers.threshold"], "outliers.threshold")
    max_ordinary = _gt_number(t["outliers.max_non_outlier"], "outliers.max_non_outlier")
    costs = [_gt_number(r["case_cost_eur"], "outliers.case_cost_eur") for r in rows]
    services = sorted({str(r["service"]) for r in rows})
    ids = sorted(int(r["record_id"]) for r in rows)
    ref = (_response_ref(ev),)

    if not _parsed(ev):
        text_answer = c.indeterminate("response not parsed")
    else:
        text_answer = (c.not_met if ev.response.type == "image" else c.met)("text/table answer", ev.response.type)
    s.check(text_answer, "fe12.not_plot_only", D.UNDERSTANDING, MUST, DIRECT, ref, F.UNDERSTANDING)

    records = _records(ev) or []
    cost_cols = [k for k in (records[0] if records else {})
                 if all(c.to_number(r.get(k)).ok for r in records)
                 and any(abs(c.to_number(r.get(k)).value - x) <= TOL_MONEY for r in records for x in costs)]
    cost_col = cost_cols[0] if len(cost_cols) == 1 else None

    def matched(cost: float) -> list[dict[str, Any]]:
        return [r for r in records if abs(c.to_number(r[cost_col]).value - cost) <= TOL_MONEY]

    lo, hi = min(costs), max(costs)
    if cost_col is None:
        found = c.indeterminate("no structured cost column")
    else:
        missing = [x for x in costs if not matched(x)]
        found = (c.not_met if missing else c.met)(costs, len(costs) - len(missing),
                                                    f"missing={missing}" if missing else None)
    s.fact(found, "fe12.extremes_found", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer identify a small number of extreme cost cases with values around "
           f"€{lo:,.0f}–€{hi:,.0f} (±€{_fmt(TOL_MONEY)})?")

    columns = list(records[0]) if records else []
    service_col = next((k for k in columns if _norm(k) in ("service", "servizio")), None) or next(
        (k for k in columns if any(_norm(r.get(k)) in {_norm(x) for x in services} for r in records)), None)
    if cost_col is None or service_col is None or found.status is not MET:
        located = c.indeterminate("services of the extreme rows not structured")
    else:
        located = c.compare_set(services, sorted({str(r[service_col]) for x in costs for r in matched(x)}))
    s.fact(located, "fe12.services", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer locate the extreme cases in {' and '.join(services)}?")
    s.review("fe12.ordinary_not_anomalous", D.INTERPRETATION, MUST,
             f"Does the answer avoid presenting ordinary high costs (≤ €{max_ordinary:,.2f}, the largest "
             f"non-extreme value) as anomalies?",
             ref, F.INTERPRETATION, focus=_lexical(_text(ev), _ANOMALY))

    extreme = [r for r in records if cost_col and c.to_number(r[cost_col]).value >= threshold]
    s.check(c.compare_exact(len(costs), len(extreme)) if cost_col else c.indeterminate("no structured cost column"),
            "fe12.four_cases", D.ANALYTICAL, SHOULD, DIRECT, ref)
    observed_ids = [c.to_number(r.get("record_id")) for r in extreme]
    s.check(c.compare_set([str(i) for i in ids], [str(int(p.value)) for p in observed_ids])
            if cost_col and extreme and all(p.ok and p.value == int(p.value) for p in observed_ids)
            else c.indeterminate("record IDs not structured"),
            "fe12.record_ids", D.ANALYTICAL, SHOULD, DIRECT, ref)


def _shares(pairs: c.Extracted) -> dict[str, float] | None:
    """Region → share as a fraction, only when the unit is unambiguous (fractions, or explicit %)."""
    if not pairs.ok:
        return None
    fractions = _numeric(pairs.value)
    if fractions is not None and all(0 <= v <= 1 for v in fractions.values()):
        return fractions
    return _numeric(pairs.value, percent=True) if all(
        isinstance(v, str) and v.strip().endswith("%") for _, v in pairs.value) else None


def _fe13(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    shares = _gt_map(t["semantic.negative_share_by_region"], "semantic.negative_share_by_region")
    regions = _gt_records(t["aggregates.region"], "aggregates.region", "region", "comments_with_text")
    texts = {str(r["region"]): r["comments_with_text"] for r in regions}
    comment = _gt_map(t["missingness.comment"], "missingness.comment")
    top = _top(shares)
    if top is None:
        raise GroundTruthError("semantic.negative_share_by_region: no unique top region")
    zero = sorted(k for k, v in shares.items() if v == 0)
    ref = (_response_ref(ev),)

    s.check(_usage_contribution(ev), "fe13.semantic_provider_call", D.HONESTY, MUST, DIRECT, (_usage_ref(ev),),
            F.HONESTY)
    s.check(_tool_use(ev, "sentiment_analysis"), "fe13.sentiment_tool", D.HONESTY, MUST, INFERRED,
            _tool_refs(ev), F.HONESTY)

    pairs = _labelled_values(ev, shares, value_columns=("share", "negative_share", "pct", "percent"))
    observed = _shares(pairs)
    highest = (_top_outcome(c.Extracted(tuple(observed.items())), top) if observed is not None
               else c.indeterminate("negative shares not structured (unit unknown)", top))
    if highest.status is NOT_MET:
        highest = _with_note(highest, "anomalous provider labelling, if any, is decided by review")
    s.fact(highest, "fe13.top_region", D.ANALYTICAL, MUST, ref, F.ANALYTICAL,
           f"Does the answer name {top} as the region with the highest share of negative comments?")

    # Regions with no negative comment (Central) may legitimately drop out of a negatives-only table.
    required = [k for k, v in shares.items() if v > 0]
    labels = [label for label, _ in pairs.value] if pairs.ok else None
    compared = c.check_subset(required, labels) if labels is not None else c.indeterminate("regions not structured")
    if compared.status is IND and _readable(ev):
        s.review("fe13.regions_compared", D.UNDERSTANDING, MUST,
                 f"Does the answer compare the negative share across regions ({', '.join(shares)})?",
                 ref, F.UNDERSTANDING)
    else:
        s.check(compared, "fe13.regions_compared", D.CHAINING, MUST, DIRECT, ref, F.CHAINING)

    s.review("fe13.nulls_not_neutral", D.HONESTY, MUST,
             f"Does the answer avoid counting the {_fmt(comment['missing'])} missing and {_fmt(comment['blank'])} "
             "blank comments as neutral (or as any sentiment)?",
             ref, F.HONESTY, focus=_lexical(_text(ev), _NULLS))
    s.review("fe13.small_region_qualified", D.INTERPRETATION, MUST,
             "If the answer presents " + ", ".join(f"{k} ({_fmt(texts.get(k, 0))} comments with text, 0% negative)"
                                                   for k in zero)
             + " as the best region, does it qualify the small sample? Answer NOT_APPLICABLE if it does not.",
             ref, F.INTERPRETATION)

    low, high = FE13_SOUTH_SHARE_RANGE
    band = (c.check_range(observed[_norm(top)], min_value=low, max_value=high)
            if observed is not None and _norm(top) in observed
            else c.indeterminate("share not structured"))
    s.check(band, "fe13.top_share_band", D.ANALYTICAL, SHOULD, DIRECT, ref)
    s.review("fe13.denominator_disclosed", D.HONESTY, SHOULD,
             "Does the answer disclose the denominator / counting convention of the shares (e.g. analysed "
             "comments only, missing and blank excluded)?", ref, F.HONESTY)
    charts = c.charts(ev.response)
    by_region = (c.indeterminate(f"extraction failed: {charts.error}") if not charts.ok
                 else (c.met if _chart_with_labels(ev, shares) is not None else c.not_met)(
                     "chart by region", len(charts.value)))
    s.check(by_region, "fe13.chart_by_region", D.UNDERSTANDING, SHOULD, DIRECT, ref)


def _category_counts(ev: ScenarioEvidence) -> c.Outcome:
    """Category → count from a chart, or a table with one label column and one integer column."""
    found = c.charts(ev.response)
    for chart in found.value if found.ok else ():
        pairs = c.chart_mapping(chart)
        values = _numeric(pairs.value) if pairs.ok else None
        if values and len(values) >= 2 and all(v >= 0 and v == int(v) for v in values.values()):
            return c.met("counts per category", {k: int(v) for k, v in values.items()})
    records = _records(ev)
    if records:
        text_cols = [k for k in records[0] if all(isinstance(r.get(k), str) for r in records)]
        int_cols = [k for k in records[0] if all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool)
                                                  for r in records)]
        if len(text_cols) == 1 and len(int_cols) == 1 and len(records) >= 2:
            return c.met("counts per category", {str(r[text_cols[0]]): r[int_cols[0]] for r in records})
    return c.indeterminate("category counts not structured", "counts per category")


def _fe14(s: _Sheet, t: dict[str, Any]) -> None:
    ev = s.ev
    taxonomy = t["semantic.taxonomy"]
    if not isinstance(taxonomy, list) or not taxonomy or not all(isinstance(x, str) for x in taxonomy):
        raise GroundTruthError("semantic.taxonomy: expected a non-empty list of theme names")
    comment = _gt_map(t["missingness.comment"], "missingness.comment")
    ref = (_response_ref(ev),)
    text = _text(ev)

    s.check(_tool_use(ev, "classify_match"), "fe14.classify_match", D.TOOL_SELECTION, MUST, INFERRED,
            _tool_refs(ev), F.TOOL_SELECTION)
    s.check(_usage_contribution(ev), "fe14.semantic_provider_call", D.TOOL_SELECTION, MUST, DIRECT,
            (_usage_ref(ev),), F.TOOL_SELECTION)
    s.fact(_category_counts(ev), "fe14.counts_per_category", D.UNDERSTANDING, MUST, ref, F.UNDERSTANDING,
           "Does the answer give a comment count for each proposed category?")
    s.review("fe14.proposal_framing", D.HONESTY, MUST,
             "Does the answer present the categories as its own proposal derived from the comments, not as a "
             "given or objective taxonomy?", ref, F.HONESTY, focus=_lexical(text, _PROPOSAL))
    s.review("fe14.null_handling", D.HONESTY, MUST,
             f"Are unclassified comments and the {_fmt(comment['missing'])} missing / {_fmt(comment['blank'])} "
             "blank comments kept distinct from the themes (not counted as a theme, not folded into 'altro')?",
             ref, F.HONESTY, focus=_lexical(text, _NULLS))
    s.review("fe14.taxonomy_coherence", D.INTERPRETATION, MUST,
             "Do the proposed categories group the comments into coherent themes rather than arbitrary or "
             "overlapping labels?", ref, F.INTERPRETATION)
    s.review("fe14.theme_coverage", D.INTERPRETATION, SHOULD,
             f"Do the categories semantically cover at least {len(taxonomy) - 1} of the {len(taxonomy)} designed "
             f"themes ({', '.join(taxonomy)}), with waiting time present and among the largest?",
             ref, F.INTERPRETATION)

    keyword_events, classify_events = _tool_events(ev, "keywords"), _tool_events(ev, "classify_match")
    code = _code(ev)
    if keyword_events and classify_events:
        first = (c.met if keyword_events[0].line_no < classify_events[0].line_no else c.not_met)(
            "keywords before classify_match", "runtime event order")
    elif code is not None and re.search(r"\bkeywords\s*\(", code):
        k, m = re.search(r"\bkeywords\s*\(", code).start(), re.search(r"\bclassify_match\s*\(", code)
        first = (c.met if m is None or k < m.start() else c.not_met)("keywords before classify_match",
                                                                      "generated code order")
    elif code is not None or ev.runtime is not None:
        first = c.not_met("keywords before classify_match", "no keywords call")
    else:
        first = c.indeterminate("no generated code and no runtime log", "keywords before classify_match")
    s.check(first, "fe14.keywords_first", D.TOOL_SELECTION, SHOULD, INFERRED, _tool_refs(ev))


CATALOGUE: dict[str, Callable[[_Sheet, dict[str, Any]], None]] = {
    "FE-01": _fe01, "FE-02": _fe02, "FE-03": _fe03, "FE-04": _fe04, "FE-05": _fe05, "FE-06": _fe06,
    "FE-07": _fe07, "FE-08": _fe08, "FE-09": _fe09, "FE-10": _fe10, "FE-11": _fe11, "FE-12": _fe12,
    "FE-13": _fe13, "FE-14": _fe14,
}

if tuple(CATALOGUE) != SCENARIO_ORDER or tuple(TRUTH_BINDINGS) != SCENARIO_ORDER:
    raise CatalogueError("catalogue does not cover the frozen scenario order")
