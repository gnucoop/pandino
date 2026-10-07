"""Deterministic, scenario-agnostic grading primitives.

Pure functions only: no files, no run loading, no live systems, no scenario
knowledge. Each comparison returns an ``Outcome``; the caller (which owns the
policy: check ID, evidence class, failure
category) turns it into a ``CheckResult`` with ``make_check``.

Status discipline:

- ``MET`` / ``NOT_MET`` only when the observed value was found and parsed
  reliably;
- ``INDETERMINATE`` when a value is absent, unparseable or ambiguous, so a
  parser limitation never becomes a model failure;
- ``NOT_APPLICABLE`` only when the caller says so (``not_applicable``).

Exceptions are reserved for programmer misuse (invalid arguments).
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    EvidenceClass,
    TechnicalFailure,
)
from evaluation.datachat.grading.reader import NormalizedResponse, ParseStatus


# --- Outcome and CheckResult -------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """Result of one deterministic comparison, before policy is attached."""

    status: CheckStatus
    expected: Any = None
    observed: Any = None
    note: str | None = None


def met(expected: Any = None, observed: Any = None, note: str | None = None) -> Outcome:
    return Outcome(CheckStatus.MET, expected, observed, note)


def not_met(expected: Any = None, observed: Any = None, note: str | None = None) -> Outcome:
    return Outcome(CheckStatus.NOT_MET, expected, observed, note)


def indeterminate(note: str, expected: Any = None, observed: Any = None) -> Outcome:
    return Outcome(CheckStatus.INDETERMINATE, expected, observed, note)


def not_applicable(note: str) -> Outcome:
    return Outcome(CheckStatus.NOT_APPLICABLE, note=note)


def make_check(
    outcome: Outcome,
    *,
    check_id: str,
    evidence_class: EvidenceClass,
    failure: TechnicalFailure,
    evidence_refs: Sequence[str] = (),
) -> CheckResult:
    """Attach caller-owned policy. ``failure`` is kept only on NOT_MET."""
    return CheckResult(
        check_id=check_id,
        evidence_class=evidence_class,
        status=outcome.status,
        expected=outcome.expected,
        observed=outcome.observed,
        evidence_refs=tuple(evidence_refs),
        failure=failure if outcome.status is CheckStatus.NOT_MET else None,
        note=outcome.note,
    )


# --- Exact values and labels -------------------------------------------------


def normalize_label(value: Any) -> str | None:
    """Strip, collapse internal whitespace, casefold. No synonyms or fuzzy matching."""
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.split())
    return collapsed.casefold() if collapsed else None


def compare_exact(expected: Any, observed: Any, *, normalize: bool = False) -> Outcome:
    """Exact equality. ``normalize=True`` applies ``normalize_label`` to both strings.

    ``observed is None`` means not found → INDETERMINATE.
    """
    if observed is None:
        return indeterminate("observed value not found", _small(expected))
    if isinstance(expected, bool) != isinstance(observed, bool):
        return not_met(_small(expected), _small(observed), "type differs")
    a, b = expected, observed
    if normalize:
        a, b = normalize_label(expected), normalize_label(observed)
        if a is None or b is None:
            return indeterminate("label not normalizable", _small(expected), _small(observed))
    return (met if a == b else not_met)(_small(expected), _small(observed))


# --- Structured response extraction ------------------------------------------


@dataclass(frozen=True)
class Extracted:
    """A structurally extracted value, or the reason it could not be extracted."""

    value: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _response_error(response: NormalizedResponse | None) -> str | None:
    if response is None:
        return "no response"
    if response.parse_status is not ParseStatus.OK:
        return f"response parse status {response.parse_status.value}"
    return None


def charts(response: NormalizedResponse | None) -> Extracted:
    """Backend-recorded charts. An OK response without charts yields an empty tuple."""
    error = _response_error(response)
    if error:
        return Extracted(error=error)
    return Extracted(response.charts)


def check_chart_count(response: NormalizedResponse | None, *, min_count: int = 1) -> Outcome:
    """At least ``min_count`` recorded charts. Absence in a parsed response is a reliable NOT_MET."""
    found = charts(response)
    if not found.ok:
        return indeterminate(found.error, f">= {min_count}")
    count = len(found.value)
    return (met if count >= min_count else not_met)(f">= {min_count}", count)


# --- Structural validity (grading design §9.1) --------------------------------
# Shape only, per the CURRENT DataChat contract: values are never compared to anything.

CHART_KINDS = ("bar", "line", "pie", "doughnut", "scatter")
_TABLE_COUNTS = ("result_rows", "result_columns", "preview_rows", "preview_columns")


def _chart_value(value: Any) -> bool:
    """A finite number, or null (a recorded gap)."""
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))


def _coordinate(value: Any) -> bool:
    return value is not None and _chart_value(value)


def chart_structure_error(chart: Any) -> str | None:
    """Why one ``charts[]`` entry is not a structurally usable chart spec; None when it is."""
    if not isinstance(chart, dict):
        return "entry is not an object"
    kind = chart.get("type")
    if kind not in CHART_KINDS:
        return f"unsupported type {_small(kind)!r}"
    datasets = chart.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return "datasets missing or empty"
    if not all(isinstance(d, dict) and isinstance(d.get("data"), list) for d in datasets):
        return "dataset without a data list"
    if kind == "scatter":
        for d in datasets:
            if not all(isinstance(p, dict) and _coordinate(p.get("x")) and _coordinate(p.get("y")) for p in d["data"]):
                return "scatter point without numeric x/y"
        return None
    labels = chart.get("labels")
    if not isinstance(labels, list):
        return "labels missing"
    for i, d in enumerate(datasets, 1):
        if len(d["data"]) != len(labels):
            return f"dataset {i} has {len(d['data'])} values for {len(labels)} labels"
        if not all(_chart_value(v) for v in d["data"]):
            return f"dataset {i} has a non-numeric value"
    return None


def dataframe_structure_error(response: Any) -> str | None:
    """Why a ``dataframe`` response's records and declared metadata disagree; None when consistent.

    Mirrors the CURRENT table response: ``value`` is a list of records,
    ``preview_*`` count them, ``result_*`` bound the preview and
    ``truncated`` says the preview hides rows or columns. Absent metadata is
    not checked.
    """
    if not isinstance(response, dict):
        return "response is not an object"
    rows = response.get("value")
    if not isinstance(rows, list):
        return "value is not a list of records"
    if not all(isinstance(r, dict) for r in rows):
        return "a row is not a record"
    for key in _TABLE_COUNTS:
        n = response.get(key)
        if key in response and (isinstance(n, bool) or not isinstance(n, int) or n < 0):
            return f"{key} is not a non-negative integer"
    if "truncated" in response and not isinstance(response["truncated"], bool):
        return "truncated is not a boolean"
    columns = len(set().union(*rows)) if rows else 0
    pr, pc = response.get("preview_rows"), response.get("preview_columns")
    rr, rc = response.get("result_rows"), response.get("result_columns")
    if pr is not None and pr != len(rows):
        return f"preview_rows {pr} but {len(rows)} records"
    if pc is not None and pc != columns:
        return f"preview_columns {pc} but {columns} columns in records"
    if None not in (pr, rr) and pr > rr:
        return "preview_rows exceeds result_rows"
    if None not in (pc, rc) and pc > rc:
        return "preview_columns exceeds result_columns"
    truncated = response.get("truncated")
    if truncated is not None and None not in (pr, pc, rr, rc):
        if truncated != (pr < rr or (pr > 0 and pc < rc)):
            return "truncated contradicts the row/column counts"
    return None


# --- Helpers -----------------------------------------------------------------


def _small(value: Any) -> Any:
    """A JSON-safe scalar view for ``observed``/``expected``; long strings are cut."""
    if isinstance(value, str):
        return value if len(value) <= 120 else value[:117] + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return type(value).__name__
