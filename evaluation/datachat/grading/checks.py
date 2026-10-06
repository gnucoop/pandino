"""Deterministic, scenario-agnostic grading primitives.

Pure functions only: no files, no run loading, no live systems, no scenario
knowledge. Each comparison returns an ``Outcome``; the caller (which owns the
policy: check ID, dimension, requirement level, evidence class, failure
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
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    Dimension,
    EvidenceClass,
    FailureCategory,
    RequirementLevel,
)
from evaluation.datachat.grading.reader import NormalizedResponse, ParseStatus

# Absorbs binary floating-point noise at an inclusive tolerance boundary
# (e.g. |0.15 - 0.10| computing to 0.05000000000000002).
_FLOAT_EPS = 1e-9


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
    dimension: Dimension,
    requirement_level: RequirementLevel,
    evidence_class: EvidenceClass,
    evidence_refs: Sequence[str] = (),
    failure_category: FailureCategory | None = None,
) -> CheckResult:
    """Attach caller-owned policy. ``failure_category`` is kept only on NOT_MET."""
    return CheckResult(
        check_id=check_id,
        dimension=dimension,
        requirement_level=requirement_level,
        evidence_class=evidence_class,
        status=outcome.status,
        expected=outcome.expected,
        observed=outcome.observed,
        evidence_refs=tuple(evidence_refs),
        failure_category=failure_category if outcome.status is CheckStatus.NOT_MET else None,
        note=outcome.note,
    )


# --- Numbers -----------------------------------------------------------------


@dataclass(frozen=True)
class ParsedNumber:
    """``value`` as written (``"45%"`` → 45.0, ``is_percent=True``), or ``error``."""

    value: float | None
    is_percent: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


_CURRENCY = "€"
_MINUS = "-−"  # ASCII hyphen-minus and U+2212
_GROUPED_DOT = re.compile(r"[1-9]\d{0,2}(\.\d{3})+")
_GROUPED_COMMA = re.compile(r"[1-9]\d{0,2}(,\d{3})+")
_DIGITS = re.compile(r"\d+")
_AMBIGUOUS = re.compile(r"[1-9]\d{0,2}[.,]\d{3}")


def _parse_body(body: str) -> float | str:
    """Unsigned numeric body → float, or an error string.

    Accepted conventions:

    - plain digits: ``1234``;
    - a single decimal separator, dot or comma: ``12.5``, ``12,5``, ``0.579``;
    - both separators: the last one is the decimal separator and the other
      must form valid thousands groups: ``1,234.50``, ``1.234,50``;
    - a single separator type used only as thousands groups, repeated:
      ``1.234.567``.

    ``1.234`` / ``1,234`` (one separator followed by exactly three digits,
    leading group 1–999) are rejected as ambiguous: the value could be a
    decimal or a thousands-grouped integer.
    """
    if _DIGITS.fullmatch(body):
        return float(body)
    has_dot, has_comma = "." in body, "," in body
    if has_dot and has_comma:
        decimal = "." if body.rfind(".") > body.rfind(",") else ","
        group = "," if decimal == "." else "."
        integer, _, fraction = body.rpartition(decimal)
        grouped = _GROUPED_DOT if group == "." else _GROUPED_COMMA
        if not _DIGITS.fullmatch(fraction) or not (grouped.fullmatch(integer) or _DIGITS.fullmatch(integer)):
            return "malformed number"
        if group in integer and not grouped.fullmatch(integer):
            return "malformed thousands grouping"
        return float(integer.replace(group, "") + "." + fraction)
    sep = "." if has_dot else ","
    if body.count(sep) > 1:
        grouped = _GROUPED_DOT if sep == "." else _GROUPED_COMMA
        return float(body.replace(sep, "")) if grouped.fullmatch(body) else "malformed number"
    if _AMBIGUOUS.fullmatch(body):
        return "ambiguous separator"
    integer, _, fraction = body.partition(sep)
    if not _DIGITS.fullmatch(integer) or not _DIGITS.fullmatch(fraction):
        return "malformed number"
    return float(f"{integer}.{fraction}")


def parse_number(value: Any) -> ParsedNumber:
    """Parse one unambiguously numeric value. Never extracts numbers from prose.

    Accepts ``int``/``float`` (finite) and strings of a single number with
    optional surrounding whitespace, leading sign, ``€`` symbol (before or
    after, spaces allowed) and trailing ``%``. ``bool`` is rejected.
    """
    if isinstance(value, bool):
        return ParsedNumber(None, error="boolean is not a number")
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ParsedNumber(None, error="non-finite number")
        return ParsedNumber(float(value))
    if not isinstance(value, str):
        return ParsedNumber(None, error=f"not a number: {type(value).__name__}")
    text = value.strip()
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].rstrip()
    negative = False
    if text and text[0] in _MINUS + "+":
        negative = text[0] in _MINUS
        text = text[1:].lstrip()
    if text.startswith(_CURRENCY):
        text = text[1:].lstrip()
    elif text.endswith(_CURRENCY):
        text = text[:-1].rstrip()
    if not negative and text and text[0] in _MINUS:  # "€ -12"
        negative, text = True, text[1:].lstrip()
    if is_percent and _CURRENCY in value:
        return ParsedNumber(None, error="both percent and currency")
    if not text:
        return ParsedNumber(None, error="empty")
    parsed = _parse_body(text)
    if isinstance(parsed, str):
        return ParsedNumber(None, error=parsed)
    return ParsedNumber(-parsed if negative else parsed, is_percent=is_percent)


def to_number(value: Any, *, percent: bool = False) -> ParsedNumber:
    """Parse ``value`` for comparison against a number in one declared unit.

    - ``percent=False``: percent notation is rejected, so ``"45%"`` is never
      compared with ``45`` or ``0.45`` by accident;
    - ``percent=True``: the expected value is a fraction; ``"45%"`` becomes
      ``0.45`` and plain numbers are taken as already being fractions.
    """
    parsed = parse_number(value)
    if not parsed.ok:
        return parsed
    if parsed.is_percent:
        if not percent:
            return ParsedNumber(None, error="unexpected percent notation")
        return ParsedNumber(parsed.value / 100.0)
    return parsed


def _within(diff: float, tolerance: float) -> bool:
    return diff <= tolerance or math.isclose(diff, tolerance, abs_tol=_FLOAT_EPS)


def compare_number(expected: float, observed: Any, *, tolerance: float = 0.0, percent: bool = False) -> Outcome:
    """MET when ``abs(observed - expected) <= tolerance`` (inclusive, absolute)."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    parsed = to_number(observed, percent=percent)
    if not parsed.ok:
        return indeterminate(f"observed not parseable: {parsed.error}", expected, _small(observed))
    diff = abs(parsed.value - expected)
    note = f"|diff|={diff:.6g} tolerance={tolerance:g}"
    return (met if _within(diff, tolerance) else not_met)(expected, parsed.value, note)


_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def check_threshold(observed: Any, op: str, threshold: float, *, absolute: bool = False) -> Outcome:
    """``observed <op> threshold`` with ``op`` in ``<, <=, >, >=``; the caller owns the threshold."""
    if op not in _OPS:
        raise ValueError(f"unsupported operator: {op!r}")
    parsed = to_number(observed)
    if not parsed.ok:
        return indeterminate(f"observed not parseable: {parsed.error}", f"{op} {threshold:g}", _small(observed))
    value = abs(parsed.value) if absolute else parsed.value
    expected = f"{'|x| ' if absolute else ''}{op} {threshold:g}"
    return (met if _OPS[op](value, threshold) else not_met)(expected, parsed.value)


def check_range(
    observed: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    include_min: bool = True,
    include_max: bool = True,
    absolute: bool = False,
) -> Outcome:
    """``observed`` (or ``|observed|``) within explicit inclusive/exclusive bounds."""
    if min_value is None and max_value is None:
        raise ValueError("a range needs at least one bound")
    lower = "" if min_value is None else f"{'[' if include_min else '('}{min_value:g}"
    upper = "" if max_value is None else f"{max_value:g}{']' if include_max else ')'}"
    expected = f"{'|x| in ' if absolute else ''}{lower or '(-inf'}, {upper or '+inf)'}"
    parsed = to_number(observed)
    if not parsed.ok:
        return indeterminate(f"observed not parseable: {parsed.error}", expected, _small(observed))
    value = abs(parsed.value) if absolute else parsed.value
    ok = True
    if min_value is not None:
        ok &= value >= min_value if include_min else value > min_value
    if max_value is not None:
        ok &= value <= max_value if include_max else value < max_value
    return (met if ok else not_met)(expected, parsed.value)


def check_sign(observed: Any, sign: str) -> Outcome:
    """``sign`` in ``positive`` (> 0), ``negative`` (< 0), ``zero`` (== 0)."""
    tests = {"positive": lambda v: v > 0, "negative": lambda v: v < 0, "zero": lambda v: v == 0}
    if sign not in tests:
        raise ValueError(f"unsupported sign: {sign!r}")
    parsed = to_number(observed)
    if not parsed.ok:
        return indeterminate(f"observed not parseable: {parsed.error}", sign, _small(observed))
    return (met if tests[sign](parsed.value) else not_met)(sign, parsed.value)


def compare_order(a: Any, op: str, b: Any) -> Outcome:
    """Mathematical relation between two observed values: ``a <op> b``."""
    if op not in _OPS:
        raise ValueError(f"unsupported operator: {op!r}")
    pa, pb = to_number(a), to_number(b)
    if not (pa.ok and pb.ok):
        return indeterminate("operand not parseable", f"a {op} b", [_small(a), _small(b)])
    return (met if _OPS[op](pa.value, pb.value) else not_met)(f"a {op} b", [pa.value, pb.value])


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


def _labels(values: Iterable[Any], normalize: bool) -> tuple[list[Any], list[Any]]:
    """(normalized labels, duplicates after normalization)."""
    out, seen, dups = [], set(), []
    for value in values:
        label = normalize_label(value) if normalize else value
        if label is None:
            dups.append(value)  # unusable label is treated like an unsafe structure
            continue
        if label in seen:
            dups.append(value)
        seen.add(label)
        out.append(label)
    return out, dups


def compare_set(expected: Iterable[Any], observed: Iterable[Any] | None, *, normalize: bool = True) -> Outcome:
    """Unordered equality of label sets. Duplicate observed labels → INDETERMINATE."""
    if observed is None:
        return indeterminate("observed labels not found")
    exp, _ = _labels(expected, normalize)
    obs, dups = _labels(observed, normalize)
    if dups:
        return indeterminate("duplicate or invalid observed labels", sorted(map(str, exp)), [str(d) for d in dups])
    missing = sorted(set(exp) - set(obs))
    extra = sorted(set(obs) - set(exp))
    note = f"missing={missing} unexpected={extra}" if missing or extra else None
    return (met if not missing and not extra else not_met)(sorted(exp), sorted(obs), note)


def check_subset(required: Iterable[Any], observed: Iterable[Any] | None, *, normalize: bool = True) -> Outcome:
    """Every required label is present in ``observed`` (extras allowed)."""
    if observed is None:
        return indeterminate("observed labels not found")
    req, _ = _labels(required, normalize)
    obs, dups = _labels(observed, normalize)
    if dups:
        return indeterminate("duplicate or invalid observed labels", sorted(map(str, req)), [str(d) for d in dups])
    missing = sorted(set(req) - set(obs))
    return (met if not missing else not_met)(sorted(req), sorted(obs), f"missing={missing}" if missing else None)


def compare_sequence(expected: Sequence[Any], observed: Sequence[Any] | None, *, normalize: bool = False) -> Outcome:
    """Ordered equality, element by element."""
    if observed is None:
        return indeterminate("observed sequence not found")
    exp = [normalize_label(v) if normalize else v for v in expected]
    obs = [normalize_label(v) if normalize else v for v in observed]
    if normalize and None in obs:
        return indeterminate("observed element not normalizable", list(map(_small, expected)), list(map(_small, observed)))
    return (met if exp == obs else not_met)(list(map(_small, expected)), list(map(_small, observed)))


# --- Label → value mappings --------------------------------------------------

Pairs = Sequence[tuple[Any, Any]]


def compare_mapping(
    expected: Mapping[str, float],
    observed: Pairs | Mapping[Any, Any] | None,
    *,
    tolerance: float | None = None,
    percent: bool = False,
    allow_extra: bool = False,
) -> Outcome:
    """Compare label → value results, ignoring order, with normalized labels.

    ``tolerance=None`` compares values exactly as numbers; otherwise
    ``abs(observed - expected) <= tolerance``.

    - any reliably wrong value, missing label or (unless ``allow_extra``)
      unexpected label → NOT_MET;
    - otherwise, duplicate labels after normalization or unparseable values
      → INDETERMINATE (never silently merged or skipped).
    """
    if observed is None:
        return indeterminate("observed mapping not found", dict(expected))
    pairs = list(observed.items()) if isinstance(observed, Mapping) else list(observed)
    exp = {}
    for label, value in expected.items():
        key = normalize_label(label)
        if key is None or key in exp:
            raise ValueError(f"invalid or duplicate expected label: {label!r}")
        exp[key] = value
    obs: dict[str, Any] = {}
    dups: list[str] = []
    for label, value in pairs:
        key = normalize_label(label)
        if key is None or key in obs:
            dups.append(str(label))
            continue
        obs[key] = value
    tol = 0.0 if tolerance is None else tolerance
    wrong, unparsed = [], []
    shown: dict[str, Any] = {}
    for key, value in exp.items():
        if key not in obs:
            continue
        parsed = to_number(obs[key], percent=percent)
        if not parsed.ok:
            unparsed.append(key)
            shown[key] = _small(obs[key])
            continue
        shown[key] = parsed.value
        if not _within(abs(parsed.value - value), tol):
            wrong.append(key)
    missing = sorted(set(exp) - set(obs))
    extra = sorted(set(obs) - set(exp))
    for key in extra:
        shown[key] = _small(obs[key])
    parts = [f"{name}={items}" for name, items in (
        ("wrong", wrong), ("missing", missing), ("unexpected", extra if not allow_extra else []),
        ("duplicate", dups), ("unparseable", unparsed),
    ) if items]
    note = "; ".join(parts) or None
    expected_view = dict(exp)
    if wrong or missing or (extra and not allow_extra):
        return not_met(expected_view, shown, note)
    if dups or unparsed:
        return indeterminate(note, expected_view, shown)
    return met(expected_view, shown, note)


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


def table_rows(response: NormalizedResponse | None, *, allow_truncated: bool = False) -> Extracted:
    """The answer table rows. A truncated preview is refused unless allowed."""
    error = _response_error(response)
    if error:
        return Extracted(error=error)
    if response.table_rows is None:
        return Extracted(error="answer is not a table")
    if response.table_meta.get("truncated") and not allow_truncated:
        return Extracted(error="table preview is truncated")
    return Extracted(response.table_rows)


def table_column(response: NormalizedResponse | None, column: str, *, allow_truncated: bool = False) -> Extracted:
    """Values of one named column, in row order."""
    rows = table_rows(response, allow_truncated=allow_truncated)
    if not rows.ok:
        return rows
    values = []
    for row in rows.value:
        if not isinstance(row, dict) or column not in row:
            return Extracted(error=f"row without column {column!r}")
        values.append(row[column])
    return Extracted(tuple(values))


def table_mapping(
    response: NormalizedResponse | None, label_column: str, value_column: str, *, allow_truncated: bool = False
) -> Extracted:
    """``(label, value)`` pairs from two named columns. Duplicates are left for the comparison to detect."""
    labels = table_column(response, label_column, allow_truncated=allow_truncated)
    if not labels.ok:
        return labels
    values = table_column(response, value_column, allow_truncated=allow_truncated)
    if not values.ok:
        return values
    return Extracted(tuple(zip(labels.value, values.value)))


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


def chart_type(chart: Any) -> Extracted:
    if not isinstance(chart, dict) or not isinstance(chart.get("type"), str):
        return Extracted(error="chart without a type")
    return Extracted(chart["type"])


def check_chart_type(chart: Any, allowed: Iterable[str]) -> Outcome:
    """Chart ``type`` is one of ``allowed`` (exact, normalized)."""
    allowed_set = sorted({normalize_label(a) for a in allowed})
    found = chart_type(chart)
    if not found.ok:
        return indeterminate(found.error, allowed_set)
    kind = normalize_label(found.value)
    return (met if kind in allowed_set else not_met)(allowed_set, found.value)


def chart_mapping(chart: Any, *, dataset_index: int = 0) -> Extracted:
    """``(label, value)`` pairs from ``labels`` and one dataset's ``data``.

    Only when structurally safe: a dict chart, list labels, the dataset
    present, and ``data`` the same length as ``labels``.
    """
    if not isinstance(chart, dict):
        return Extracted(error="chart is not an object")
    labels = chart.get("labels")
    datasets = chart.get("datasets")
    if not isinstance(labels, list) or not isinstance(datasets, list):
        return Extracted(error="chart without labels/datasets")
    if not 0 <= dataset_index < len(datasets) or not isinstance(datasets[dataset_index], dict):
        return Extracted(error=f"chart has no dataset {dataset_index}")
    data = datasets[dataset_index].get("data")
    if not isinstance(data, list) or len(data) != len(labels):
        return Extracted(error="chart labels and data differ in length")
    return Extracted(tuple(zip(labels, data)))


def from_extracted(extracted: Extracted, compare, *args, **kwargs) -> Outcome:
    """Run ``compare(*args, extracted.value, **kwargs)``, or INDETERMINATE if extraction failed."""
    if not extracted.ok:
        return indeterminate(f"extraction failed: {extracted.error}")
    return compare(*args, extracted.value, **kwargs)


# --- Helpers -----------------------------------------------------------------


def _small(value: Any) -> Any:
    """A JSON-safe scalar view for ``observed``/``expected``; long strings are cut."""
    if isinstance(value, str):
        return value if len(value) <= 120 else value[:117] + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return type(value).__name__
