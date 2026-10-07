"""FE-01 … FE-14 technical grading contract (technical redesign delta §4, grading design §9.2).

    ScenarioEvidence + frozen policy + checks.py primitives → ScenarioCheckSet(checks)

The catalogue answers "which technical requirements belong to FE-xx" and
produces grading ingredients only. It decides no outcome and persists
nothing: that is the outcome builder's job.

Scope: only requirements decidable from execution state, structured fields,
runtime/tool events, Usage rows and artifacts, without interpreting
free-form prose. Semantic correctness, ground-truth values, method choice,
tool order, chart kind and wording are not graded here; semantic evaluation
happens outside automatic grading.

Scenario requirements (grading design §9.2), failure on reliable violation:

- FE-01, FE-09: a chart recorded in ``charts[]`` → ARTIFACT;
- FE-11: an image answer → RESPONSE CONTRACT; whose artifact decodes → ARTIFACT;
- FE-13: provider-backed semantic work (Usage extra rows) and
  ``sentiment_analysis`` activity → TOOL EXECUTION;
- FE-14: provider-backed classification (Usage extra rows) and
  ``classify_match`` activity → TOOL EXECUTION.

Other scenarios carry no scenario-specific check. Tool activity from
runtime events or generated code is INFERRED: an INFERRED negative never
fails a scenario by itself (outcome rules).

Only GRADEABLE scenarios are accepted. Output is deterministic.
"""

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from evaluation.datachat.grading import checks as c
from evaluation.datachat.grading.model import (
    CheckResult,
    CheckStatus,
    EvidenceClass,
    Gate,
    TechnicalFailure,
)
from evaluation.datachat.grading.reader import ParseStatus, ScenarioEvidence
from evaluation.datachat.model import Availability
from evaluation.datachat.scenarios import SCENARIO_ORDER

DIRECT, INFERRED = EvidenceClass.DIRECT, EvidenceClass.INFERRED
MET, NOT_MET = CheckStatus.MET, CheckStatus.NOT_MET
F = TechnicalFailure


class CatalogueError(Exception):
    """Programmer/configuration misuse: unknown scenario ID or non-gradeable evidence."""


# --- Output ---------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioCheckSet:
    """Grading ingredients of one scenario. Not an outcome."""

    scenario_id: str
    checks: tuple[CheckResult, ...]


def grade_scenario_checks(evidence: ScenarioEvidence) -> ScenarioCheckSet:
    """Run the technical FE contract of ``evidence.scenario_id`` on GRADEABLE evidence."""
    sid = evidence.scenario_id
    if sid not in CATALOGUE:
        raise CatalogueError(f"unknown scenario: {sid!r}")
    if evidence.gate is not Gate.GRADEABLE:
        raise CatalogueError(f"{sid}: gate {evidence.gate.value} is not gradeable")
    sheet = _Sheet(evidence)
    CATALOGUE[sid](sheet)
    return sheet.result()


class _Sheet:
    """Collects one scenario's checks in emission order."""

    def __init__(self, ev: ScenarioEvidence) -> None:
        self.ev = ev
        self._checks: list[CheckResult] = []
        self._ids: set[str] = set()

    def check(
        self,
        outcome: c.Outcome,
        check_id: str,
        evidence_class: EvidenceClass,
        refs: Iterable[str | None],
        failure: TechnicalFailure,
    ) -> CheckResult:
        """A mandatory technical check."""
        if check_id in self._ids:
            raise CatalogueError(f"duplicate check id: {check_id}")
        self._ids.add(check_id)
        result = c.make_check(
            outcome,
            check_id=check_id,
            evidence_class=evidence_class,
            failure=failure,
            evidence_refs=_refs(refs),
        )
        self._checks.append(result)
        return result

    def result(self) -> ScenarioCheckSet:
        return ScenarioCheckSet(self.ev.scenario_id, tuple(self._checks))


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


def _with_note(outcome: c.Outcome, note: str) -> c.Outcome:
    return replace(outcome, note=f"{outcome.note}; {note}" if outcome.note else note)


def _chart_present(s: _Sheet, check_id: str) -> None:
    """The question requests a chart: at least one entry in ``charts[]`` (structural validity is generic)."""
    ev = s.ev
    present = c.check_chart_count(ev.response)
    if present.status is NOT_MET:
        # INFERRED attempt is context only: the DIRECT empty charts[] is the ARTIFACT failure either way.
        attempt = _tool_use(ev, "chart")
        present = _with_note(present, f"chart attempted ({attempt.observed}) but charts[] is empty"
                             if attempt.status is MET else "no chart attempt in generated code or runtime events")
    s.check(present, check_id, DIRECT, (_response_ref(ev), *_tool_refs(ev)), F.ARTIFACT)


def _no_requirement(s: _Sheet) -> None:
    """No scenario-specific technical requirement (grading design §9.2)."""


def _fe01(s: _Sheet) -> None:
    _chart_present(s, "fe01.chart_present")


def _fe09(s: _Sheet) -> None:
    _chart_present(s, "fe09.chart_present")


def _fe11(s: _Sheet) -> None:
    ev = s.ev
    art = ev.artifacts
    ref = (_response_ref(ev), art.image_meta_ref if art else None)

    # Response kind (RESPONSE CONTRACT) and artifact validity (ARTIFACT) are separate checks.
    if not _parsed(ev):
        kind = c.indeterminate("response not parsed", "image")
    elif ev.response.type == "image":
        kind = c.met("image", "image")
    elif ev.response.type == "str" and ".png" in (ev.response.text or ""):
        kind = c.not_met("image", "str", "a .png server path instead of a delivered image")
    else:
        kind = c.not_met("image", ev.response.type, "no image answer")
    s.check(kind, "fe11.image_delivered", DIRECT, ref, F.RESPONSE_CONTRACT)

    if kind.status is not MET:
        valid = c.not_applicable("no image answer to validate")
    else:
        meta = art.image_meta if art else None
        status = meta.get("decode_status") if meta else None
        valid = (c.indeterminate("image metadata unavailable", "decode_status OK") if status is None
                 else c.compare_exact("OK", status))
    s.check(valid, "fe11.image_valid", DIRECT, ref, F.ARTIFACT)


def _semantic_capability(s: _Sheet, prefix: str, tool: str) -> None:
    """Provider-backed semantic work (DIRECT, Usage rows) and the capability's activity (INFERRED)."""
    ev = s.ev
    s.check(_usage_contribution(ev), f"{prefix}.semantic_provider_call", DIRECT,
            (_usage_ref(ev),), F.TOOL_EXECUTION)
    s.check(_tool_use(ev, tool), f"{prefix}.{tool}", INFERRED, _tool_refs(ev), F.TOOL_EXECUTION)


def _fe13(s: _Sheet) -> None:
    _semantic_capability(s, "fe13", "sentiment_analysis")


def _fe14(s: _Sheet) -> None:
    _semantic_capability(s, "fe14", "classify_match")


CATALOGUE: dict[str, Callable[[_Sheet], None]] = {
    "FE-01": _fe01, "FE-02": _no_requirement, "FE-03": _no_requirement, "FE-04": _no_requirement,
    "FE-05": _no_requirement, "FE-06": _no_requirement, "FE-07": _no_requirement, "FE-08": _no_requirement,
    "FE-09": _fe09, "FE-10": _no_requirement, "FE-11": _fe11, "FE-12": _no_requirement,
    "FE-13": _fe13, "FE-14": _fe14,
}

if tuple(CATALOGUE) != SCENARIO_ORDER:
    raise CatalogueError("catalogue does not cover the frozen scenario order")
