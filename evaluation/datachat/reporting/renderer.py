"""Markdown renderer of the operator report (Reporting v1 Slice 3).

``render_markdown(report)`` turns an ``OperatorRunReport`` into the text of
``<run>/reports/report.md``:

    OperatorRunReport → Markdown

Pure: no I/O, no clock, no environment. It trusts the typed operator model:
every status, technical outcome, gate, assessment, failure and attention
reason shown is a persisted or Slice 2 value; nothing is compared, ranked or
recomputed here. The Agent result is shown verbatim from the Slice 1 source
(text in a fenced block, tables and chart specs structurally). Links to
persisted artifacts are relative to ``reports/report.md`` so the run package
stays portable.
"""

import json
from typing import Any, Iterable

from evaluation.datachat.grading.model import CheckResult, ExecutionAssessment
from evaluation.datachat.model import Availability
from evaluation.datachat.reporting.operator_report import (
    AttentionReason,
    OperatorAgentResult,
    OperatorRunReport,
    ResultShape,
    ScenarioReport,
)

REPORT_DIR = "reports"
REPORT_NAME = "report.md"

# Run-relative path → path relative to reports/report.md.
_UP = "../"

# One-to-one labels of the Slice 2 attention reasons; listing order is the enum order, not a rank.
_ATTENTION_LABELS = {
    AttentionReason.FAIL_OUTCOME: "technical FAIL",
    AttentionReason.INDETERMINATE_OUTCOME: "technical INDETERMINATE",
    AttentionReason.NOT_GRADEABLE: "non-gradeable gate",
    AttentionReason.EXECUTION_NOT_OK: "execution assessment not OK",
    AttentionReason.TEARDOWN_NOT_VERIFIED: "teardown not verified",
}

_CHART_SERIES_KEYS = ("labels", "datasets")


# --- Primitives --------------------------------------------------------------------------


def _value(value: Any) -> str:
    """A persisted value as text: strings as-is, everything else as compact JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "value") and isinstance(value.value, str):  # enums
        return value.value
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def _cell(value: Any) -> str:
    return _value(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def _table(header: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    header = list(header)
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return lines + [""]


def _fence(text: str, info: str = "text") -> list[str]:
    """Verbatim block: the fence is longer than any backtick run inside ``text``."""
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{info}", text, fence, ""]


def _yes(value: bool | None) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _link(run_relative: str) -> str:
    return f"[`{run_relative}`]({_UP}{run_relative})"


def _field(label: str, value: Any) -> str:
    return f"- **{label}:** {_value(value)}"


# --- Run ---------------------------------------------------------------------------------


def _overview(r: OperatorRunReport) -> list[str]:
    out = ["## Run overview", "",
           _field("Run", f"`{r.run_id}`"),
           _field("Official", _yes(r.official)),
           _field("Comparable", _yes(r.comparable)),
           _field("Run status", r.detail.status)]
    if r.detail.failure is not None:
        f = r.detail.failure
        out.append(_field("Run failure", f"{f.code.value} — {f.summary}" + (f" (phase {f.phase.value})" if f.phase else "")))
    out.append("")
    if r.attention_scenarios:
        out.append(f"**Scenarios requiring operator attention:** {', '.join(r.attention_scenarios)}")
    else:
        out.append("**Scenarios requiring operator attention:** none")
    out.append("")
    out += ["### Technical outcome counts", "",
            "A TECHNICAL PASS says nothing about semantic correctness.", ""]
    out += _table(("Outcome", "Scenarios"), r.outcome_counts)
    out += ["### Technical failure counts", "",
            "A FAIL scenario counts once per established failure category.", ""]
    out += _table(("Failure", "Scenarios"), r.failure_counts)
    out += ["### Non-gradeable gate counts", "",
            "Gates are not outcomes and are not counted as FAIL.", ""]
    out += _table(("Gate", "Scenarios"), ((g, n) for g, n in r.gate_counts))
    return out


def _inventory(r: OperatorRunReport) -> list[str]:
    attention = {s.scenario_id: s.attention for s in r.scenarios}
    rows = []
    for e in r.inventory:
        a = attention[e.scenario_id]
        rows.append((f"[{e.scenario_id}](#{e.scenario_id.lower()})", e.gate,
                     e.label if e.outcome is not None else "no technical outcome",
                     ", ".join(_ATTENTION_LABELS[x] for x in a.reasons) if a.required else ""))
    return ["## Scenario summary", ""] + _table(("Scenario", "Gate", "Technical outcome", "Attention"), rows)


def _run_provenance(r: OperatorRunReport) -> list[str]:
    p, d = r.provenance, r.detail
    contract = dict(p.contract)
    out = ["## Technical provenance", "",
           f"- **Source `grading/results/run_grade.json` SHA-256:** `{p.run_grade_sha256}`",
           _field("Grading schema version", p.grading_schema_version),
           _field("Run manifest schema version", p.manifest_schema_version),
           _field("Run started", d.started_at),
           _field("Run ended", d.ended_at),
           _field("Scenario order", ", ".join(d.scenario_order)),
           "", "### Grader", ""]
    out += _table(("Field", "Value"), p.grader.items())
    out += ["### Maui", ""] + _table(("Field", "Value"), d.maui.items())
    out += ["### DataChat configuration", ""] + _table(("Field", "Value"), d.datachat.items())
    artifacts = contract.pop("artifacts", None)
    executions = contract.pop("scenario_execution_sha256", None)
    out += ["### Evaluation contract", ""] + _table(("Field", "Value"), contract.items())
    if isinstance(artifacts, dict):
        out += _table(("Contract artifact", "SHA-256"), artifacts.items())
    if isinstance(executions, dict):
        out += _table(("Scenario", "execution.json SHA-256"), executions.items())
    return out


# --- Agent result ------------------------------------------------------------------------


def _artifact_path(s: ScenarioReport, scenario_relative: str) -> str:
    return f"{s.provenance.dir}/{scenario_relative}"


def _data_table(rows: Any) -> list[str]:
    if not (isinstance(rows, list) and all(isinstance(x, dict) for x in rows)):
        return _fence(json.dumps(rows, ensure_ascii=False, indent=2), "json")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    if not columns:
        return ["_The returned table has no columns._", ""]
    return _table(columns, ([row.get(c) for c in columns] for row in rows))


def _chart(index: int, spec: dict[str, Any]) -> list[str]:
    out = [f"**Chart {index}**", ""]
    out += [_field(k, v) for k, v in spec.items() if k not in _CHART_SERIES_KEYS]
    out.append("")
    labels = spec.get("labels")
    datasets = spec.get("datasets")
    if isinstance(labels, list) and isinstance(datasets, list) and all(isinstance(d, dict) for d in datasets):
        series = [d.get("data") if isinstance(d.get("data"), list) else [] for d in datasets]
        header = ["label"] + [_value(d.get("label")) or f"dataset {i}" for i, d in enumerate(datasets, 1)]
        length = max([len(labels), *(len(x) for x in series)])
        rows = ([labels[i] if i < len(labels) else None, *(x[i] if i < len(x) else None for x in series)]
                for i in range(length))
        out += _table(header, rows)
        extra = [{k: v for k, v in d.items() if k not in ("label", "data")} for d in datasets]
        out += [_field(f"dataset {i} other fields", e) for i, e in enumerate(extra, 1) if e]
        if any(extra):
            out.append("")
    else:
        out += [_field("labels", labels), _field("datasets", datasets), ""]
    return out


def _image(s: ScenarioReport, a: OperatorAgentResult) -> list[str]:
    image = a.image
    if image is None or (image.availability is Availability.NOT_APPLICABLE and a.shape is not ResultShape.IMAGE):
        return []
    meta = image.metadata
    status = meta.decode_status if meta else None
    out = ["**Image artifact**", "", _field("Availability", image.availability)]
    if image.reason:
        out.append(_field("Reason", image.reason))
    if meta is not None:
        out.append(_field("Recorded decode status", status))
    out.append("")
    if image.file is not None and status == "OK":
        path = _artifact_path(s, image.file.path)
        out += [f"![DataChat image]({_UP}{path})", "", f"Persisted image: {_link(path)}", ""]
    elif image.availability is not Availability.NOT_APPLICABLE:
        out += ["No decodable persisted image is available.", ""]
    return out


def _export(s: ScenarioReport, a: OperatorAgentResult) -> list[str]:
    export = a.export
    if export is None or export.availability is Availability.NOT_APPLICABLE:
        return []
    out = ["**Export artifact** (full delivered export, separate from the inline preview)", "",
           _field("Availability", export.availability)]
    if export.reason:
        out.append(_field("Reason", export.reason))
    out += [_field(k, v) for k, v in export.locator.items()]
    if export.file is not None:
        out.append(f"- **File:** {_link(_artifact_path(s, export.file.path))}")
    return out + [""]


def _agent_result(s: ScenarioReport) -> list[str]:
    a = s.agent_result
    out = ["### Agent Result", ""]
    if a.shape is ResultShape.ABSENT:
        return out + ["_No Agent Result: no `/datachat` request was recorded for this scenario._", ""]
    src = a.source
    if a.shape is ResultShape.NO_RESPONSE:
        out.append("_No Agent Result was captured: no `/datachat` response was persisted._")
        out.append("")
        if src.transport_error is not None:
            out.append(_field("Transport error", f"{src.transport_error.type}: {src.transport_error.message}"))
        if src.availability is not None:
            out.append(_field("Evidence availability", src.availability))
        if src.reason:
            out.append(_field("Reason", src.reason))
        return out + [_field("Recorded attempts", src.attempts), ""]
    c = src.content
    if a.shape is ResultShape.TEXT:
        out += _fence(c.value)
    elif a.shape is ResultShape.TABLE:
        out += ["**Inline table (as returned)**", ""] + _data_table(c.value)
        out += [_field(k, v) for k, v in c.table_meta.items()]
        if c.more_rows_available is not None:
            out.append(_field("more_rows_available", _value(c.more_rows_available)))
        if c.table_meta or c.more_rows_available is not None:
            out.append("")
    elif a.shape is ResultShape.IMAGE:
        pass  # rendered from the persisted artifact below; the base64 value is never decoded here
    elif a.shape is ResultShape.STRUCTURED:
        out += ["**Structured value (as returned)**", ""] + _fence(json.dumps(c.value, ensure_ascii=False, indent=2), "json")
    else:  # ERROR_BODY / RAW: verbatim body, not interpreted
        label = "Error body" if a.shape is ResultShape.ERROR_BODY else f"Raw body ({c.parse_status.value})"
        out += [f"**{label}** — HTTP {src.http_status}, content type `{src.content_type}`", ""]
        out += _fence(src.body.decode("utf-8", errors="replace"))
    if c.note is not None:
        out += [_field("note", c.note), ""]
    if c.explanation is not None:
        out += ["**Explanation (as returned)**", ""] + _fence(_value(c.explanation))
    out += _image(s, a)
    out += _export(s, a)
    for i, spec in enumerate(a.charts, 1):
        out += _chart(i, spec)
    return out


# --- Grading -----------------------------------------------------------------------------


def _check_rows(checks: Iterable[CheckResult]) -> list[str]:
    return _table(("Check", "Evidence class", "Status", "Expected", "Observed", "Failure", "Note"),
                  ((c.check_id, c.evidence_class, c.status, c.expected, c.observed, c.failure, c.note)
                   for c in checks))


def _check_detail(c: CheckResult) -> list[str]:
    out = [f"- `{c.check_id}` — status **{c.status.value}**",
           f"  - evidence class {c.evidence_class.value}",
           f"  - expected: `{_value(c.expected)}`",
           f"  - observed: `{_value(c.observed)}`"]
    if c.failure is not None:
        out.append(f"  - failure category: {c.failure.value}")
    if c.note:
        out.append(f"  - note: {c.note}")
    if c.evidence_refs:
        out.append(f"  - evidence: {', '.join(_link(x) for x in c.evidence_refs)}")
    return out


def _grading(s: ScenarioReport) -> list[str]:
    out = ["### Grading", "", _field("Gate", s.gate)]
    if s.has_outcome:
        out.append(f"- **Technical outcome:** **{s.label}**")
    else:
        out.append(f"- **Technical outcome:** none (gate {s.gate.value})")
    if s.execution_assessment is not None:
        reason = s.detail.execution_assessment_reason
        text = s.execution_assessment.value
        if s.execution_assessment is not ExecutionAssessment.OK and reason is not None:
            text += f" ({reason.value})"
        out.append(_field("Execution assessment", text))
    if s.check_status_counts:
        out.append(_field("Checks", ", ".join(f"{n} {st.value}" for st, n in s.check_status_counts)))
    if s.attention.required:
        out.append(_field("Requires operator attention",
                          "yes — " + "; ".join(_ATTENTION_LABELS[x] for x in s.attention.reasons)))
    else:
        out.append(_field("Requires operator attention", "no"))
    out.append("")

    if s.execution_issue is not None:
        i = s.execution_issue
        out += ["#### Execution issue", "",
                f"Execution assessment **{i.assessment.value}**, reason `{i.reason.value}`.", ""]
    if s.gate_execution_failure is not None:
        f = s.gate_execution_failure
        out += ["#### Execution context", "",
                _field("Execution status", s.detail.execution_status),
                _field("Failure", f"{f.code.value} — {f.summary}"),
                _field("Phase", f.phase)]
        if f.detail:
            out.append(_field("Detail", f.detail))
        out.append("")
    elif not s.has_outcome and s.detail.execution_status is not None:
        out += [_field("Execution status", s.detail.execution_status), ""]
    if s.failures:
        out += ["#### Failures", ""]
        for f in s.failures:
            out += [f"**{f.category.value}**", ""]
            if f.checks_carrying_category:
                out.append("Checks carrying this failure category: "
                           + ", ".join(f"`{c.check_id}`" for c in f.checks_carrying_category))
            else:
                out.append("No check carries this failure category (see the execution assessment).")
            out.append("")
    if s.attention_checks:
        out += ["#### Checks requiring attention (NOT_MET / INDETERMINATE)", ""]
        for c in s.attention_checks:
            out += _check_detail(c)
        out.append("")
    if s.detail.checks:
        out += ["#### All checks", ""] + _check_rows(s.detail.checks)
    return out


# --- Scenario ----------------------------------------------------------------------------


def _question(s: ScenarioReport) -> list[str]:
    out = ["### Question", ""]
    if s.question is None:
        return out + ["_No question recorded: the scenario package was never written._", ""]
    if not s.question_sent:
        out += ["_Not sent to DataChat._", ""]
    return out + _fence(s.question)


def _scenario_provenance(s: ScenarioReport) -> list[str]:
    p = s.provenance
    out = ["#### Scenario provenance", "",
           _field("Scenario directory", _link(p.dir)),
           _field("execution.json SHA-256", p.execution_sha256 and f"`{p.execution_sha256}`"),
           _field("Question SHA-256", p.question_sha256 and f"`{p.question_sha256}`"),
           _field("Request IDs", ", ".join(f"`{x}`" for x in p.request_ids)),
           _field("Teardown verified", _yes(p.teardown_verified))]
    src = s.agent_result.source
    if src is not None and src.exchange is not None:
        x = src.exchange
        out += [_field("/datachat HTTP status", src.http_status),
                _field("/datachat content type", src.content_type),
                _field("/datachat attempts", src.attempts),
                _field("/datachat duration ms", x.duration_ms),
                _field("/datachat sent / completed", f"{x.sent_at} / {x.completed_at}")]
        if src.content is not None and src.content.log_id is not None:
            out.append(_field("log_id", src.content.log_id))
        if src.body_file:
            out.append(_field("Response body", _link(_artifact_path(s, src.body_file))))
    if p.evidence_refs:
        out.append(_field("Evidence refs (checks)", ", ".join(_link(x) for x in p.evidence_refs)))
    out.append("")
    if p.evidence:
        rows = ((e.source, e.availability, e.reason,
                 "<br>".join(_link(_artifact_path(s, f.path)) for f in e.files)) for e in p.evidence)
        # Links are already Markdown; build the table without escaping the file column.
        out += ["| Evidence source | Availability | Reason | Files |", "|---|---|---|---|"]
        out += [f"| {_cell(a)} | {_cell(b)} | {_cell(c)} | {d} |" for a, b, c, d in rows]
        out.append("")
    return out


def _scenario(s: ScenarioReport) -> list[str]:
    out = [f"## {s.scenario_id}", ""]
    out += _question(s)
    out += _agent_result(s)
    out += _grading(s)
    out += _scenario_provenance(s)
    return out


# --- Entry point -------------------------------------------------------------------------


def render_markdown(report: OperatorRunReport) -> str:
    """The Markdown report for one run. Pure and deterministic for a given model."""
    lines = ["# DataChat Functional Evaluation Report", "",
             f"Derived from `grading/results/run_grade.json` (SHA-256 `{report.provenance.run_grade_sha256}`). "
             "Regenerable; not a source of truth. There is no run-level outcome or score.", ""]
    lines += _overview(report)
    lines += _inventory(report)
    for s in report.scenarios:
        lines += _scenario(s)
    lines += _run_provenance(report)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
