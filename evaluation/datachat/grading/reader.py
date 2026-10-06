"""Read-only loader of completed execution packages for grading.

``load_run_for_grading(run_dir)`` turns a finished run directory into a
normalized, integrity-checked, scenario-agnostic view:

    run package → integrity verification → RunForGrading

The reader never writes, repairs, retries or queries anything live (Maui,
database, providers). It decides no verdict: parse problems are represented
faithfully (``ParseStatus``) for later checks.

Two failure kinds are kept apart:

- recorded unavailability (an evidence source the execution manifest marks
  ``UNAVAILABLE``/``EXPECTED_ABSENT``/``NOT_APPLICABLE``) is a valid state and is
  exposed as such;
- integrity corruption (a referenced file missing, a hash or size mismatch, a
  path escaping the run, an inconsistent manifest, an unfinished run, a
  changed evaluation contract) raises ``GradingInputError``. It is never a
  scenario verdict.

Epistemics: runtime ``tool_call_*`` events and generated agent code are
evidence about tool *use* only. Exact tool outputs and agent observations are
not persisted by the execution layer and are not reconstructed here.
"""

import hashlib
import json
import re
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.datachat import contract
from evaluation.datachat.evidence import RunLayout, read_json, safe_child, sha256_file
from evaluation.datachat.grading.model import Gate
from evaluation.datachat.model import (
    Availability,
    EvaluationRun,
    HttpExchange,
    ScenarioExecution,
    ScenarioStatus,
    SuiteStatus,
)


class GradingInputError(Exception):
    """The run package cannot be graded: unfinished, inconsistent or corrupted."""


class ContractMismatchError(GradingInputError):
    """The run was executed under a different evaluation contract than the CURRENT one."""


# --- Runtime log -------------------------------------------------------------

_RUNTIME_LINE = re.compile(r"^(\S+) (\S+) (\S+) request_id=(\S+) app_id=(\S+) ?(.*)$")
_FIELD = re.compile(r"(\w+)=(\S*)")


@dataclass(frozen=True)
class RuntimeRecord:
    """One parsed stderr record. ``fields`` are the message's ``key=value`` tokens."""

    line_no: int
    timestamp: str
    level: str
    logger: str
    request_id: str
    event: str | None
    fields: dict[str, str]


@dataclass(frozen=True)
class ToolEvent:
    """A runtime ``tool_call_*`` event.

    Proves that a tool invocation was logged, with the arguments the tool
    chose to log. It is corroborating evidence for INFERRED tool-use claims,
    not proof of the tool's output or of chaining.
    """

    line_no: int
    tool: str
    outcome: str  # suffix of tool_call_<outcome>: result, failed, rejected, empty
    fields: dict[str, str]


def _bool_field(value: str | None) -> bool | None:
    return {"True": True, "False": False}.get(value) if value is not None else None


@dataclass(frozen=True)
class RuntimeEvidence:
    """Generic facts from ``runtime/datachat.log`` (last occurrence wins)."""

    records: tuple[RuntimeRecord, ...] = ()
    unparsed_lines: int = 0
    final_kind: str | None = None
    final_answer_passed: bool | None = None
    response_kind: str | None = None
    adapter_fallback_used: bool | None = None
    trace_present: bool | None = None
    request_end_status: str | None = None
    request_end_http_status: int | None = None
    request_end_error_code: str | None = None
    tool_events: tuple[ToolEvent, ...] = ()


def parse_runtime_log(text: str) -> RuntimeEvidence:
    records, tool_events, unparsed = [], [], 0
    last: dict[str, dict[str, str]] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = _RUNTIME_LINE.match(line)
        if not match:
            if line.strip():
                unparsed += 1
            continue
        timestamp, level, logger, request_id, _app, message = match.groups()
        fields = dict(_FIELD.findall(message))
        first = message.split(" ", 1)[0] if message else ""
        event = fields.get("event") or (first if first and "=" not in first else None)
        records.append(RuntimeRecord(line_no, timestamp, level, logger, request_id, event, fields))
        if event:
            last[event] = fields
        if event and event.startswith("tool_call_"):
            tool = fields.get("tool") or logger.rsplit(".", 1)[-1].removesuffix("_tool")
            tool_events.append(ToolEvent(line_no, tool, event[len("tool_call_"):], fields))
    final = last.get("final_answer_check", {})
    end = last.get("datachat_request_end", {})
    http_status = end.get("http_status")
    return RuntimeEvidence(
        records=tuple(records),
        unparsed_lines=unparsed,
        final_kind=final.get("final_kind"),
        final_answer_passed=_bool_field(final.get("passed")),
        response_kind=(last.get("chat_end") or last.get("datachat_engine_done") or {}).get("response_kind"),
        adapter_fallback_used=_bool_field(last.get("datachat_adapter_status", {}).get("adapter_fallback_used")),
        trace_present=_bool_field(last.get("datachat_trace_status", {}).get("trace_present")),
        request_end_status=end.get("status"),
        request_end_http_status=int(http_status) if http_status and http_status.isdigit() else None,
        request_end_error_code=end.get("error_code"),
        tool_events=tuple(tool_events),
    )


# --- /datachat response ------------------------------------------------------


class ParseStatus(str, Enum):
    OK = "OK"
    MALFORMED_JSON = "MALFORMED_JSON"
    NON_JSON = "NON_JSON"
    ERROR_BODY = "ERROR_BODY"
    UNKNOWN_SHAPE = "UNKNOWN_SHAPE"


KNOWN_RESPONSE_TYPES = frozenset({"str", "dataframe", "image", "dict"})
_TABLE_META = ("result_rows", "result_columns", "preview_rows", "preview_columns", "truncated")


@dataclass(frozen=True)
class NormalizedResponse:
    """The ``/datachat`` answer: raw parsed body plus normalized access.

    ``raw`` keeps the whole parsed body for checks that need other fields.
    ``final_kind``/``adapter_fallback_used`` come from the runtime log: an
    engine ``kind:error`` reaches HTTP as an ordinary ``type: "str"``.
    """

    http_status: int | None
    content_type: str | None
    parse_status: ParseStatus
    raw: Any = None
    type: str | None = None
    text: str | None = None
    table_rows: tuple[Any, ...] | None = None
    table_meta: dict[str, Any] = field(default_factory=dict)
    charts: tuple[dict[str, Any], ...] = ()
    note: str | None = None
    more_rows_available: bool | None = None
    download_url: str | None = None
    log_id: int | None = None
    error: str | None = None
    final_kind: str | None = None
    adapter_fallback_used: bool | None = None

    @property
    def runtime_error(self) -> bool:
        """The runtime explicitly recorded an error final answer."""
        return self.final_kind == "error"

    @property
    def answer_kind(self) -> str | None:
        """``error`` when the runtime says so; otherwise text/table/image/dict."""
        if self.runtime_error:
            return "error"
        return {"str": "text", "dataframe": "table", "image": "image", "dict": "dict"}.get(self.type or "")


def normalize_response(
    body: bytes,
    http_status: int | None,
    content_type: str | None,
    runtime: RuntimeEvidence | None = None,
) -> NormalizedResponse:
    base = {
        "http_status": http_status,
        "content_type": content_type,
        "final_kind": runtime.final_kind if runtime else None,
        "adapter_fallback_used": runtime.adapter_fallback_used if runtime else None,
    }
    if content_type is not None and "json" not in content_type.lower():
        return NormalizedResponse(parse_status=ParseStatus.NON_JSON, **base)
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return NormalizedResponse(parse_status=ParseStatus.MALFORMED_JSON, **base)
    if not isinstance(parsed, dict):
        return NormalizedResponse(parse_status=ParseStatus.UNKNOWN_SHAPE, raw=parsed, **base)
    log_id = parsed.get("log_id") if isinstance(parsed.get("log_id"), int) else None
    response = parsed.get("response")
    if "response" not in parsed and "error" in parsed:
        error = parsed["error"]
        return NormalizedResponse(
            parse_status=ParseStatus.ERROR_BODY, raw=parsed, error=str(error), log_id=log_id, **base
        )
    if not isinstance(response, dict) or response.get("type") not in KNOWN_RESPONSE_TYPES:
        return NormalizedResponse(parse_status=ParseStatus.UNKNOWN_SHAPE, raw=parsed, log_id=log_id, **base)
    rtype = response["type"]
    value = response.get("value")
    charts = response.get("charts")
    note = response.get("note")
    more = response.get("more_rows_available")
    url = response.get("download_url")
    return NormalizedResponse(
        parse_status=ParseStatus.OK,
        raw=parsed,
        type=rtype,
        text=value if rtype == "str" and isinstance(value, str) else None,
        table_rows=tuple(value) if rtype == "dataframe" and isinstance(value, list) else None,
        table_meta={k: response[k] for k in _TABLE_META if k in response} if rtype == "dataframe" else {},
        charts=tuple(c for c in charts if isinstance(c, dict)) if isinstance(charts, list) else (),
        note=note if isinstance(note, str) else None,
        more_rows_available=more if isinstance(more, bool) else None,
        download_url=url if isinstance(url, str) else None,
        log_id=log_id,
        **base,
    )


# --- agent_runs, Usage, artifacts --------------------------------------------


@dataclass(frozen=True)
class AgentRunEvidence:
    """Generic view of the ``agent_runs`` record. No observations exist in CURRENT records."""

    availability: Availability
    reason: str | None = None
    ref: str | None = None
    parse_ok: bool = False
    state: str | None = None
    steps_count: int | None = None
    code_blocks: tuple[str, ...] = ()
    response_kind: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class UsageRows:
    """Persisted Usage rows (no DB access). Extra rows are non-primary contributions."""

    availability: Availability
    reason: str | None = None
    ref: str | None = None
    rows: tuple[dict[str, Any], ...] = ()
    primary_log_id: int | None = None
    primary_row_present: bool = False

    @property
    def extra_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.rows if row.get("id") != self.primary_log_id)


@dataclass(frozen=True)
class ArtifactEvidence:
    """Image/export state and references. No image interpretation happens here."""

    image_availability: Availability | None = None
    image_reason: str | None = None
    image_ref: str | None = None
    image_meta_ref: str | None = None
    image_meta: dict[str, Any] | None = None
    export_availability: Availability | None = None
    export_reason: str | None = None
    export_ref: str | None = None
    export_locator: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolEvidence:
    """Tool-use evidence, kept epistemically separate by source.

    - ``code_blocks``: generated ``python_interpreter`` code (INFERRED);
    - ``runtime_events``: runtime ``tool_call_*`` events (corroboration only);
    - ``usage_extra_rows``: provider rows beyond the primary row (a semantic
      provider call happened; which tool made it is not recorded).
    """

    code_blocks: tuple[str, ...] = ()
    runtime_events: tuple[ToolEvent, ...] = ()
    usage_extra_rows: int = 0


# --- Scenario and run views --------------------------------------------------


@dataclass(frozen=True)
class ScenarioEvidence:
    """Integrity-checked, normalized view of one scenario package.

    ``refs`` are run-relative paths for citing evidence. Views are None when
    the gate is not GRADEABLE or the source was not captured.
    """

    scenario_id: str
    ordinal: int | None
    question: str | None
    status: ScenarioStatus
    gate: Gate
    execution_sha256: str | None
    failure_code: str | None = None
    flags: dict[str, bool] = field(default_factory=dict)
    availability: dict[str, tuple[Availability, str | None]] = field(default_factory=dict)
    datachat_request_id: str | None = None
    refs: dict[str, str] = field(default_factory=dict)
    response: NormalizedResponse | None = None
    runtime: RuntimeEvidence | None = None
    agent_run: AgentRunEvidence | None = None
    usage: UsageRows | None = None
    artifacts: ArtifactEvidence | None = None
    tools: ToolEvidence | None = None

    @property
    def teardown_verified(self) -> bool | None:
        return self.flags.get("teardown_verified")

    @property
    def evidence_complete(self) -> bool | None:
        return self.flags.get("evidence_complete")


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    status: SuiteStatus
    official: bool
    comparable: bool
    failure_code: str | None
    maui: dict[str, Any]
    contract_hashes: dict[str, str]

    @property
    def dataset_sha256(self) -> str | None:
        return self.contract_hashes.get(contract.DATASET_NAME)

    @property
    def ground_truth_sha256(self) -> str | None:
        return self.contract_hashes.get(contract.GROUND_TRUTH_JSON_NAME)


@dataclass(frozen=True)
class RunForGrading:
    root: Path
    provenance: RunProvenance
    scenarios: tuple[ScenarioEvidence, ...]

    def scenario(self, scenario_id: str) -> ScenarioEvidence:
        for item in self.scenarios:
            if item.scenario_id == scenario_id:
                return item
        raise KeyError(scenario_id)


# --- Loading -----------------------------------------------------------------


def _refuse(message: str) -> GradingInputError:
    return GradingInputError(message)


def _read_manifest_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise _refuse(f"{what} missing: {path.name}")
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        raise _refuse(f"{what} unreadable: {type(exc).__name__}") from exc


def _safe(root: Path, relative: str, what: str) -> Path:
    try:
        return safe_child(root, relative)
    except (ValueError, TypeError) as exc:
        raise _refuse(f"{what}: unsafe path {relative!r}") from exc


def _verify_file(path: Path, sha256: str, size: int | None, what: str) -> None:
    if not path.is_file():
        raise _refuse(f"{what}: referenced file missing")
    if size is not None and path.stat().st_size != size:
        raise _refuse(f"{what}: size mismatch")
    if sha256_file(path) != sha256:
        raise _refuse(f"{what}: sha256 mismatch")


def _verify_contract(run: EvaluationRun, repo_root: Path) -> dict[str, str]:
    recorded = {a.get("name"): a for a in run.evaluation_contract.get("artifacts", []) if isinstance(a, dict)}
    if run.evaluation_contract.get("required_match") is not True:
        raise ContractMismatchError("run was executed against a non-matching evaluation contract")
    hashes: dict[str, str] = {}
    for check in contract.verify_artifacts(repo_root):
        entry = recorded.get(check.name)
        if not check.required:
            if check.actual_sha256 is not None:
                hashes[check.name] = check.actual_sha256
            continue
        if not check.matches:
            raise ContractMismatchError(f"evaluation contract artifact changed or missing: {check.name}")
        if entry is None or entry.get("actual_sha256") != check.actual_sha256:
            raise ContractMismatchError(f"run manifest does not record the current contract hash: {check.name}")
        hashes[check.name] = check.actual_sha256
    return hashes


def _gate(execution: ScenarioExecution) -> Gate:
    if execution.status is ScenarioStatus.NOT_RUN:
        return Gate.NOT_RUN
    if execution.status is ScenarioStatus.NOT_SCOREABLE_INFRA:
        return Gate.NOT_SCOREABLE_INFRA
    if execution.status is ScenarioStatus.EXECUTED:
        return Gate.GRADEABLE if execution.flags.scoreable else Gate.NOT_SCOREABLE
    raise _refuse(f"{execution.scenario_id}: scenario is not finished ({execution.status.value})")


def _verify_scenario_files(execution: ScenarioExecution, scenario_dir: Path) -> dict[str, HttpExchange]:
    sid = execution.scenario_id
    for source, record in execution.evidence.items():
        for ref in record.files:
            path = _safe(scenario_dir, ref.path, f"{sid} {source}")
            _verify_file(path, ref.sha256, ref.size, f"{sid} {source} {ref.path}")
    exchanges: dict[str, HttpExchange] = {}
    for key, meta_paths in execution.requests.items():
        for meta_rel in meta_paths:
            meta_path = _safe(scenario_dir, meta_rel, f"{sid} http {key}")
            data = _read_manifest_json(meta_path, f"{sid} http metadata {meta_rel}")
            try:
                exchange = HttpExchange.from_dict(data)
            except (KeyError, TypeError, ValueError) as exc:
                raise _refuse(f"{sid} http metadata {meta_rel} malformed") from exc
            for part in (exchange.request, exchange.response):
                if part is None or part.body_file is None:
                    continue
                body = _safe(scenario_dir, part.body_file, f"{sid} http {key} body")
                _verify_file(body, part.body_sha256, getattr(part, "size", None), f"{sid} http {part.body_file}")
            exchanges[key] = exchange  # last attempt wins
    return exchanges


def _source_ref(execution: ScenarioExecution, source: str, suffix: str) -> str | None:
    record = execution.evidence.get(source)
    if record is None:
        return None
    for ref in record.files:
        if ref.path.endswith(suffix):
            return ref.path
    return None


def _load_agent_run(execution: ScenarioExecution, scenario_dir: Path, run_rel: str) -> AgentRunEvidence:
    record = execution.evidence.get("agent_runs")
    if record is None:
        return AgentRunEvidence(Availability.UNAVAILABLE, "not recorded")
    path = _source_ref(execution, "agent_runs", "record.jsonl")
    if record.availability is not Availability.PRESENT or path is None:
        return AgentRunEvidence(record.availability, record.reason)
    lines = [ln for ln in (scenario_dir / path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    ref = f"{run_rel}/{path}"
    try:
        raw = json.loads(lines[0]) if len(lines) == 1 else None
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        return AgentRunEvidence(record.availability, record.reason, ref, parse_ok=False)
    code = []
    for call in raw.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name = call.get("tool", call.get("tool_name"))
        args = call.get("args", call.get("arguments"))
        if name == "python_interpreter" and isinstance(args, str):
            code.append(args)
    extra = raw.get("extra") if isinstance(raw.get("extra"), dict) else {}
    steps = raw.get("steps_count")
    return AgentRunEvidence(
        availability=record.availability,
        reason=record.reason,
        ref=ref,
        parse_ok=True,
        state=raw.get("state") if isinstance(raw.get("state"), str) else None,
        steps_count=steps if isinstance(steps, int) else None,
        code_blocks=tuple(code),
        response_kind=extra.get("response_kind"),
        raw=raw,
    )


def _load_usage(execution: ScenarioExecution, scenario_dir: Path, run_rel: str) -> UsageRows:
    record = execution.evidence.get("usage")
    if record is None:
        return UsageRows(Availability.UNAVAILABLE, "not recorded")
    path = _source_ref(execution, "usage", "rows.json")
    if record.availability is not Availability.PRESENT or path is None:
        return UsageRows(record.availability, record.reason)
    data = _read_manifest_json(scenario_dir / path, f"{execution.scenario_id} usage rows")
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise _refuse(f"{execution.scenario_id} usage rows malformed")
    return UsageRows(
        availability=record.availability,
        reason=record.reason,
        ref=f"{run_rel}/{path}",
        rows=tuple(r for r in rows if isinstance(r, dict)),
        primary_log_id=data.get("primary_log_id"),
        primary_row_present=bool(data.get("primary_row_present")),
    )


def _load_artifacts(execution: ScenarioExecution, scenario_dir: Path, run_rel: str) -> ArtifactEvidence:
    image = execution.evidence.get("image")
    export = execution.evidence.get("export")
    png = _source_ref(execution, "image", "image.png")
    meta_path = _source_ref(execution, "image", "image.json")
    meta = _read_manifest_json(scenario_dir / meta_path, "image metadata") if meta_path else None
    csv = _source_ref(execution, "export", "export.csv")
    return ArtifactEvidence(
        image_availability=image.availability if image else None,
        image_reason=image.reason if image else None,
        image_ref=f"{run_rel}/{png}" if png else None,
        image_meta_ref=f"{run_rel}/{meta_path}" if meta_path else None,
        image_meta=meta if isinstance(meta, dict) else None,
        export_availability=export.availability if export else None,
        export_reason=export.reason if export else None,
        export_ref=f"{run_rel}/{csv}" if csv else None,
        export_locator=dict(export.locator) if export else {},
    )


def _load_scenario(root: Path, entry: dict[str, Any], run_id: str) -> ScenarioEvidence:
    sid = entry.get("scenario_id")
    layout = RunLayout(root)
    try:
        scenario_dir = layout.scenario_dir(sid)
    except ValueError as exc:
        raise _refuse(f"invalid scenario id in manifest: {sid!r}") from exc
    run_rel = f"scenarios/{sid}"
    if entry.get("dir") != run_rel:
        raise _refuse(f"{sid}: manifest dir {entry.get('dir')!r} is not {run_rel!r}")
    execution_path = layout.execution(sid)
    if not execution_path.is_file():
        raise _refuse(f"{sid}: execution.json missing")
    execution_sha256 = sha256_file(execution_path)
    if execution_sha256 != entry.get("execution_sha256"):
        raise _refuse(f"{sid}: execution.json sha256 does not match the run manifest")
    try:
        execution = ScenarioExecution.from_dict(_read_manifest_json(execution_path, f"{sid} execution"))
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(f"{sid}: execution.json malformed") from exc
    if execution.run_id != run_id or execution.scenario_id != sid:
        raise _refuse(f"{sid}: execution.json identity does not match the run")
    if execution.status.value != entry.get("status"):
        raise _refuse(f"{sid}: status differs between manifest and execution.json")
    gate = _gate(execution)
    exchanges = _verify_scenario_files(execution, scenario_dir)

    flags = execution.flags.to_dict()
    availability = {name: (rec.availability, rec.reason) for name, rec in execution.evidence.items()}
    refs = {}
    for source, record in execution.evidence.items():
        for ref in record.files:
            refs[ref.path] = f"{run_rel}/{ref.path}"
    common = dict(
        scenario_id=sid,
        ordinal=execution.ordinal,
        question=execution.question,
        status=execution.status,
        gate=gate,
        execution_sha256=execution_sha256,
        failure_code=execution.failure.code.value if execution.failure else None,
        flags=flags,
        availability=availability,
        refs=refs,
    )
    if gate is not Gate.GRADEABLE:
        return ScenarioEvidence(**common)

    runtime = None
    runtime_path = _source_ref(execution, "runtime_datachat", "datachat.log")
    if runtime_path:
        runtime = parse_runtime_log((scenario_dir / runtime_path).read_text(encoding="utf-8"))

    exchange = exchanges.get("datachat")
    response = None
    if exchange is not None and exchange.response is not None:
        body = (scenario_dir / exchange.response.body_file).read_bytes()
        response = normalize_response(body, exchange.response.status, exchange.response.content_type, runtime)

    agent_run = _load_agent_run(execution, scenario_dir, run_rel)
    usage = _load_usage(execution, scenario_dir, run_rel)
    tools = ToolEvidence(
        code_blocks=agent_run.code_blocks,
        runtime_events=runtime.tool_events if runtime else (),
        usage_extra_rows=len(usage.extra_rows) if usage.primary_log_id is not None else len(usage.rows),
    )
    return ScenarioEvidence(
        **common,
        datachat_request_id=exchange.request_id if exchange else None,
        response=response,
        runtime=runtime,
        agent_run=agent_run,
        usage=usage,
        artifacts=_load_artifacts(execution, scenario_dir, run_rel),
        tools=tools,
    )


def evidence_digest(run_dir: Path, refs: tuple[str, ...] | list[str]) -> str:
    """SHA-256 binding a manual-review answer to the evidence it was made against.

    Combines each run-relative ref and its file SHA-256, in ``refs`` order.
    """
    root = Path(run_dir)
    digest = hashlib.sha256()
    for ref in refs:
        path = _safe(root, ref, "evidence ref")
        if not path.is_file():
            raise _refuse(f"evidence ref missing: {ref}")
        digest.update(f"{ref}\0{sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def load_run_for_grading(run_dir: Path, repo_root: Path = contract.REPO_ROOT) -> RunForGrading:
    """Load and verify one finished run directory. Read-only.

    ``repo_root`` locates the frozen evaluation artifacts whose hashes must
    still match both the pins and the run manifest.

    Raises ``GradingInputError`` when the package cannot be trusted.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise _refuse(f"run directory not found: {root}")
    data = _read_manifest_json(RunLayout(root).run_manifest, "run manifest")
    try:
        run = EvaluationRun.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse("run manifest malformed") from exc
    if run.status is SuiteStatus.RUNNING:
        raise _refuse("run is still RUNNING (interrupted or in progress)")
    if root.name != run.run_id:
        raise _refuse("run directory name does not match run_id")

    hashes = _verify_contract(run, Path(repo_root))

    order = run.scenario_order
    if len(set(order)) != len(order):
        raise _refuse("duplicate scenario ids in scenario_order")
    index_ids = [entry.get("scenario_id") for entry in run.scenarios]
    if len(set(index_ids)) != len(index_ids) or any(sid not in order for sid in index_ids):
        raise _refuse("scenario index inconsistent with scenario_order")
    if index_ids != [sid for sid in order if sid in index_ids]:
        raise _refuse("scenario index out of catalogue order")
    missing = [sid for sid in order if sid not in index_ids]
    if missing and run.status is not SuiteStatus.ABORTED_INFRA:
        raise _refuse(f"scenarios missing from a {run.status.value} run: {missing}")

    by_id = {entry["scenario_id"]: entry for entry in run.scenarios}
    scenarios = []
    for sid in order:
        if sid in by_id:
            scenarios.append(_load_scenario(root, by_id[sid], run.run_id))
        else:
            # Suite aborted before this scenario's package was written.
            scenarios.append(
                ScenarioEvidence(
                    scenario_id=sid, ordinal=None, question=None,
                    status=ScenarioStatus.NOT_RUN, gate=Gate.NOT_RUN, execution_sha256=None,
                )
            )
    if run.status is SuiteStatus.COMPLETED and any(s.status is not ScenarioStatus.EXECUTED for s in scenarios):
        raise _refuse("COMPLETED run contains non-EXECUTED scenarios")

    provenance = RunProvenance(
        run_id=run.run_id,
        status=run.status,
        official=run.official,
        comparable=run.comparable,
        failure_code=run.failure.code.value if run.failure else None,
        maui=dict(run.maui),
        contract_hashes=hashes,
    )
    return RunForGrading(root=root, provenance=provenance, scenarios=tuple(scenarios))
