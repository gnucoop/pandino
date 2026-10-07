"""Run-centric projection of a graded run (reporting design §5–§7).

``build_operator_report(run_input, references)`` turns a validated
``ReportingRunInput`` and the scenario Evaluation References into a typed,
renderer-independent model whose scenario fields follow the seven report
sections:

    Domanda · Risposta dell'agente · Attività osservata · Artefatti ·
    Validazione tecnica · Riferimento di valutazione · Provenienza tecnica

Pure: no I/O, no clock, no environment, no mutation of the input. It
reorganizes persisted values only; it never grades, compares the answer with
the reference, ranks failures or derives a run verdict, score or rate.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from evaluation.datachat.grading.model import (
    CheckResult,
    ExecutionAssessment,
    ExecutionAssessmentReason,
    Gate,
    ObservedActivity,
    ScenarioIndexEntry,
    TechnicalFailure,
    TechnicalOutcome,
)
from evaluation.datachat.grading.reader import ParseStatus
from evaluation.datachat.model import EvidenceSourceRecord, Failure, ScenarioStatus, SuiteStatus
from evaluation.datachat.reporting.charts import chart_filename
from evaluation.datachat.reporting.model import (
    AgentResult,
    DeliveredArtifact,
    ReportingRunInput,
    ReportingScenarioInput,
)
from evaluation.datachat.reporting.reference import EvaluationReference

# Manifest config keys shown as DataChat identity; nothing else is copied.
_DATACHAT_IDENTITY_KEYS = ("DATACHAT_PROVIDER", "DATACHAT_MODEL", "DATACHAT_ENGINE")


class OperatorReportError(Exception):
    """A validated reporting input violates an invariant the operator model relies on."""


# --- Agent result -----------------------------------------------------------------------


class ResultShape(str, Enum):
    """Structural shape of the persisted response (design §6.1). Says nothing about correctness."""

    TEXT = "TEXT"                # response.type == "str"
    TABLE = "TABLE"              # response.type == "dataframe"
    IMAGE = "IMAGE"              # response.type == "image"
    STRUCTURED = "STRUCTURED"    # response.type == "dict"
    ERROR_BODY = "ERROR_BODY"    # {"error": ...}
    RAW = "RAW"                  # non-JSON, malformed JSON or unknown envelope: body verbatim
    NO_RESPONSE = "NO_RESPONSE"  # request recorded, no response persisted
    ABSENT = "ABSENT"            # no /datachat record at all (e.g. NOT_RUN)


_SHAPE_BY_TYPE = {"str": ResultShape.TEXT, "dataframe": ResultShape.TABLE,
                  "image": ResultShape.IMAGE, "dict": ResultShape.STRUCTURED}


@dataclass(frozen=True)
class OperatorAgentResult:
    """The persisted response plus its structural shape. ``source`` is None only for ``ABSENT``."""

    shape: ResultShape
    source: AgentResult | None

    @property
    def charts(self) -> tuple[dict[str, Any], ...]:
        return self.source.content.charts if self.source and self.source.content else ()

    @property
    def image(self) -> DeliveredArtifact | None:
        return self.source.image if self.source else None

    @property
    def export(self) -> DeliveredArtifact | None:
        return self.source.export if self.source else None


def _agent_result(result: AgentResult | None) -> OperatorAgentResult:
    if result is None:
        return OperatorAgentResult(ResultShape.ABSENT, None)
    content = result.content
    if content is None:
        shape = ResultShape.NO_RESPONSE
    elif content.parse_status is ParseStatus.ERROR_BODY:
        shape = ResultShape.ERROR_BODY
    elif content.parse_status is ParseStatus.OK and content.response_type in _SHAPE_BY_TYPE:
        shape = _SHAPE_BY_TYPE[content.response_type]
    else:
        shape = ResultShape.RAW
    return OperatorAgentResult(shape, result)


# --- Scenario ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartArtifact:
    """One recorded ``charts[]`` spec and the stable name of its derived drawing under ``reports/charts/``."""

    index: int  # 1-based, recorded order
    spec: dict[str, Any]
    filename: str


@dataclass(frozen=True)
class TechnicalValidation:
    """Persisted technical grading of one scenario, unchanged. ``outcome`` None: non-gradeable gate."""

    gate: Gate
    outcome: TechnicalOutcome | None
    failures: tuple[TechnicalFailure, ...]  # recorded order, unranked
    execution_assessment: ExecutionAssessment | None
    execution_assessment_reason: ExecutionAssessmentReason | None
    checks: tuple[CheckResult, ...]


@dataclass(frozen=True)
class ScenarioProvenance:
    """Secondary identities and navigation targets in the run package."""

    dir: str
    execution_status: ScenarioStatus | None
    execution_failure: Failure | None
    execution_sha256: str | None
    question_sha256: str | None
    teardown_verified: bool | None
    request_ids: tuple[str, ...]
    evidence: tuple[EvidenceSourceRecord, ...]  # execution.json records with FileRefs
    evidence_refs: tuple[str, ...]  # run-relative, from checks, first-seen order


@dataclass(frozen=True)
class ScenarioReport:
    """One scenario, one field per report section."""

    scenario_id: str
    question: str | None                     # 1 Domanda
    question_sent: bool
    agent_result: OperatorAgentResult        # 2 Risposta dell'agente
    activity: tuple[ObservedActivity, ...]   # 3 Attività osservata (persisted order)
    charts: tuple[ChartArtifact, ...]        # 4 Artefatti (image/export via agent_result)
    validation: TechnicalValidation          # 5 Validazione tecnica
    reference: EvaluationReference | None    # 6 Riferimento di valutazione
    provenance: ScenarioProvenance           # 7 Provenienza tecnica


def _unique(values) -> tuple:
    return tuple(dict.fromkeys(v for v in values if v is not None))


def _scenario(s: ReportingScenarioInput, reference: EvaluationReference | None) -> ScenarioReport:
    g, execution, result = s.grade, s.execution, s.agent_result
    agent = _agent_result(result)
    return ScenarioReport(
        scenario_id=s.scenario_id,
        question=s.question,
        question_sent=bool(result and result.attempts),
        agent_result=agent,
        activity=g.activity,
        charts=tuple(ChartArtifact(i, spec, chart_filename(s.scenario_id, i))
                     for i, spec in enumerate(agent.charts, 1)),
        validation=TechnicalValidation(g.gate, g.outcome, g.failures, g.execution_assessment,
                                       g.execution_assessment_reason, g.checks),
        reference=reference,
        provenance=ScenarioProvenance(
            dir=s.dir,
            execution_status=execution.status if execution else None,
            execution_failure=execution.failure if execution else None,
            execution_sha256=s.execution_sha256,
            question_sha256=s.question_sha256,
            teardown_verified=g.teardown_verified,
            request_ids=_unique([
                result.exchange.request_id if result and result.exchange else None,
                result.image.metadata.source_request_id if result and result.image and result.image.metadata else None,
            ]),
            evidence=tuple(execution.evidence.values()) if execution else (),
            evidence_refs=_unique(ref for c in g.checks for ref in c.evidence_refs),
        ),
    )


# --- Run ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorRunReport:
    """One graded run. No run outcome, score or rate.

    ``outcome_counts`` / ``gate_counts`` split the persisted ``counts_by_outcome``
    by key kind; ``failure_counts`` is ``counts_by_failure``; persisted order.
    """

    run_id: str
    official: bool
    comparable: bool
    status: SuiteStatus
    failure: Failure | None
    started_at: str | None
    ended_at: str | None
    maui: dict[str, Any]
    datachat: dict[str, Any]  # provider / model / engine only
    grader: dict[str, Any]
    contract: dict[str, Any]
    grading_schema_version: int
    run_grade_sha256: str
    inventory: tuple[ScenarioIndexEntry, ...]  # frozen run order
    outcome_counts: tuple[tuple[TechnicalOutcome, int], ...]
    failure_counts: tuple[tuple[TechnicalFailure, int], ...]
    gate_counts: tuple[tuple[Gate, int], ...]
    scenarios: tuple[ScenarioReport, ...]  # frozen run order


def _split_counts(counts: dict[str, int]) -> tuple[tuple, tuple]:
    outcomes, gates = [], []
    for key, value in counts.items():
        if key in TechnicalOutcome._value2member_map_:
            outcomes.append((TechnicalOutcome(key), value))
        elif key in Gate._value2member_map_ and key != Gate.GRADEABLE.value:
            gates.append((Gate(key), value))
        else:
            raise OperatorReportError(f"counts_by_outcome has an unknown key {key!r}")
    return tuple(outcomes), tuple(gates)


def _failure_counts(counts: dict[str, int]) -> tuple[tuple[TechnicalFailure, int], ...]:
    try:
        return tuple((TechnicalFailure(key), value) for key, value in counts.items())
    except ValueError as exc:
        raise OperatorReportError(f"counts_by_failure has an unknown key: {exc}") from exc


def build_operator_report(
    run_input: ReportingRunInput, references: Mapping[str, EvaluationReference]
) -> OperatorRunReport:
    """Project a validated ``ReportingRunInput`` with its Evaluation References. Pure."""
    rg, manifest = run_input.run_grade, run_input.manifest
    if [e.scenario_id for e in rg.scenarios] != [s.scenario_id for s in run_input.scenarios]:
        raise OperatorReportError("scenario inputs do not follow RunGrade.scenarios")
    outcome_counts, gate_counts = _split_counts(rg.counts_by_outcome)
    config = manifest.datachat_config.get("values", {})
    return OperatorRunReport(
        run_id=rg.run_id,
        official=rg.official,
        comparable=rg.comparable,
        status=manifest.status,
        failure=manifest.failure,
        started_at=manifest.started_at,
        ended_at=manifest.ended_at,
        maui=dict(manifest.maui),
        datachat={k: config.get(k) for k in _DATACHAT_IDENTITY_KEYS},
        grader=dict(rg.grader),
        contract=dict(rg.contract),
        grading_schema_version=rg.schema_version,
        run_grade_sha256=run_input.run_grade_sha256,
        inventory=rg.scenarios,
        outcome_counts=outcome_counts,
        failure_counts=_failure_counts(rg.counts_by_failure),
        gate_counts=gate_counts,
        scenarios=tuple(_scenario(s, references.get(s.scenario_id)) for s in run_input.scenarios),
    )
