"""Reporting source model of the DataChat Functional Evaluation.

Typed, renderer-independent view of the persisted inputs of one graded run,
produced by ``reader.load_run_for_reporting``. Each scenario keeps three
levels apart:

- Question: what the run's own ``execution.json`` recorded;
- Agent result: what ``/datachat`` returned to the user (persisted HTTP
  response plus delivered artifacts), never ``agent_runs``, runtime logs or
  Usage;
- Grading: the persisted ``ScenarioGrade``, an opaque authoritative judgment.

Persisted domain records (``RunGrade``, ``ScenarioGrade``, ``EvaluationRun``,
``ScenarioExecution``, ``HttpExchange``, ``FileRef``, ``ArtifactRecord``) are
reused as-is. Nothing here decides prominence, labels, order or attention.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.datachat.grading.model import RunGrade, ScenarioGrade, ScenarioIndexEntry
from evaluation.datachat.grading.reader import ParseStatus
from evaluation.datachat.model import (
    ArtifactRecord,
    Availability,
    EvaluationRun,
    FileRef,
    HttpExchange,
    ScenarioExecution,
    TransportError,
)


@dataclass(frozen=True)
class DatachatBody:
    """Structural view of the stored ``/datachat`` response body.

    ``parsed`` is the whole parsed JSON (None unless the body is JSON). The
    other fields are direct reads of the CURRENT envelope
    ``{"response": {...}, "explanation": ..., "log_id"?}`` or of an error body
    ``{"error": ...}``; they are never rewritten or summarized.
    """

    parse_status: ParseStatus
    parsed: Any = None
    response_type: str | None = None
    value: Any = None
    table_meta: dict[str, Any] = field(default_factory=dict)
    charts: tuple[dict[str, Any], ...] = ()
    note: str | None = None
    more_rows_available: bool | None = None
    download_url: str | None = None
    explanation: Any = None
    log_id: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeliveredArtifact:
    """A user-visible artifact (``image`` or ``export``) as the harness recorded it.

    ``file`` is the hash-verified ``image.png`` / ``export.csv`` when one was
    persisted. ``metadata`` is the image's ``artifacts/image.json`` record
    (``decode_status`` etc.); exports carry their facts in ``locator``.
    """

    kind: str
    availability: Availability
    reason: str | None = None
    file: FileRef | None = None
    metadata: ArtifactRecord | None = None
    metadata_file: FileRef | None = None
    locator: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    """What ``/datachat`` returned in this run, from persisted user-visible sources only.

    ``availability``/``reason`` come from the ``http_datachat`` evidence record
    (None when the scenario never recorded one). ``exchange`` is the last
    recorded ``/datachat`` attempt; without a response, ``body`` and ``content``
    are None and ``transport_error`` explains why.
    """

    availability: Availability | None
    reason: str | None = None
    exchange: HttpExchange | None = None
    attempts: int = 0
    body: bytes | None = None
    body_file: str | None = None  # scenario-relative
    content: DatachatBody | None = None
    image: DeliveredArtifact | None = None
    export: DeliveredArtifact | None = None

    @property
    def http_status(self) -> int | None:
        return self.exchange.response.status if self.exchange and self.exchange.response else None

    @property
    def content_type(self) -> str | None:
        return self.exchange.response.content_type if self.exchange and self.exchange.response else None

    @property
    def transport_error(self) -> TransportError | None:
        return self.exchange.transport_error if self.exchange else None


@dataclass(frozen=True)
class ReportingScenarioInput:
    """One indexed scenario: Question, Agent result and Grading, plus provenance.

    ``execution`` is None only for a scenario whose package was never written
    (suite aborted before it); then there is no question and no Agent result.
    """

    scenario_id: str
    index: ScenarioIndexEntry
    grade: ScenarioGrade
    execution: ScenarioExecution | None
    execution_sha256: str | None
    dir: str  # run-relative scenario directory
    agent_result: AgentResult | None = None

    @property
    def question(self) -> str | None:
        return self.execution.question if self.execution else None

    @property
    def question_sha256(self) -> str | None:
        return self.execution.question_sha256 if self.execution else None

    def run_path(self, scenario_relative: str) -> str:
        """Run-relative path of a scenario-relative ``FileRef.path``."""
        return f"{self.dir}/{scenario_relative}"


@dataclass(frozen=True)
class ReportingRunInput:
    """A graded run package, integrity-checked for reporting. Scenarios follow ``RunGrade.scenarios``."""

    root: Path
    run_grade: RunGrade
    run_grade_sha256: str
    manifest: EvaluationRun
    scenarios: tuple[ReportingScenarioInput, ...]

    @property
    def run_id(self) -> str:
        return self.run_grade.run_id

    def scenario(self, scenario_id: str) -> ReportingScenarioInput:
        for item in self.scenarios:
            if item.scenario_id == scenario_id:
                return item
        raise KeyError(scenario_id)
