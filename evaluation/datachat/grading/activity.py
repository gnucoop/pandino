"""Observed-activity summary of one GRADEABLE scenario (grading design §5.1).

    ScenarioEvidence (normalized tool/Usage evidence) → tuple[ObservedActivity]

Pure and scenario-agnostic. The summary is descriptive evidence for the
report, never grading policy: no check or outcome reads it.

Each source stays separate and never claims more than it supports:

- RUNTIME: runtime ``tool_call_<outcome>`` events, one record per
  (tool, outcome) with its count. Logged fields are not persisted (they may
  carry paths or values);
- GENERATED_CODE: calls to a CURRENT DataChat tool written in generated
  ``python_interpreter`` code, one record per tool with its call count. The
  agent wrote the call; execution and success are not implied;
- USAGE: Usage rows beyond the primary row, as one ``provider_backed_call``
  record. Only recorded when the primary row id is known; the tool that made
  the provider calls is never attributed.

Records are listed by source (declaration order), then name, then outcome.
"""

import re

from evaluation.datachat.grading.model import ActivitySource, ObservedActivity
from evaluation.datachat.grading.reader import ScenarioEvidence
from evaluation.datachat.model import Availability

# ``name`` of each CURRENT DataChat tool (``datachat/tools/*_tool.py``).
DATACHAT_TOOLS = (
    "aggregate", "chart", "classify_match", "compare_groups", "correlation", "crosstab", "describe",
    "filter_rows", "keywords", "missing_values", "plot", "row_count", "sample_rows", "sentiment_analysis",
    "sql_engine", "top_rows", "trend", "unique_values",
)

PROVIDER_BACKED_CALL = "provider_backed_call"

# A bare call: not an attribute (``df.plot(``) and not part of a longer identifier.
_CALL = {tool: re.compile(rf"(?<![\w.]){re.escape(tool)}\s*\(") for tool in DATACHAT_TOOLS}


def _runtime(ev: ScenarioEvidence) -> list[ObservedActivity]:
    counts: dict[tuple[str, str], int] = {}
    for event in ev.tools.runtime_events:
        key = (event.tool, event.outcome)
        counts[key] = counts.get(key, 0) + 1
    return [ObservedActivity(tool, ActivitySource.RUNTIME, n, outcome) for (tool, outcome), n in counts.items()]


def _generated_code(ev: ScenarioEvidence) -> list[ObservedActivity]:
    if ev.agent_run is None or not ev.agent_run.parse_ok:
        return []
    out = []
    for tool, pattern in _CALL.items():
        n = sum(len(pattern.findall(block)) for block in ev.agent_run.code_blocks)
        if n:
            out.append(ObservedActivity(tool, ActivitySource.GENERATED_CODE, n))
    return out


def _usage(ev: ScenarioEvidence) -> list[ObservedActivity]:
    usage = ev.usage
    if usage is None or usage.availability is not Availability.PRESENT or usage.primary_log_id is None:
        return []
    n = len(usage.extra_rows)
    return [ObservedActivity(PROVIDER_BACKED_CALL, ActivitySource.USAGE, n)] if n else []


_SOURCE_ORDER = {source: i for i, source in enumerate(ActivitySource)}


def observed_activity(ev: ScenarioEvidence) -> tuple[ObservedActivity, ...]:
    """Deterministic, aggregated activity summary. Empty when nothing was observed or loaded."""
    if ev.tools is None:
        return ()
    records = _runtime(ev) + _generated_code(ev) + _usage(ev)
    return tuple(sorted(records, key=lambda a: (_SOURCE_ORDER[a.source], a.name, a.outcome or "")))
