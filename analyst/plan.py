"""
plan.py
-------

The plan: what the analyst will do, in one fixed shape, whatever the brief looks like.

A brief is written by the interviewer's model, so its shape is not guaranteed:
keys get renamed, nested, dropped or added from one brief to the next. Nothing
after planning reads the brief's keys. The planner agent (planner_agent.py)
reads the brief, whatever its shape, and writes a plan; code checks it here:

    {"title", "language", "audience",
     "relations": [relation names the analysis reads],
     "tasks": [{"id", "name", "mission", "depends_on": [earlier ids],
                "outputs": [keys], "exploratory": bool}],
     "charts": [{"id", "description"}],
     "report": {"sections": [...], "tone", "length"},
     "metrics_config": null | {"metrics": {key: {label, type, better}},
                               "categories": {key: {label, color, icon}}}}

validate_plan() is the planner's guard. plan_from_brief() is the fallback when
the planner fails: plain code, tolerant of any shape, that always returns a
valid plan, so no brief can stop a run. brief_context_text() is how the brief
itself reaches the task agents: a compact text of every key that is not about
tasks, charts or the report, whatever those keys are called.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

MAX_TASKS = 10

METRIC_TYPES = ("currency", "number", "percent")
METRIC_BETTER = ("higher", "lower")
CATEGORY_COLORS = ("danger", "warning", "success", "info", "secondary")

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Brief keys, matched case-insensitively, that hold the work, the charts, the
# report and the tracked metrics. Best effort: an unknown name only means the
# planner has to find it, and that the context text carries it too.
_TASK_KEYS = ("sub_tasks", "subtasks", "tasks", "phases", "steps", "analysis_phases", "plan")
_REPORT_KEYS = ("report", "report_requirements", "output_requirements", "deliverable")
_CHART_KEYS = ("charts", "graphs", "visualizations", "visualisations", "plots")
_METRIC_KEYS = ("metrics_config", "metrics", "kpis", "kpi")
_NOT_CONTEXT = {*_TASK_KEYS, *_REPORT_KEYS, *_CHART_KEYS, *_METRIC_KEYS}

# Fields a task-like dict may carry its instructions under.
_MISSION_FIELDS = ("mission", "description", "instructions", "goal", "objective", "task")


# ----------------------------
# Small helpers
# ----------------------------

def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def slug(value: Any, fallback: str) -> str:
    """'Baseline 2026!' -> 'baseline_2026'; a leading digit gets a 't_' prefix."""
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        return fallback
    return text if text[0].isalpha() else f"t_{text}"


def _get(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    """The value of the first key matching one of `names`, case-insensitively."""
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _texts(value: Any) -> list[str]:
    """A list of strings out of a string, a list of strings or a list of dicts."""
    if _is_text(value):
        return [value.strip()]
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if _is_text(item):
            items.append(item.strip())
        elif isinstance(item, dict):
            text = _get(item, ("description", "title", "name", "chart", "text", "type"))
            items.append(str(text).strip() if text else json.dumps(item, ensure_ascii=False))
    return [i for i in items if i]


def _is_task_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(t, dict) for t in value)
        and any(_get(t, _MISSION_FIELDS) for t in value)
    )


def find_task_list(brief: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The brief's task list: under a known key, one level down, or any list of task-like dicts."""
    value = _get(brief, _TASK_KEYS)
    if _is_task_list(value):
        return value
    for nested in brief.values():
        if isinstance(nested, dict):
            value = _get(nested, _TASK_KEYS)
            if _is_task_list(value):
                return value
    for value in brief.values():
        if _is_task_list(value):
            return value
    return []


def brief_task_ids(brief: Mapping[str, Any]) -> list[str]:
    """The ids the brief gave its tasks, which a plan must keep."""
    return [t["id"] for t in find_task_list(brief) if isinstance(t.get("id"), str) and t["id"].strip()]


def _report_part(brief: Mapping[str, Any]) -> dict[str, Any]:
    report = _get(brief, _REPORT_KEYS)
    return report if isinstance(report, dict) else {}


# ----------------------------
# Validation
# ----------------------------

def normalize_plan(plan: Any) -> Any:
    """Fill the harmless defaults (empty lists, exploratory=False) before validation."""
    if not isinstance(plan, dict):
        return plan
    plan = dict(plan)
    plan.setdefault("audience", "")
    plan.setdefault("relations", [])
    plan.setdefault("metrics_config", None)
    report = plan.get("report") if isinstance(plan.get("report"), dict) else {}
    plan["report"] = {
        "sections": report.get("sections") or [],
        "tone": report.get("tone") or "",
        "length": report.get("length") or "",
    }
    if isinstance(plan.get("tasks"), list):
        plan["tasks"] = [
            {"depends_on": [], "exploratory": False, **t} if isinstance(t, dict) else t
            for t in plan["tasks"]
        ]
    return plan


def _metrics_errors(config: Any, output_keys: set[str]) -> list[str]:
    if config is None:
        return []
    if not isinstance(config, dict):
        return ["metrics_config must be an object or null."]
    errors = []
    sections = (
        ("metrics", {"type": METRIC_TYPES, "better": METRIC_BETTER}),
        ("categories", {"color": CATEGORY_COLORS}),
    )
    for section, enums in sections:
        entries = config.get(section) or {}
        if not isinstance(entries, dict):
            errors.append(f"metrics_config.{section} must be an object.")
            continue
        for key, spec in entries.items():
            where = f"metrics_config.{section}.{key}"
            if key not in output_keys:
                errors.append(f"{where}: key must be one of the tasks' output keys.")
            if not isinstance(spec, dict) or not _is_text(spec.get("label")):
                errors.append(f"{where} must be an object with a label.")
                continue
            for field, allowed in enums.items():
                if spec.get(field) not in allowed:
                    errors.append(f"{where}.{field} must be one of: {', '.join(allowed)}.")
    return errors


def validate_plan(
    plan: Any,
    *,
    identifiers: Mapping[str, Iterable[str]],
    brief_ids: Iterable[str] = (),
    max_extra: int = 0,
) -> list[str]:
    """
    Errors in a plan, worded for the planner (who receives them verbatim);
    empty when it is valid. Call normalize_plan() first.

    :param identifiers: the visible relations (name -> columns).
    :param brief_ids: task ids the brief defined: the plan must keep them all.
    :param max_extra: how many exploratory tasks the plan may add.
    """
    if not isinstance(plan, dict):
        return ["The plan must be a JSON object."]
    errors: list[str] = []

    for field in ("title", "language"):
        if not _is_text(plan.get(field)):
            errors.append(f"{field} must be a non-empty string.")

    tasks = plan.get("tasks")
    output_keys: set[str] = set()
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
        errors.append(f"tasks must be a list of 1 to {MAX_TASKS} tasks.")
        tasks = []

    seen: list[str] = []
    exploratory_ids: set[str] = set()
    exploratory = 0
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"tasks[{i}] must be an object.")
            continue
        task_id = task.get("id")
        where = f"tasks[{task_id or i}]"
        if not isinstance(task_id, str) or not _SNAKE_RE.match(task_id) or task_id in seen:
            errors.append(f"{where}.id must be a unique snake_case string.")
        for field in ("name", "mission"):
            if not _is_text(task.get(field)):
                errors.append(f"{where}.{field} must be a non-empty string.")
        depends_on = task.get("depends_on")
        if not isinstance(depends_on, list) or any(d not in seen for d in depends_on):
            errors.append(f"{where}.depends_on must list only tasks that come before it.")
        elif task.get("exploratory") is not True and (optional := [d for d in depends_on if d in exploratory_ids]):
            # A failed task skips its dependents: an optional task must never take
            # a task of the brief down with it.
            errors.append(
                f"{where} is a brief task and may not depend on exploratory tasks "
                f"({', '.join(optional)}): they are optional extras, and their findings "
                "reach the report as insights."
            )
        outputs = task.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            errors.append(f"{where}.outputs must be a non-empty list of keys.")
        else:
            for key in outputs:
                if not isinstance(key, str) or not _SNAKE_RE.match(key):
                    errors.append(f"{where}: output key {key!r} must be snake_case.")
                elif key in output_keys:
                    errors.append(f"{where}: output key {key!r} is already produced by another task.")
                else:
                    output_keys.add(key)
        if task.get("exploratory") is True:
            exploratory += 1
            if isinstance(task_id, str):
                exploratory_ids.add(task_id)
        if isinstance(task_id, str):
            seen.append(task_id)

    if exploratory > max_extra:
        errors.append(f"At most {max_extra} tasks may be exploratory, the plan has {exploratory}.")
    missing = [t for t in brief_ids if t not in seen]
    if missing:
        errors.append(f"The brief's tasks must all be kept (same ids); missing: {', '.join(missing)}.")

    relations = plan.get("relations")
    if not isinstance(relations, list) or not all(isinstance(r, str) for r in relations):
        errors.append("relations must be a list of relation names.")
    else:
        known = {name.casefold() for name in identifiers}
        unknown = [r for r in relations if r.casefold() not in known]
        if unknown:
            errors.append(f"Unknown relations {unknown}: use names from the schema.")

    charts = plan.get("charts")
    if not isinstance(charts, list) or not charts:
        errors.append("charts must list at least one chart: design them if the brief names none.")
    else:
        ids = [c.get("id") for c in charts if isinstance(c, dict)]
        if len(ids) != len(charts) or len(set(ids)) != len(ids) or not all(_is_text(c) for c in ids):
            errors.append("Every chart must be an object with a unique id.")
        if not all(isinstance(c, dict) and _is_text(c.get("description")) for c in charts):
            errors.append("Every chart needs a description of what it shows.")

    report = plan.get("report")
    if not isinstance(report, dict) or not isinstance(report.get("sections"), list):
        errors.append("report must be an object with a sections list.")

    errors.extend(_metrics_errors(plan.get("metrics_config"), output_keys))
    return errors


# ----------------------------
# Fallback: a plan from the brief, by code
# ----------------------------

def _keys(raw: Any) -> list[str]:
    keys = [slug(k, "") for k in raw if isinstance(k, str)] if isinstance(raw, list) else []
    return list(dict.fromkeys(k for k in keys if k))


def _outputs(task: Mapping[str, Any], task_id: str) -> tuple[list[str], list[str]]:
    """(required, optional) output keys: a list, a {required, optional} dict, or none."""
    raw = task.get("outputs")
    if isinstance(raw, dict):
        return _keys(raw.get("required")) or [f"{task_id}_result"], _keys(raw.get("optional"))
    return _keys(raw) or [f"{task_id}_result"], []


def _fallback_tasks(brief: Mapping[str, Any]) -> list[dict[str, Any]]:
    def order(task: dict[str, Any]) -> float:
        value = task.get("order")
        return float(value) if isinstance(value, (int, float)) else float("inf")

    # sorted() is stable: without any order the brief's own order stands.
    found = sorted(find_task_list(brief), key=order)

    tasks: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for i, raw in enumerate(found[:MAX_TASKS]):
        name = _get(raw, ("name", "title"))
        task_id = slug(raw.get("id") or name, f"task_{i + 1}")
        if any(t["id"] == task_id for t in tasks):
            task_id = f"{task_id}_{i + 1}"
        mission = _get(raw, _MISSION_FIELDS)
        earlier = [t["id"] for t in tasks]
        depends = raw.get("depends_on")
        if isinstance(depends, list):
            depends = [slug(d, "") for d in depends if slug(d, "") in earlier]
        else:
            # No dependencies stated: the previous task's results are the safest input.
            depends = earlier[-1:]
        required, optional = _outputs(raw, task_id)
        outputs = [k for k in required if k not in used_keys] or [f"{task_id}_result"]
        used_keys.update(outputs)
        mission = str(mission).strip() if mission else json.dumps(raw, ensure_ascii=False)
        if optional:
            # A plan has required outputs only: optional ones become a request in the mission.
            mission += f"\nAlso save, when available: {', '.join(optional)}."
        tasks.append({
            "id": task_id,
            "name": str(name or task_id).strip(),
            "mission": mission,
            "depends_on": depends,
            "outputs": outputs,
            "exploratory": False,
        })

    if not tasks:
        tasks = [
            {"id": "analysis", "name": "Analysis",
             "mission": "Carry out the analysis the brief context describes: compute its figures "
                        "from the data and save them.",
             "depends_on": [], "outputs": ["key_figures", "analysis_findings"], "exploratory": False},
            {"id": "final_summary", "name": "Summary",
             "mission": "Synthesize the key findings of the analysis, with the figures that support them.",
             "depends_on": ["analysis"], "outputs": ["key_findings"], "exploratory": False},
        ]
    return tasks


def _relations(brief: Mapping[str, Any], identifiers: Mapping[str, Iterable[str]]) -> list[str]:
    """Visible relations the brief names, wherever it names them."""
    known = {name.casefold(): name for name in identifiers}
    text = json.dumps(brief, ensure_ascii=False).casefold()
    return [real for folded, real in known.items() if f'"{folded}"' in text or f"\\\"{folded}\\\"" in text]


def plan_from_brief(brief: Mapping[str, Any], identifiers: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """A valid plan from any brief, by code alone. Used when the planner fails."""
    report = _report_part(brief)
    tasks = _fallback_tasks(brief)

    charts = _texts(_get(report, _CHART_KEYS)) or _texts(_get(brief, _CHART_KEYS))
    if not charts:
        charts = ["The chart that best shows the main result of the analysis."]

    language = _get(brief, ("language", "report_language")) or _get(report, ("language",))
    plan = normalize_plan({
        "title": str(_get(brief, ("name", "title", "subject")) or "Analysis").strip(),
        "language": str(language).strip() if _is_text(language) else "English",
        "audience": str(_get(brief, ("audience",)) or _get(report, ("audience",)) or ""),
        "relations": _relations(brief, identifiers),
        "tasks": tasks,
        "charts": [{"id": f"c{i + 1}", "description": d} for i, d in enumerate(charts[:20])],
        "report": {
            "sections": _texts(_get(report, ("sections",)) or _get(brief, ("sections",))),
            "tone": str(_get(report, ("tone",)) or ""),
            "length": str(_get(report, ("length",)) or ""),
        },
        "metrics_config": _get(brief, _METRIC_KEYS),
    })
    output_keys = {k for t in tasks for k in t["outputs"]}
    if _metrics_errors(plan["metrics_config"], output_keys):
        plan["metrics_config"] = None
    return plan


# ----------------------------
# The brief as context
# ----------------------------

def _render(value: Any, indent: str = "") -> list[str]:
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{indent}{key}:")
                lines.extend(_render(item, indent + "  "))
            else:
                lines.append(f"{indent}{key}: {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = _render(item, indent + "  ")
                lines.append(f"{indent}- " + (rendered[0].strip() if rendered else ""))
                lines.extend(rendered[1:])
            else:
                lines.append(f"{indent}- {item}")
        return lines
    return [f"{indent}{value}"]


def brief_context_text(brief: Mapping[str, Any], max_chars: int = 12000) -> str:
    """
    The brief as the task agents read it: every key except the tasks, charts,
    report and metrics (the plan carries those), compact and capped.
    """
    context = {k: v for k, v in brief.items() if str(k).lower() not in _NOT_CONTEXT}
    text = "\n".join(_render(context)).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n… [brief context truncated]"
    return text or "(the brief carries no further context)"

