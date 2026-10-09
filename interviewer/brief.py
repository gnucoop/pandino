"""
brief.py
--------

The analysis brief: what the interviewer produces and the data analyst consumes.

A brief is a plain JSON-compatible dict, modelled on digest-pipeline's PROFILE
and adapted to an analyst that reads the database directly. It is written by
the model, so nothing about it is trusted: validate_brief() checks every field
against the schema below *and* against the database the interviewer was shown
— a brief naming a column that does not exist would send the analyst after
data that is not there. Its error list is replayed to the model, which fixes
the brief and proposes it again (the validate -> repair loop digest-pipeline
runs by hand).

render_brief_prompt() turns a valid brief into the long prompt text handed to
the data analyst. The dict stays the source of truth; the text is derived.

Shape (every key required unless marked optional):

    name, description, category, tags[], language, audience, business_context
    data_scope: {relations: [{name, columns[], role?}], joins[]?, filters[]?,
                 time_range?, granularity?, sql_hints[]?}
    data_notes[]
    mission
    sub_tasks: [{id, name, order, depends_on[], mission,
                 outputs: {required[], optional[]?}}]      last id: final_summary
    metrics_config: null | {categories: {key: {label, color, icon}},
                            metrics:    {key: {label, type, better}}}
    report: {format: "html", sections[], charts[], tone, length}
    open_questions[]
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from datachat.sql_guard import validate_select
from datachat.sql_identifiers import quote_ident

BRIEF_CATEGORIES = ("discovery", "relationships", "quality", "prediction", "segmentation")
METRIC_COLORS = ("danger", "warning", "success", "info", "secondary")
METRIC_TYPES = ("currency", "number", "percent")
METRIC_BETTER = ("higher", "lower")
REPORT_FORMATS = ("html",)

FINAL_TASK_ID = "final_summary"
MIN_SUB_TASKS = 3
MAX_SUB_TASKS = 8
MIN_TAGS = 3
MAX_TAGS = 8

_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_REQUIRED_TEXT_FIELDS = (
    "name",
    "description",
    "category",
    "language",
    "audience",
    "business_context",
    "mission",
)


# ----------------------------
# Small checks
# ----------------------------

def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text_list_errors(value: Any, field: str, *, required: bool = False) -> list[str]:
    """Errors for a field that must be a list of non-blank strings."""
    if value is None and not required:
        return []
    if not isinstance(value, list):
        return [f"{field} must be a list of strings."]
    if required and not value:
        return [f"{field} must not be empty."]
    if any(not _is_text(item) for item in value):
        return [f"{field} must contain only non-empty strings."]
    return []


def _bare_name(name: str) -> str:
    """A relation name as the model may write it: '"Clienti"' -> 'Clienti'."""
    return str(name).strip().strip('"')


def _resolve_relation(
    name: str, identifiers: Mapping[str, Sequence[str]]
) -> Optional[str]:
    """The real relation name for `name`, matched exactly, then case-insensitively."""
    bare = _bare_name(name)
    if bare in identifiers:
        return bare
    folded = {real.lower(): real for real in identifiers}
    return folded.get(bare.lower())


def _cte_names(sql: str) -> set[str]:
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except SqlglotError:
        return set()
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}


# ----------------------------
# Section validators
# ----------------------------

def _validate_data_scope(
    scope: Any, identifiers: Mapping[str, Sequence[str]]
) -> tuple[list[str], set[str]]:
    """
    Errors for data_scope, and the real column names it references.

    The column set is what the mission is later checked against.
    """
    if not isinstance(scope, dict):
        return ["data_scope must be an object."], set()

    errors: list[str] = []
    columns_in_scope: set[str] = set()

    relations = scope.get("relations")
    if not isinstance(relations, list) or not relations:
        errors.append("data_scope.relations must be a non-empty list.")
        relations = []

    for i, relation in enumerate(relations):
        where = f"data_scope.relations[{i}]"
        if not isinstance(relation, dict) or not _is_text(relation.get("name")):
            errors.append(f"{where} must be an object with a non-empty name.")
            continue

        real = _resolve_relation(relation["name"], identifiers)
        if real is None:
            errors.append(
                f"{where}: relation {relation['name']!r} is not in the database schema. "
                "Use only relations listed in the schema."
            )
            continue

        columns = relation.get("columns")
        if not isinstance(columns, list) or not columns:
            errors.append(f"{where} ({real}): columns must be a non-empty list.")
            continue

        known = {c: c for c in identifiers[real]}
        folded = {c.lower(): c for c in identifiers[real]}
        for column in columns:
            bare = _bare_name(column) if isinstance(column, str) else ""
            real_column = known.get(bare) or folded.get(bare.lower())
            if real_column is None:
                errors.append(
                    f"{where}: column {column!r} does not exist in {real}. "
                    f"Its columns are: {', '.join(identifiers[real])}."
                )
            else:
                columns_in_scope.add(real_column)

    for field in ("joins", "filters"):
        errors.extend(_text_list_errors(scope.get(field), f"data_scope.{field}"))

    for field in ("time_range", "granularity"):
        value = scope.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"data_scope.{field} must be a string or null.")

    errors.extend(_validate_sql_hints(scope.get("sql_hints"), identifiers))
    return errors, columns_in_scope


def _validate_sql_hints(
    hints: Any, identifiers: Mapping[str, Sequence[str]]
) -> list[str]:
    """
    Every reference query must pass the same guard sql_engine applies, and read
    only relations the interviewer was shown.
    """
    errors = _text_list_errors(hints, "data_scope.sql_hints")
    if errors or not hints:
        return errors

    known = {name.lower() for name in identifiers}
    for i, sql in enumerate(hints):
        result = validate_select(sql)
        if not result.ok:
            errors.append(
                f"data_scope.sql_hints[{i}] is not an accepted read-only query "
                f"({result.code}): {result.message}"
            )
            continue
        unknown = sorted(set(result.tables) - known - _cte_names(sql))
        if unknown:
            errors.append(
                f"data_scope.sql_hints[{i}] reads relations not in the schema: "
                f"{', '.join(unknown)}."
            )
    return errors


def _validate_sub_tasks(sub_tasks: Any) -> tuple[list[str], set[str]]:
    """Errors for sub_tasks, and every output key they declare."""
    if not isinstance(sub_tasks, list):
        return ["sub_tasks must be a list."], set()
    if not MIN_SUB_TASKS <= len(sub_tasks) <= MAX_SUB_TASKS:
        return [
            f"sub_tasks must hold between {MIN_SUB_TASKS} and {MAX_SUB_TASKS} "
            f"tasks, got {len(sub_tasks)}."
        ], set()

    errors: list[str] = []
    output_keys: set[str] = set()
    orders: dict[str, int] = {}

    for i, task in enumerate(sub_tasks):
        where = f"sub_tasks[{i}]"
        if not isinstance(task, dict):
            errors.append(f"{where} must be an object.")
            continue

        task_id = task.get("id")
        if not isinstance(task_id, str) or not _SNAKE_RE.match(task_id):
            errors.append(f"{where}.id must be a snake_case string.")
            continue
        where = f"sub_tasks[{task_id}]"
        if task_id in orders:
            errors.append(f"{where}: duplicate id.")
            continue

        order = task.get("order")
        if not isinstance(order, int) or isinstance(order, bool):
            errors.append(f"{where}.order must be an integer.")
            order = i + 1
        if order in orders.values():
            errors.append(f"{where}.order {order} is used by another task.")
        orders[task_id] = order

        if not _is_text(task.get("name")):
            errors.append(f"{where}.name must be a non-empty string.")
        if not _is_text(task.get("mission")):
            errors.append(f"{where}.mission must be a non-empty string.")

        outputs = task.get("outputs")
        if not isinstance(outputs, dict):
            errors.append(f"{where}.outputs must be an object with 'required' keys.")
            continue
        required = outputs.get("required")
        optional = outputs.get("optional") or []
        if not isinstance(required, list) or not required:
            errors.append(f"{where}.outputs.required must be a non-empty list.")
            required = []
        if not isinstance(optional, list):
            errors.append(f"{where}.outputs.optional must be a list.")
            optional = []
        for key in [*required, *optional]:
            if not isinstance(key, str) or not _SNAKE_RE.match(key):
                errors.append(f"{where}: output key {key!r} must be snake_case.")
            else:
                output_keys.add(key)

    # Dependencies are checked once every id and order is known.
    for task in sub_tasks:
        if not isinstance(task, dict) or task.get("id") not in orders:
            continue
        task_id = task["id"]
        depends_on = task.get("depends_on")
        if not isinstance(depends_on, list):
            errors.append(f"sub_tasks[{task_id}].depends_on must be a list.")
            continue
        for dep in depends_on:
            if dep not in orders:
                errors.append(f"sub_tasks[{task_id}] depends on unknown task {dep!r}.")
            elif orders[dep] >= orders[task_id]:
                errors.append(
                    f"sub_tasks[{task_id}] depends on {dep!r}, which does not come before it."
                )

    if orders and max(orders, key=orders.__getitem__) != FINAL_TASK_ID:
        errors.append(
            f"The last sub-task (highest order) must be {FINAL_TASK_ID!r}: it "
            "synthesizes the findings of the previous tasks."
        )
    return errors, output_keys


def _validate_metrics_config(config: Any, output_keys: set[str]) -> list[str]:
    """
    Metric and category keys must match sub-task output keys exactly: the
    analyst extracts their values from the task outputs by key name.
    """
    if config is None:
        return []
    if not isinstance(config, dict):
        return ["metrics_config must be an object or null."]

    errors: list[str] = []
    sections = (
        ("categories", {"color": METRIC_COLORS}, ("label", "icon")),
        ("metrics", {"type": METRIC_TYPES, "better": METRIC_BETTER}, ("label",)),
    )
    for section, enums, texts in sections:
        entries = config.get(section) or {}
        if not isinstance(entries, dict):
            errors.append(f"metrics_config.{section} must be an object.")
            continue
        for key, spec in entries.items():
            where = f"metrics_config.{section}.{key}"
            if key not in output_keys:
                errors.append(
                    f"{where}: key must match a sub-task output key exactly "
                    f"(known keys: {', '.join(sorted(output_keys)) or 'none'})."
                )
            if not isinstance(spec, dict):
                errors.append(f"{where} must be an object.")
                continue
            for field in texts:
                if not _is_text(spec.get(field)):
                    errors.append(f"{where}.{field} must be a non-empty string.")
            for field, allowed in enums.items():
                if spec.get(field) not in allowed:
                    errors.append(f"{where}.{field} must be one of: {', '.join(allowed)}.")
    return errors


def _validate_report(report: Any) -> list[str]:
    if not isinstance(report, dict):
        return ["report must be an object."]
    errors: list[str] = []
    if report.get("format") not in REPORT_FORMATS:
        errors.append(f"report.format must be one of: {', '.join(REPORT_FORMATS)}.")
    errors.extend(_text_list_errors(report.get("sections"), "report.sections", required=True))
    errors.extend(_text_list_errors(report.get("charts"), "report.charts", required=True))
    for field in ("tone", "length"):
        if not _is_text(report.get(field)):
            errors.append(f"report.{field} must be a non-empty string.")
    return errors


# ----------------------------
# Public API
# ----------------------------

def validate_brief(
    brief: Any, identifiers: Mapping[str, Sequence[str]]
) -> list[str]:
    """
    Check a brief against its schema and against the database schema.

    :param brief: the brief as produced by the model.
    :param identifiers: real relation name -> real column names, exactly the
        relations the interviewer was shown (allow/deny filtering included).
    :return: human-readable errors, empty when the brief is valid. Worded for
        the model, which receives them verbatim.
    """
    if not isinstance(brief, dict):
        return ["The brief must be a JSON object."]

    errors: list[str] = []

    for field in _REQUIRED_TEXT_FIELDS:
        if not _is_text(brief.get(field)):
            errors.append(f"{field} must be a non-empty string.")

    if _is_text(brief.get("category")) and brief["category"] not in BRIEF_CATEGORIES:
        errors.append(f"category must be one of: {', '.join(BRIEF_CATEGORIES)}.")

    tags = brief.get("tags")
    tag_errors = _text_list_errors(tags, "tags", required=True)
    if not tag_errors and not MIN_TAGS <= len(tags) <= MAX_TAGS:
        tag_errors.append(f"tags must hold between {MIN_TAGS} and {MAX_TAGS} keywords.")
    errors.extend(tag_errors)

    scope_errors, columns_in_scope = _validate_data_scope(brief.get("data_scope"), identifiers)
    errors.extend(scope_errors)

    mission = brief.get("mission")
    if _is_text(mission) and columns_in_scope and not any(
        column in mission for column in columns_in_scope
    ):
        errors.append(
            "mission must reference the real columns it analyses by name "
            "(none of the data_scope columns appears in it)."
        )

    errors.extend(_text_list_errors(brief.get("data_notes"), "data_notes", required=True))

    sub_task_errors, output_keys = _validate_sub_tasks(brief.get("sub_tasks"))
    errors.extend(sub_task_errors)

    if "metrics_config" not in brief:
        errors.append("metrics_config is required (use null when no metrics are tracked).")
    else:
        errors.extend(_validate_metrics_config(brief["metrics_config"], output_keys))

    errors.extend(_validate_report(brief.get("report")))
    errors.extend(_text_list_errors(brief.get("open_questions"), "open_questions"))
    if "open_questions" not in brief:
        errors.append("open_questions is required (use [] when there are none).")

    return errors


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"


def render_brief_prompt(brief: Mapping[str, Any]) -> str:
    """
    Render a validated brief into the prompt text for the data analyst.

    :param brief: a brief for which validate_brief() returned no errors.
    :return: the full prompt, as plain text with markdown-style headings.
    """
    scope = brief.get("data_scope") or {}
    parts: list[str] = [
        f"# ANALYSIS BRIEF: {brief['name']}",
        brief["description"],
        f"Category: {brief['category']} | Tags: {', '.join(brief.get('tags') or [])}",
        f"Report language: {brief['language']} | Audience: {brief['audience']}",
        "## BUSINESS CONTEXT",
        brief["business_context"],
        "## MISSION",
        brief["mission"],
        "## DATA SCOPE",
    ]

    relation_lines = []
    for relation in scope.get("relations") or []:
        columns = ", ".join(quote_ident(_bare_name(c)) for c in relation.get("columns") or [])
        role = f" ({relation['role']})" if relation.get("role") else ""
        relation_lines.append(f"{quote_ident(_bare_name(relation['name']))}{role}: {columns}")
    parts.append("Relations:\n" + _bullets(relation_lines))
    if scope.get("joins"):
        parts.append("Joins:\n" + _bullets(scope["joins"]))
    if scope.get("filters"):
        parts.append("Filters:\n" + _bullets(scope["filters"]))
    if scope.get("time_range"):
        parts.append(f"Time range: {scope['time_range']}")
    if scope.get("granularity"):
        parts.append(f"Granularity: {scope['granularity']}")
    if scope.get("sql_hints"):
        parts.append(
            "Reference queries (validated read-only SELECTs, a starting point, not the analysis):\n"
            + "\n\n".join(f"```sql\n{sql.strip()}\n```" for sql in scope["sql_hints"])
        )

    parts.extend(["## DATA NOTES", _bullets(brief.get("data_notes") or [])])

    parts.append("## ANALYSIS PHASES")
    for task in sorted(brief.get("sub_tasks") or [], key=lambda t: t["order"]):
        depends = ", ".join(task.get("depends_on") or []) or "none"
        outputs = task.get("outputs") or {}
        lines = [
            f"### {task['order']}. {task['name']} [{task['id']}]",
            f"Depends on: {depends}",
            task["mission"],
            f"Required outputs: {', '.join(outputs.get('required') or [])}",
        ]
        if outputs.get("optional"):
            lines.append(f"Optional outputs: {', '.join(outputs['optional'])}")
        parts.append("\n".join(lines))

    metrics_config = brief.get("metrics_config")
    if metrics_config:
        lines = []
        for key, spec in (metrics_config.get("metrics") or {}).items():
            lines.append(
                f"{key}: {spec['label']} ({spec['type']}, {spec['better']} is better)"
            )
        for key, spec in (metrics_config.get("categories") or {}).items():
            lines.append(f"{key}: category {spec['icon']} {spec['label']} ({spec['color']})")
        parts.extend(["## TRACKED METRICS", _bullets(lines)])

    report = brief.get("report") or {}
    parts.extend(
        [
            "## REPORT REQUIREMENTS",
            f"Format: {str(report.get('format', '')).upper()}\n"
            f"Tone: {report.get('tone', '')}\n"
            f"Length: {report.get('length', '')}",
            "Sections:\n" + _bullets(report.get("sections") or []),
            "Charts:\n" + _bullets(report.get("charts") or []),
            "## OPEN QUESTIONS AND ASSUMPTIONS",
            _bullets(brief.get("open_questions") or []),
        ]
    )
    return "\n\n".join(parts) + "\n"
