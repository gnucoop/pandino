"""
Tests for the analysis brief schema: a brief is written by the model, so every
field is checked, and so is every relation, column and query it names against
the schema the interviewer was shown.
"""

import copy

import pytest

from interviewer.brief import render_brief_prompt, validate_brief

IDENTIFIERS = {
    "Ordini": ("id", "Cliente", "Importo", "DataOrdine"),
    "clienti": ("id", "regione"),
}


def make_brief(**overrides):
    brief = {
        "name": "Revenue by region",
        "description": "Monthly revenue trends per region, with the top clients.",
        "category": "discovery",
        "tags": ["revenue", "regions", "trends"],
        "language": "Italian",
        "audience": "Sales management",
        "business_context": "The sales team plans next year's regional targets.",
        "data_scope": {
            "relations": [
                {"name": "Ordini", "columns": ["Cliente", "Importo", "DataOrdine"], "role": "facts"},
                {"name": "clienti", "columns": ["id", "regione"], "role": "dimension"},
            ],
            "joins": ['"Ordini"."Cliente" = "clienti"."id"'],
            "filters": ['"DataOrdine" >= 2023-01-01'],
            "time_range": "2023-01 to 2025-12",
            "granularity": "month",
            "sql_hints": [
                'SELECT "regione", SUM("Importo") FROM "Ordini" o '
                'JOIN "clienti" c ON o."Cliente" = c."id" GROUP BY 1'
            ],
        },
        "data_notes": ["Importo is in EUR, VAT excluded."],
        "mission": "Analyse monthly Importo per regione over DataOrdine and flag declines.",
        "sub_tasks": [
            {
                "id": "data_preparation", "name": "Data preparation", "order": 1,
                "depends_on": [], "mission": "Check nulls in Importo.",
                "outputs": {"required": ["row_count"], "optional": []},
            },
            {
                "id": "regional_trends", "name": "Regional trends", "order": 2,
                "depends_on": ["data_preparation"], "mission": "Monthly totals per regione.",
                "outputs": {"required": ["total_revenue", "declining_regions"]},
            },
            {
                "id": "final_summary", "name": "Summary", "order": 3,
                "depends_on": ["regional_trends"], "mission": "Synthesize findings.",
                "outputs": {"required": ["key_findings"]},
            },
        ],
        "metrics_config": {
            "categories": {
                "declining_regions": {"label": "Declining", "color": "danger", "icon": "🔴"}
            },
            "metrics": {
                "total_revenue": {"label": "Total revenue", "type": "currency", "better": "higher"}
            },
        },
        "report": {
            "format": "html",
            "sections": ["Executive summary", "Regional trends"],
            "charts": ["Monthly revenue line chart per region"],
            "tone": "executive",
            "length": "short",
        },
        "open_questions": [],
    }
    brief.update(overrides)
    return brief


def _errors_for(mutate):
    brief = make_brief()
    mutate(brief)
    return validate_brief(brief, IDENTIFIERS)


# ---------------------------------------------------------------------------
# A valid brief
# ---------------------------------------------------------------------------


def test_a_complete_brief_is_valid():
    assert validate_brief(make_brief(), IDENTIFIERS) == []


def test_relation_and_column_names_match_case_insensitively_and_quoted():
    brief = make_brief()
    brief["data_scope"]["relations"][0] = {"name": '"ordini"', "columns": ['"importo"']}

    assert validate_brief(brief, IDENTIFIERS) == []


def test_metrics_config_may_be_null():
    assert validate_brief(make_brief(metrics_config=None), IDENTIFIERS) == []


def test_a_cte_in_a_reference_query_is_not_mistaken_for_a_table():
    brief = make_brief()
    brief["data_scope"]["sql_hints"] = [
        'WITH monthly AS (SELECT "Importo" FROM "Ordini") SELECT * FROM monthly'
    ]

    assert validate_brief(brief, IDENTIFIERS) == []


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_not_an_object_is_rejected():
    assert validate_brief(["nope"], IDENTIFIERS) == ["The brief must be a JSON object."]


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda b: b.pop("mission"), "mission must be a non-empty string"),
        (lambda b: b.update(category="astrology"), "category must be one of"),
        (lambda b: b.update(tags=["one"]), "tags must hold between"),
        (
            lambda b: b["data_scope"]["relations"].append({"name": "segreti", "columns": ["x"]}),
            "relation 'segreti' is not in the database schema",
        ),
        (
            lambda b: b["data_scope"]["relations"][0]["columns"].append("Sconto"),
            "column 'Sconto' does not exist in Ordini",
        ),
        (
            lambda b: b["data_scope"].update(sql_hints=['DELETE FROM "Ordini"']),
            "is not an accepted read-only query",
        ),
        (
            lambda b: b["data_scope"].update(sql_hints=['SELECT * FROM "segreti"']),
            "reads relations not in the schema: segreti",
        ),
        (
            lambda b: b.update(mission="Look at the data and say something clever."),
            "mission must reference the real columns",
        ),
        (
            lambda b: b["sub_tasks"][2].update(id="wrap_up"),
            "must be 'final_summary'",
        ),
        (
            lambda b: b["sub_tasks"][1].update(depends_on=["ghost_task"]),
            "depends on unknown task 'ghost_task'",
        ),
        (
            lambda b: b["sub_tasks"][0].update(depends_on=["final_summary"]),
            "which does not come before it",
        ),
        (lambda b: b["sub_tasks"][1].update(id="Regional Trends"), "must be a snake_case string"),
        (lambda b: b.update(sub_tasks=b["sub_tasks"][:2]), "sub_tasks must hold between"),
        (
            lambda b: b["metrics_config"]["metrics"].update(
                revenue={"label": "Revenue", "type": "currency", "better": "higher"}
            ),
            "metrics_config.metrics.revenue: key must match a sub-task output key",
        ),
        (
            lambda b: b["metrics_config"]["metrics"]["total_revenue"].update(type="money"),
            "total_revenue.type must be one of",
        ),
        (lambda b: b.pop("metrics_config"), "metrics_config is required"),
        (lambda b: b["report"].update(format="pdf"), "report.format must be one of: html"),
        (lambda b: b["report"].update(sections=[]), "report.sections must not be empty"),
        (lambda b: b.update(data_notes=[]), "data_notes must not be empty"),
        (lambda b: b.pop("open_questions"), "open_questions is required"),
    ],
)
def test_invalid_briefs_are_rejected_with_a_usable_reason(mutate, expected):
    errors = _errors_for(mutate)

    assert any(expected in error for error in errors), errors


def test_an_unknown_column_error_lists_the_real_ones():
    """The model must be able to fix the brief from the error alone."""
    errors = _errors_for(lambda b: b["data_scope"]["relations"][1]["columns"].append("zona"))

    assert any("Its columns are: id, regione" in error for error in errors)


def test_every_error_is_reported_at_once():
    """One repair round per problem would burn the agent's step budget."""
    errors = _errors_for(lambda b: (b.update(category="x"), b["report"].update(format="pdf")))

    assert len(errors) == 2


def test_validation_does_not_mutate_the_brief():
    brief = make_brief()
    snapshot = copy.deepcopy(brief)

    validate_brief(brief, IDENTIFIERS)

    assert brief == snapshot


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_prompt_carries_every_section_the_analyst_needs():
    prompt = render_brief_prompt(make_brief())

    for heading in (
        "# ANALYSIS BRIEF: Revenue by region",
        "## BUSINESS CONTEXT",
        "## MISSION",
        "## DATA SCOPE",
        "## DATA NOTES",
        "## ANALYSIS PHASES",
        "## TRACKED METRICS",
        "## REPORT REQUIREMENTS",
        "## OPEN QUESTIONS AND ASSUMPTIONS",
    ):
        assert heading in prompt


def test_the_prompt_quotes_identifiers_and_orders_the_phases():
    brief = make_brief()
    brief["sub_tasks"] = list(reversed(brief["sub_tasks"]))

    prompt = render_brief_prompt(brief)

    assert '"Ordini" (facts): "Cliente", "Importo", "DataOrdine"' in prompt
    assert prompt.index("[data_preparation]") < prompt.index("[final_summary]")
    assert "```sql\nSELECT" in prompt


def test_the_prompt_omits_metrics_when_none_are_tracked():
    assert "## TRACKED METRICS" not in render_brief_prompt(make_brief(metrics_config=None))
