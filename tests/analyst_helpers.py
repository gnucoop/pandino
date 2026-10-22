"""Builders shared by the analyst tests."""

from analyst.plan import plan_from_brief
from analyst.state import RunState
from config import AnalystConfig

COLUMNS = [
    {"column": "id", "type": "INTEGER", "nullable": False, "primary_key": True},
    {"column": "country", "type": "VARCHAR(2)", "nullable": True, "primary_key": False},
    {"column": "amount", "type": "NUMERIC", "nullable": True, "primary_key": False},
]


def make_config(**overrides):
    params = dict(
        provider="Deepinfra",
        model="task/model",
        report_provider="Deepinfra",
        report_model="report/model",
        plan_max_steps=5,
        task_max_steps=5,
        chart_max_steps=5,
        report_max_steps=5,
        task_retries=1,
        max_extra_tasks=2,
        max_insights=5,
        sql_max_rows=50,
        dataset_max_rows=1000,
        output_max_chars=5000,
        preview_max_chars=500,
        max_charts=10,
        chart_max_points=100,
        reports_dir="unused",
    )
    params.update(overrides)
    return AnalystConfig(**params)


def valid_brief():
    """A brief validate_brief() accepts against a schema of orders/customers (COLUMNS)."""
    return {
        "name": "Revenue by country",
        "description": "Revenue per country.",
        "category": "discovery",
        "tags": ["revenue", "country", "orders"],
        "language": "Italian",
        "audience": "Sales",
        "business_context": "Planning targets.",
        "data_scope": {
            "relations": [{"name": "orders", "columns": ["country", "amount"]}],
            "filters": ["Only 2026."],
            "sql_hints": ['SELECT "country", SUM("amount") FROM "orders" GROUP BY 1'],
        },
        "data_notes": ["amount is in EUR."],
        "mission": "Sum amount per country.",
        "sub_tasks": [
            {"id": "data_preparation", "name": "Prep", "order": 1, "depends_on": [],
             "mission": "Check amount.", "outputs": {"required": ["row_count"]}},
            {"id": "by_country", "name": "By country", "order": 2, "depends_on": ["data_preparation"],
             "mission": "Totals.", "outputs": {"required": ["total_revenue"], "optional": ["top_country"]}},
            {"id": "final_summary", "name": "Summary", "order": 3, "depends_on": ["by_country"],
             "mission": "Synthesize.", "outputs": {"required": ["key_findings"]}},
        ],
        "metrics_config": {
            "metrics": {"total_revenue": {"label": "Revenue", "type": "currency", "better": "higher"}},
        },
        "report": {"format": "html", "sections": ["Summary"], "charts": ["Bar chart"],
                   "tone": "plain", "length": "short"},
        "open_questions": ["Revenue excludes VAT."],
    }


class Run:
    """What CodeAgent.run(..., return_full_result=True) returns, reduced to what is read."""

    def __init__(self, output, token_usage=None):
        self.output = output
        self.token_usage = token_usage


def make_state(tmp_path, brief=None, *, plan=None, run_id="r1", brief_id=3):
    """A RunState for `brief`, planned by the code fallback unless `plan` is given."""
    brief = brief if brief is not None else valid_brief()
    plan = plan if plan is not None else plan_from_brief(brief, {"orders": ("country", "amount")})
    return RunState.from_plan(
        run_id=run_id, brief_id=brief_id, brief=brief, plan=plan, run_dir=tmp_path / run_id
    )
