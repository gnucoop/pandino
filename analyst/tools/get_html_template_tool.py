import json
from typing import Any, ClassVar, Optional

from smolagents import Tool

from analyst.report_template import SECTION_STYLES, render_report
from analyst.state import RunState


def _sections_problem(sections: Any, state: RunState) -> Optional[str]:
    """Why `sections` cannot be rendered, or None."""
    if not isinstance(sections, list) or not sections:
        return "sections must be a non-empty list."
    for i, section in enumerate(sections):
        where = f"sections[{i}]"
        if not isinstance(section, dict):
            return f"{where} must be a dict."
        if not str(section.get("title") or "").strip():
            return f"{where}.title is empty."
        if not isinstance(section.get("content", ""), str):
            return f"{where}.content must be a markdown string."
        style = section.get("style")
        if style is not None and style not in SECTION_STYLES:
            return f"{where}.style must be one of: {', '.join(SECTION_STYLES)}."
        for field, registry in (("charts", state.charts), ("tables", state.tables)):
            names = section.get(field) or []
            if not isinstance(names, list):
                return f"{where}.{field} must be a list of names."
            unknown = [n for n in names if n not in registry]
            if unknown:
                return (
                    f"{where}.{field}: unknown names {unknown}. Saved: "
                    f"{', '.join(registry) or 'none'} (see get_all_saved_charts)."
                )
    return None


class GetHtmlTemplateTool(Tool):
    """
    Render the report from the sections the agent wrote.

    The page is rendered and kept in the run state;
    the agent sends markdown and names and gets back a short summary.
    KPI cards, the comparison with the previous run and an
    appendix with any chart or table left unplaced are added automatically.
    """

    name = "get_html_template"
    description = (
        "Render the HTML report. sections is a list of dicts, in reading order: "
        "{'title': str, 'content': markdown text (no HTML), 'style': 'danger'|'warning'|"
        "'success'|'info'|'secondary' (optional), 'charts': [chart names], 'tables': "
        "[table names]}. KPI cards from save_run_metrics and the previous-run comparison "
        "are added on top, and charts or tables you do not place go to an appendix. "
        "title defaults to the plan's title; give a one-line subtitle. Call "
        "save_html_report afterwards; calling this again replaces the rendering."
    )
    output_type = "object"

    inputs: ClassVar[dict[str, Any]] = {
        "sections": {"type": "any", "description": "The report sections (a list, or its JSON)."},
        "title": {"type": "string", "description": "Report title.", "nullable": True},
        "subtitle": {"type": "string", "description": "Report subtitle.", "nullable": True},
    }

    def __init__(self, state: RunState, *, appendix_title: str = "Appendix", footer: str = "") -> None:
        super().__init__()
        self._state = state
        self._appendix_title = appendix_title
        self._footer = footer

    def forward(
        self, sections: Any, title: Optional[str] = None, subtitle: Optional[str] = None
    ) -> dict[str, Any]:
        try:
            if isinstance(sections, str):
                try:
                    sections = json.loads(sections)
                except ValueError as e:
                    return {"kind": "error", "message": f"sections is not valid JSON: {e}", "code": "INVALID_SECTIONS"}

            problem = _sections_problem(sections, self._state)
            if problem is not None:
                return {"kind": "error", "message": problem, "code": "INVALID_SECTIONS"}

            self._state.html = render_report(
                self._state,
                title=title,
                subtitle=subtitle,
                sections=sections,
                appendix_title=self._appendix_title,
                footer=self._footer,
            )
            placed = {n for s in sections for n in (s.get("charts") or []) + (s.get("tables") or [])}
            unplaced = [n for n in [*self._state.charts, *self._state.tables] if n not in placed]
            return {
                "kind": "text",
                "text": (
                    f"Report rendered: {len(sections)} sections, {len(self._state.charts)} charts, "
                    f"{len(self._state.tables)} tables"
                    + (f" (in the appendix: {', '.join(unplaced)})" if unplaced else "")
                    + ". Call save_html_report() to write it."
                ),
            }
        except Exception as e:
            return {"kind": "error", "message": f"get_html_template failed: {e}", "code": "TOOL_FAILED"}
