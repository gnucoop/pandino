"""
report_template.py
------------------

Rendering of the HTML report from the run state and the sections the report
agent wrote.

The agent sends only section titles, markdown text and the names
of the charts and tables to place, and this module does the rest: the KPI
cards (metrics_config + save_run_metrics), the comparison with the previous
run, Chart.js charts and escaped tables. Charts and tables no section placed
land in a final appendix, so nothing computed is lost.

The output is one self-contained HTML file; only Chart.js comes from a CDN.
Nothing the agents wrote reaches the page as raw HTML: markdown is rendered
with HTML disabled, everything else is autoescaped or embedded as JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from markdown_it import MarkdownIt

from analyst.state import RunState

SECTION_STYLES = ("danger", "warning", "success", "info", "secondary")

# The currency of KPI cards typed "currency" in metrics_config.
REPORT_CURRENCY = "EUR"

# The plan's language (an English language name) -> number/date locale.
_LOCALES = {
    "italian": "it-IT",
    "english": "en-GB",
    "spanish": "es-ES",
    "french": "fr-FR",
    "german": "de-DE",
    "portuguese": "pt-PT",
}

_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
    autoescape=select_autoescape(["html"]),
)
_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")


def locale_for(language: Any) -> str:
    return _LOCALES.get(str(language or "").strip().lower(), "en-GB")


def _kpis(state: RunState) -> list[dict[str, Any]]:
    """One card per configured metric, then one per category (its item count)."""
    config = state.metrics_config or {}
    values = (state.metrics or {}).get("metrics", {})
    categories = (state.metrics or {}).get("categories", {})
    deltas = (state.delta or {}).get("metrics", {})
    previous = (state.delta or {}).get("previous_run_at", "")

    cards = []
    for key, spec in (config.get("metrics") or {}).items():
        value = values.get(key)
        card = {
            "label": spec.get("label", key),
            "type": spec.get("type", "number"),
            "value": value if isinstance(value, (int, float)) else None,
            "display": "—" if value is None else str(value),
            "icon": "",
            "delta": None,
        }
        delta = deltas.get(key)
        if delta and delta.get("change") is not None:
            card["delta"] = {
                "change": delta["change"],
                "css": {True: "good", False: "bad"}.get(delta.get("improved"), ""),
                "suffix": f"vs {previous}" if previous else "",
                "text": f"{delta['change']:+} vs {previous}".strip(),
            }
        cards.append(card)

    for key, spec in (config.get("categories") or {}).items():
        items = categories.get(key)
        count = len(items) if isinstance(items, (list, dict)) else None
        cards.append({
            "label": spec.get("label", key),
            "type": "number",
            "value": count,
            "display": "—" if count is None else str(count),
            "icon": spec.get("icon", ""),
            "delta": None,
        })
    return cards


def _place(names: list[str], registry: dict[str, dict[str, Any]], placed: set[str]) -> list[dict[str, Any]]:
    items = []
    for name in names:
        if name in registry and name not in placed:
            placed.add(name)
            items.append({"name": name, **registry[name]})
    return items


def render_report(
    state: RunState,
    *,
    title: Optional[str],
    subtitle: Optional[str],
    sections: list[dict[str, Any]],
    appendix_title: str = "Appendix",
    footer: str = "",
) -> str:
    """
    The report as a complete HTML document.

    :param sections: validated sections, {title, content, style, charts, tables}.
    """
    placed_charts: set[str] = set()
    placed_tables: set[str] = set()

    rendered = []
    for section in sections:
        rendered.append({
            "title": section["title"],
            "style": section.get("style") if section.get("style") in SECTION_STYLES else "",
            "html": Markup(_markdown.render(section.get("content") or "")),
            "charts": _place(section.get("charts") or [], state.charts, placed_charts),
            "tables": _place(section.get("tables") or [], state.tables, placed_tables),
        })

    leftover_charts = _place(list(state.charts), state.charts, placed_charts)
    leftover_tables = _place(list(state.tables), state.tables, placed_tables)
    if leftover_charts or leftover_tables:
        rendered.append({
            "title": appendix_title,
            "style": "secondary",
            "html": "",
            "charts": leftover_charts,
            "tables": leftover_tables,
        })

    return _env.get_template("report.html").render(
        title=(title or "").strip() or state.title,
        subtitle=(subtitle or "").strip(),
        audience=state.audience,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        lang=locale_for(state.language).split("-")[0],
        locale=locale_for(state.language),
        currency=REPORT_CURRENCY,
        kpis=_kpis(state),
        sections=rendered,
        charts={name: chart["spec"] for name, chart in state.charts.items()},
        footer=footer,
    )
