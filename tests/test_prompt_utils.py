"""
Tests for infrastructure.prompt_utils.render_prompt.

render_prompt substitutes strict ``{identifier}`` placeholders in a single
regex pass. Every Datachat prompt carries literal JSON braces in its
final-answer contract; a naive .format() read those as replacement fields and
raised, which used to make the whole template come back unrendered with its
real placeholders still in place.

Two properties are pinned here and must not regress silently:

1. literal JSON survives, and never counts as a missing placeholder;
2. a *recognizable* placeholder the caller did not supply stays literal in the
   output but is still reported via ``event=prompt_placeholder_missing`` -
   partial rendering must not become silent rendering.
"""

import logging
import sys
import types

import pytest

# prompt_utils imports get_prompt_from_db at module scope, which pulls in psycopg
# and a live connection. Nothing here exercises load_prompt, so the database
# module is stubbed before the import rather than requiring a driver.
if "infrastructure.database_pg" not in sys.modules:
    _stub = types.ModuleType("infrastructure.database_pg")
    _stub.get_prompt_from_db = lambda *a, **k: None
    sys.modules["infrastructure.database_pg"] = _stub

from infrastructure.prompt_utils import render_prompt  # noqa: E402


# The shape that actually occurs in datachat/smolagents_engine.py: a real
# placeholder next to the literal JSON of the final-answer contract.
TEMPLATE_WITH_JSON = (
    "- The dataset has the following columns: {columns}.\n"
    'Final-answer schemas (strict):\n'
    '- kind="text"  -> {"kind":"text","text":"..."}\n'
    '- kind="table" -> {"kind":"table","data":[...]}\n'
)


def test_placeholder_is_substituted_next_to_literal_json_braces():
    """The regression this whole fix exists for."""
    out = render_prompt(TEMPLATE_WITH_JSON, columns=["age", "city"])

    assert "{columns}" not in out
    assert "['age', 'city']" in out


def test_literal_json_braces_survive_unchanged():
    out = render_prompt(TEMPLATE_WITH_JSON, columns=[])

    assert '{"kind":"text","text":"..."}' in out
    assert '{"kind":"table","data":[...]}' in out
    # Escaping must not leak into the output.
    assert "{{" not in out
    assert "}}" not in out


def test_multiple_placeholders_are_all_substituted():
    out = render_prompt(
        "Hello {name}, today is {day}. Contract: {\"kind\":\"text\"}",
        name="Gustavo",
        day="Thursday",
    )

    assert out == 'Hello Gustavo, today is Thursday. Contract: {"kind":"text"}'


def test_no_kwargs_returns_the_template_unchanged():
    """
    Several callers render a template with no substitutions at all
    (infrastructure/ai.py, services/completion_service.py, ...). Escaping and
    unescaping must round-trip exactly for them.
    """
    template = 'Answer as JSON: {"kind":"text"} and keep {braces} intact.'

    assert render_prompt(template) == template


def test_template_without_any_braces_is_unaffected():
    assert render_prompt("plain text, no braces") == "plain text, no braces"


def test_unused_kwarg_is_ignored():
    """A stored prompt that predates a new placeholder must still render."""
    out = render_prompt("no placeholder here", sql_schema="TABLES\n  orders(id:int4)")

    assert out == "no placeholder here"


def test_repeated_placeholder_is_substituted_everywhere():
    out = render_prompt("{x} and again {x}", x="v")

    assert out == "v and again v"


@pytest.mark.parametrize("value", ["", None, 0, []])
def test_falsy_values_are_substituted_not_skipped(value):
    out = render_prompt("value=[{v}]", v=value)

    assert out == f"value=[{value}]"


def test_unknown_placeholder_is_left_literal_and_does_not_block_the_others():
    """
    A DB-stored override may reference a placeholder the caller does not supply.
    It is treated as literal text, and the placeholders that *were* supplied
    still render. The old all-or-nothing behaviour returned the whole template
    unrendered instead.
    """
    out = render_prompt("Hello {name}, {unsupplied} stays", name="Gustavo")

    assert out == "Hello Gustavo, {unsupplied} stays"


# ---------------------------------------------------------------------------
# Missing-placeholder diagnostics
# ---------------------------------------------------------------------------


def _missing_keys(caplog):
    """The keys reported by event=prompt_placeholder_missing in caplog."""
    return [
        record.args[0]
        for record in caplog.records
        if record.getMessage().startswith("event=prompt_placeholder_missing")
    ]


def test_missing_placeholder_emits_the_structured_diagnostic(caplog):
    with caplog.at_level(logging.WARNING, logger="infrastructure.prompt_utils"):
        out = render_prompt("Hello {username}, use {old_variable}", username="Mario")

    assert out == "Hello Mario, use {old_variable}"
    assert _missing_keys(caplog) == ["old_variable"]


def test_literal_json_does_not_emit_a_missing_placeholder_warning(caplog):
    """
    The whole point of the strict placeholder definition: {"kind":"text"} opens
    with a quote, not an identifier, so it is not a candidate at all.
    """
    with caplog.at_level(logging.WARNING, logger="infrastructure.prompt_utils"):
        out = render_prompt(TEMPLATE_WITH_JSON, columns=["age"])

    assert '{"kind":"text","text":"..."}' in out
    assert _missing_keys(caplog) == []


def test_supplied_placeholder_emits_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="infrastructure.prompt_utils"):
        out = render_prompt("Schema: {sql_schema}", sql_schema="users(id)")

    assert out == "Schema: users(id)"
    assert _missing_keys(caplog) == []


def test_repeated_missing_placeholder_is_reported_once(caplog):
    with caplog.at_level(logging.WARNING, logger="infrastructure.prompt_utils"):
        render_prompt("{gone} then {gone}")

    assert _missing_keys(caplog) == ["gone"]


def test_stale_db_prompt_placeholder_is_visible_and_reported(caplog):
    """
    Regression cover for the real failure mode: a prompt stored in the DB still
    references a placeholder the code no longer passes. The operator must be
    able to see it - both in the rendered text and in the logs - instead of the
    substitution silently going missing.
    """
    stored = "Columns: {columns}. Legacy: {dataframe_head}."

    with caplog.at_level(logging.WARNING, logger="infrastructure.prompt_utils"):
        out = render_prompt(stored, columns=["a"])

    assert out == "Columns: ['a']. Legacy: {dataframe_head}."
    assert _missing_keys(caplog) == ["dataframe_head"]


def test_malformed_and_non_identifier_braces_are_never_placeholders(caplog):
    """
    Unbalanced braces, spaces, dotted/indexed names and format specs all fall
    outside the strict definition, so they stay literal and silent rather than
    being half-parsed by a permissive brace scanner.
    """
    template = "{ spaced } {a.b} {c[0]} {x:>10} {y!r} {unclosed and 100% {"

    with caplog.at_level(logging.WARNING, logger="infrastructure.prompt_utils"):
        out = render_prompt(template, a="A", c="C", x="X", y="Y")

    assert out == template
    assert _missing_keys(caplog) == []


def test_render_failure_falls_back_to_the_template(caplog):
    """
    event=prompt_render_failed is still reachable: str() on a value is caller
    controlled and can raise.
    """

    class Explodes:
        def __str__(self):
            raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="infrastructure.prompt_utils"):
        out = render_prompt("value={v}", v=Explodes())

    assert out == "value={v}"
    assert any(
        record.getMessage().startswith("event=prompt_render_failed")
        for record in caplog.records
    )
