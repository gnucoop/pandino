"""
Tests for infrastructure.prompt_utils.render_prompt.

render_prompt is str.format() based, and every Datachat prompt carries literal
JSON braces in its final-answer contract. A naive .format() reads those as
replacement fields and raises, which used to make the whole template come back
unrendered with its real placeholders still in place. These tests pin the
substitution down so that cannot regress silently again.
"""

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
