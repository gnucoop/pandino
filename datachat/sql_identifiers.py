"""
sql_identifiers.py
------------------
Repair PostgreSQL identifier case in LLM-authored SQL.

PostgreSQL folds every unquoted identifier to lowercase, so `FROM Trasporti`
reaches the planner as `trasporti` and fails with 42P01 against a mixed-case
schema. Models write unquoted SQL by default and recover from the error at best
inconsistently, at the cost of a whole extra step each time.

This module rewrites bare identifiers to their real, quoted spelling before the
query is validated and executed. Like sql_guard it depends on sqlglot alone —
no database, no SQLAlchemy, no snapshot type — so every rule here is testable
from literal dicts without a live PostgreSQL.

Two mechanisms, and the split between them is the whole design:

- **The AST decides what may be rewritten.** Only an unquoted Identifier sitting
  in relation or column position is eligible. This is default-deny: a construct
  sqlglot does not report as a relation/column reference is left alone, whatever
  it looks like. Deciding from tokens instead would be default-allow and wrong —
  `count`, `sum` and the `YEAR` in `EXTRACT(YEAR FROM d)` are all bare VAR tokens
  that are not column references, and this schema really does have a `Data`
  column that would collide with one.
- **The tokens decide where to cut.** Only the tokenizer reports source offsets,
  so it supplies the spans; sqlglot expressions carry no positional metadata.

The rewrite is a surgical patch of the original text, never a round trip through
sqlglot's generator: `sql_guard` rejects re-emitting a parsed query because the
round trip is not guaranteed lossless, and that reasoning holds here too. Every
byte outside a replaced span is copied verbatim, so string literals, comments and
already-quoted identifiers survive untouched by construction rather than by rule.

Everything here fails open: on any doubt the input is returned unchanged, because
the caller validates whatever comes out. The caller must validate the *rewritten*
text and execute that same text, which is what preserves sql_guard's guarantee
that what was checked is exactly what runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType

# Identifier positions that name a real relation or column. `Table.this` is the
# relation, `Column.this` the column, and `Column.table` the qualifier in
# `Clienti.Idcliente` — which needs quoting too when it is a relation name rather
# than an alias. `Table.db` / `Table.catalog` are deliberately absent: the schema
# comes from configuration, not from the agent, and is never ours to rewrite.
_REWRITABLE_POSITIONS: tuple[tuple[type, frozenset[str]], ...] = (
    (exp.Table, frozenset({"this"})),
    (exp.Column, frozenset({"this", "table"})),
)


def quote_ident(name: str) -> str:
    """
    Quote an identifier for literal interpolation into SQL.

    Only ever applied to names that came back from catalog reflection: the
    agent's spelling is used to *look up* a real name, and it is the real name
    that gets quoted.
    """
    return '"' + str(name).replace('"', '""') + '"'


def _resolve(names: Iterable[str]) -> dict[str, str]:
    """
    Map lowercase spelling -> real spelling, for names that can be fixed safely.

    Two kinds of name are deliberately left out:

    - Names already all-lowercase. They need no quoting, and skipping them keeps
      the rewrite off the overwhelming majority of tokens.
    - Names that resolve to more than one spelling. This schema really does hold
      both `Anno` and `anno`, and both `Cliente` and `cliente`; picking between
      them would silently change the meaning of queries that work today. They are
      left bare for the agent to quote from the schema it was given.
    """
    by_lower: dict[str, set[str]] = {}
    for name in names:
        by_lower.setdefault(name.lower(), set()).add(name)

    return {
        lower: next(iter(spellings))
        for lower, spellings in by_lower.items()
        if len(spellings) == 1 and next(iter(spellings)) != lower
    }


def _qualifier_relations(tree: exp.Expression) -> dict[str, str]:
    """
    Alias (and bare relation name) -> the real relation it stands for.

    `FROM "Sedi" s` makes `s` mean Sedi, which is what lets a qualified column be
    resolved against one relation instead of against everything the query
    touches — the difference between fixing `s.IdSede` and giving up on it.
    """
    mapping: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = table.name
        if not name:
            continue
        mapping[name.lower()] = name
        alias = table.alias
        if alias:
            mapping[alias.lower()] = name
    return mapping


def _qualifier_of(tokens: list, index: int) -> Optional[str]:
    """The `x` of `x.column`, read straight off the token stream."""
    if index < 2:
        return None
    if tokens[index - 1].token_type is not TokenType.DOT:
        return None
    if tokens[index - 2].token_type not in (TokenType.VAR, TokenType.IDENTIFIER):
        return None
    return tokens[index - 2].text


def _rewritable_names(tree: exp.Expression) -> set[str]:
    """
    Lowercased names that appear *only* in relation or column position.

    A name used anywhere else in the same query — as a table alias, a column
    alias, a CTE name, or a bare keyword argument such as the `YEAR` of an
    EXTRACT — is withheld everywhere in that query. The suppression is by name
    rather than by occurrence, which costs the odd rewrite when a query aliases a
    column to the name of another column, and fails in the safe direction: the
    query then behaves exactly as it does today.
    """
    allowed: set[str] = set()
    denied: set[str] = set()

    for identifier in tree.find_all(exp.Identifier):
        if identifier.quoted:
            continue
        name = (identifier.name or "").lower()
        if not name:
            continue
        parent = identifier.parent
        if any(
            isinstance(parent, node) and identifier.arg_key in keys
            for node, keys in _REWRITABLE_POSITIONS
        ):
            allowed.add(name)
        else:
            denied.add(name)

    # exp.Var covers bare keyword arguments that are not identifiers at all.
    denied |= {
        (var.name or "").lower() for var in tree.find_all(exp.Var) if var.name
    }

    return allowed - denied


def quote_identifiers(
    sql: str,
    relations: Mapping[str, Sequence[str]],
) -> tuple[str, tuple[str, ...]]:
    """
    Rewrite bare identifiers in `sql` to their real, quoted spelling.

    :param sql: The SQL text authored by the agent.
    :param relations: Real relation name -> its real column names. Must already
                      be filtered to what the agent may see: a name that reaches
                      this map is a name the rewriter may reveal.
    :return: (rewritten sql, the real names substituted). The input is returned
             unchanged, with an empty tuple, whenever nothing needs fixing or the
             query cannot be parsed as a single statement — an unparseable or
             multi-statement query is sql_guard's to reject with a proper
             message, not this module's to mangle.
    """
    text = sql or ""
    if not text.strip() or not relations:
        return sql, ()

    try:
        statements = [stmt for stmt in sqlglot.parse(text, read="postgres") if stmt]
        tokens = sqlglot.tokenize(text, read="postgres")
    except SqlglotError:
        return sql, ()
    except Exception:
        return sql, ()

    if len(statements) != 1:
        return sql, ()

    tree = statements[0]
    rewritable = _rewritable_names(tree)
    if not rewritable:
        return sql, ()

    referenced = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}

    relation_names = _resolve(relations.keys())

    # Columns resolve only against the relations this query actually references.
    # That scoping is what lets `anno` become "Anno" in a Budgets query and stay
    # bare in a report_mensile one, instead of having to give up on both.
    column_names = _resolve(
        column
        for relation, columns in relations.items()
        if relation.lower() in referenced
        for column in columns
    )

    # A qualified column narrows the scope further, to the single relation the
    # qualifier names. This is what rescues a join key like `s.IdSede = t.idSede`,
    # where the two relations spell the same column differently and the
    # query-wide map has to abstain.
    qualifiers = _qualifier_relations(tree)
    # Keyed by lowercase, because the qualifier names the relation as the agent
    # spelled it, which is exactly the spelling we are here to correct.
    by_lower: dict[str, list[str]] = {}
    for name in relations:
        by_lower.setdefault(name.lower(), []).append(name)
    per_relation = {
        name: _resolve(columns) for name, columns in relations.items()
    }

    def columns_of(qualifier: str) -> Optional[dict[str, str]]:
        owner = qualifiers.get(qualifier.lower())
        if owner is None:
            return None
        candidates = by_lower.get(owner.lower(), [])
        if len(candidates) != 1:
            return None
        return per_relation.get(candidates[0])

    pieces: list[str] = []
    rewritten: list[str] = []
    cursor = 0

    for index, token in enumerate(tokens):
        # VAR is a bare word. IDENTIFIER is already double-quoted and STRING is a
        # literal, so both are skipped here rather than by any rule of our own.
        if token.token_type is not TokenType.VAR:
            continue

        lowered = token.text.lower()
        if lowered not in rewritable:
            continue

        qualifier = _qualifier_of(tokens, index)
        owned = columns_of(qualifier) if qualifier else None
        if owned is not None:
            # Qualified: the owning relation is the only sensible scope, so a
            # miss here is a miss, not a reason to guess query-wide.
            real = owned.get(lowered)
        else:
            real = relation_names.get(lowered) or column_names.get(lowered)

        if real is None:
            continue

        pieces.append(text[cursor : token.start])
        pieces.append(quote_ident(real))
        cursor = token.end + 1
        rewritten.append(real)

    if not rewritten:
        return sql, ()

    pieces.append(text[cursor:])
    return "".join(pieces), tuple(dict.fromkeys(rewritten))


def suggest_identifier(name: str, candidates: Iterable[str]) -> Optional[str]:
    """
    Find the real spelling of `name`, for the error path.

    PostgreSQL reports the folded name it failed to find ("trasporti"), which is
    exactly the form that matches case-insensitively against the real one. Used
    to turn a bare 42P01/42703 into a message the agent can act on.
    """
    lowered = (name or "").strip().lower()
    if not lowered:
        return None
    for candidate in candidates:
        if candidate.lower() == lowered and candidate != name:
            return candidate
    return None
