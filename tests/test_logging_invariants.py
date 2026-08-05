"""
INVARIANT A0
The test suite contains a permanent test, driven by the `ast` module of the
Python standard library, which walks the whole repository (excluding `venv/`)
and asserts:

  violations(A1) union violations(A2) union violations(A5a) union
  violations(A5b)  is a subset of  BASELINE union DECLARED_EXCEPTIONS

where BASELINE is an explicit, machine-readable registry of the call sites
that are known to violate an invariant and are scheduled to be fixed by a
later axis, and DECLARED_EXCEPTIONS is a separate, explicit registry of the
call sites that are permitted to violate an invariant permanently, each with
a stated reason.

The test fails if a violation appears that is in neither registry. The test
also fails if a BASELINE entry no longer corresponds to an existing violation,
so that a fixed site cannot be left in the registry.

BASELINE and DECLARED_EXCEPTIONS must be two distinct registries. A site is
never moved from BASELINE to DECLARED_EXCEPTIONS as a way of closing an axis.

INVARIANT A1(scope)
Within <scope>:

(a) No call expression uses the `logging` module or `current_app.logger` as
    the receiver of any of the level methods
    `debug|info|warning|warn|error|exception|critical|log`.

(b) Every module that emits log records binds exactly one module-level logger
    as `logger = logging.getLogger(__name__)`, and all its emissions go
    through that object.

The chained form `logging.getLogger(...).<level>(...)` satisfies (a) but not
(b) and is permitted only via DECLARED_EXCEPTIONS.

Loggers that are deliberately named and isolated — `agent_runs` and
`datachat.runtime`, reached through `_agent_logger`, `runtime_logger` and
`_logger` — are NOT in scope for this invariant and must not be migrated.

INVARIANT A2(scope)
Within <scope>, the first argument of no logging call begins with a bracketed
manual area prefix matching ^(\\[[^\\]]+\\])+ .

Exempt while Axis 3 is incomplete: the sites listed in the CARRY-OVER register
(Appendix B), whose prefix encodes a flow finer-grained than the module and
whose content is relocated into the event name by Axis 3, not deleted. Once
Axis 3 is complete the exemption is void and A2 holds unconditionally over the
whole repository.

INVARIANT A5a
`main.py` contains no `print(` call.

INVARIANT A5b
In `infrastructure/database_pg.py`, no `print(` call and no logging call has,
anywhere in its arguments, a reference to a variable holding decrypted key
material or stored key ciphertext. Concretely, the identifiers
`decrypted_api_key`, `decrypted_key` and `encrypted_key` do not appear inside
the arguments of any `print(` or logging call in the file.

This invariant is satisfied by REDACTION, not by deletion: the functions that
contain these calls must keep emitting their non-credential output.

INVARIANT A5c
No `print(` call exists anywhere in the repository, excluding `venv/`,
except at the call sites listed in DECLARED_EXCEPTIONS.
"""

import ast
import os
import re
from collections import Counter
from pathlib import Path

from tests.logging_baseline import BASELINE
from tests.logging_declared_exceptions import DECLARED_EXCEPTIONS

REPO_ROOT = Path(__file__).resolve().parent.parent

LEVELS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
DEDICATED_VARS = {"_agent_logger", "runtime_logger", "_logger"}
KEY_MATERIAL_IDENTIFIERS = {"decrypted_api_key", "decrypted_key", "encrypted_key"}
EXCLUDED_DIR_NAMES = {"venv", "__pycache__", ".git", "docs"}

PREFIX_RE = re.compile(r"^(\[[^\]]+\])+")


def _get_full_attr(node):
    """Dotted string for an Attribute/Name chain, or None if not simple."""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _get_str_literal_value(node):
    """Best-effort literal text of a Constant string or f-string."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                parts.append("{...}")
        return "".join(parts)
    return None


def _args_contain_identifier(args, keywords, identifiers):
    nodes = list(args) + [kw.value for kw in keywords]
    for n in nodes:
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name) and sub.id in identifiers:
                return True
    return False


class _FileVisitor(ast.NodeVisitor):
    """Walks one module, recording level-method calls, print() calls,
    imports of the `logging` module, and whether a module-level
    `logger = logging.getLogger(__name__)` binding exists."""

    def __init__(self, relpath):
        self.relpath = relpath
        self.func_stack = []
        self.calls = []
        self.module_level_logger_bound = False
        self.aliased_logging_imports = []

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "logging" and alias.asname is not None:
                self.aliased_logging_imports.append(f"import logging as {alias.asname}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "logging":
            names = ", ".join(a.name for a in node.names)
            self.aliased_logging_imports.append(f"from logging import {names}")
        self.generic_visit(node)

    def visit_Assign(self, node):
        if not self.func_stack and node.col_offset == 0:
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "logger"
                and isinstance(node.value, ast.Call)
            ):
                val = node.value
                if isinstance(val.func, ast.Attribute):
                    fullattr = _get_full_attr(val.func)
                elif isinstance(val.func, ast.Name):
                    fullattr = val.func.id
                else:
                    fullattr = None
                if fullattr in ("logging.getLogger", "getLogger") and (
                    len(val.args) == 1
                    and isinstance(val.args[0], ast.Name)
                    and val.args[0].id == "__name__"
                ):
                    self.module_level_logger_bound = True
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        enclosing = self.func_stack[-1] if self.func_stack else "MODULE"
        if isinstance(func, ast.Attribute):
            fullattr = _get_full_attr(func)
            attr = func.attr
            if fullattr is None and attr in LEVELS and isinstance(func.value, ast.Call):
                inner = func.value
                if isinstance(inner.func, ast.Attribute):
                    inner_full = _get_full_attr(inner.func)
                elif isinstance(inner.func, ast.Name):
                    inner_full = inner.func.id
                else:
                    inner_full = None
                if inner_full in ("logging.getLogger", "getLogger"):
                    self.calls.append(
                        {"form": "chained", "enclosing": enclosing, "args": node.args, "keywords": node.keywords}
                    )
            elif fullattr and fullattr.startswith("logging.") and attr in LEVELS:
                self.calls.append(
                    {"form": "a1a", "enclosing": enclosing, "args": node.args, "keywords": node.keywords}
                )
            elif fullattr and fullattr.endswith("current_app.logger." + attr) and attr in LEVELS:
                self.calls.append(
                    {"form": "a1a", "enclosing": enclosing, "args": node.args, "keywords": node.keywords}
                )
            elif attr in LEVELS and fullattr and "." in fullattr:
                varname = fullattr.rsplit(".", 1)[0]
                self.calls.append(
                    {
                        "form": "logger_object",
                        "varname": varname,
                        "enclosing": enclosing,
                        "args": node.args,
                        "keywords": node.keywords,
                    }
                )
        elif isinstance(func, ast.Name):
            if func.id == "print":
                self.calls.append(
                    {"form": "print", "enclosing": enclosing, "args": node.args, "keywords": node.keywords}
                )
        self.generic_visit(node)


def _walk_py_files(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
        for fn in sorted(files):
            if fn.endswith(".py"):
                yield os.path.join(base, fn)


def scan(root):
    """Returns (observed, alias_violations).

    observed: Counter keyed (invariant, posix_relative_path, enclosing_function_name)
    alias_violations: list of "<relpath>: <description>" strings, one per
    `import logging as X` or `from logging import ...` occurrence.

    A SyntaxError in any scanned file propagates rather than being swallowed,
    so a broken file fails the test loudly instead of silently shrinking the
    scanned set.
    """
    observed = Counter()
    alias_violations = []

    for fpath in _walk_py_files(root):
        rel = os.path.relpath(fpath, root)
        relposix = rel.replace(os.sep, "/")
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=fpath)

        visitor = _FileVisitor(relposix)
        visitor.visit(tree)

        alias_violations.extend(f"{relposix}: {d}" for d in visitor.aliased_logging_imports)

        in_scope_emission = False
        for call in visitor.calls:
            form = call["form"]
            enclosing = call["enclosing"]
            args = call["args"]
            keywords = call["keywords"]
            lit = _get_str_literal_value(args[0]) if args else None

            if form == "a1a":
                observed[("A1a", relposix, enclosing)] += 1
                in_scope_emission = True
            elif form == "chained":
                observed[("A1b-chained", relposix, enclosing)] += 1
                in_scope_emission = True
            elif form == "logger_object" and call["varname"] not in DEDICATED_VARS:
                in_scope_emission = True

            if form in ("a1a", "chained", "logger_object") and lit and PREFIX_RE.match(lit):
                observed[("A2", relposix, enclosing)] += 1

            if relposix == "main.py" and form == "print":
                observed[("A5a", relposix, enclosing)] += 1

            if relposix == "infrastructure/database_pg.py" and form in (
                "print",
                "a1a",
                "chained",
                "logger_object",
            ):
                if _args_contain_identifier(args, keywords, KEY_MATERIAL_IDENTIFIERS):
                    observed[("A5b", relposix, enclosing)] += 1

            if form == "print":
                observed[("A5c", relposix, enclosing)] += 1

        if in_scope_emission and not visitor.module_level_logger_bound:
            observed[("A1b-binding", relposix, "MODULE")] += 1

    return observed, alias_violations


_OBSERVED, _ALIAS_VIOLATIONS = scan(REPO_ROOT)


def _format_key(key):
    invariant, relpath, func = key
    return f"invariant={invariant} file={relpath} function={func}"


def test_no_unregistered_violations():
    allowed = set(BASELINE) | set(DECLARED_EXCEPTIONS)
    failures = []
    for key, count in sorted(_OBSERVED.items()):
        if key not in allowed:
            failures.append(
                f"{_format_key(key)}: {count} observed violation(s) but this key is "
                f"registered in neither tests/logging_baseline.py nor "
                f"tests/logging_declared_exceptions.py. If this is scheduled to be "
                f"fixed by a later axis, add it to BASELINE; if it is permanently "
                f"permitted, add it to DECLARED_EXCEPTIONS with a reason."
            )
            continue
        registered = BASELINE.get(key)
        if registered is None:
            registered = DECLARED_EXCEPTIONS[key][0]
        if count > registered:
            failures.append(
                f"{_format_key(key)}: observed {count} violation(s), but only "
                f"{registered} are registered. Update the registered count in "
                f"tests/logging_baseline.py or tests/logging_declared_exceptions.py "
                f"to match, or fix the new violation(s)."
            )
    assert not failures, "\n".join(failures)


def test_no_stale_baseline_entries():
    failures = []
    for key, registered in sorted(BASELINE.items()):
        observed_count = _OBSERVED.get(key, 0)
        if observed_count < registered:
            failures.append(
                f"{_format_key(key)}: tests/logging_baseline.py registers {registered} "
                f"violation(s) but only {observed_count} are observed now. Lower the "
                f"registered count to match, or remove the entry if the site was fixed."
            )
    for key, (registered, _reason) in sorted(DECLARED_EXCEPTIONS.items()):
        observed_count = _OBSERVED.get(key, 0)
        if observed_count < registered:
            failures.append(
                f"{_format_key(key)}: tests/logging_declared_exceptions.py registers "
                f"{registered} violation(s) but only {observed_count} are observed now. "
                f"Lower the registered count to match, or remove the entry if the site "
                f"no longer exists."
            )
    assert not failures, "\n".join(failures)


def test_logging_is_imported_unaliased():
    assert not _ALIAS_VIOLATIONS, (
        "The scanner's A1a detection assumes `logging` is always imported "
        "unaliased and `from logging import ...` is never used, so it can match "
        "on `logging.<level>(...)` textually instead of resolving aliases. The "
        "following occurrences break that assumption:\n"
        + "\n".join(_ALIAS_VIOLATIONS)
    )
