"""
INVARIANT A0
The test suite contains a permanent test, driven by the `ast` module of the
Python standard library, which walks the whole repository (excluding `venv/`)
and asserts:

  violations(A1) union violations(A2) union violations(A5a) union
  violations(A5b) union violations(A5c)  is a subset of  BASELINE union
  DECLARED_EXCEPTIONS

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


"""
INVARIANT O1-O5 (FOUNDATION INTERVENTION I7)

Operational Persistence static invariants, additive to A0-A5c above and to
the same BASELINE/DECLARED_EXCEPTIONS registries' spirit (though O1-O5 start
with zero known violations and therefore need no BASELINE entries: current
production adoption of the persistent metadata surface is exactly zero).

Scope for O1-O4 is PRODUCTION code only: the same repository walk as A0-A5c,
excluding venv/__pycache__/.git/docs (as above) and ALSO excluding `tests/`
and the two persistence-implementation modules themselves
(utils/operational_event.py, utils/operational_persistence.py). Tests are
explicitly sanctioned to construct synthetic marked LogRecords and literal
extra= mappings for their own fixtures (e.g. tests/test_operational_event_
contract.py's canonical-form check, tests/test_operational_persistence_
bootstrap.py's synthetic marked events) - forbidding that in test fixtures
would make the persistence subsystem untestable, not safer. O1-O4 forbid it
only where it matters: at production call sites.

O1  No production logging call (`logger.<level>(...)`, any of the A1
    forms) passes a DICT LITERAL, or anything other than a bare Name, as
    `extra=`. The only sanctioned value is a Name bound to the extra half of
    a `message, extra = build_operational_event(...)` unpacking (O2).

O2  For a logging call passing `extra=<Name>`, the SAME function scope must
    bind that Name via a two-element tuple/list unpacking whose value is a
    call to `build_operational_event(...)`, and the logging call's first
    positional argument must be the OTHER (message) name from that same
    unpacking. This is the static half of the anti-drift requirement: the
    rendered text and the structured event can only diverge if two
    independent values are constructed, and the canonical form constructs
    both from one call.

O3  request_id/app_id are infrastructure-owned and unfalsifiable by a call
    site: no call to `build_operational_event` may pass `request_id=` or
    `app_id=`, and no dict literal used as `extra=` may declare either key.

O4  No production logging call declares any `maui_*` metadata key directly
    inside a literal `extra=` mapping. The only sanctioned way to place
    `maui_*` attributes on a LogRecord is through the builder's returned
    mapping (O1 already forbids the literal mapping itself; O4 additionally
    names the specific keys, so a future relaxation of O1 for some other
    reason would not silently reopen marker forgery).

O5  Neither utils/operational_persistence.py nor infrastructure/database_pg.py
    may call build_operational_event, and neither may contain the literal
    `maui_persist` anywhere in its source. This is the static form of the
    recursion barrier: neither the transport nor the writer may manufacture
    a persistent event or self-mark its own diagnostics.

Keeping the analysis conservative and non-data-flow: O2's pairing check is a
same-function, name-equality check, not a control-flow or type analysis. A
builder result assigned in one function and passed to logger.<level>() in
another is out of scope for O2 by design (ARCH: builder output must not
"escape across unrelated scopes before logging" - such a call site simply
fails O2, which is the intended, conservative outcome).
"""

O_EXCLUDED_DIR_NAMES = EXCLUDED_DIR_NAMES | {"tests"}

O_SANCTIONED_RELPATHS = {
    "utils/operational_event.py",
    "utils/operational_persistence.py",
}

MAUI_KEYS = {
    "maui_persist",
    "maui_event",
    "maui_provider",
    "maui_model",
    "maui_duration_ms",
    "maui_error_type",
    "maui_details",
    "maui_message",
}

CONTEXT_OWNERSHIP_KEYS = {"request_id", "app_id"}


def _walk_o_family_py_files(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in O_EXCLUDED_DIR_NAMES]
        for fn in sorted(files):
            if fn.endswith(".py"):
                yield os.path.join(base, fn)


def _o_family_full_attr_or_name(func_node):
    if isinstance(func_node, ast.Attribute):
        return _get_full_attr(func_node)
    if isinstance(func_node, ast.Name):
        return func_node.id
    return None


def _is_build_operational_event_call(func_node):
    full = _o_family_full_attr_or_name(func_node)
    return full == "build_operational_event" or (
        full is not None and full.endswith(".build_operational_event")
    )


class _OFamilyVisitor(ast.NodeVisitor):
    """Walks one production module, recording (per function scope):
    - logging calls in the A1 forms, with their args/keywords;
    - `name, name = build_operational_event(...)` unpacking bindings;
    - direct calls to build_operational_event, for the O3 request_id/app_id
      keyword check.
    """

    def __init__(self, relpath):
        self.relpath = relpath
        self.func_stack = []
        self.logging_calls = []
        self.builder_bindings = []
        self.builder_calls = []

    def _enclosing(self):
        return self.func_stack[-1] if self.func_stack else "MODULE"

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        target_ok = (
            len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and len(node.targets[0].elts) == 2
            and all(isinstance(e, ast.Name) for e in node.targets[0].elts)
        )
        if (
            target_ok
            and isinstance(node.value, ast.Call)
            and _is_build_operational_event_call(node.value.func)
        ):
            msg_name, extra_name = (e.id for e in node.targets[0].elts)
            self.builder_bindings.append(
                {
                    "enclosing": self._enclosing(),
                    "msg_name": msg_name,
                    "extra_name": extra_name,
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func

        if _is_build_operational_event_call(func):
            self.builder_calls.append(
                {"enclosing": self._enclosing(), "keywords": node.keywords}
            )

        if isinstance(func, ast.Attribute) and func.attr in LEVELS:
            fullattr = _get_full_attr(func)
            is_logging_call = False
            if fullattr and fullattr.startswith("logging."):
                is_logging_call = True
            elif fullattr and fullattr.endswith("current_app.logger." + func.attr):
                is_logging_call = True
            elif fullattr and "." in fullattr:
                is_logging_call = True  # logger_object form: <name>.<level>()
            elif isinstance(func.value, ast.Call):
                inner_full = _o_family_full_attr_or_name(func.value.func)
                if inner_full in ("logging.getLogger", "getLogger"):
                    is_logging_call = True

            if is_logging_call:
                self.logging_calls.append(
                    {
                        "enclosing": self._enclosing(),
                        "args": node.args,
                        "keywords": node.keywords,
                    }
                )

        self.generic_visit(node)


def _dict_string_keys(dict_node):
    for key in dict_node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            yield key.value


def scan_o_family(root):
    """Returns a Counter keyed (invariant, posix_relative_path,
    enclosing_function_name), covering O1-O4. O5 is checked separately
    (test_o5_*) as a direct source-text assertion over exactly two files.
    """
    observed = Counter()

    for fpath in _walk_o_family_py_files(root):
        rel = os.path.relpath(fpath, root)
        relposix = rel.replace(os.sep, "/")
        if relposix in O_SANCTIONED_RELPATHS:
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=fpath)

        visitor = _OFamilyVisitor(relposix)
        visitor.visit(tree)

        bindings_by_func = {}
        for binding in visitor.builder_bindings:
            bindings_by_func.setdefault(binding["enclosing"], []).append(binding)

        for call in visitor.logging_calls:
            enclosing = call["enclosing"]
            extra_kw = next(
                (kw for kw in call["keywords"] if kw.arg == "extra"), None
            )
            if extra_kw is None:
                continue

            if not isinstance(extra_kw.value, ast.Name):
                observed[("O1", relposix, enclosing)] += 1
                if isinstance(extra_kw.value, ast.Dict):
                    keys = set(_dict_string_keys(extra_kw.value))
                    if keys & CONTEXT_OWNERSHIP_KEYS:
                        observed[("O3", relposix, enclosing)] += 1
                    if keys & MAUI_KEYS:
                        observed[("O4", relposix, enclosing)] += 1
                continue

            extra_name = extra_kw.value.id
            matching = [
                b
                for b in bindings_by_func.get(enclosing, [])
                if b["extra_name"] == extra_name
            ]
            if not matching:
                observed[("O2", relposix, enclosing)] += 1
                continue

            first_arg = call["args"][0] if call["args"] else None
            msg_name = matching[0]["msg_name"]
            if not (isinstance(first_arg, ast.Name) and first_arg.id == msg_name):
                observed[("O2", relposix, enclosing)] += 1

        for call in visitor.builder_calls:
            for kw in call["keywords"]:
                if kw.arg in CONTEXT_OWNERSHIP_KEYS:
                    observed[("O3", relposix, call["enclosing"])] += 1

    return observed


_O_OBSERVED = scan_o_family(REPO_ROOT)


def _o_family_failures(invariant):
    return [
        f"{_format_key(key)}: {count} observed violation(s) of {invariant}. "
        "Production persistent-event call sites must use the canonical "
        "`message, extra = build_operational_event(...)` form (see "
        "docs/logging/operational_persistence_foundation_tdd.md section 16)."
        for key, count in sorted(_O_OBSERVED.items())
        if key[0] == invariant
    ]


def test_o1_no_literal_extra_mappings():
    failures = _o_family_failures("O1")
    assert not failures, "\n".join(failures)


def test_o2_builder_pairing():
    failures = _o_family_failures("O2")
    assert not failures, "\n".join(failures)


def test_o3_context_ownership_is_not_call_site_supplied():
    failures = _o_family_failures("O3")
    assert not failures, "\n".join(failures)


def test_o4_no_raw_maui_metadata_outside_the_builder():
    failures = _o_family_failures("O4")
    assert not failures, "\n".join(failures)


def test_o5_persistence_subsystem_does_not_self_mark():
    """utils/operational_persistence.py legitimately reads the
    ``maui_persist`` attribute off the LogRecord (the marker check in
    ``emit()`` IS the recursion barrier), so a blanket text search for the
    string would flag the barrier's own implementation. What O5 forbids is
    narrower and load-bearing: neither file may CALL build_operational_event,
    and neither file's OWN diagnostic logging calls may carry "maui_persist"
    as a literal argument (i.e. hand-roll the marker onto a subsystem
    diagnostic instead of leaving it, correctly, unmarked).
    """
    failures = []
    for relpath in ("utils/operational_persistence.py", "infrastructure/database_pg.py"):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(REPO_ROOT / relpath))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_build_operational_event_call(
                node.func
            ):
                failures.append(
                    f"{relpath}: calls build_operational_event() - the "
                    "transport/writer must never manufacture a persistent "
                    "event itself."
                )

        visitor = _OFamilyVisitor(relpath)
        visitor.visit(tree)
        for call in visitor.logging_calls:
            call_nodes = list(call["args"]) + [kw.value for kw in call["keywords"]]
            for arg_node in call_nodes:
                for sub in ast.walk(arg_node):
                    if (
                        isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str)
                        and "maui_persist" in sub.value
                    ):
                        failures.append(
                            f"{relpath} function={call['enclosing']}: a "
                            "diagnostic logging call carries the literal "
                            "'maui_persist' - subsystem diagnostics must "
                            "remain unmarked."
                        )

    assert not failures, "\n".join(failures)
