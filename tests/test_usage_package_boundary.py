"""Structural guards for the Usage package boundary.

The Usage subsystem lives in the root-level ``usage/`` package. Its logical
boundary was verified before the package existed; these guards keep the
physical boundary from drifting back:

    G1  routes/services reach Usage only through its adopter surface, plus
        the one declared ``get_usage_log_id`` compatibility accessor
    G2  the provider capture adapter imports only the producer surface
    G3  ``main.py`` knows exactly one Usage name, the lifecycle composition
    G4  no module imports the pre-package ``utils.*`` Usage paths

Plus one property the package's shape depends on: ``usage/__init__.py``
performs no imports, so importing a producer-facing accounting module does
not drag in Flask or the database layer (the capture adapter runs on a
background event-loop thread and depends on staying free of both).

These assert import *direction*, deliberately not the internal dependency
shape, module role labels or ``__all__`` contents.
"""

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# --- the ratified surfaces -------------------------------------------------

#: Adopter surface: routes and services may import anything public here.
ADOPTER_MODULES = frozenset(
    {"usage.recording", "usage.attribution", "usage.embedding_operation_context"}
)

#: Compatibility surface: exactly one accessor, because existing HTTP
#: contracts return ``log_id``. Setters and the multi-id internals stay in.
COMPATIBILITY = {"usage.request_state": frozenset({"get_usage_log_id"})}

#: Producer surface: what the embedding capture adapter is allowed to know.
PRODUCER = {
    "usage.embedding_accounting": frozenset(
        {
            "EmbeddingAccountingContribution",
            "COST_PROVIDER_AUTHORITATIVE",
            "ORIGIN_PROVIDER_REPORTED",
            "QUANTITY_UNIT_INPUT_TOKENS",
        }
    ),
    "usage.embedding_accounting_sink": frozenset({"get_embedding_accounting_sink"}),
    "usage.embedding_operation_context": frozenset({"get_embedding_operation"}),
}

#: Composition-root surface.
LIFECYCLE = {"usage.lifecycle": frozenset({"register_usage_lifecycle_hooks"})}

#: Pre-package paths of the modules that moved into ``usage/``.
STALE_PREFIXES = (
    "utils.usage_",
    "utils.embedding_accounting",
    "utils.embedding_operation_context",
    "utils.embedding_usage_",
)

PRODUCTION_DIRS = ("routes", "services", "infrastructure", "utils", "usage")


# --- helpers ---------------------------------------------------------------


def _python_files():
    yield REPO / "main.py"
    for name in PRODUCTION_DIRS + ("tests",):
        yield from sorted((REPO / name).glob("*.py"))


def _usage_imports(path):
    """Every ``usage.*`` import in one file as (module, imported_names).

    ``import usage.attribution as x`` yields no names: it binds the module
    itself, so only the module is constrained.
    """
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "usage" or alias.name.startswith("usage."):
                    found.append((alias.name, frozenset()))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "usage" or module.startswith("usage."):
                found.append((module, frozenset(a.name for a in node.names)))
    return found


def _rel(path):
    return str(path.relative_to(REPO))


# --- G1 --------------------------------------------------------------------


def test_g1_adopters_use_only_the_public_usage_surface():
    """Routes and services import Usage only through its declared surfaces."""
    violations = []
    for path in sorted(
        list((REPO / "routes").glob("*.py")) + list((REPO / "services").glob("*.py"))
    ):
        for module, names in _usage_imports(path):
            if module in ADOPTER_MODULES:
                continue
            if module in COMPATIBILITY:
                extra = names - COMPATIBILITY[module]
                if extra or not names:
                    violations.append(
                        f"{_rel(path)}: {module} exposes only "
                        f"{sorted(COMPATIBILITY[module])} to adopters, got "
                        f"{sorted(names) or 'a bare module import'}"
                    )
                continue
            violations.append(
                f"{_rel(path)}: {module} is Usage-internal; adopters use "
                f"{sorted(ADOPTER_MODULES)} or "
                f"usage.request_state.get_usage_log_id"
            )
    assert not violations, "adopter boundary violated:\n" + "\n".join(violations)


# --- G2 --------------------------------------------------------------------


def test_g2_provider_capture_imports_only_the_producer_surface():
    """The capture adapter knows the contribution contract, nothing deeper."""
    path = REPO / "infrastructure" / "embedding_capture.py"
    violations = []
    for module, names in _usage_imports(path):
        if module not in PRODUCER:
            violations.append(
                f"{module} is not part of the producer surface "
                f"{sorted(PRODUCER)}"
            )
            continue
        extra = names - PRODUCER[module]
        if extra:
            violations.append(f"{module}: unapproved names {sorted(extra)}")
    assert not violations, "producer boundary violated:\n" + "\n".join(violations)


# --- G3 --------------------------------------------------------------------


def test_g3_composition_root_imports_only_the_lifecycle_boundary():
    """``main.py`` owns no Usage registration order of its own."""
    imports = _usage_imports(REPO / "main.py")
    assert imports, "main.py must register the Usage lifecycle"
    violations = []
    for module, names in imports:
        if module not in LIFECYCLE:
            violations.append(f"{module} is not the composition-root surface")
            continue
        extra = names - LIFECYCLE[module]
        if extra:
            violations.append(
                f"{module}: main.py must not import lifecycle internals "
                f"{sorted(extra)}"
            )
    assert not violations, "composition root violated:\n" + "\n".join(violations)


# --- G4 --------------------------------------------------------------------


def test_g4_no_module_imports_the_pre_package_usage_paths():
    """The modules moved into ``usage/``; nothing imports their old paths."""
    violations = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [node.module or ""]
            else:
                continue
            for target in targets:
                if target.startswith(STALE_PREFIXES):
                    violations.append(f"{_rel(path)}:{node.lineno}: {target}")
    assert not violations, "stale pre-package imports:\n" + "\n".join(violations)


def test_g4_shared_timing_primitive_stays_outside_usage():
    """Usage orders the request timer; it does not own timing semantics."""
    assert (REPO / "utils" / "request_duration.py").is_file()
    assert not (REPO / "usage" / "request_duration.py").exists()


# --- package shape ---------------------------------------------------------


def test_usage_package_init_performs_no_imports():
    """A docstring-only ``__init__`` keeps producer-facing modules light."""
    init = REPO / "usage" / "__init__.py"
    assert init.is_file(), "usage/ must be a real package, not a namespace one"
    body = ast.parse(init.read_text()).body
    assert body, "usage/__init__.py should carry the surface-map docstring"
    offenders = [
        type(node).__name__
        for node in body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    assert not offenders, (
        "usage/__init__.py must contain only its docstring; found "
        f"{offenders}. Re-exporting here would make every usage.* import "
        "pull in the database layer."
    )
