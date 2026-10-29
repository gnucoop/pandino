"""
sandbox.py
----------

The Python executor of the task agents: smolagents' local executor plus a guard.

Task agents compute on DataFrames placed in their namespace (`datasets`,
`inputs`), so that rows never travel through the model's context. smolagents
already refuses unauthorized imports, dunder access and modules such as os,
io or subprocess, but it only checks what a call *returns*: whatever pandas or
numpy do internally goes unchecked, and the DataFrame API reads and writes
files (to_csv, to_pickle, ndarray.tofile/dump, Figure.savefig ...).

So before any code runs, check_code() walks its syntax tree and rejects:

- every attribute starting with "_" (pandas/numpy internals) or named in
  _FORBIDDEN_ATTRIBUTES, and any `to_*` / `read_*` / `load*` / `save*` /
  `dump*` outside _ALLOWED_METHODS;
- to_string/to_markdown given a positional argument or a `buf`: both write to
  a file when given one, and both are otherwise the natural way to print a
  small table;
- names reaching objects dynamically (getattr, vars, open, eval, ...).

pandas and numpy are not importable at all (see AUTHORIZED_IMPORTS): with no
module object around, pd.read_csv or np.load simply do not exist. A rejected
snippet never starts, and the model is told why.

The executor also repairs one model slip (unescape_code): code written as an
escaped string, with literal \\n for line breaks and \\" for quotes, which
Python refuses before running anything. Every analyst agent and the interviewer
run in GuardedPythonExecutor. The guard matters most for the agents holding SQL
tools: smolagents' own executor lets code reach a tool's private attributes
(sql_engine._datasource), so it could query past the SQL guard or read the
database configuration.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

from smolagents.local_python_executor import CodeOutput, InterpreterError, LocalPythonExecutor

# Modules the agent code may import. Everything heavier (pandas, numpy,
# sklearn) is reached only through the injected objects and the tools.
AUTHORIZED_IMPORTS = ["json", "math", "statistics", "datetime"]

# Exceptions to _FORBIDDEN_PREFIXES. json.dumps/loads only handle strings.
_ALLOWED_METHODS = frozenset({
    "dumps",
    "loads",
    "to_dict",
    "to_list",
    "tolist",
    "to_numpy",
    "to_frame",
    "to_period",
    "to_timestamp",
    "to_pydatetime",
    "to_records",
    "to_string",
    "to_markdown",
})
# Allowed only without a positional argument or a `buf`, which would be a path.
_PRINT_ONLY_METHODS = frozenset({"to_string", "to_markdown"})

_FORBIDDEN_PREFIXES = ("to_", "read_", "load", "save", "dump")
_FORBIDDEN_ATTRIBUTES = frozenset({
    "tofile",
    "fromfile",
    "memmap",
    "eval",
    "pipe",
    "ctypes",
    "style",
    "plot",
})
_FORBIDDEN_NAMES = frozenset({
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "globals",
    "locals",
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "input",
})


def _forbidden_attribute(attr: str) -> bool:
    if attr.startswith("_") or attr in _FORBIDDEN_ATTRIBUTES:
        return True
    if attr in _ALLOWED_METHODS:
        return False
    return attr.startswith(_FORBIDDEN_PREFIXES)


def check_code(code: str) -> Optional[str]:
    """
    Why the code may not run, or None when it may.

    Code that does not parse is let through: the executor reports the syntax
    error to the model as usual.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _forbidden_attribute(node.attr):
            return f"`.{node.attr}` is not allowed"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return f"`{node.id}` is not allowed"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _PRINT_ONLY_METHODS
            and (node.args or any(k.arg == "buf" for k in node.keywords))
        ):
            return f"`.{node.func.attr}()` may only return text: call it with no file argument"
    return None


# One escape of a code step written as an escaped string: \n and \t, quotes,
# the backslash itself, and the markdown escapes \[ \] the model adds.
_ESCAPE_RE = re.compile(r"""\\(n|t|"|'|\\|\[|\])""")
_UNESCAPED = {"n": "\n", "t": "\t"}

# A \n escape split by a real line break: backslash, newline, "n". After a
# comment, or on a line of its own, the backslash is swallowed and the code
# parses with a stray name (`nprint(...)`), so those cases are repaired whenever
# they occur: no real code ends a comment with a backslash, or holds a lone
# backslash on a line, before a line starting with "n". Anywhere else it would
# be a valid continuation, so it is only repaired in code that does not parse.
_COMMENT_SPLIT_RE = re.compile(r"(#[^\n]*)\\\r?\nn")
_LONE_BACKSLASH_SPLIT_RE = re.compile(r"(^|\n)[ \t]*\\\r?\nn")
_SPLIT_RE = re.compile(r"\\\r?\nn")


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def unescape_code(code: str) -> Optional[str]:
    """
    The code with its escapes decoded, when the model wrote it (or part of it)
    as an escaped string: literal \\n for line breaks, or a \\n split by a
    real line break. None when there is nothing to repair or the repaired code
    does not parse either. Valid code is never touched (but for the split
    after a comment or a lone backslash), so a "\\n" inside one of its strings
    keeps its meaning.
    """
    fixed = _COMMENT_SPLIT_RE.sub(lambda m: m.group(1) + "\n", code)
    fixed = _LONE_BACKSLASH_SPLIT_RE.sub(lambda m: m.group(1) + "\n", fixed)
    if _parses(fixed):
        return fixed if fixed != code else None
    fixed = _SPLIT_RE.sub("\n", fixed)
    if "\\n" in fixed and not _parses(fixed):
        # One left-to-right pass: an escaped backslash followed by n stays a string escape.
        fixed = _ESCAPE_RE.sub(lambda m: _UNESCAPED.get(m.group(1), m.group(1)), fixed)
    return fixed if fixed != code and _parses(fixed) else None


class GuardedPythonExecutor(LocalPythonExecutor):
    """
    LocalPythonExecutor that runs an escaped code step decoded (unescape_code)
    and refuses code check_code() rejects: the check sees the code that runs.
    """

    def __init__(self, max_print_outputs_length: Optional[int] = None) -> None:
        super().__init__(AUTHORIZED_IMPORTS, max_print_outputs_length=max_print_outputs_length)

    def __call__(self, code_action: str) -> CodeOutput:
        code_action = unescape_code(code_action) or code_action
        problem = check_code(code_action)
        if problem is not None:
            # The previous snippet's prints would otherwise be replayed with the error.
            self.state.pop("_print_outputs", None)
            raise InterpreterError(
                f"Code rejected by the sandbox: {problem}. The code may not touch files, "
                "the network or objects' internals: compute with DataFrame methods and "
                "the tools, and print short summaries only."
            )
        return super().__call__(code_action)
