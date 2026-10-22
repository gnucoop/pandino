"""Tests for the task agents' guarded Python executor."""

import pandas as pd
import pytest
from smolagents.local_python_executor import InterpreterError

from analyst.sandbox import GuardedPythonExecutor, check_code, unescape_code


@pytest.mark.parametrize(
    "code",
    [
        "df.to_csv('/tmp/x.csv')",
        "df.to_pickle('/tmp/x.pkl')",
        "df.to_parquet('x')",
        "df.to_json('x.json')",
        "df.values.tofile('x')",
        "df.values.dump('x')",
        "df.plot().figure.savefig('x.png')",
        "df._mgr",
        "df.values.ctypes",
        "df.eval('a + 1')",
        "df.pipe(print)",
        "df.style",
        "f = getattr(df, 'to_csv')",
        "open('/etc/passwd')",
        "df.to_string('/tmp/out.txt')",
        "df.to_markdown(buf='/tmp/out.md')",
    ],
)
def test_file_and_dynamic_access_is_rejected(code):
    assert check_code(code) is not None


@pytest.mark.parametrize(
    "code",
    [
        "df.groupby('m')['v'].sum().to_dict()",
        "df['v'].tolist()",
        "df.to_numpy()",
        "print(df.head(3).to_string())",
        "print(df.head(3).to_markdown(index=False))",
        "import json\nprint(json.dumps({'a': 1}))",
        "json.loads('{}')",
        "df.query('v > 1')",
        "df.pivot_table(index='m', values='v', aggfunc='sum')",
        "this is not python",
    ],
)
def test_analysis_code_is_accepted(code):
    assert check_code(code) is None


def executor_with(frame):
    executor = GuardedPythonExecutor()
    executor.send_tools({})
    executor.send_variables({"datasets": {"trips": frame}})
    return executor


def test_computations_run_on_injected_dataframes():
    frame = pd.DataFrame({"m": ["a", "a", "b"], "v": [1, 2, 4]})
    executor = executor_with(frame)

    result = executor("totals = datasets['trips'].groupby('m')['v'].sum().to_dict()\nprint(totals)")

    assert "{'a': 3, 'b': 4}" in result.logs
    assert executor.state["totals"] == {"a": 3, "b": 4}


def test_a_rejected_snippet_never_runs():
    frame = pd.DataFrame({"v": [1]})
    executor = executor_with(frame)

    with pytest.raises(InterpreterError, match="rejected by the sandbox"):
        executor("x = 1\ndatasets['trips'].to_csv('/tmp/leak.csv')")
    assert "x" not in executor.state


def test_pandas_and_numpy_cannot_be_imported():
    executor = executor_with(pd.DataFrame())
    for module in ("pandas", "numpy", "os"):
        with pytest.raises(InterpreterError):
            executor(f"import {module}")


# ---------------------------------------------------------------------------
# Escaped code steps
# ---------------------------------------------------------------------------

# A code step as the model sometimes writes it: one line, literal \n for line
# breaks, \" for quotes and the markdown escape \] (shortened from a real run).
ESCAPED = (
    r'import json\n\nqr_str = """{\n  \"fonti\": {\"report_mensile\": {\"row_count\": 527, '
    r'\"mesi\": [\"202601\", \"202602\"\]}}\n}"""\n\nqr = json.loads(qr_str)\n'
    r'print(\"fonti:\", list(qr[\"fonti\"].keys()))\n'
)


def test_an_escaped_code_step_is_decoded_and_runs():
    executor = executor_with(pd.DataFrame())

    result = executor(ESCAPED)

    assert "fonti: ['report_mensile']" in result.logs
    assert executor.state["qr"]["fonti"]["report_mensile"]["mesi"] == ["202601", "202602"]


def test_a_single_escaped_line_in_real_code_is_repaired():
    code = 'a = 1\nb = 2\\nprint(\\"sum\\", a + b)\n'
    assert unescape_code(code) == 'a = 1\nb = 2\nprint("sum", a + b)\n'


def test_valid_code_is_never_touched():
    code = 'print("a\\nb")'
    assert unescape_code(code) is None

    result = executor_with(pd.DataFrame())(code)
    assert result.logs == "a\nb\n"


def test_code_still_broken_once_decoded_keeps_its_error():
    code = r"x = (1,\n"
    assert unescape_code(code) is None
    with pytest.raises(InterpreterError, match="Code parsing failed"):
        executor_with(pd.DataFrame())(code)


def test_the_guard_checks_the_decoded_code():
    frame = pd.DataFrame({"v": [1]})
    with pytest.raises(InterpreterError, match="rejected by the sandbox"):
        executor_with(frame)(r"x = 1\ndatasets[\"trips\"].to_csv(\"/tmp/leak.csv\")")


def test_a_split_escape_after_a_comment_is_repaired():
    # As seen in a run: `'nprint' is not among the explicitly allowed tools`.
    code = "r = 41 + 1\n# show it\\\nnprint(r)\n"
    assert unescape_code(code) == "r = 41 + 1\n# show it\nprint(r)\n"

    result = executor_with(pd.DataFrame())(code)
    assert result.logs == "42\n"


def test_a_split_escape_in_code_that_does_not_parse_is_repaired():
    code = 'cov = {"a": 1}\\\nnprint(cov)\n'
    assert unescape_code(code) == 'cov = {"a": 1}\nprint(cov)\n'


def test_a_real_continuation_is_left_alone():
    code = "a, n_rows = 1, 2\ntotal = (a +\\\nn_rows)\n"
    assert unescape_code(code) is None
    executor = executor_with(pd.DataFrame())
    executor(code)
    assert executor.state["total"] == 3


@pytest.mark.parametrize(
    "code, expected",
    [
        # As seen in a run: a line holding only the backslash of a split \n.
        ("x = 1\n\\\nnprint(x)\n", "x = 1\n\nprint(x)\n"),
        ("x = 1\n    \\\nnprint(x)\n", "x = 1\n\nprint(x)\n"),
        ("\\\nnprint(1)\n", "\nprint(1)\n"),
    ],
)
def test_a_lone_backslash_split_is_repaired(code, expected):
    assert unescape_code(code) == expected


def test_a_lone_backslash_split_runs_as_meant():
    result = executor_with(pd.DataFrame())("x = 1\n\\\nnprint(x)\n")
    assert result.logs == "1\n"


@pytest.mark.parametrize(
    "code",
    [
        "d = {'a': 1}\nfor k in d:\\\n    print(k)\n",
        "n_rows = 2\ntotal = (1 +\\\nn_rows)\n",
    ],
)
def test_continuations_without_a_lone_backslash_are_left_alone(code):
    assert unescape_code(code) is None


class _FakeDatasource:
    password = "s3cret"

    def run_select(self, query, max_rows):
        return ["ran"], [(query,)], False


@pytest.mark.parametrize(
    "code",
    [
        "r = sql_engine._datasource.run_select('SELECT * FROM hidden_table', 5)",
        "r = sql_engine._datasource.password",
        "r = getattr(sql_engine, '_datasource')",
    ],
)
def test_code_cannot_reach_past_a_tool_to_its_datasource(code):
    from smolagents import Tool

    class SqlTool(Tool):
        name = "sql_engine"
        description = "Runs a guarded query."
        inputs = {"query": {"type": "string", "description": "SQL"}}
        output_type = "string"

        def __init__(self):
            super().__init__()
            self._datasource = _FakeDatasource()

        def forward(self, query):
            return "guarded"

    executor = GuardedPythonExecutor()
    executor.send_tools({"sql_engine": SqlTool()})
    executor.send_variables({})

    with pytest.raises(InterpreterError, match="rejected by the sandbox"):
        executor(code)
    assert "r" not in executor.state
