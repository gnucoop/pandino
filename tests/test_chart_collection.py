"""Charts produced during a run must reach the client even if the model forgets them.

Regression origin: asked for a detailed comment plus two charts, the agent built both
successfully in one step, then called final_answer with a bare text payload in the next. The
specs stayed in the interpreter's local variables and the user got prose describing charts
that were never sent.

Attaching them was the model's job by design, which was the defect: it is bookkeeping the
model gains nothing from. Delivery is now deterministic.
"""

import os
import tempfile
from types import SimpleNamespace

import pandas as pd
import pytest

import datachat.smolagents_engine as se
from datachat.tools.chart_tool import ChartTool


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "programma": [f"Corso {i % 4}" for i in range(40)],
            "voto": [float(i % 4 + 1) for i in range(40)],
            "ruolo": ["Operatore" if i % 2 else "Coordinatore" for i in range(40)],
        }
    )


class Collector:
    """Minimal stand-in for the engine's record_chart sink."""

    def __init__(self) -> None:
        self.specs: list[dict] = []

    def record_chart(self, spec):
        self.specs.append(spec)


class ExplodingCollector:
    def record_chart(self, spec):
        raise RuntimeError("bookkeeping is on fire")


# ---------------------------------------------------------------------------
# Tool -> collector contract
# ---------------------------------------------------------------------------


def test_each_successful_chart_is_reported(df):
    collector = Collector()
    tool = ChartTool(df, collector=collector)

    tool.forward(kind="bar", x="programma")
    tool.forward(kind="pie", x="ruolo")

    assert len(collector.specs) == 2
    assert [s["type"] for s in collector.specs] == ["bar", "pie"]


def test_failed_charts_are_not_reported(df):
    collector = Collector()
    tool = ChartTool(df, collector=collector)

    tool.forward(kind="box", x="voto")       # belongs to plot
    tool.forward(kind="bar", x="nope")       # unknown column
    tool.forward(kind="scatter", x="voto")   # missing y

    assert collector.specs == []


def test_reported_spec_is_the_one_returned(df):
    collector = Collector()
    result = ChartTool(df, collector=collector).forward(kind="bar", x="programma")

    assert collector.specs[0] == result["chart"]


def test_the_tool_works_without_a_collector(df):
    result = ChartTool(df).forward(kind="bar", x="programma")

    assert result["kind"] == "chart"


def test_a_broken_collector_does_not_cost_the_chart(df):
    """Bookkeeping must never be able to fail the user's request."""
    result = ChartTool(df, collector=ExplodingCollector()).forward(kind="bar", x="programma")

    assert result["kind"] == "chart"
    assert result["chart"]["datasets"]


# ---------------------------------------------------------------------------
# Engine buffer
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine(df, monkeypatch):
    monkeypatch.setenv("DATACHAT_PLOTS_DIR", tempfile.mkdtemp())
    monkeypatch.setattr(se, "load_prompt", lambda t, default_text="", **k: default_text)
    monkeypatch.setattr(se, "build_litellm_model", lambda **k: object())
    return se.SmolagentsEngine(api_key="k", user_name="Test User", llm=None, data=df)


def _spec(label="x", chart_type="bar"):
    return {
        "type": chart_type,
        "labels": ["a", "b"],
        "datasets": [{"label": label, "data": [1, 2]}],
    }


def test_identical_specs_are_recorded_once(engine):
    engine.record_chart(_spec())
    engine.record_chart(_spec())

    assert len(engine._run_charts) == 1


def test_different_specs_are_both_kept(engine):
    engine.record_chart(_spec("a"))
    engine.record_chart(_spec("b"))

    assert len(engine._run_charts) == 2


def test_the_ceiling_holds(engine):
    for i in range(20):
        engine.record_chart(_spec(f"series {i}"))

    assert len(engine._run_charts) == se._MAX_RUN_CHARTS


def test_specs_without_datasets_are_ignored(engine):
    engine.record_chart({"type": "bar", "labels": ["a"]})
    engine.record_chart({"type": "bar", "datasets": []})
    engine.record_chart("nonsense")

    assert engine._run_charts == []


def test_close_clears_the_buffer(engine):
    engine.record_chart(_spec())
    engine.close()

    assert engine._run_charts == []


# ---------------------------------------------------------------------------
# Attachment to the final answer
# ---------------------------------------------------------------------------


class FakeAgent:
    """Builds `charts` charts, then returns `final` — the shape of the reported run."""

    def __init__(self, tool, final, charts=2):
        self._tool = tool
        self._final = final
        self._charts = charts

    def run(self, message, reset=True, return_full_result=True):
        specs = [("bar", "programma"), ("pie", "ruolo")]
        for kind, x in specs[: self._charts]:
            self._tool.forward(kind=kind, x=x)
        return SimpleNamespace(output=self._final)


def _wire(engine, df, final, charts=2):
    engine._agent = FakeAgent(ChartTool(df, collector=engine), final, charts)
    return engine


def test_charts_are_attached_when_the_model_forgets(engine, df):
    """The regression: two charts built, a bare text final answer."""
    _wire(engine, df, {"kind": "text", "text": "### Analisi ... i grafici mostrano ..."})

    result = engine.chat("analizza e crea 2 grafici")

    assert result["kind"] == "text"
    assert len(result["charts"]) == 2
    assert [c["type"] for c in result["charts"]] == ["bar", "pie"]


def test_a_model_supplied_list_is_left_alone(engine, df):
    """An explicit list states intent, including order — do not merge or reorder."""
    mine = _spec("mine", "line")
    _wire(engine, df, {"kind": "text", "text": "x", "charts": [mine]})

    result = engine.chat("q")

    assert result["charts"] == [mine]


def test_a_table_answer_also_receives_charts(engine, df):
    _wire(engine, df, {"kind": "table", "data": [{"a": 1}]})

    result = engine.chat("q")

    assert len(result["charts"]) == 2


def test_a_chart_answer_receives_the_other_charts(engine, df):
    """`chart` is a host too: returning one chart must not discard the rest."""
    _wire(engine, df, {"kind": "chart", "chart": _spec()})

    result = engine.chat("q")

    assert len(result["charts"]) == 2


def test_an_error_answer_is_not_decorated(engine, df):
    _wire(engine, df, {"kind": "error", "message": "boom"})

    result = engine.chat("q")

    assert "charts" not in result


def test_no_charts_built_means_no_charts_key(engine, df):
    _wire(engine, df, {"kind": "text", "text": "hi"}, charts=0)

    result = engine.chat("q")

    assert "charts" not in result


def test_charts_do_not_leak_between_requests(engine, df):
    _wire(engine, df, {"kind": "text", "text": "first"})
    first = engine.chat("first question")
    assert len(first["charts"]) == 2

    # A second run that builds nothing must come back clean.
    _wire(engine, df, {"kind": "text", "text": "second"}, charts=0)
    second = engine.chat("second question")

    assert "charts" not in second


def test_a_chart_answer_carries_the_other_charts(engine, df):
    """The model returns one chart and assumes the rest surface. They must."""
    engine._agent = FakeAgent(ChartTool(df, collector=engine), None, charts=2)

    # The fake agent returns its first chart as the answer, as the real one did.
    class ReturnsFirst(FakeAgent):
        def run(self, message, reset=True, return_full_result=True):
            built = [
                self._tool.forward(kind=k, x=x)
                for k, x in [("bar", "programma"), ("pie", "ruolo")]
            ]
            return SimpleNamespace(output=built[0])

    engine._agent = ReturnsFirst(ChartTool(df, collector=engine), None, charts=2)
    result = engine.chat("due grafici")

    assert result["kind"] == "chart"
    assert len(result["charts"]) == 1
    # The primary is never repeated among the extras.
    assert result["charts"][0] != result["chart"]
    assert result["charts"][0]["type"] == "pie"


def test_a_lone_chart_answer_gets_no_extras(engine, df):
    class ReturnsOnly(FakeAgent):
        def run(self, message, reset=True, return_full_result=True):
            return SimpleNamespace(output=self._tool.forward(kind="bar", x="programma"))

    engine._agent = ReturnsOnly(ChartTool(df, collector=engine), None, charts=1)
    result = engine.chat("un grafico")

    assert result["kind"] == "chart"
    assert "charts" not in result


def test_chart_extras_survive_the_normalizer(engine, df):
    from datachat.output_normalizer import normalize_datachat_response

    class ReturnsFirst(FakeAgent):
        def run(self, message, reset=True, return_full_result=True):
            built = [
                self._tool.forward(kind=k, x=x)
                for k, x in [("bar", "programma"), ("pie", "ruolo")]
            ]
            return SimpleNamespace(output=built[0])

    engine._agent = ReturnsFirst(ChartTool(df, collector=engine), None, charts=2)
    response = normalize_datachat_response(engine.chat("q"), exporter=engine)

    assert response["type"] == "chart"
    assert response["value"]["type"] == "bar"
    assert [c["type"] for c in response["charts"]] == ["pie"]


def test_attached_charts_survive_the_normalizer(engine, df):
    from datachat.output_normalizer import normalize_datachat_response

    _wire(engine, df, {"kind": "text", "text": "### Analisi"})
    result = engine.chat("q")

    response = normalize_datachat_response(result, exporter=engine)

    assert response["type"] == "str"
    assert len(response["charts"]) == 2
