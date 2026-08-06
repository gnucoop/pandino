"""Tests for the chart tool — charts as data rather than base64 images.

Regression origin: asked for "un commento dettagliato + 2 grafici", the agent produced both
charts and the user saw neither. The contract carried exactly one `kind`, so prose and charts
could not coexist and the PNGs were dropped. Charts are now small enough to travel together.
"""

import numpy as np
import pandas as pd
import pytest

from datachat.tools.aggregate_tool import AggregateTool
from datachat.tools.chart_tool import ChartTool
from datachat.tools.unique_values_tool import UniqueValuesTool


@pytest.fixture()
def survey_df():
    """Italian survey shape: 1-4 ratings, a programme, a role, and a date."""
    n = 240
    return pd.DataFrame(
        {
            "programma": [f"Corso {i % 8}" for i in range(n)],
            "ruolo": ["Operatore" if i % 2 else "Coordinatore" for i in range(n)],
            "soddisfazione": [float(i % 4 + 1) for i in range(n)],
            "chiarezza": [float((i * 3) % 4 + 1) for i in range(n)],
            "mese": [f"2026-{(i % 12) + 1:02d}" for i in range(n)],
        }
    )


@pytest.fixture()
def tool(survey_df):
    return ChartTool(survey_df)


def _first(spec):
    return spec["datasets"][0]


# ---------------------------------------------------------------------------
# Every kind produces a well-formed spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,kwargs,expected_type",
    [
        ("bar", {"x": "programma"}, "bar"),
        ("line", {"x": "mese", "y": "soddisfazione"}, "line"),
        ("area", {"x": "mese", "y": "soddisfazione"}, "line"),
        ("pie", {"x": "ruolo"}, "pie"),
        ("doughnut", {"x": "ruolo"}, "doughnut"),
        ("hist", {"x": "soddisfazione"}, "bar"),
        ("kde", {"x": "soddisfazione"}, "line"),
    ],
)
def test_kind_produces_parallel_labels_and_data(tool, kind, kwargs, expected_type):
    result = tool.forward(kind=kind, **kwargs)

    assert result["kind"] == "chart"
    spec = result["chart"]
    assert spec["type"] == expected_type
    assert spec["labels"]
    assert len(_first(spec)["data"]) == len(spec["labels"])


def test_every_emitted_type_is_a_real_chartjs_type(tool):
    """No client-side translation step: 'area' must arrive as a filled line."""
    chartjs_types = {"bar", "line", "pie", "doughnut", "scatter"}

    for kind in ("bar", "line", "area", "pie", "doughnut", "hist", "kde"):
        kwargs = {"x": "mese", "y": "soddisfazione"} if kind in {"line", "area"} else {"x": "programma"}
        if kind in {"hist", "kde"}:
            kwargs = {"x": "soddisfazione"}
        spec = tool.forward(kind=kind, **kwargs)["chart"]
        assert spec["type"] in chartjs_types, kind


def test_area_is_a_filled_line(tool):
    spec = tool.forward(kind="area", x="mese", y="soddisfazione")["chart"]

    assert spec["type"] == "line"
    assert _first(spec)["fill"] is True


def test_scatter_carries_points_and_no_labels(tool):
    spec = tool.forward(kind="scatter", x="chiarezza", y="soddisfazione")["chart"]

    assert spec["type"] == "scatter"
    assert spec["labels"] is None
    assert all(set(p) == {"x", "y"} for p in _first(spec)["data"])


# ---------------------------------------------------------------------------
# The numbers must match the tools that answer the same question in table form
# ---------------------------------------------------------------------------


def test_counts_match_unique_values(tool, survey_df):
    """A chart must never disagree with the table printed beside it."""
    chart = tool.forward(kind="bar", x="programma")["chart"]
    table = UniqueValuesTool(survey_df).forward(column="programma")["data"]

    from_chart = dict(zip(chart["labels"], _first(chart)["data"]))
    from_table = {r["value"]: r["count"] for r in table}

    assert from_chart == from_table


def test_aggregates_match_aggregate_tool(tool, survey_df):
    chart = tool.forward(kind="bar", x="programma", y="soddisfazione", agg="mean")["chart"]
    table = AggregateTool(survey_df).forward(
        group_by="programma", op="mean", metric="soddisfazione"
    )["data"]

    from_chart = dict(zip(chart["labels"], _first(chart)["data"]))
    for row in table:
        assert from_chart[row["programma"]] == pytest.approx(row["mean_soddisfazione"], abs=1e-6)


def test_counts_serialise_as_integers(tool):
    """20.0 in a tooltip reads wrong; a count is a whole number."""
    data = _first(tool.forward(kind="bar", x="programma")["chart"])["data"]

    assert all(isinstance(v, int) for v in data)


# ---------------------------------------------------------------------------
# Multiple series
# ---------------------------------------------------------------------------


def test_series_by_yields_one_dataset_per_value(tool):
    spec = tool.forward(
        kind="bar", x="programma", y="soddisfazione", series_by="ruolo"
    )["chart"]

    assert {d["label"] for d in spec["datasets"]} == {"Operatore", "Coordinatore"}
    for dataset in spec["datasets"]:
        assert len(dataset["data"]) == len(spec["labels"])


def test_series_by_stacks_only_for_multi_series_bars(tool):
    grouped = tool.forward(kind="bar", x="programma", y="soddisfazione", series_by="ruolo")["chart"]
    plain = tool.forward(kind="bar", x="programma", y="soddisfazione")["chart"]

    assert grouped["stacked"] is True
    assert plain["stacked"] is False


def test_missing_combination_is_null_not_zero():
    """On a 1-4 scale a zero is not a possible answer, so a gap must stay a gap."""
    df = pd.DataFrame(
        {"g": ["A", "A", "B"], "s": ["X", "Y", "X"], "v": [4.0, 3.0, 2.0]}
    )
    spec = ChartTool(df).forward(kind="bar", x="g", y="v", series_by="s")["chart"]

    by_label = {d["label"]: dict(zip(spec["labels"], d["data"])) for d in spec["datasets"]}
    assert by_label["Y"]["B"] is None


def test_pie_rejects_multiple_series(tool):
    result = tool.forward(kind="pie", x="programma", series_by="ruolo")

    assert result["kind"] == "error"
    assert result["code"] == "NO_CHART_DATA"


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def test_hist_respects_bins(tool):
    spec = tool.forward(kind="hist", x="soddisfazione", bins=5)["chart"]

    assert len(spec["labels"]) == 5
    assert sum(_first(spec)["data"]) == 240


def test_kde_returns_a_monotonic_grid(tool):
    spec = tool.forward(kind="kde", x="soddisfazione")["chart"]
    grid = [float(v) for v in spec["labels"]]

    assert grid == sorted(grid)
    assert len(grid) == len(_first(spec)["data"])


def test_kde_needs_variation():
    df = pd.DataFrame({"v": [3.0] * 10})
    result = ChartTool(df).forward(kind="kde", x="v")

    assert result["kind"] == "error"


# ---------------------------------------------------------------------------
# Ordering, limits, disclosure
# ---------------------------------------------------------------------------


def test_sort_by_value_is_the_default(tool):
    data = _first(tool.forward(kind="bar", x="programma", y="soddisfazione")["chart"])["data"]

    assert data == sorted(data, reverse=True)


def test_natural_order_is_available_for_scales_and_dates(tool):
    """A 1-4 rating axis must not be reordered by frequency."""
    spec = tool.forward(kind="bar", x="soddisfazione", sort_by_value=False)["chart"]

    assert spec["labels"] == ["1.0", "2.0", "3.0", "4.0"]


def test_category_limit_is_disclosed(tool):
    result = tool.forward(kind="bar", x="programma", n=3)

    assert len(result["chart"]["labels"]) == 3
    assert "categories" in result["note"]


def test_no_limit_returns_every_category(tool):
    result = tool.forward(kind="bar", x="programma")

    assert len(result["chart"]["labels"]) == 8
    assert "note" not in result


def test_large_scatter_is_capped_and_disclosed():
    df = pd.DataFrame({"a": np.arange(3000.0), "b": np.arange(3000.0)})
    result = ChartTool(df).forward(kind="scatter", x="a", y="b")

    assert len(_first(result["chart"])["data"]) == 2000
    assert "2000" in result["note"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_box_points_at_the_plot_tool(tool):
    result = tool.forward(kind="box", x="soddisfazione")

    assert result["code"] == "KIND_NEEDS_PLOT"
    assert "plot" in result["message"]


def test_hexbin_points_at_the_plot_tool(tool):
    assert tool.forward(kind="hexbin", x="soddisfazione")["code"] == "KIND_NEEDS_PLOT"


def test_unknown_kind_is_rejected(tool):
    assert tool.forward(kind="violin", x="soddisfazione")["code"] == "INVALID_KIND"


def test_scatter_requires_y(tool):
    assert tool.forward(kind="scatter", x="soddisfazione")["code"] == "MISSING_Y"


def test_unknown_column_is_rejected(tool):
    assert tool.forward(kind="bar", x="nope")["code"] == "INVALID_COLUMN"


def test_agg_count_means_the_same_as_omitting_agg(tool):
    """The model carried aggregate's op="count" across and the call failed, killing a request."""
    explicit = tool.forward(kind="bar", x="programma", agg="count")
    implicit = tool.forward(kind="bar", x="programma")

    assert explicit["kind"] == "chart"
    assert explicit["chart"] == implicit["chart"]


def test_agg_count_ignores_y_like_the_aggregate_tool(tool, survey_df):
    """aggregate's op="count" ignores metric; chart must behave the same way."""
    result = tool.forward(kind="bar", x="programma", y="soddisfazione", agg="count")
    table = UniqueValuesTool(survey_df).forward(column="programma")["data"]

    assert result["kind"] == "chart"
    from_chart = dict(zip(result["chart"]["labels"], _first(result["chart"])["data"]))
    assert from_chart == {r["value"]: r["count"] for r in table}


def test_an_unusable_agg_cannot_fail_the_call(tool):
    """With no y there is nothing to aggregate, so the value is irrelevant, not invalid."""
    result = tool.forward(kind="bar", x="programma", agg="median")

    assert result["kind"] == "chart"


def test_a_bad_agg_is_still_rejected_when_it_applies(tool):
    result = tool.forward(kind="bar", x="programma", y="soddisfazione", agg="median")

    assert result["code"] == "INVALID_AGG"
    assert "count" in result["message"]


def test_non_numeric_metric_is_reported(tool):
    result = tool.forward(kind="bar", x="programma", y="ruolo", agg="mean")

    assert result["kind"] == "error"


def test_blank_categories_are_labelled_not_dropped():
    """The sentinel is language-neutral: the backend does not pick a display language."""
    df = pd.DataFrame({"g": ["A", None, "  ", "B"]})
    spec = ChartTool(df).forward(kind="bar", x="g")["chart"]

    assert "(empty)" in spec["labels"]


def test_upstream_data_is_used(tool):
    records = [{"g": "A", "v": 4.0}, {"g": "A", "v": 2.0}, {"g": "B", "v": 3.0}]
    spec = tool.forward(kind="bar", x="g", y="v", agg="mean", data=records)["chart"]

    assert dict(zip(spec["labels"], _first(spec)["data"])) == {"A": 3, "B": 3}


def test_title_and_axis_labels_are_carried(tool):
    spec = tool.forward(
        kind="bar", x="programma", y="soddisfazione", title="Soddisfazione per corso"
    )["chart"]

    assert spec["title"] == "Soddisfazione per corso"
    assert spec["x_label"] == "programma"
    assert spec["y_label"] == "mean(soddisfazione)"


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def _long_labels_df(count=6, length=30):
    return pd.DataFrame(
        {
            "categoria": [f"{'C' * length} {i}" for i in range(count)],
            "valore": [float(i + 1) for i in range(count)],
        }
    )


def test_long_labels_default_to_horizontal():
    """Vertical bars with 30-character labels either truncate or rotate to unreadability."""
    spec = ChartTool(_long_labels_df()).forward(
        kind="bar", x="categoria", y="valore"
    )["chart"]

    assert spec["horizontal"] is True


def test_short_labels_stay_vertical(tool):
    spec = tool.forward(kind="bar", x="soddisfazione", sort_by_value=False)["chart"]

    assert spec["horizontal"] is False


def test_many_categories_default_to_horizontal():
    df = pd.DataFrame({"c": [f"g{i}" for i in range(20)], "v": [1.0] * 20})
    spec = ChartTool(df).forward(kind="bar", x="c", y="v")["chart"]

    assert spec["horizontal"] is True


def test_a_single_long_outlier_does_not_flip_the_axis():
    """One freak value can be truncated; a whole axis of long labels cannot."""
    df = pd.DataFrame(
        {
            "c": ["a", "b", "c", "d", "UNA DOMANDA MOLTO LUNGA CHE OCCUPA TANTISSIMO SPAZIO"],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    spec = ChartTool(df).forward(kind="bar", x="c", y="v")["chart"]

    assert spec["horizontal"] is False


def test_histograms_are_always_vertical(tool):
    """Bins belong on the x axis: that is what makes it read as a distribution."""
    spec = tool.forward(kind="hist", x="soddisfazione", bins=30)["chart"]

    assert spec["horizontal"] is False


@pytest.mark.parametrize("kind,kwargs", [
    ("pie", {"x": "programma"}),
    ("doughnut", {"x": "programma"}),
    ("line", {"x": "mese", "y": "soddisfazione"}),
    ("scatter", {"x": "chiarezza", "y": "soddisfazione"}),
])
def test_non_bar_kinds_have_no_orientation(tool, kind, kwargs):
    spec = tool.forward(kind=kind, **kwargs)["chart"]

    assert spec["horizontal"] is False


def test_explicit_horizontal_overrides_the_heuristic(tool):
    forced_on = tool.forward(kind="bar", x="programma", horizontal=True)["chart"]
    forced_off = ChartTool(_long_labels_df()).forward(
        kind="bar", x="categoria", y="valore", horizontal=False
    )["chart"]

    assert forced_on["horizontal"] is True
    assert forced_off["horizontal"] is False


def test_horizontal_bars_put_the_largest_first():
    """With horizontal bars the first entry renders at the top, so descending == top-down."""
    spec = ChartTool(_long_labels_df()).forward(
        kind="bar", x="categoria", y="valore"
    )["chart"]
    data = _first(spec)["data"]

    assert spec["horizontal"] is True
    assert data == sorted(data, reverse=True)


def test_no_colours_are_emitted(tool):
    """Palette and theming belong to the client; the backend must not dictate them."""
    spec = tool.forward(kind="bar", x="programma", y="soddisfazione", series_by="ruolo")["chart"]

    forbidden = {"backgroundColor", "borderColor", "color", "colors", "options"}
    assert not (forbidden & set(spec))
    for dataset in spec["datasets"]:
        assert not (forbidden & set(dataset))
