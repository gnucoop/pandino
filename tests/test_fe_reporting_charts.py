"""Tests for the deterministic drawing of recorded chart specs. Offline."""

import copy

import pytest

from evaluation.datachat.reporting.charts import chart_filename, render_chart_png

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def spec(kind="bar", **extra):
    return {"type": kind, "labels": ["North", "South", "Central"],
            "datasets": [{"label": "count", "data": [140, 110, 12]}],
            "title": "Richieste per regione", "x_label": "region", "y_label": "count",
            "stacked": False, "horizontal": False, **extra}


@pytest.mark.parametrize("value", [
    spec(), spec(horizontal=True), spec("line"), spec("pie"), spec("doughnut"),
    spec(stacked=True, datasets=[{"label": "a", "data": [1, 2, 3]}, {"label": "b", "data": [3, None, 1]}]),
    spec(datasets=[{"label": "a", "data": [1, 2, 3]}, {"label": "b", "data": [3, 2, 1]}]),
    {"type": "scatter", "labels": [], "datasets": [{"label": "pts", "data": [{"x": 1, "y": 2}, {"x": 3, "y": 1}]}],
     "title": None, "x_label": "wait_days", "y_label": "satisfaction", "stacked": False, "horizontal": False},
])
def test_supported_specs_draw_deterministically(value):
    before = copy.deepcopy(value)
    first, second = render_chart_png(value), render_chart_png(value)
    assert first is not None and first.startswith(PNG_SIGNATURE)
    assert first == second and value == before  # byte-identical, spec untouched


def test_different_values_draw_differently():
    other = spec(datasets=[{"label": "count", "data": [140, 110, 13]}])
    assert render_chart_png(spec()) != render_chart_png(other)


def test_no_timestamp_or_software_metadata():
    png = render_chart_png(spec())
    assert b"Software" not in png and b"tIME" not in png


@pytest.mark.parametrize("value", [
    None, "bar", spec("radar"),
    spec(datasets=[{"label": "count", "data": [1, 2]}]),            # length mismatch
    spec(datasets=[{"label": "count", "data": [1, "due", 3]}]),     # non-numeric
    spec(datasets=[{"label": "count", "data": [1, float("nan"), 3]}]),
    spec("pie", datasets=[{"data": [1, 2, 3]}, {"data": [1, 2, 3]}]),  # pie with two series
    spec(datasets=[{"label": str(i), "data": [1, 2, 3]} for i in range(9)]),  # beyond fixed colours
])
def test_specs_that_cannot_be_drawn_faithfully_are_refused(value):
    assert render_chart_png(value) is None


def test_stable_filenames():
    assert chart_filename("FE-01", 1) == "FE-01_chart_1.png"
    assert chart_filename("FE-13", 2) == "FE-13_chart_2.png"
