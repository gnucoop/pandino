"""Tests for table preview + full-result CSV export in the DataChat normalizer.

The point of these tests is the guarantee that makes the preview honest: the client
gets a small labelled sample, and the CSV written alongside it holds *everything*.
"""

import os
import tempfile

import pandas as pd
import pytest

from datachat.output_normalizer import (
    _PREVIEW_COLUMNS,
    _PREVIEW_ROWS,
    normalize_datachat_response,
)


class FakeExporter:
    """Stands in for the engine's register_export sink."""

    def __init__(self, tmpdir: str) -> None:
        self._tmpdir = tmpdir
        self.calls: list[tuple[int, str]] = []
        self.last_path: str | None = None

    def register_export(self, records, hint="export"):
        self.calls.append((len(records), hint))
        token = f"tok{len(self.calls)}"
        path = os.path.join(self._tmpdir, f"{token}.csv")
        pd.DataFrame(records).to_csv(path, index=False)
        self.last_path = path
        return token, f"{hint}.csv"


class ExplodingExporter:
    def register_export(self, records, hint="export"):
        raise RuntimeError("disk on fire")


@pytest.fixture()
def exporter():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FakeExporter(tmpdir)


def _rows(n: int, cols: int = 2) -> list[dict]:
    return [{f"c{c}": f"r{r}c{c}" for c in range(cols)} for r in range(n)]


# ---------------------------------------------------------------------------
# Truncated results
# ---------------------------------------------------------------------------


def test_large_table_is_previewed_and_reports_the_real_total(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(340)}, exporter=exporter
    )

    assert response["type"] == "dataframe"
    assert len(response["value"]) == _PREVIEW_ROWS
    assert response["preview_rows"] == _PREVIEW_ROWS
    assert response["total_rows"] == 340
    assert response["truncated"] is True
    assert response["download_url"] == "/datachat/export/tok1"


def test_exported_csv_contains_every_row_and_column(exporter):
    """The core guarantee: nothing the preview drops is lost from the download."""
    normalize_datachat_response(
        {"kind": "table", "data": _rows(340, cols=15)}, exporter=exporter
    )

    exported = pd.read_csv(exporter.last_path)
    assert len(exported) == 340
    assert len(exported.columns) == 15


def test_export_name_becomes_the_download_filename(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(100), "export_name": "sentiment_reviews"},
        exporter=exporter,
    )

    assert response["download_filename"] == "sentiment_reviews.csv"
    assert exporter.calls == [(100, "sentiment_reviews")]


def test_too_many_columns_also_counts_as_truncated(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(3, cols=15)}, exporter=exporter
    )

    assert response["total_columns"] == 15
    assert len(response["value"][0]) == _PREVIEW_COLUMNS
    assert response["truncated"] is True
    assert response["download_url"] is not None


def test_dataframe_payload_takes_the_same_path(exporter):
    """A DataFrame used to bypass the caps entirely."""
    response = normalize_datachat_response(
        {"kind": "table", "data": pd.DataFrame(_rows(120))}, exporter=exporter
    )

    assert len(response["value"]) == _PREVIEW_ROWS
    assert response["total_rows"] == 120
    assert response["truncated"] is True


# ---------------------------------------------------------------------------
# Small results are untouched
# ---------------------------------------------------------------------------


def test_small_table_is_not_truncated_and_gets_no_download(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(12)}, exporter=exporter
    )

    assert len(response["value"]) == 12
    assert response["total_rows"] == 12
    assert response["truncated"] is False
    assert response["download_url"] is None
    assert exporter.calls == []


def test_table_at_exactly_the_preview_limit_is_not_truncated(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(_PREVIEW_ROWS)}, exporter=exporter
    )

    assert response["truncated"] is False
    assert exporter.calls == []


def test_empty_table_is_not_truncated(exporter):
    response = normalize_datachat_response({"kind": "table", "data": []}, exporter=exporter)

    assert response["value"] == []
    assert response["total_rows"] == 0
    assert response["truncated"] is False


# ---------------------------------------------------------------------------
# Degradation: a broken export must never cost the user their answer
# ---------------------------------------------------------------------------


def test_without_an_exporter_the_preview_still_works():
    response = normalize_datachat_response({"kind": "table", "data": _rows(340)})

    assert len(response["value"]) == _PREVIEW_ROWS
    assert response["total_rows"] == 340
    assert response["truncated"] is True
    assert response["download_url"] is None


def test_failing_exporter_degrades_to_a_labelled_preview():
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(340)}, exporter=ExplodingExporter()
    )

    assert len(response["value"]) == _PREVIEW_ROWS
    assert response["total_rows"] == 340
    assert response["truncated"] is True
    assert response["download_url"] is None


def test_nested_cell_values_are_still_flattened(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": [{"a": {"nested": 1}, "b": [1, 2]}]}, exporter=exporter
    )

    assert isinstance(response["value"][0]["a"], str)
    assert isinstance(response["value"][0]["b"], str)


# ---------------------------------------------------------------------------
# Downloads attached to text answers (export_csv tool)
# ---------------------------------------------------------------------------


def test_tool_note_is_forwarded_to_the_client(exporter):
    """A tool's caveat about its own result must not be dropped in transport."""
    response = normalize_datachat_response(
        {
            "kind": "table",
            "data": _rows(5),
            "note": "40 rows could not be analyzed.",
        },
        exporter=exporter,
    )

    assert response["note"] == "40 rows could not be analyzed."


def test_absent_note_is_omitted(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(5), "note": None}, exporter=exporter
    )

    assert "note" not in response


def test_text_response_forwards_a_download():
    response = normalize_datachat_response(
        {
            "kind": "text",
            "text": "Export pronto: 500 righe.",
            "download_url": "/datachat/export/abc",
            "download_filename": "dataset.csv",
        }
    )

    assert response["type"] == "str"
    assert response["value"] == "Export pronto: 500 righe."
    assert response["download_url"] == "/datachat/export/abc"
    assert response["download_filename"] == "dataset.csv"


def test_plain_text_response_is_unchanged():
    response = normalize_datachat_response({"kind": "text", "text": "hello"})

    assert response == {"type": "str", "value": "hello"}


# ---------------------------------------------------------------------------
# Charts (see DINO_CLIENT_SPEC.md)
#
# Regression origin: a response carried exactly one `kind`, so an answer could not be prose
# *and* charts. The agent generated two charts, returned only its commentary, and described
# images the user could never see.
# ---------------------------------------------------------------------------


def _chart(label="Risposte"):
    return {
        "type": "bar",
        "labels": ["1", "2"],
        "datasets": [{"label": label, "data": [10, 20]}],
        "title": None,
        "x_label": "voto",
        "y_label": "numero di risposte",
        "stacked": False,
    }


def test_text_can_carry_several_charts():
    """The reported bug: commentary plus two charts in one answer."""
    response = normalize_datachat_response(
        {"kind": "text", "text": "### Analisi", "charts": [_chart("a"), _chart("b")]}
    )

    assert response["type"] == "str"
    assert response["value"] == "### Analisi"
    assert len(response["charts"]) == 2
    assert [c["datasets"][0]["label"] for c in response["charts"]] == ["a", "b"]


def test_a_table_can_carry_a_chart(exporter):
    response = normalize_datachat_response(
        {"kind": "table", "data": _rows(5), "charts": [_chart()]}, exporter=exporter
    )

    assert response["type"] == "dataframe"
    assert len(response["value"]) == 5
    assert len(response["charts"]) == 1


def test_chart_only_response():
    response = normalize_datachat_response({"kind": "chart", "chart": _chart()})

    assert response["type"] == "chart"
    assert response["value"]["type"] == "bar"


def test_a_single_chart_passed_unwrapped_is_accepted():
    response = normalize_datachat_response({"kind": "text", "text": "x", "charts": _chart()})

    assert len(response["charts"]) == 1


def test_too_many_charts_are_capped_and_disclosed():
    response = normalize_datachat_response(
        {"kind": "text", "text": "x", "charts": [_chart()] * 9}
    )

    assert len(response["charts"]) == 6
    assert "only the first 6" in response["note"]


def test_malformed_charts_are_dropped_not_fatal():
    """One bad spec must not cost the user the whole answer."""
    response = normalize_datachat_response(
        {"kind": "text", "text": "x", "charts": [_chart(), {"type": "bar"}, "nonsense"]}
    )

    assert len(response["charts"]) == 1
    assert "could not be rendered" in response["note"]


def test_an_incomplete_chart_kind_degrades_to_text():
    response = normalize_datachat_response({"kind": "chart", "chart": {"type": "bar"}})

    assert response["type"] == "str"
    assert "could not be rendered" in response["value"]


def test_absent_charts_key_leaves_the_payload_untouched():
    assert normalize_datachat_response({"kind": "text", "text": "hi"}) == {
        "type": "str",
        "value": "hi",
    }


def test_charts_note_combines_with_a_tool_note():
    response = normalize_datachat_response(
        {"kind": "text", "text": "x", "note": "12 rows unscored.", "charts": [{"bad": 1}]}
    )

    assert "12 rows unscored." in response["note"]
    assert "could not be rendered" in response["note"]
