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
