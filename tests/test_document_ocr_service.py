"""
Tests for the provider-independent PDF rendering pipeline.

The service under test only renders PDF pages to PNG bytes. It must remain
independent from Flask, database setup, and AI/Vision provider configuration.
"""

import importlib
import io
from typing import Any, cast

import numpy as np
import pytest
import pymupdf
from PIL import Image, ImageDraw

from services.document_ocr_service import (
    RenderedPdfPage,
    is_rendered_page_blank,
    render_pdf_pages_to_png,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _make_pdf(page_count: int) -> bytes:
    with pymupdf.open() as document:
        document_for_typing = cast(Any, document)
        for index in range(page_count):
            page = document_for_typing.new_page(width=160, height=120)
            page.insert_text((24, 48), f"Page {index + 1}")

        return document.tobytes()


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_render_one_page_pdf_returns_one_rendered_page():
    """Verify the basic PDF-to-rendered-page contract."""
    pages = render_pdf_pages_to_png(_make_pdf(1))

    assert len(pages) == 1
    assert isinstance(pages[0], RenderedPdfPage)
    assert pages[0].page_number == 1
    assert pages[0].image_bytes


def test_render_multi_page_pdf_preserves_order_and_one_based_page_numbers():
    """Verify deterministic document order and 1-based page numbering."""
    pages = render_pdf_pages_to_png(_make_pdf(3))

    assert [page.page_number for page in pages] == [1, 2, 3]


def test_rendered_image_bytes_are_png_bytes():
    """Verify rendered pages are provider-ready PNG bytes."""
    pages = render_pdf_pages_to_png(_make_pdf(1))

    assert pages[0].image_bytes.startswith(PNG_SIGNATURE)


def test_empty_pdf_bytes_raise_value_error():
    """Verify empty input is rejected with a controlled error."""
    with pytest.raises(ValueError, match="pdf_bytes must not be empty"):
        render_pdf_pages_to_png(b"")


def test_max_pages_limit_is_enforced():
    """Verify the future OCR page limit is enforced before rendering."""
    with pytest.raises(ValueError, match="exceeding max_pages=1"):
        render_pdf_pages_to_png(_make_pdf(2), max_pages=1)


@pytest.mark.parametrize("max_pages", [0, -1, 1.5, True])
def test_invalid_max_pages_raises_value_error(max_pages):
    """Verify invalid page-limit options are rejected."""
    with pytest.raises(ValueError, match="max_pages must be a positive integer"):
        render_pdf_pages_to_png(_make_pdf(1), max_pages=max_pages)


@pytest.mark.parametrize("zoom", [0, -1, 0.0, True])
def test_invalid_zoom_raises_value_error(zoom):
    """Verify invalid render-scale options are rejected."""
    with pytest.raises(ValueError, match="zoom must be a positive number"):
        render_pdf_pages_to_png(_make_pdf(1), zoom=zoom)


def test_zero_page_pdf_raises_value_error(monkeypatch):
    """Verify an empty PyMuPDF document is handled as a controlled error."""
    class EmptyPdfDocument:
        page_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        "services.document_ocr_service.pymupdf.open",
        lambda **kwargs: EmptyPdfDocument(),
    )

    with pytest.raises(ValueError, match="PDF contains no pages"):
        render_pdf_pages_to_png(b"%PDF fake")


def test_service_imports_without_flask_db_or_ai_provider_configuration():
    """Verify importability without Flask, DB, or AI provider configuration."""
    module = importlib.import_module("services.document_ocr_service")

    assert module.RenderedPdfPage is RenderedPdfPage
    assert callable(module.render_pdf_pages_to_png)


def test_pure_white_png_is_blank():
    image = Image.new("RGB", (120, 80), "white")

    assert is_rendered_page_blank(_png_bytes(image)) is True


def test_off_white_lightly_noisy_png_is_blank():
    rng = np.random.default_rng(123)
    pixels = rng.integers(246, 252, size=(100, 140), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="L")

    assert is_rendered_page_blank(_png_bytes(image)) is True


def test_png_with_clear_dark_text_like_pixels_is_not_blank():
    image = Image.new("L", (200, 120), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 32, 150, 38), fill=20)
    draw.rectangle((30, 48, 120, 54), fill=20)
    draw.rectangle((30, 64, 170, 70), fill=20)

    assert is_rendered_page_blank(_png_bytes(image)) is False


def test_png_with_meaningful_sparse_marks_is_not_blank():
    image = Image.new("L", (100, 100), 255)
    draw = ImageDraw.Draw(image)
    draw.line((45, 50, 54, 50), fill=0, width=1)

    assert is_rendered_page_blank(_png_bytes(image)) is False


@pytest.mark.parametrize("image_bytes", [b"", b"not an image"])
def test_blank_page_detection_rejects_empty_or_invalid_bytes(image_bytes):
    with pytest.raises(ValueError, match="image_bytes must"):
        is_rendered_page_blank(image_bytes)
