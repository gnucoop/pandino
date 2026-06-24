"""
Provider-independent PDF page rendering for future OCR fallback.

This module intentionally stops at PDF-to-PNG conversion. It does not perform
OCR, call Vision providers, import AI infrastructure, or decide when fallback
should be used. Later OCR steps can consume the PNG bytes returned here.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import pymupdf


DEFAULT_MAX_OCR_PAGES = 10
DEFAULT_RENDER_ZOOM = 2.0


@dataclass(frozen=True)
class RenderedPdfPage:
    """A rendered PDF page represented as PNG bytes with 1-based numbering."""

    page_number: int
    image_bytes: bytes


def _validate_render_options(max_pages: int, zoom: float) -> None:
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
        raise ValueError("max_pages must be a positive integer")

    if isinstance(zoom, bool) or not isinstance(zoom, Real) or zoom <= 0:
        raise ValueError("zoom must be a positive number")


def render_pdf_pages_to_png(
    pdf_bytes: bytes,
    *,
    max_pages: int = DEFAULT_MAX_OCR_PAGES,
    zoom: float = DEFAULT_RENDER_ZOOM,
) -> list[RenderedPdfPage]:
    """
    Render PDF pages to PNG bytes for a future OCR/Vision pipeline.

    The function opens a PDF from bytes, renders pages in deterministic document
    order, and returns one PNG image per page. It does not perform OCR or call
    any AI/Vision provider.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes must not be empty")

    _validate_render_options(max_pages=max_pages, zoom=zoom)

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf_document:
        page_count = pdf_document.page_count

        if page_count == 0:
            raise ValueError("PDF contains no pages")

        if page_count > max_pages:
            raise ValueError(f"PDF has {page_count} pages, exceeding max_pages={max_pages}")

        matrix = pymupdf.Matrix(float(zoom), float(zoom))
        rendered_pages: list[RenderedPdfPage] = []

        for page_index in range(page_count):
            page = cast(Any, pdf_document.load_page(page_index))
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            rendered_pages.append(
                RenderedPdfPage(
                    page_number=page_index + 1,
                    image_bytes=pixmap.tobytes("png"),
                )
            )

    return rendered_pages
