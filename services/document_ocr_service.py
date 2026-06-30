"""
Provider-independent PDF page rendering for future OCR fallback.

This module intentionally stops at PDF-to-PNG conversion. It does not perform
OCR, call Vision providers, import AI infrastructure, or decide when fallback
should be used. Later OCR steps can consume the PNG bytes returned here.
"""

import io
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import numpy as np
from PIL import Image, UnidentifiedImageError
import pymupdf


DEFAULT_MAX_OCR_PAGES = 10
DEFAULT_RENDER_ZOOM = 2.0
BLANK_PAGE_CROP_MARGIN_RATIO = 0.01
BLANK_PAGE_BACKGROUND_PERCENTILE = 95.0
BLANK_PAGE_MIN_BACKGROUND_GRAY = 235
BLANK_PAGE_FOREGROUND_DELTA = 20
BLANK_PAGE_MAX_FOREGROUND_GRAY = 245
BLANK_PAGE_MAX_FOREGROUND_RATIO = 0.001
BLANK_PAGE_DARK_GRAY_THRESHOLD = 220
BLANK_PAGE_MAX_DARK_RATIO = 0.0002


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


def _crop_margin(pixels: np.ndarray) -> np.ndarray:
    height, width = pixels.shape
    margin_y = int(height * BLANK_PAGE_CROP_MARGIN_RATIO)
    margin_x = int(width * BLANK_PAGE_CROP_MARGIN_RATIO)

    if margin_y == 0 and margin_x == 0:
        return pixels

    if height <= margin_y * 2 or width <= margin_x * 2:
        return pixels

    return pixels[margin_y : height - margin_y, margin_x : width - margin_x]


def is_rendered_page_blank(image_bytes: bytes) -> bool:
    """
    Return True when rendered PNG bytes are visually blank with high confidence.

    The heuristic is intentionally conservative: uncertain pages are treated as
    non-blank so the OCR provider can still attempt extraction.
    """
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            grayscale = image.convert("L")
            pixels = np.asarray(grayscale, dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("image_bytes must be valid image data") from error

    if pixels.size == 0:
        raise ValueError("image_bytes must contain image pixels")

    pixels = _crop_margin(pixels)
    background = float(np.percentile(pixels, BLANK_PAGE_BACKGROUND_PERCENTILE))
    foreground_threshold = min(
        BLANK_PAGE_MAX_FOREGROUND_GRAY,
        background - BLANK_PAGE_FOREGROUND_DELTA,
    )
    foreground_ratio = float(np.mean(pixels <= foreground_threshold))
    dark_ratio = float(np.mean(pixels < BLANK_PAGE_DARK_GRAY_THRESHOLD))

    return (
        background >= BLANK_PAGE_MIN_BACKGROUND_GRAY
        and foreground_ratio <= BLANK_PAGE_MAX_FOREGROUND_RATIO
        and dark_ratio <= BLANK_PAGE_MAX_DARK_RATIO
    )


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
