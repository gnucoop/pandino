"""
Document extraction orchestration.

This service is the application-layer boundary between route handlers and the
pure local parser in services.document_text_service. It delegates to local
extraction first, then uses OCR only as a configured fallback for PDFs whose
local text is empty or insufficient.

Uploaded files are read once here because upload streams are consumable. The
captured bytes are wrapped back into an in-memory FileStorage before delegating
to the local parser, which lets OCR fallback reuse the same PDF bytes without
changing the public NormalizedDocument contract.
"""

import io
import logging
import os
import time
from typing import TypedDict

from infrastructure.ai import TokenUsage, extract_text_from_image_with_usage
from services.document_ocr_service import (
    is_rendered_page_blank,
    render_pdf_pages_to_png,
)
from werkzeug.datastructures import FileStorage

from services.document_text_service import (
    DocumentInput,
    NormalizedDocument,
    extract_and_normalize_document,
)
from utils.operational_event import build_operational_event

logger = logging.getLogger(__name__)

MIN_EXTRACTED_TEXT_CHARS = 50

#: Closed `reason` domain for the OCR fallback Operational events (design E2/E6).
OCR_FALLBACK_REASON_EMPTY_LOCAL_TEXT = "empty_local_text"
OCR_FALLBACK_REASON_INSUFFICIENT_LOCAL_TEXT = "insufficient_local_text"
OCR_FAILURE_REASON_BLANK_PAGE = "blank_page"
OCR_FAILURE_REASON_EMPTY_PAGE_TEXT = "empty_page_text"


class DocumentExtractionResult(TypedDict):
    document: NormalizedDocument
    ocr_token_usage: TokenUsage


def _zero_token_usage() -> TokenUsage:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _add_token_usage(total: TokenUsage, usage: TokenUsage) -> None:
    total["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    total["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
    total["total_tokens"] += int(usage.get("total_tokens", 0) or 0)


def _is_pdf_filename(filename: str | None) -> bool:
    return os.path.splitext((filename or "").lower())[1] == ".pdf"


def _has_sufficient_text(text: str) -> bool:
    return len(text.strip()) >= MIN_EXTRACTED_TEXT_CHARS


def _is_empty_file_error(error: ValueError) -> bool:
    return str(error) == "File is empty"


def _resolve_ocr_config(
    ocr_provider: str | None,
    ocr_model: str | None,
) -> tuple[str, str] | None:
    provider = ocr_provider.strip() if ocr_provider else ""
    model = ocr_model.strip() if ocr_model else ""

    if provider and model:
        return provider, model

    return None


def _file_storage_from_bytes(
    original_file: FileStorage,
    file_bytes: bytes,
    filename: str,
) -> FileStorage:
    """
    Rebuild a consumable upload object from bytes captured by the orchestrator.
    """
    return FileStorage(
        stream=io.BytesIO(file_bytes),
        filename=filename,
        name=original_file.name,
        content_type=original_file.content_type,
        headers=original_file.headers,
    )


def _extract_pdf_text_with_ocr(
    *,
    pdf_bytes: bytes,
    filename: str,
    role: str | None,
    ocr_provider: str,
    ocr_model: str,
    ocr_api_key: str | None,
    reason: str,
) -> DocumentExtractionResult:
    """
    Run the OCR fallback for one PDF and report its outcome operationally.

    ``reason`` is module-private plumbing: the fallback *decision* is taken by
    extract_document_text_with_metadata, while the single natural emission
    point for the fallback's outcome is here. It carries one of the two
    OCR_FALLBACK_REASON_* literals and never reaches a public contract.
    """
    started = time.perf_counter()
    rendered_pages = render_pdf_pages_to_png(pdf_bytes)
    page_texts = []
    ocr_token_usage = _zero_token_usage()

    for page in rendered_pages:
        if is_rendered_page_blank(page.image_bytes):
            message, extra = build_operational_event(
                event="document_ocr_fallback_failed",
                provider=ocr_provider,
                model=ocr_model,
                duration_ms=round((time.perf_counter() - started) * 1000),
                details={
                    "page_number": page.page_number,
                    "page_count": len(rendered_pages),
                    "reason": OCR_FAILURE_REASON_BLANK_PAGE,
                },
            )
            logger.warning(message, extra=extra)
            raise ValueError(
                f"OCR did not extract text from PDF page {page.page_number}"
            )

        page_result = extract_text_from_image_with_usage(
            page.image_bytes,
            ocr_provider,
            ocr_model,
            api_key=ocr_api_key,
        )
        page_text = page_result["text"].strip()

        if not page_text:
            message, extra = build_operational_event(
                event="document_ocr_fallback_failed",
                provider=ocr_provider,
                model=ocr_model,
                duration_ms=round((time.perf_counter() - started) * 1000),
                details={
                    "page_number": page.page_number,
                    "page_count": len(rendered_pages),
                    "reason": OCR_FAILURE_REASON_EMPTY_PAGE_TEXT,
                },
            )
            logger.warning(message, extra=extra)
            raise ValueError(
                f"OCR did not extract text from PDF page {page.page_number}"
            )

        page_texts.append(page_text)
        _add_token_usage(ocr_token_usage, page_result["token_usage"])

    text = "\n\n".join(page_texts).strip()

    if not text:
        raise ValueError("OCR did not extract text from PDF")

    message, extra = build_operational_event(
        event="document_ocr_fallback_completed",
        provider=ocr_provider,
        model=ocr_model,
        duration_ms=round((time.perf_counter() - started) * 1000),
        details={
            "page_count": len(rendered_pages),
            "extracted_chars": len(text),
            "reason": reason,
        },
    )
    logger.info(message, extra=extra)

    return {
        "document": {"text": text, "filename": filename, "role": role},
        "ocr_token_usage": ocr_token_usage,
    }


def extract_document_text_with_metadata(
    document_input: DocumentInput,
    *,
    ocr_provider: str | None = None,
    ocr_model: str | None = None,
    ocr_api_key: str | None = None,
) -> DocumentExtractionResult:
    """
    Return extracted document text plus OCR token metadata.

    Non-file inputs keep the current direct delegation path. File inputs are
    byte-captured once, delegated to document_text_service.py through a rebuilt
    FileStorage, and then checked for recoverable PDF cases that OCR can handle
    when both provider and model are configured.
    """
    if document_input["source_type"] != "file":
        return {
            "document": extract_and_normalize_document(document_input),
            "ocr_token_usage": _zero_token_usage(),
        }

    file = document_input.get("content")

    if not isinstance(file, FileStorage):
        return {
            "document": extract_and_normalize_document(document_input),
            "ocr_token_usage": _zero_token_usage(),
        }

    filename = document_input.get("filename") or file.filename

    if not filename:
        return {
            "document": extract_and_normalize_document(document_input),
            "ocr_token_usage": _zero_token_usage(),
        }

    file_bytes = file.read()
    delegated_input: DocumentInput = {
        **document_input,
        "content": _file_storage_from_bytes(file, file_bytes, filename),
        "filename": filename,
    }

    is_pdf = _is_pdf_filename(filename)
    ocr_config = _resolve_ocr_config(ocr_provider, ocr_model)

    try:
        normalized = extract_and_normalize_document(delegated_input)
    except ValueError as error:
        if is_pdf and file_bytes and _is_empty_file_error(error):
            if ocr_config:
                provider, model = ocr_config
                return _extract_pdf_text_with_ocr(
                    pdf_bytes=file_bytes,
                    filename=filename,
                    role=document_input.get("role"),
                    ocr_provider=provider,
                    ocr_model=model,
                    ocr_api_key=ocr_api_key,
                    reason=OCR_FALLBACK_REASON_EMPTY_LOCAL_TEXT,
                )
        raise

    if is_pdf and file_bytes and not _has_sufficient_text(normalized["text"]):
        if ocr_config:
            provider, model = ocr_config
            return _extract_pdf_text_with_ocr(
                pdf_bytes=file_bytes,
                filename=filename,
                role=normalized["role"],
                ocr_provider=provider,
                ocr_model=model,
                ocr_api_key=ocr_api_key,
                reason=OCR_FALLBACK_REASON_INSUFFICIENT_LOCAL_TEXT,
            )

    return {"document": normalized, "ocr_token_usage": _zero_token_usage()}


def extract_document_text(
    document_input: DocumentInput,
    *,
    ocr_provider: str | None = None,
    ocr_model: str | None = None,
    ocr_api_key: str | None = None,
) -> NormalizedDocument:
    """
    Return a NormalizedDocument using local extraction plus optional PDF OCR.

    Compatibility wrapper for callers that do not need extraction metadata.
    """
    return extract_document_text_with_metadata(
        document_input,
        ocr_provider=ocr_provider,
        ocr_model=ocr_model,
        ocr_api_key=ocr_api_key,
    )["document"]
