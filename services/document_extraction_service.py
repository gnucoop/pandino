"""
Document extraction orchestration.

This service is the application-layer boundary between route handlers and the
pure local parser in services.document_text_service. Step 1 keeps OCR inactive:
PDFs that look recoverable are only marked at an isolated decision point, then
the current local-parser result or exception is preserved.

Uploaded files are read once here because upload streams are consumable. The
captured bytes are wrapped back into an in-memory FileStorage before delegating
to the local parser, which lets future OCR fallback code reuse the same PDF
bytes without changing the public NormalizedDocument contract.
"""

import io
import os
from typing import Literal

from werkzeug.datastructures import FileStorage

from services.document_text_service import (
    DocumentInput,
    NormalizedDocument,
    extract_and_normalize_document,
)


MIN_EXTRACTED_TEXT_CHARS = 50

RecoverablePdfReason = Literal["insufficient_text", "empty_file"]


def _is_pdf_filename(filename: str | None) -> bool:
    return os.path.splitext((filename or "").lower())[1] == ".pdf"


def _has_sufficient_text(text: str) -> bool:
    return len(text.strip()) >= MIN_EXTRACTED_TEXT_CHARS


def _is_empty_file_error(error: ValueError) -> bool:
    return str(error) == "File is empty"


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


def _mark_pdf_ocr_fallback_candidate(
    *,
    reason: RecoverablePdfReason,
    pdf_bytes: bytes,
    filename: str,
    local_result: NormalizedDocument | None = None,
    local_error: ValueError | None = None,
) -> None:
    """
    Future OCR fallback decision point.

    Step 1 keeps fallback inactive: callers still receive the same local parser
    result or exception, while this isolated hook owns the reusable PDF bytes
    that a later OCR implementation will need.
    """


def extract_document_text(document_input: DocumentInput) -> NormalizedDocument:
    """
    Return a NormalizedDocument using local extraction plus inactive OCR hooks.

    Non-file inputs keep the current direct delegation path. File inputs are
    byte-captured once, delegated to document_text_service.py through a rebuilt
    FileStorage, and then checked for recoverable PDF cases that a later OCR
    implementation can handle.
    """
    if document_input["source_type"] != "file":
        return extract_and_normalize_document(document_input)

    file = document_input.get("content")

    if not isinstance(file, FileStorage):
        return extract_and_normalize_document(document_input)

    filename = document_input.get("filename") or file.filename

    if not filename:
        return extract_and_normalize_document(document_input)

    file_bytes = file.read()
    delegated_input: DocumentInput = {
        **document_input,
        "content": _file_storage_from_bytes(file, file_bytes, filename),
        "filename": filename,
    }

    is_pdf = _is_pdf_filename(filename)

    try:
        normalized = extract_and_normalize_document(delegated_input)
    except ValueError as error:
        if is_pdf and file_bytes and _is_empty_file_error(error):
            _mark_pdf_ocr_fallback_candidate(
                reason="empty_file",
                pdf_bytes=file_bytes,
                filename=filename,
                local_error=error,
            )
        raise

    if is_pdf and file_bytes and not _has_sufficient_text(normalized["text"]):
        _mark_pdf_ocr_fallback_candidate(
            reason="insufficient_text",
            pdf_bytes=file_bytes,
            filename=filename,
            local_result=normalized,
        )

    return normalized
