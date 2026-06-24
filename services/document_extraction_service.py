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
import os

from infrastructure.ai import extract_text_from_image
from services.document_ocr_service import render_pdf_pages_to_png
from werkzeug.datastructures import FileStorage

from services.document_text_service import (
    DocumentInput,
    NormalizedDocument,
    extract_and_normalize_document,
)


MIN_EXTRACTED_TEXT_CHARS = 50


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
) -> NormalizedDocument:
    rendered_pages = render_pdf_pages_to_png(pdf_bytes)
    page_texts = []

    for page in rendered_pages:
        page_text = extract_text_from_image(
            page.image_bytes,
            ocr_provider,
            ocr_model,
            api_key=ocr_api_key,
        ).strip()

        if not page_text:
            raise ValueError(
                f"OCR did not extract text from PDF page {page.page_number}"
            )

        page_texts.append(page_text)

    text = "\n\n".join(page_texts).strip()

    if not text:
        raise ValueError("OCR did not extract text from PDF")

    return {"text": text, "filename": filename, "role": role}


def extract_document_text(
    document_input: DocumentInput,
    *,
    ocr_provider: str | None = None,
    ocr_model: str | None = None,
    ocr_api_key: str | None = None,
) -> NormalizedDocument:
    """
    Return a NormalizedDocument using local extraction plus optional PDF OCR.

    Non-file inputs keep the current direct delegation path. File inputs are
    byte-captured once, delegated to document_text_service.py through a rebuilt
    FileStorage, and then checked for recoverable PDF cases that OCR can handle
    when both provider and model are configured.
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
            )

    return normalized
