"""
Tests for the Step 1 document extraction orchestration boundary.

The new service does not perform OCR yet. These tests pin the inactive fallback
decision points, byte reuse behavior, and unchanged delegation to the local
document_text_service parser.
"""

import io

import pytest
from werkzeug.datastructures import FileStorage

from services import document_extraction_service as service
from services.document_text_service import DocumentInput


def _upload(filename: str, content: bytes) -> FileStorage:
    return FileStorage(stream=io.BytesIO(content), filename=filename)


def test_text_input_delegates_to_local_parser(monkeypatch):
    input_doc: DocumentInput = {
        "content": "hello",
        "filename": None,
        "source_type": "text",
        "role": "candidate",
    }
    expected = {"text": "hello", "filename": None, "role": "candidate"}

    def fake_extract(doc):
        assert doc is input_doc
        return expected

    monkeypatch.setattr(service, "extract_and_normalize_document", fake_extract)

    assert service.extract_document_text(input_doc) == expected


def test_file_input_reads_upload_once_and_delegates_reusable_bytes(monkeypatch):
    """Upload bytes are captured once and replayed through a fresh FileStorage."""
    original_file = _upload("notes.txt", b"local text")
    input_doc: DocumentInput = {
        "content": original_file,
        "filename": "notes.txt",
        "source_type": "file",
        "role": None,
    }

    def fake_extract(doc):
        delegated_file = doc["content"]
        assert delegated_file is not original_file
        assert delegated_file.read() == b"local text"
        return {"text": "local text", "filename": "notes.txt", "role": None}

    monkeypatch.setattr(service, "extract_and_normalize_document", fake_extract)

    assert service.extract_document_text(input_doc)["text"] == "local text"
    assert original_file.read() == b""


def test_short_pdf_text_marks_recoverable_case_but_returns_local_result(monkeypatch):
    """Insufficient PDF text is marked for future OCR without changing output."""
    markers = []
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": "cv",
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": "cv"}

    monkeypatch.setattr(service, "extract_and_normalize_document", lambda doc: local_result)
    monkeypatch.setattr(
        service,
        "_mark_pdf_ocr_fallback_candidate",
        lambda **kwargs: markers.append(kwargs),
    )

    assert service.extract_document_text(input_doc) == local_result
    assert markers[0]["reason"] == "insufficient_text"
    assert markers[0]["pdf_bytes"] == b"%PDF bytes"
    assert markers[0]["local_result"] == local_result


def test_empty_pdf_error_marks_recoverable_case_and_reraises(monkeypatch):
    """The current empty-PDF error remains visible while bytes are preserved."""
    markers = []
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }

    def fake_extract(doc):
        raise ValueError("File is empty")

    monkeypatch.setattr(service, "extract_and_normalize_document", fake_extract)
    monkeypatch.setattr(
        service,
        "_mark_pdf_ocr_fallback_candidate",
        lambda **kwargs: markers.append(kwargs),
    )

    with pytest.raises(ValueError, match="File is empty"):
        service.extract_document_text(input_doc)

    assert markers[0]["reason"] == "empty_file"
    assert markers[0]["pdf_bytes"] == b"%PDF bytes"
    assert isinstance(markers[0]["local_error"], ValueError)


def test_non_pdf_empty_file_error_is_not_marked_recoverable(monkeypatch):
    markers = []
    input_doc: DocumentInput = {
        "content": _upload("empty.txt", b""),
        "filename": "empty.txt",
        "source_type": "file",
        "role": None,
    }

    def fake_extract(doc):
        raise ValueError("File is empty")

    monkeypatch.setattr(service, "extract_and_normalize_document", fake_extract)
    monkeypatch.setattr(
        service,
        "_mark_pdf_ocr_fallback_candidate",
        lambda **kwargs: markers.append(kwargs),
    )

    with pytest.raises(ValueError, match="File is empty"):
        service.extract_document_text(input_doc)

    assert markers == []
