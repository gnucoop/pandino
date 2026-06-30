"""Tests for document extraction orchestration with optional PDF OCR fallback."""

import io
from types import SimpleNamespace

import pytest
from werkzeug.datastructures import FileStorage

from services import document_extraction_service as service
from services.document_text_service import DocumentInput


def _upload(filename: str, content: bytes) -> FileStorage:
    return FileStorage(stream=io.BytesIO(content), filename=filename)


def _rendered_page(page_number: int, image_bytes: bytes):
    return SimpleNamespace(page_number=page_number, image_bytes=image_bytes)


def _ocr_result(text: str, input_tokens: int = 0, output_tokens: int = 0):
    return {
        "text": text,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _fail_render(*args, **kwargs):
    pytest.fail("PDF rendering should not be called")


def _fail_ocr(*args, **kwargs):
    pytest.fail("Vision OCR should not be called")


@pytest.fixture(autouse=True)
def _default_rendered_pages_non_blank(monkeypatch):
    monkeypatch.setattr(service, "is_rendered_page_blank", lambda image_bytes: False)


def test_non_pdf_input_with_ocr_config_still_delegates_to_local_parser(monkeypatch):
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
    monkeypatch.setattr(service, "render_pdf_pages_to_png", _fail_render)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", _fail_ocr)

    assert (
        service.extract_document_text(
            input_doc,
            ocr_provider="Deepinfra",
            ocr_model="vision-model",
        )
        == expected
    )


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


def test_pdf_with_sufficient_local_text_and_ocr_config_does_not_call_ocr(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("text.pdf", b"%PDF bytes"),
        "filename": "text.pdf",
        "source_type": "file",
        "role": "cv",
    }
    local_result = {
        "text": "x" * service.MIN_EXTRACTED_TEXT_CHARS,
        "filename": "text.pdf",
        "role": "cv",
    }

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(service, "render_pdf_pages_to_png", _fail_render)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", _fail_ocr)

    assert (
        service.extract_document_text(
            input_doc,
            ocr_provider="Deepinfra",
            ocr_model="vision-model",
        )
        == local_result
    )


def test_short_pdf_without_ocr_config_returns_local_result_without_ocr(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": "cv",
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": "cv"}

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(service, "render_pdf_pages_to_png", _fail_render)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", _fail_ocr)

    assert service.extract_document_text(input_doc) == local_result


def test_local_pdf_path_exposes_zero_ocr_usage_metadata(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("text.pdf", b"%PDF bytes"),
        "filename": "text.pdf",
        "source_type": "file",
        "role": "cv",
    }
    local_result = {
        "text": "x" * service.MIN_EXTRACTED_TEXT_CHARS,
        "filename": "text.pdf",
        "role": "cv",
    }

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(service, "render_pdf_pages_to_png", _fail_render)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", _fail_ocr)

    assert service.extract_document_text_with_metadata(
        input_doc,
        ocr_provider="Deepinfra",
        ocr_model="vision-model",
    ) == {
        "document": local_result,
        "ocr_token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def test_short_pdf_with_ocr_config_renders_pages_and_returns_ocr_text(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": "cv",
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": "cv"}
    calls = {}

    def fake_render(pdf_bytes):
        calls["pdf_bytes"] = pdf_bytes
        return [_rendered_page(1, b"page image")]

    def fake_ocr(image_bytes, provider, model, *, api_key=None, **kwargs):
        calls["ocr"] = {
            "image_bytes": image_bytes,
            "provider": provider,
            "model": model,
            "api_key": api_key,
        }
        return _ocr_result("  OCR text  ", input_tokens=11, output_tokens=3)

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(service, "render_pdf_pages_to_png", fake_render)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", fake_ocr)

    assert service.extract_document_text(
        input_doc,
        ocr_provider=" Deepinfra ",
        ocr_model=" vision-model ",
        ocr_api_key="test-key",
    ) == {"text": "OCR text", "filename": "scan.pdf", "role": "cv"}
    assert calls["pdf_bytes"] == b"%PDF bytes"
    assert calls["ocr"] == {
        "image_bytes": b"page image",
        "provider": "Deepinfra",
        "model": "vision-model",
        "api_key": "test-key",
    }


def test_single_page_ocr_exposes_usage_metadata(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": "cv",
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": "cv"}

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [_rendered_page(1, b"page image")],
    )
    monkeypatch.setattr(
        service,
        "extract_text_from_image_with_usage",
        lambda *args, **kwargs: _ocr_result("OCR text", 12, 4),
    )

    assert service.extract_document_text_with_metadata(
        input_doc,
        ocr_provider="Deepinfra",
        ocr_model="vision-model",
    ) == {
        "document": {"text": "OCR text", "filename": "scan.pdf", "role": "cv"},
        "ocr_token_usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        },
    }


def test_empty_pdf_error_without_ocr_config_reraises_original_error(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }

    def fake_extract(doc):
        raise ValueError("File is empty")

    monkeypatch.setattr(service, "extract_and_normalize_document", fake_extract)
    monkeypatch.setattr(service, "render_pdf_pages_to_png", _fail_render)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", _fail_ocr)

    with pytest.raises(ValueError, match="File is empty"):
        service.extract_document_text(input_doc)


def test_empty_pdf_error_with_ocr_config_recovers_via_ocr(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": "cv",
    }

    def fake_extract(doc):
        raise ValueError("File is empty")

    monkeypatch.setattr(service, "extract_and_normalize_document", fake_extract)
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [_rendered_page(1, b"page image")],
    )
    monkeypatch.setattr(
        service,
        "extract_text_from_image_with_usage",
        lambda *args, **kwargs: _ocr_result("Recovered OCR text"),
    )

    assert service.extract_document_text(
        input_doc,
        ocr_provider="Deepinfra",
        ocr_model="vision-model",
    ) == {"text": "Recovered OCR text", "filename": "scan.pdf", "role": "cv"}


def test_multi_page_ocr_joins_text_in_render_order(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": None}

    def fake_ocr(image_bytes, *args, **kwargs):
        return {
            b"first page": _ocr_result("  First page text  ", 5, 1),
            b"second page": _ocr_result("Second page text", 7, 2),
        }[image_bytes]

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [
            _rendered_page(1, b"first page"),
            _rendered_page(2, b"second page"),
        ],
    )
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", fake_ocr)

    result = service.extract_document_text(
        input_doc,
        ocr_provider="Deepinfra",
        ocr_model="vision-model",
    )

    assert result["text"] == "First page text\n\nSecond page text"


def test_multi_page_ocr_sums_usage_metadata(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": None}

    def fake_ocr(image_bytes, *args, **kwargs):
        return {
            b"first page": _ocr_result("First page text", 5, 1),
            b"second page": _ocr_result("Second page text", 7, 2),
        }[image_bytes]

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [
            _rendered_page(1, b"first page"),
            _rendered_page(2, b"second page"),
        ],
    )
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", fake_ocr)

    result = service.extract_document_text_with_metadata(
        input_doc,
        ocr_provider="Deepinfra",
        ocr_model="vision-model",
    )

    assert result["ocr_token_usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }


def test_blank_rendered_page_fails_before_ocr_provider_call(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": None}
    blank_checks = []
    ocr_calls = []

    def fake_blank_detector(image_bytes):
        blank_checks.append(image_bytes)
        return image_bytes == b"blank page"

    def fake_ocr(image_bytes, *args, **kwargs):
        ocr_calls.append(image_bytes)
        return _ocr_result("First page text")

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [
            _rendered_page(1, b"first page"),
            _rendered_page(2, b"blank page"),
            _rendered_page(3, b"third page"),
        ],
    )
    monkeypatch.setattr(service, "is_rendered_page_blank", fake_blank_detector)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", fake_ocr)

    with pytest.raises(ValueError) as error:
        service.extract_document_text(
            input_doc,
            ocr_provider="Deepinfra",
            ocr_model="vision-model",
        )

    assert str(error.value) == "OCR did not extract text from PDF page 2"
    assert blank_checks == [b"first page", b"blank page"]
    assert ocr_calls == [b"first page"]


def test_empty_ocr_output_raises_page_value_error(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": None}

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [_rendered_page(1, b"page image")],
    )
    monkeypatch.setattr(
        service,
        "extract_text_from_image_with_usage",
        lambda *args, **kwargs: _ocr_result("  "),
    )

    with pytest.raises(ValueError, match="OCR did not extract text from PDF page 1"):
        service.extract_document_text(
            input_doc,
            ocr_provider="Deepinfra",
            ocr_model="vision-model",
        )


def test_empty_ocr_output_on_one_page_fails_instead_of_returning_partial_text(
    monkeypatch,
):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": None}

    def fake_ocr(image_bytes, *args, **kwargs):
        return {
            b"first page": _ocr_result("First page text"),
            b"second page": _ocr_result("  "),
        }[image_bytes]

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [
            _rendered_page(1, b"first page"),
            _rendered_page(2, b"second page"),
        ],
    )
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", fake_ocr)

    with pytest.raises(ValueError, match="OCR did not extract text from PDF page 2"):
        service.extract_document_text(
            input_doc,
            ocr_provider="Deepinfra",
            ocr_model="vision-model",
        )


def test_render_value_error_propagates_before_ocr_calls(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("large.pdf", b"%PDF bytes"),
        "filename": "large.pdf",
        "source_type": "file",
        "role": None,
    }
    local_result = {"text": "short", "filename": "large.pdf", "role": None}

    def fake_render(pdf_bytes):
        raise ValueError("PDF has 11 pages, exceeding max_pages=10")

    monkeypatch.setattr(
        service, "extract_and_normalize_document", lambda doc: local_result
    )
    monkeypatch.setattr(service, "render_pdf_pages_to_png", fake_render)
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", _fail_ocr)

    with pytest.raises(ValueError, match="exceeding max_pages=10"):
        service.extract_document_text(
            input_doc,
            ocr_provider="Deepinfra",
            ocr_model="vision-model",
        )


def test_ocr_provider_exception_on_any_page_fails_whole_extraction(monkeypatch):
    input_doc: DocumentInput = {
        "content": _upload("scan.pdf", b"%PDF bytes"),
        "filename": "scan.pdf",
        "source_type": "file",
        "role": None,
    }
    local_result = {"text": "short", "filename": "scan.pdf", "role": None}

    def fake_ocr(image_bytes, *args, **kwargs):
        if image_bytes == b"second page":
            raise RuntimeError("provider down")
        return _ocr_result("First page text")

    monkeypatch.setattr(service, "extract_and_normalize_document", lambda doc: local_result)
    monkeypatch.setattr(
        service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [
            _rendered_page(1, b"first page"),
            _rendered_page(2, b"second page"),
        ],
    )
    monkeypatch.setattr(service, "extract_text_from_image_with_usage", fake_ocr)

    with pytest.raises(RuntimeError, match="provider down"):
        service.extract_document_text(
            input_doc,
            ocr_provider="Deepinfra",
            ocr_model="vision-model",
        )
