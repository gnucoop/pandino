"""Integration-light tests for /compare_docs route wiring."""

import io
import json
from types import SimpleNamespace

import pytest
from flask import Flask
from werkzeug.datastructures import MultiDict

from routes import documents as documents_route


def _make_app() -> Flask:
    app = Flask(__name__)
    app.config["MAUI_CONFIG"] = SimpleNamespace(
        compare_docs_token_cost=1,
        models=SimpleNamespace(
            compare_docs_provider="Google",
            compare_docs_model="gemini-2.5-flash",
            vision_provider="Deepinfra",
            vision_model="vision-ocr-model",
        ),
    )
    app.register_blueprint(documents_route.documents_bp)
    return app


def _patch_success_dependencies(monkeypatch):
    monkeypatch.setattr(documents_route, "assert_valid_api_key", lambda *args: None)
    monkeypatch.setattr(
        documents_route.database_pg, "get_user_tokens", lambda user_email: 10
    )
    monkeypatch.setattr(
        documents_route.database_pg,
        "get_user_by_username",
        lambda user_email: {"id": 123, "username": user_email},
    )
    monkeypatch.setattr(documents_route, "edit_tokens", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        documents_route, "log_token_usage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(documents_route.os, "getenv", lambda *args, **kwargs: None)


def test_compare_docs_uses_document_extraction_service_and_preserves_response(
    monkeypatch,
):
    app = _make_app()
    _patch_success_dependencies(monkeypatch)

    assert not hasattr(documents_route, "extract_and_normalize_document")

    extraction_calls = []
    comparison_call = {}

    def fake_extract_document_text(
        doc_input,
        *,
        ocr_provider=None,
        ocr_model=None,
        ocr_api_key=None,
    ):
        extraction_calls.append(
            {
                "doc_input": doc_input,
                "ocr_provider": ocr_provider,
                "ocr_model": ocr_model,
                "ocr_api_key": ocr_api_key,
            }
        )
        if doc_input["source_type"] == "text":
            return {
                "text": "normalized text document",
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            }

        return {
            "text": "normalized file document",
            "filename": doc_input["filename"],
            "role": doc_input["role"],
        }

    def fake_compare_documents(**kwargs):
        comparison_call.update(kwargs)
        return {
            "comparison": {
                "score": 87,
                "summary": "Documents are compatible",
                "reasoning": "Mocked comparison reasoning",
            },
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }

    monkeypatch.setattr(
        documents_route, "extract_document_text", fake_extract_document_text
    )
    monkeypatch.setattr(documents_route, "compare_documents", fake_compare_documents)

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("additional_context", "Prefer recent experience"),
            ("language", "ITA"),
            (
                "text_documents",
                json.dumps(
                    [
                        {
                            "content": "raw text document",
                            "filename": "reference.txt",
                            "role": "reference",
                        }
                    ]
                ),
            ),
            ("file_roles", json.dumps(["candidate"])),
            ("files", (io.BytesIO(b"%PDF fake bytes"), "candidate.pdf")),
        ]
    )

    response = app.test_client().post(
        "/compare_docs",
        data=data,
        content_type="multipart/form-data",
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "score": 87,
        "summary": "Documents are compatible",
        "reasoning": "Mocked comparison reasoning",
    }

    assert len(extraction_calls) == 2
    assert extraction_calls[0] == {
        "doc_input": {
            "content": "raw text document",
            "filename": "reference.txt",
            "source_type": "text",
            "role": "reference",
        },
        "ocr_provider": "Deepinfra",
        "ocr_model": "vision-ocr-model",
        "ocr_api_key": None,
    }

    file_call = extraction_calls[1]
    assert file_call["ocr_provider"] == "Deepinfra"
    assert file_call["ocr_model"] == "vision-ocr-model"
    assert file_call["ocr_api_key"] is None
    assert file_call["doc_input"]["filename"] == "candidate.pdf"
    assert file_call["doc_input"]["source_type"] == "file"
    assert file_call["doc_input"]["role"] == "candidate"
    assert file_call["doc_input"]["content"].filename == "candidate.pdf"

    assert comparison_call == {
        "documents": [
            {
                "text": "normalized text document",
                "filename": "reference.txt",
                "role": "reference",
            },
            {
                "text": "normalized file document",
                "filename": "candidate.pdf",
                "role": "candidate",
            },
        ],
        "prompt": "Compare these documents",
        "llm_type": "Google",
        "model": "gemini-2.5-flash",
        "additional_context": "Prefer recent experience",
        "language": "ITA",
        "api_key": None,
    }


def test_compare_docs_extraction_value_error_keeps_existing_400_mapping(
    monkeypatch,
):
    app = _make_app()
    _patch_success_dependencies(monkeypatch)

    def fake_extract_document_text(*args, **kwargs):
        raise ValueError("File is empty")

    monkeypatch.setattr(
        documents_route, "extract_document_text", fake_extract_document_text
    )
    monkeypatch.setattr(
        documents_route,
        "compare_documents",
        lambda **kwargs: pytest.fail("comparison should not be called"),
    )

    data = {
        "prompt": "Compare these documents",
        "text_documents": json.dumps(
            [
                {
                    "content": "first document",
                    "filename": "first.txt",
                    "role": "reference",
                },
                {
                    "content": "second document",
                    "filename": "second.txt",
                    "role": "candidate",
                },
            ]
        ),
    }

    response = app.test_client().post(
        "/compare_docs",
        data=data,
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid request",
        "details": "File is empty",
    }
