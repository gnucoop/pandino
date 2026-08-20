"""Integration-light tests for /compare_docs route wiring."""

import io
import json
import logging
from types import SimpleNamespace

import pytest
from flask import Flask
from werkzeug.datastructures import MultiDict

from routes import documents as documents_route
from services import document_comparison_service as comparison_service
from services import document_extraction_service as extraction_service
from utils.logging_config import ContextDefaultsFilter, register_request_context_hooks
from utils.operational_persistence import OperationalPersistenceHandler


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
    register_request_context_hooks(app)
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
        lambda user_email: {"id": 123, "username": user_email, "client": "coopi"},
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
    log_calls = []
    edit_calls = []

    def fake_extract_document_text_with_metadata(
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
                "document": {
                    "text": "normalized text document",
                    "filename": doc_input["filename"],
                    "role": doc_input["role"],
                },
                "ocr_token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }

        return {
            "document": {
                "text": "normalized file document",
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            },
            "ocr_token_usage": {
                "input_tokens": 20,
                "output_tokens": 7,
                "total_tokens": 27,
            },
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
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )
    monkeypatch.setattr(documents_route, "compare_documents", fake_compare_documents)
    monkeypatch.setattr(
        documents_route, "log_token_usage", lambda **kwargs: log_calls.append(kwargs)
    )
    monkeypatch.setattr(
        documents_route,
        "edit_tokens",
        lambda user_email, token_delta: edit_calls.append((user_email, token_delta)),
    )

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
    assert log_calls == [
        {
            "user_id": 123,
            "token_input": 30,
            "token_output": 12,
            "model": "gemini-2.5-flash",
            "provider": "Google",
            "service": "/compare_docs",
            "request_id": response.headers["X-Request-ID"],
            "source": "coopi",
        }
    ]
    assert edit_calls == [("user@example.com", -1)]


def test_compare_docs_with_zero_ocr_usage_logs_comparison_usage_once(monkeypatch):
    app = _make_app()
    _patch_success_dependencies(monkeypatch)

    log_calls = []
    edit_calls = []

    def fake_extract_document_text_with_metadata(doc_input, **kwargs):
        return {
            "document": {
                "text": f"normalized {doc_input['filename']}",
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            },
            "ocr_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    def fake_compare_documents(**kwargs):
        return {
            "comparison": {
                "score": 91,
                "summary": "Documents match",
                "reasoning": "Mocked comparison",
            },
            "token_usage": {
                "input_tokens": 13,
                "output_tokens": 6,
                "total_tokens": 19,
            },
        }

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )
    monkeypatch.setattr(documents_route, "compare_documents", fake_compare_documents)
    monkeypatch.setattr(
        documents_route, "log_token_usage", lambda **kwargs: log_calls.append(kwargs)
    )
    monkeypatch.setattr(
        documents_route,
        "edit_tokens",
        lambda user_email, token_delta: edit_calls.append((user_email, token_delta)),
    )

    data = {
        "prompt": "Compare these documents",
        "text_documents": json.dumps(
            [
                {"content": "first document", "filename": "first.txt", "role": "a"},
                {"content": "second document", "filename": "second.txt", "role": "b"},
            ]
        ),
    }

    response = app.test_client().post(
        "/compare_docs",
        data=data,
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "score": 91,
        "summary": "Documents match",
        "reasoning": "Mocked comparison",
    }
    assert log_calls == [
        {
            "user_id": 123,
            "token_input": 13,
            "token_output": 6,
            "model": "gemini-2.5-flash",
            "provider": "Google",
            "service": "/compare_docs",
            "request_id": response.headers["X-Request-ID"],
            "source": "coopi",
        }
    ]
    assert edit_calls == [("user@example.com", -1)]


def test_compare_docs_context_window_error_returns_413_without_accounting(
    monkeypatch,
):
    app = _make_app()
    _patch_success_dependencies(monkeypatch)

    log_calls = []
    edit_calls = []

    def fake_extract_document_text_with_metadata(doc_input, **kwargs):
        return {
            "document": {
                "text": f"normalized {doc_input['filename']}",
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            },
            "ocr_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    def fake_compare_documents(**kwargs):
        raise documents_route.DocumentComparisonPayloadTooLargeError(
            documents_route.CONTEXT_WINDOW_ERROR_MESSAGE
        )

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )
    monkeypatch.setattr(documents_route, "compare_documents", fake_compare_documents)
    monkeypatch.setattr(
        documents_route, "log_token_usage", lambda **kwargs: log_calls.append(kwargs)
    )
    monkeypatch.setattr(
        documents_route,
        "edit_tokens",
        lambda user_email, token_delta: edit_calls.append((user_email, token_delta)),
    )

    data = {
        "prompt": "Compare these documents",
        "text_documents": json.dumps(
            [
                {"content": "first document", "filename": "first.txt", "role": "a"},
                {"content": "second document", "filename": "second.txt", "role": "b"},
            ]
        ),
    }

    response = app.test_client().post(
        "/compare_docs",
        data=data,
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )

    assert response.status_code == 413
    assert response.get_json() == {
        "error": "Payload too large",
        "details": (
            "The extracted document text is too large for the configured comparison "
            "model. Reduce document size or split the comparison."
        ),
    }
    assert log_calls == []
    assert edit_calls == []


def test_compare_docs_extraction_value_error_keeps_existing_400_mapping(
    monkeypatch,
):
    app = _make_app()
    _patch_success_dependencies(monkeypatch)
    comparison_calls = []
    log_calls = []
    edit_calls = []

    def fake_extract_document_text_with_metadata(*args, **kwargs):
        raise ValueError("OCR did not extract text from PDF page 2")

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )
    monkeypatch.setattr(
        documents_route,
        "compare_documents",
        lambda **kwargs: comparison_calls.append(kwargs),
    )
    monkeypatch.setattr(
        documents_route, "log_token_usage", lambda **kwargs: log_calls.append(kwargs)
    )
    monkeypatch.setattr(
        documents_route,
        "edit_tokens",
        lambda user_email, token_delta: edit_calls.append((user_email, token_delta)),
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
        "details": "OCR did not extract text from PDF page 2",
    }
    assert comparison_calls == []
    assert log_calls == []
    assert edit_calls == []


def test_compare_docs_hands_off_log_id_without_exposing_it(monkeypatch):
    """Usage Duration Slice B3: /compare_docs previously discarded the
    returned log_id entirely. It must now capture it internally and hand
    it off via set_usage_log_id(), while the response contract - no
    log_id field - stays exactly as before."""
    app = _make_app()
    _patch_success_dependencies(monkeypatch)

    handoff_calls = []

    def fake_extract_document_text_with_metadata(doc_input, **kwargs):
        return {
            "document": {
                "text": f"normalized {doc_input['filename']}",
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            },
            "ocr_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    def fake_compare_documents(**kwargs):
        return {
            "comparison": {
                "score": 91,
                "summary": "Documents match",
                "reasoning": "Mocked comparison",
            },
            "token_usage": {"input_tokens": 13, "output_tokens": 6, "total_tokens": 19},
        }

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )
    monkeypatch.setattr(documents_route, "compare_documents", fake_compare_documents)
    monkeypatch.setattr(documents_route, "log_token_usage", lambda **kwargs: 555)
    monkeypatch.setattr(
        documents_route,
        "set_usage_log_id",
        lambda log_id: handoff_calls.append(log_id),
    )
    monkeypatch.setattr(documents_route, "edit_tokens", lambda *a, **k: None)

    data = {
        "prompt": "Compare these documents",
        "text_documents": json.dumps(
            [
                {"content": "first document", "filename": "first.txt", "role": "a"},
                {"content": "second document", "filename": "second.txt", "role": "b"},
            ]
        ),
    }

    response = app.test_client().post(
        "/compare_docs",
        data=data,
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )

    assert response.status_code == 200
    assert handoff_calls == [555]
    assert "log_id" not in response.get_json()


def test_compare_docs_usage_write_failure_registers_no_log_id(monkeypatch):
    """Usage Duration Slice B3 invariant: the handoff only ever follows a
    successful Usage INSERT. When log_token_usage() raises, /compare_docs
    keeps its existing behavior (caught, logged, 200 preserved) and
    set_usage_log_id() must not be called."""
    app = _make_app()
    _patch_success_dependencies(monkeypatch)

    handoff_calls = []

    def fake_extract_document_text_with_metadata(doc_input, **kwargs):
        return {
            "document": {
                "text": f"normalized {doc_input['filename']}",
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            },
            "ocr_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    def fake_compare_documents(**kwargs):
        return {
            "comparison": {"score": 91, "summary": "s", "reasoning": "r"},
            "token_usage": {"input_tokens": 13, "output_tokens": 6, "total_tokens": 19},
        }

    def raising_log_token_usage(**kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )
    monkeypatch.setattr(documents_route, "compare_documents", fake_compare_documents)
    monkeypatch.setattr(documents_route, "log_token_usage", raising_log_token_usage)
    monkeypatch.setattr(
        documents_route,
        "set_usage_log_id",
        lambda log_id: handoff_calls.append(log_id),
    )
    monkeypatch.setattr(documents_route, "edit_tokens", lambda *a, **k: None)

    data = {
        "prompt": "Compare these documents",
        "text_documents": json.dumps(
            [
                {"content": "first document", "filename": "first.txt", "role": "a"},
                {"content": "second document", "filename": "second.txt", "role": "b"},
            ]
        ),
    }

    response = app.test_client().post(
        "/compare_docs",
        data=data,
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
    )

    # Existing behavior preserved: the exception is caught inside
    # compare_docs's own try/except, response stays a valid 200.
    assert response.status_code == 200
    assert handoff_calls == []


# ---------------------------------------------------------------------------
# Persistent Operational events — PILOT SLICE P2 (E1, E5)
#
# Assertions are on LogRecord metadata captured via caplog: the emission
# boundary is logger.<level>(message, extra=extra), and the persistence
# queue/DB are deliberately out of scope here (they have their own foundation
# tests). E1 is INFO, so those tests raise the capture level explicitly; that
# is a test mechanic only and changes no runtime configuration.
# ---------------------------------------------------------------------------

ROUTE_LOGGER_NAME = documents_route.__name__


def _persistent_records(caplog, event=None):
    records = [r for r in caplog.records if getattr(r, "maui_persist", False)]
    if event is not None:
        records = [r for r in records if getattr(r, "maui_event", None) == event]
    return records


def _patch_full_success_flow(monkeypatch):
    """Success-path doubles for extraction and comparison, on top of the
    shared dependency patches, so a request reaches the 200 response."""
    _patch_success_dependencies(monkeypatch)

    def fake_extract_document_text_with_metadata(doc_input, **kwargs):
        return {
            "document": {
                "text": "normalized document",
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            },
            "ocr_token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )
    monkeypatch.setattr(
        documents_route,
        "compare_documents",
        lambda **kwargs: {
            "comparison": {"score": 91, "summary": "ok", "reasoning": "because"},
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        },
    )


def _two_text_documents():
    return json.dumps(
        [
            {"content": "first document", "filename": "first.txt", "role": "reference"},
            {
                "content": "second document",
                "filename": "second.txt",
                "role": "candidate",
            },
        ]
    )


def _post(app, data, content_type=None):
    kwargs = {}
    if content_type is not None:
        kwargs["content_type"] = content_type
    return app.test_client().post(
        "/compare_docs",
        data=data,
        headers={"X-API-KEY": "test-key", "X-USER-EMAIL": "user@example.com"},
        **kwargs,
    )


def test_compare_docs_emits_one_started_event_with_exact_fields(monkeypatch, caplog):
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("additional_context", "SENTINEL-CONTEXT-do-not-persist"),
            ("language", "SENTINEL-LANGUAGE-do-not-persist"),
            (
                "text_documents",
                json.dumps(
                    [{"content": "raw text document", "filename": "reference.txt"}]
                ),
            ),
            ("file_roles", json.dumps(["candidate"])),
            ("files", (io.BytesIO(b"%PDF fake bytes"), "SENTINEL-FILENAME.pdf")),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data, content_type="multipart/form-data")

    assert response.status_code == 200

    started = _persistent_records(caplog, "compare_docs_started")
    assert len(started) == 1

    record = started[0]
    assert record.maui_persist is True
    assert record.maui_event == "compare_docs_started"
    assert record.maui_provider == "Google"
    assert record.maui_model == "gemini-2.5-flash"
    assert record.maui_details == {
        "file_count": 1,
        "text_document_count": 1,
        "ocr_configured": True,
        "language_present": True,
        "additional_context_present": True,
    }
    assert record.levelno == logging.INFO
    assert record.name == ROUTE_LOGGER_NAME

    # E1 declares no duration_ms, no error_type and no message.
    assert not hasattr(record, "maui_duration_ms")
    assert not hasattr(record, "maui_error_type")
    assert not hasattr(record, "maui_message")


def test_started_event_records_presence_booleans_not_client_content(
    monkeypatch, caplog
):
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    data = MultiDict(
        [
            ("prompt", "SENTINEL-PROMPT-do-not-persist"),
            ("additional_context", "SENTINEL-CONTEXT-do-not-persist"),
            ("language", "SENTINEL-LANGUAGE-do-not-persist"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 200

    record = _persistent_records(caplog, "compare_docs_started")[0]
    assert record.maui_details["language_present"] is True
    assert record.maui_details["additional_context_present"] is True

    # The values themselves reach neither the structured metadata nor the
    # rendered message. Sentinels are short and distinctive, so these checks
    # are exact rather than ambiguous.
    for sentinel in (
        "SENTINEL-PROMPT-do-not-persist",
        "SENTINEL-CONTEXT-do-not-persist",
        "SENTINEL-LANGUAGE-do-not-persist",
    ):
        assert sentinel not in record.getMessage()
        assert sentinel not in str(record.maui_details)


def test_started_event_absent_optionals_are_false(monkeypatch, caplog):
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 200

    record = _persistent_records(caplog, "compare_docs_started")[0]
    assert record.maui_details == {
        "file_count": 0,
        "text_document_count": 2,
        "ocr_configured": True,
        "language_present": False,
        "additional_context_present": False,
    }


def test_started_event_reports_ocr_not_configured(monkeypatch, caplog):
    app = _make_app()
    app.config["MAUI_CONFIG"].models.vision_model = None
    _patch_full_success_flow(monkeypatch)

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 200
    record = _persistent_records(caplog, "compare_docs_started")[0]
    assert record.maui_details["ocr_configured"] is False


@pytest.mark.parametrize(
    "data, expected_details",
    [
        pytest.param(
            {"text_documents": "[]"},
            "The prompt field is required.",
            id="missing_prompt",
        ),
        pytest.param(
            {
                "prompt": "Compare these documents",
                "text_documents": json.dumps([{"content": "only one"}]),
            },
            "At least two documents are required.",
            id="insufficient_document_count",
        ),
    ],
)
def test_early_rejection_emits_no_started_event(
    monkeypatch, caplog, data, expected_details
):
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 400
    assert response.get_json()["details"] == expected_details
    assert _persistent_records(caplog, "compare_docs_started") == []
    assert _persistent_records(caplog) == []


def test_value_error_path_emits_one_controlled_failure_event(monkeypatch, caplog):
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    def fake_extract_document_text_with_metadata(*args, **kwargs):
        raise ValueError("SENTINEL-ERROR-TEXT for candidate.pdf")

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    # Existing 400 contract, unchanged: the raw exception text still reaches
    # the client exactly as before.
    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid request",
        "details": "SENTINEL-ERROR-TEXT for candidate.pdf",
    }

    failures = _persistent_records(caplog, "compare_docs_controlled_failure")
    assert len(failures) == 1

    record = failures[0]
    assert record.maui_persist is True
    assert record.maui_error_type == "ValueError"
    assert record.maui_details == {"http_status": 400}
    assert record.levelno == logging.WARNING
    assert not hasattr(record, "maui_message")

    # The sentinel is in the HTTP body (unchanged contract) and nowhere in the
    # persistent metadata or the rendered operational message.
    assert "SENTINEL-ERROR-TEXT" not in record.getMessage()
    assert "SENTINEL-ERROR-TEXT" not in str(record.maui_details)
    assert "SENTINEL-ERROR-TEXT" not in str(getattr(record, "maui_error_type", ""))
    assert record.exc_info is None


def test_not_implemented_error_path_emits_one_controlled_failure_event(
    monkeypatch, caplog
):
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    def fake_extract_document_text_with_metadata(*args, **kwargs):
        raise NotImplementedError("SENTINEL-UNSUPPORTED-FORMAT .xyz")

    monkeypatch.setattr(
        documents_route,
        "extract_document_text_with_metadata",
        fake_extract_document_text_with_metadata,
    )

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 415
    assert response.get_json() == {
        "error": "Unsupported document format",
        "details": "SENTINEL-UNSUPPORTED-FORMAT .xyz",
    }

    failures = _persistent_records(caplog, "compare_docs_controlled_failure")
    assert len(failures) == 1

    record = failures[0]
    assert record.maui_error_type == "NotImplementedError"
    assert record.maui_details == {"http_status": 415}
    assert record.levelno == logging.WARNING
    assert "SENTINEL-UNSUPPORTED-FORMAT" not in record.getMessage()
    assert "SENTINEL-UNSUPPORTED-FORMAT" not in str(record.maui_details)


def test_payload_too_large_path_emits_no_controlled_failure_event(monkeypatch, caplog):
    """413 is its own terminal handler and is NOT an E5 site (E4 belongs to the
    service layer, slice P3). It must stay silent for E5 in P2."""
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    def fake_compare_documents(**kwargs):
        raise documents_route.DocumentComparisonPayloadTooLargeError("too large")

    monkeypatch.setattr(documents_route, "compare_documents", fake_compare_documents)

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 413
    assert _persistent_records(caplog, "compare_docs_controlled_failure") == []
    assert len(_persistent_records(caplog, "compare_docs_started")) == 1


def test_success_path_emits_no_controlled_failure_event(monkeypatch, caplog):
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 200
    assert _persistent_records(caplog, "compare_docs_controlled_failure") == []
    assert len(_persistent_records(caplog, "compare_docs_started")) == 1


def test_route_call_sites_declare_no_context_ownership_keys(monkeypatch, caplog):
    """request_id/app_id remain infrastructure-owned: the route cannot and does
    not declare them, and no maui_* key beyond the declared contract appears."""
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    with caplog.at_level(logging.INFO, logger=ROUTE_LOGGER_NAME):
        response = _post(app, data)

    assert response.status_code == 200
    record = _persistent_records(caplog, "compare_docs_started")[0]

    declared_maui_keys = {
        key for key in vars(record) if key.startswith("maui_")
    }
    assert declared_maui_keys == {
        "maui_persist",
        "maui_event",
        "maui_provider",
        "maui_model",
        "maui_details",
    }
    assert record.args == ()


def test_operational_persistence_failure_does_not_affect_compare_docs_response(
    monkeypatch,
):
    """Route-level fail-open, at the boundary P2 actually introduces.

    tests/test_operational_persistence_integration.py::test_e3_* already proves
    fail-open through the real handler/delivery/writer stack, but on a synthetic
    route emitting a synthetic event. P2 is the first time a REAL endpoint
    emits, so the property is asserted here at that exact seam, kept light: the
    REAL OperationalPersistenceHandler with a sink that raises. No queue, no
    database, no PostgreSQL, and no change to persistence delivery.
    """
    app = _make_app()
    _patch_full_success_flow(monkeypatch)

    captured = []

    def _raising_sink(snapshot):
        captured.append(snapshot)
        raise RuntimeError("simulated operational persistence failure")

    handler = OperationalPersistenceHandler(sink=_raising_sink)

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    data = MultiDict(
        [
            ("prompt", "Compare these documents"),
            ("text_documents", _two_text_documents()),
        ]
    )

    try:
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        response = _post(app, data)
    finally:
        root.handlers = saved_handlers
        root.level = saved_level

    # The event really did reach the persistence boundary and really did fail
    # there, and the request completed with its normal response regardless.
    assert len(captured) == 1
    assert captured[0].event == "compare_docs_started"
    assert response.status_code == 200
    assert response.get_json() == {
        "score": 91,
        "summary": "ok",
        "reasoning": "because",
    }


# ---------------------------------------------------------------------------
# Combined pilot timeline — PILOT SLICE P3
#
# The only test in the pilot that exercises route AND service instrumentation
# together. The route's own collaborators are NOT stubbed here: the real
# extract_document_text_with_metadata and compare_documents run, and only the
# leaf I/O boundaries below them (local parser, PDF renderer, Vision provider,
# comparison provider, prompt loader) are faked. That is what makes the event
# ORDER across layers meaningful rather than asserted against two mocks.
# ---------------------------------------------------------------------------

_TIMELINE_OCR_TEXT = "SENTINEL-OCR-TEXT-do-not-persist"
_TIMELINE_PDF_FILENAME = "SENTINEL-SCAN.pdf"


def _patch_real_service_leaves(monkeypatch, *, ocr_text=_TIMELINE_OCR_TEXT):
    """Fake only the leaves under the two real pilot services."""
    _patch_success_dependencies(monkeypatch)

    def fake_local_extract(doc_input):
        if doc_input["source_type"] == "text":
            return {
                "text": doc_input["content"],
                "filename": doc_input["filename"],
                "role": doc_input["role"],
            }
        # A scanned PDF: local parsing yields too little to compare.
        return {
            "text": "short",
            "filename": doc_input["filename"],
            "role": doc_input["role"],
        }

    monkeypatch.setattr(
        extraction_service, "extract_and_normalize_document", fake_local_extract
    )
    monkeypatch.setattr(
        extraction_service,
        "render_pdf_pages_to_png",
        lambda pdf_bytes: [
            SimpleNamespace(page_number=1, image_bytes=b"page one"),
            SimpleNamespace(page_number=2, image_bytes=b"page two"),
        ],
    )
    monkeypatch.setattr(
        extraction_service, "is_rendered_page_blank", lambda image_bytes: False
    )
    monkeypatch.setattr(
        extraction_service,
        "extract_text_from_image_with_usage",
        lambda *args, **kwargs: {
            "text": ocr_text,
            "token_usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
            },
        },
    )

    monkeypatch.setattr(
        comparison_service,
        "load_prompt",
        lambda title, default_text="": default_text,
    )


def _timeline_request_data():
    return MultiDict(
        [
            ("prompt", "Compare these documents"),
            (
                "text_documents",
                json.dumps(
                    [{"content": "a" * 80, "filename": "reference.txt"}]
                ),
            ),
            ("file_roles", json.dumps(["candidate"])),
            (
                "files",
                (io.BytesIO(b"%PDF fake bytes"), _TIMELINE_PDF_FILENAME),
            ),
        ]
    )


def test_successful_compare_docs_emits_the_full_three_layer_timeline(
    monkeypatch, caplog
):
    app = _make_app()
    _patch_real_service_leaves(monkeypatch)

    class SucceedingLlm:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps(
                    {"score": 88, "summary": "ok", "reasoning": "because"}
                ),
                usage_metadata={
                    "input_tokens": 40,
                    "output_tokens": 12,
                    "total_tokens": 52,
                },
            )

    monkeypatch.setattr(
        comparison_service, "choose_llm", lambda *args, **kwargs: SucceedingLlm()
    )

    # The ContextDefaultsFilter is HANDLER-owned in production
    # (utils/logging_config.py), so caplog's own handler sees records before
    # any such mutation unless the same filter is attached to it. Attaching it
    # here reproduces the real handler boundary instead of plumbing a
    # request_id through the call sites, which O3 forbids outright.
    caplog.handler.addFilter(ContextDefaultsFilter())

    with caplog.at_level(logging.INFO):
        response = _post(
            app, _timeline_request_data(), content_type="multipart/form-data"
        )

    assert response.status_code == 200
    assert response.get_json()["score"] == 88

    timeline = [r.maui_event for r in _persistent_records(caplog)]
    assert timeline == [
        "compare_docs_started",
        "document_ocr_fallback_completed",
        "document_comparison_completed",
    ]

    records = _persistent_records(caplog)

    # One request, one correlation id, across three modules.
    request_ids = {r.request_id for r in records}
    assert len(request_ids) == 1
    assert request_ids != {"-"}

    assert {r.name for r in records} == {
        documents_route.__name__,
        extraction_service.__name__,
        comparison_service.__name__,
    }

    fallback = _persistent_records(caplog, "document_ocr_fallback_completed")[0]
    assert fallback.maui_provider == "Deepinfra"
    assert fallback.maui_model == "vision-ocr-model"
    assert fallback.maui_details["page_count"] == 2
    assert fallback.maui_details["reason"] == "insufficient_local_text"

    completed = _persistent_records(caplog, "document_comparison_completed")[0]
    assert completed.maui_provider == "Google"
    assert completed.maui_details["document_count"] == 2
    assert completed.maui_details["total_tokens"] == 52

    # No failure event anywhere on a success, and no leakage across the whole
    # persisted timeline.
    assert _persistent_records(caplog, "compare_docs_controlled_failure") == []
    assert _persistent_records(caplog, "document_ocr_fallback_failed") == []
    for record in records:
        rendered = " ".join(
            [
                record.getMessage(),
                str(getattr(record, "maui_message", "")),
                str(getattr(record, "maui_details", {})),
            ]
        )
        assert "SENTINEL" not in rendered


def test_ocr_failure_emits_service_cause_then_route_controlled_failure(
    monkeypatch, caplog
):
    """E6 and E5 are layered, not duplicated: E6 names the specific OCR cause
    inside the service, E5 records the controlled HTTP outcome at the route."""
    app = _make_app()
    _patch_real_service_leaves(monkeypatch)
    monkeypatch.setattr(
        extraction_service,
        "is_rendered_page_blank",
        lambda image_bytes: image_bytes == b"page two",
    )
    monkeypatch.setattr(
        comparison_service,
        "choose_llm",
        lambda *args, **kwargs: pytest.fail("comparison must not be reached"),
    )

    with caplog.at_level(logging.INFO):
        response = _post(
            app, _timeline_request_data(), content_type="multipart/form-data"
        )

    assert response.status_code == 400

    timeline = [r.maui_event for r in _persistent_records(caplog)]
    assert timeline == [
        "compare_docs_started",
        "document_ocr_fallback_failed",
        "compare_docs_controlled_failure",
    ]

    failed = _persistent_records(caplog, "document_ocr_fallback_failed")[0]
    assert failed.levelno == logging.WARNING
    assert failed.maui_details == {
        "page_number": 2,
        "page_count": 2,
        "reason": "blank_page",
    }

    controlled = _persistent_records(caplog, "compare_docs_controlled_failure")[0]
    assert controlled.maui_error_type == "ValueError"
    assert controlled.maui_details["http_status"] == 400

    assert _persistent_records(caplog, "document_ocr_fallback_completed") == []
    assert _persistent_records(caplog, "document_comparison_completed") == []
