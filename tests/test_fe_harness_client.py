"""Tests for evaluation.datachat.client, the DataChat HTTP client of the FE harness.

Every request goes through httpx.MockTransport; nothing touches the network.
Pins the CURRENT route shapes, exact stored request bytes, X-Request-ID
capture, the /enddatachat-only connection-level retry, API-key redaction
and same-origin export URLs.
"""

import json

import httpx
import pytest

from evaluation.datachat.client import (
    DataChatClient,
    Identity,
    http_file_stem,
)
from evaluation.datachat.evidence import RunLayout, sha256_bytes, sha256_file
from evaluation.datachat.model import HttpExchange

API_KEY = "sk-fe-eval-0123456789"
BASE = "http://127.0.0.1:5099"
IDENTITY = Identity(api_key=API_KEY, user_email="fe-eval@example.invalid", user_name="fe-eval-v1")


class Recorder:
    """MockTransport handler replaying scripted responses/exceptions and recording requests."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def __call__(self, request):
        request.read()
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _client(recorder):
    return DataChatClient(BASE, IDENTITY, transport=httpx.MockTransport(recorder))


def _json(status, payload, request_id="0123456789abcdef"):
    headers = {"X-Request-ID": request_id} if request_id else {}
    return httpx.Response(status, json=payload, headers=headers)


def _stored_repr(result):
    return json.dumps(result.exchange.to_dict()) + repr(result.response_bytes)


# --- Identity / headers ------------------------------------------------------


def test_identity_and_client_repr_hide_api_key():
    client = DataChatClient(BASE, IDENTITY)
    assert API_KEY not in repr(IDENTITY)
    assert API_KEY not in repr(client)
    client.close()


@pytest.mark.parametrize("field", ["api_key", "user_email", "user_name"])
def test_identity_rejects_empty_values(field):
    values = {"api_key": "k", "user_email": "e", "user_name": "n", field: " "}
    with pytest.raises(ValueError):
        Identity(**values)


def test_base_url_normalized_and_validated():
    with DataChatClient(BASE + "/", IDENTITY) as client:
        assert client.base_url == BASE
    with pytest.raises(ValueError):
        DataChatClient("127.0.0.1:5000", IDENTITY)


def test_identity_headers_sent_and_redacted_in_metadata():
    recorder = Recorder(_json(200, {"response": {"type": "str", "value": "ok"}}))
    with _client(recorder) as client:
        result = client.datachat("ciao")

    sent = recorder.requests[0].headers
    assert sent["X-API-KEY"] == API_KEY
    assert sent["X-USER-EMAIL"] == IDENTITY.user_email
    assert sent["X-USER-NAME"] == IDENTITY.user_name
    stored = {k.lower(): v for k, v in result.exchange.request.headers_redacted.items()}
    assert stored["x-api-key"] == "<redacted>"
    assert stored["x-user-email"] == IDENTITY.user_email
    assert API_KEY not in _stored_repr(result)


# --- Request IDs -------------------------------------------------------------


def test_request_id_captured_or_null():
    recorder = Recorder(_json(200, {}, "aaaaaaaaaaaaaaaa"), _json(200, {}, None))
    with _client(recorder) as client:
        first = client.datachat("a")
        second = client.datachat("b")
    assert first.exchange.request_id == "aaaaaaaaaaaaaaaa"
    assert second.exchange.request_id is None


# --- /datachat ---------------------------------------------------------------


def test_datachat_exact_request_bytes_and_unicode():
    question = "Nel Sud, quale servizio ha il tempo medio di attesa più alto?"
    recorder = Recorder(_json(200, {"response": {"type": "str", "value": "x"}}))
    with _client(recorder) as client:
        result = client.datachat(question)

    sent = recorder.requests[0]
    expected = json.dumps({"chat": question}, ensure_ascii=False).encode("utf-8")
    assert sent.content == expected
    assert result.request_bytes == expected
    assert "più".encode() in sent.content
    assert sent.headers["Content-Type"] == "application/json"
    request = result.exchange.request
    assert request.body_file == "http/datachat.request.bin"
    assert request.body_sha256 == sha256_bytes(expected)
    assert request.content_type == "application/json"
    assert (result.exchange.method, result.exchange.path, result.exchange.key) == ("POST", "/datachat", "datachat")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b'{"response": {"type": "str", "value": "fine"}, "log_id": 7}',
                       headers={"content-type": "application/json"}),
        httpx.Response(200, content=b'{"response": {"kind": "error", "code": "RUN_FAILED"}}',
                       headers={"content-type": "application/json"}),
        httpx.Response(403, content=b"<html><body>Forbidden \xe2\x80\x94</body></html>",
                       headers={"content-type": "text/html"}),
        httpx.Response(500, content=b'{"error": "boom"}', headers={"content-type": "application/json"}),
    ],
)
def test_datachat_preserves_any_response_without_retry(response):
    raw = response.content
    recorder = Recorder(response)
    with _client(recorder) as client:
        result = client.datachat("q")

    assert len(recorder.requests) == 1
    assert result.response_bytes == raw
    meta = result.exchange.response
    assert meta.status == response.status_code
    assert meta.body_sha256 == sha256_bytes(raw)
    assert meta.size == len(raw)
    assert meta.redacted is False
    assert meta.body_file == "http/datachat.response.bin"
    assert result.exchange.transport_error is None


def test_datachat_transport_error_not_retried():
    recorder = Recorder(httpx.ConnectError("refused"), _json(200, {}))
    with _client(recorder) as client:
        result = client.datachat("q")
    assert len(recorder.requests) == 1
    assert result.exchange.response is None
    assert result.exchange.request_id is None
    assert result.exchange.transport_error.type == "ConnectError"
    assert result.response_bytes is None


def test_exchange_round_trips_through_model():
    recorder = Recorder(_json(200, {"ok": True}))
    with _client(recorder) as client:
        exchange = client.datachat("q").exchange
    data = json.loads(json.dumps(exchange.to_dict()))
    assert HttpExchange.from_dict(data) == exchange


# --- /startdatachat ----------------------------------------------------------


def test_start_sends_csv_multipart_and_references_hash(tmp_path):
    csv_path = tmp_path / "evaluation_dataset.csv"
    csv_bytes = "region,cost\nNord,1.5\nSud,2\n".encode()
    csv_path.write_bytes(csv_bytes)
    recorder = Recorder(_json(200, {"Agent active": "active"}))
    with _client(recorder) as client:
        result = client.start(csv_path)

    assert len(recorder.requests) == 1
    sent = recorder.requests[0]
    assert sent.url.path == "/startdatachat"
    assert sent.headers["Content-Type"].startswith("multipart/form-data")
    assert b'name="file"; filename="evaluation_dataset.csv"' in sent.content
    assert csv_bytes in sent.content
    assert b'name="lang"' not in sent.content
    assert b"model_name" not in sent.content and b"llm_type" not in sent.content

    assert result.request_bytes is None
    request = result.exchange.request
    assert request.body_file is None and request.body_sha256 is None
    assert request.form == {
        "file": {"filename": "evaluation_dataset.csv", "dataset_sha256": sha256_file(csv_path), "size": len(csv_bytes)}
    }
    assert result.exchange.response.status == 200


def test_start_records_lang_when_sent(tmp_path):
    csv_path = tmp_path / "d.csv"
    csv_path.write_bytes(b"a\n1\n")
    recorder = Recorder(_json(200, {"Agent active": "active"}))
    with _client(recorder) as client:
        result = client.start(csv_path, lang="ITA")
    assert b'name="lang"' in recorder.requests[0].content
    assert result.exchange.request.form["lang"] == "ITA"


def test_start_not_retried(tmp_path):
    csv_path = tmp_path / "d.csv"
    csv_path.write_bytes(b"a\n1\n")
    recorder = Recorder(httpx.ConnectError("refused"), _json(200, {}))
    with _client(recorder) as client:
        result = client.start(csv_path)
    assert len(recorder.requests) == 1
    assert result.exchange.transport_error.type == "ConnectError"


# --- /enddatachat ------------------------------------------------------------


def test_end_success_single_attempt():
    recorder = Recorder(_json(200, {"Agent deleted succesfully": "active"}, "bbbbbbbbbbbbbbbb"))
    with _client(recorder) as client:
        attempts = client.end("teardown")
    assert len(attempts) == 1
    exchange = attempts[0].exchange
    assert (exchange.key, exchange.attempt, exchange.path) == ("teardown", 1, "/enddatachat")
    assert recorder.requests[0].content == b""
    assert exchange.request.body_file is None
    assert exchange.response.redacted is False


def test_end_not_active_body_is_redacted():
    recorder = Recorder(_json(200, {"Agent was not active for this key": API_KEY}))
    with _client(recorder) as client:
        result = client.end("pre_cleanup")[0]
    assert API_KEY.encode() not in result.response_bytes
    assert result.response_bytes == b'{"Agent was not active for this key":"<redacted>"}'
    assert result.exchange.response.redacted is True
    assert result.exchange.response.body_sha256 == sha256_bytes(result.response_bytes)
    assert result.exchange.response.size == len(result.response_bytes)
    assert API_KEY not in _stored_repr(result)


def test_end_retries_once_after_connect_error():
    recorder = Recorder(httpx.ConnectError("refused"), _json(200, {"Agent deleted succesfully": "active"}, "cccccccccccccccc"))
    with _client(recorder) as client:
        attempts = client.end("teardown")
    assert [a.exchange.attempt for a in attempts] == [1, 2]
    assert attempts[0].exchange.response is None
    assert attempts[0].exchange.transport_error.type == "ConnectError"
    assert attempts[1].exchange.request_id == "cccccccccccccccc"
    assert attempts[1].exchange.response.body_file == "http/teardown.2.response.bin"


def test_end_two_connection_failures_stop_after_two():
    recorder = Recorder(httpx.ConnectError("a"), httpx.ConnectTimeout("b"), _json(200, {}))
    with _client(recorder) as client:
        attempts = client.end("teardown")
    assert len(attempts) == 2
    assert len(recorder.requests) == 2
    assert [a.exchange.transport_error.type for a in attempts] == ["ConnectError", "ConnectTimeout"]


def test_end_http_500_not_retried():
    recorder = Recorder(httpx.Response(500, text="err"), _json(200, {}))
    with _client(recorder) as client:
        attempts = client.end("teardown")
    assert len(attempts) == 1
    assert attempts[0].exchange.response.status == 500


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("closed")])
def test_end_failure_after_request_sent_not_retried(exc):
    recorder = Recorder(exc, _json(200, {}))
    with _client(recorder) as client:
        attempts = client.end("teardown")
    assert len(attempts) == 1
    assert attempts[0].exchange.transport_error.type == type(exc).__name__


def test_end_rejects_unknown_key():
    with DataChatClient(BASE, IDENTITY) as client, pytest.raises(ValueError):
        client.end("datachat")


def test_transport_error_message_redacts_api_key():
    recorder = Recorder(httpx.ConnectError(f"failed for {API_KEY}"))
    with _client(recorder) as client:
        result = client.datachat("q")
    assert API_KEY not in result.exchange.transport_error.message


# --- Export ------------------------------------------------------------------


def test_export_relative_url_resolved_and_bytes_preserved():
    csv = b"region,count\r\nNord,10\r\n"
    recorder = Recorder(httpx.Response(200, content=csv, headers={"content-type": "text/csv; charset=utf-8"}))
    with _client(recorder) as client:
        result = client.export("/datachat/export/tok123")
    sent = recorder.requests[0]
    assert str(sent.url) == f"{BASE}/datachat/export/tok123"
    assert sent.method == "GET"
    assert sent.headers["X-API-KEY"] == API_KEY
    assert result.response_bytes == csv
    assert result.exchange.key == "export"
    assert result.exchange.response.body_file == "http/export.response.bin"


def test_export_same_origin_absolute_allowed():
    recorder = Recorder(httpx.Response(200, content=b"a\n"))
    with _client(recorder) as client:
        client.export(f"{BASE}/datachat/export/tok")
    assert recorder.requests[0].url.path == "/datachat/export/tok"


def test_export_not_retried():
    recorder = Recorder(httpx.ConnectError("x"), httpx.Response(200, content=b"a\n"))
    with _client(recorder) as client:
        result = client.export("/datachat/export/tok")
    assert len(recorder.requests) == 1
    assert result.exchange.transport_error is not None


@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example/datachat/export/tok",
        "https://127.0.0.1:5099/datachat/export/tok",
        "//evil.example/datachat/export/tok",
        "/datachat",
        "/admin/export/tok",
        "/datachat/export/tok?x=1",
        "",
    ],
)
def test_export_rejects_foreign_or_unexpected_urls(url):
    recorder = Recorder()
    with _client(recorder) as client, pytest.raises(ValueError):
        client.export(url)
    assert recorder.requests == []


# --- Health ------------------------------------------------------------------


def test_health_results():
    recorder = Recorder(httpx.Response(200, json={"status": "ok"}), httpx.Response(503), httpx.ConnectError("down"))
    with _client(recorder) as client:
        ok, unavailable, down = client.health(), client.health(), client.health()
    assert (ok.reachable, ok.status, ok.error) == (True, 200, None)
    assert (unavailable.reachable, unavailable.status) == (True, 503)
    assert (down.reachable, down.status, down.error.type) == (False, None, "ConnectError")
    assert "X-API-KEY" not in recorder.requests[0].headers
    assert len(recorder.requests) == 3


# --- Naming ------------------------------------------------------------------


@pytest.mark.parametrize("key, attempt", [("datachat", 1), ("teardown", 2), ("export", 1)])
def test_file_stem_matches_run_layout(tmp_path, key, attempt):
    layout = RunLayout(tmp_path)
    _, request_path, response_path = layout.http_paths("FE-01", key, attempt)
    scenario = layout.scenario_dir("FE-01")
    stem = http_file_stem(key, attempt)
    assert f"{stem}.request.bin" == request_path.relative_to(scenario).as_posix()
    assert f"{stem}.response.bin" == response_path.relative_to(scenario).as_posix()
