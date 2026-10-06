"""DataChat HTTP client of the Functional Evaluation harness.

Sends the frozen request shapes to a harness-owned Maui and returns each
attempt as an ``HttpExchange`` plus the bytes the runner will persist.
It persists nothing, assigns no ``INFRA_*`` code and interprets no response:
any HTTP status, ``kind:error`` or HTML body is evidence, not a failure.

CURRENT route contract (routes/datachat.py, routes/system.py):

- identity headers ``X-API-KEY``, ``X-USER-EMAIL``, ``X-USER-NAME``;
- ``POST /startdatachat``: multipart ``file`` (CSV); ``model_name``,
  ``llm_type`` and ``lang`` are optional and omitted (Maui's env decides);
- ``POST /datachat``: JSON ``{"chat": ...}``;
- ``POST /enddatachat``: no body; the "not active" reply echoes the API key;
- ``GET /datachat/export/<token>``: the relative ``download_url``;
- ``GET /health``: no authentication.

Retry policy: only ``/enddatachat`` is retried, once, and only after a
connection-level failure (``httpx.ConnectError`` / ``httpx.ConnectTimeout``)
where no response was received. ``/startdatachat``, ``/datachat`` and export
are single-attempt. The API key is redacted from stored headers and from any
stored response body before hashing; it never appears in a repr.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpx

from evaluation.datachat.contract import HTTP_CLIENT_TIMEOUT_S
from evaluation.datachat.evidence import redact_bytes, redact_headers, sha256_bytes, sha256_file
from evaluation.datachat.model import (
    HttpExchange,
    HttpRequestRecord,
    HttpResponseRecord,
    TransportError,
    utc_now,
)

END_KEYS = frozenset({"pre_cleanup", "teardown"})
EXPORT_PATH_PREFIX = "/datachat/export/"
CSV_FIELD = "file"
_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout)
_MAX_ERROR_MESSAGE = 500


@dataclass(frozen=True)
class Identity:
    """Evaluation identity. The API key lives only in memory and is hidden from repr."""

    api_key: str = field(repr=False)
    user_email: str
    user_name: str

    def __post_init__(self) -> None:
        for name in ("api_key", "user_email", "user_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"identity {name} must be a non-empty string")

    def headers(self) -> dict[str, str]:
        return {
            "X-API-KEY": self.api_key,
            "X-USER-EMAIL": self.user_email,
            "X-USER-NAME": self.user_name,
        }


@dataclass(frozen=True)
class ExchangeResult:
    """One attempt: metadata plus bytes to persist (already redacted where needed)."""

    exchange: HttpExchange
    request_bytes: bytes | None
    response_bytes: bytes | None


@dataclass(frozen=True)
class HealthResult:
    reachable: bool
    status: int | None
    error: TransportError | None = None


def http_file_stem(key: str, attempt: int) -> str:
    """Same naming as ``evidence.RunLayout.http_paths`` (relative to the scenario dir)."""
    return f"http/{key}" if attempt == 1 else f"http/{key}.{attempt}"


class DataChatClient:
    def __init__(
        self,
        base_url: str,
        identity: Identity,
        timeout_s: float = HTTP_CLIENT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"invalid base url: {base_url!r}")
        self.base_url = f"{parts.scheme}://{parts.netloc}"
        self._identity = identity
        self._monotonic = monotonic
        # No transport-level retries: httpx's default transport has retries=0.
        self._http = httpx.Client(
            base_url=self.base_url, timeout=timeout_s, transport=transport, follow_redirects=False
        )

    def __repr__(self) -> str:
        return f"DataChatClient(base_url={self.base_url!r}, identity={self._identity!r})"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "DataChatClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- Public calls --------------------------------------------------------

    def health(self) -> HealthResult:
        """Single ``GET /health``; polling is the caller's job."""
        try:
            response = self._http.get("/health")
        except httpx.TransportError as exc:
            return HealthResult(False, None, self._transport_error(exc))
        return HealthResult(True, response.status_code)

    def start(self, csv_path: Path, lang: str | None = None) -> ExchangeResult:
        """``POST /startdatachat`` with the frozen CSV. Never retried (debits tokens)."""
        csv_path = Path(csv_path)
        form = {"lang": lang} if lang is not None else {}
        evidence_form = {
            **form,
            CSV_FIELD: {
                "filename": csv_path.name,
                "dataset_sha256": sha256_file(csv_path),
                "size": csv_path.stat().st_size,
            },
        }
        with open(csv_path, "rb") as handle:
            return self._send(
                "start", 1, "POST", "/startdatachat",
                data=form or None,
                files={CSV_FIELD: (csv_path.name, handle, "text/csv")},
                record_body=None,
                form_evidence=evidence_form,
            )

    def datachat(self, question: str) -> ExchangeResult:
        """Exactly one ``POST /datachat``; the stored request bytes are the bytes sent."""
        body = json.dumps({"chat": question}, ensure_ascii=False).encode("utf-8")
        return self._send(
            "datachat", 1, "POST", "/datachat",
            content=body,
            content_type="application/json",
            record_body=body,
        )

    def end(self, key: str) -> list[ExchangeResult]:
        """``POST /enddatachat`` as ``pre_cleanup`` or ``teardown``; at most one retry."""
        if key not in END_KEYS:
            raise ValueError(f"invalid end key: {key!r}")
        attempts = []
        for attempt in (1, 2):
            result, retryable = self._send_end(key, attempt)
            attempts.append(result)
            if not retryable:
                break
        return attempts

    def export(self, download_url: str) -> ExchangeResult:
        """Single ``GET`` of a same-origin ``/datachat/export/<token>`` URL."""
        path = self._export_path(download_url)
        return self._send("export", 1, "GET", path, record_body=None)

    # --- Internals -----------------------------------------------------------

    def _export_path(self, download_url: str) -> str:
        if not isinstance(download_url, str) or not download_url:
            raise ValueError("download_url must be a non-empty string")
        absolute = urljoin(self.base_url + "/", download_url)
        parts = urlsplit(absolute)
        if f"{parts.scheme}://{parts.netloc}" != self.base_url:
            raise ValueError(f"download_url is not same-origin: {download_url!r}")
        if not parts.path.startswith(EXPORT_PATH_PREFIX) or parts.query or parts.fragment:
            raise ValueError(f"download_url is not a DataChat export path: {download_url!r}")
        return parts.path

    def _send_end(self, key: str, attempt: int) -> tuple[ExchangeResult, bool]:
        retryable = False

        def note(exc: httpx.TransportError) -> None:
            nonlocal retryable
            retryable = isinstance(exc, _RETRYABLE)

        result = self._send(key, attempt, "POST", "/enddatachat", record_body=None, on_transport_error=note)
        return result, retryable and attempt == 1

    def _transport_error(self, exc: httpx.TransportError) -> TransportError:
        message = str(exc).replace(self._identity.api_key, "<redacted>")
        return TransportError(type=type(exc).__name__, message=message[:_MAX_ERROR_MESSAGE])

    def _send(
        self,
        key: str,
        attempt: int,
        method: str,
        path: str,
        *,
        record_body: bytes | None,
        content: bytes | None = None,
        content_type: str | None = None,
        data: dict | None = None,
        files: dict | None = None,
        form_evidence: dict | None = None,
        on_transport_error: Callable[[httpx.TransportError], None] | None = None,
    ) -> ExchangeResult:
        headers = self._identity.headers()
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = self._http.build_request(method, path, headers=headers, content=content, data=data, files=files)
        stem = http_file_stem(key, attempt)
        stored_headers = redact_headers(
            {k: v for k, v in request.headers.items() if k.lower() not in ("content-length", "host")}
        )
        request_record = HttpRequestRecord(
            headers_redacted=stored_headers,
            content_type=request.headers.get("content-type"),
            body_file=f"{stem}.request.bin" if record_body is not None else None,
            body_sha256=sha256_bytes(record_body) if record_body is not None else None,
            form=form_evidence,
        )

        sent_at = utc_now()
        started = self._monotonic()
        response = None
        transport_error = None
        try:
            response = self._http.send(request)
            body = response.content
        except httpx.TransportError as exc:
            transport_error = self._transport_error(exc)
            if on_transport_error is not None:
                on_transport_error(exc)
        duration_ms = int(round((self._monotonic() - started) * 1000))
        completed_at = utc_now()

        response_record = None
        stored_body = None
        request_id = None
        if response is not None:
            stored_body, redacted = redact_bytes(body, [self._identity.api_key])
            request_id = response.headers.get("X-Request-ID")
            response_record = HttpResponseRecord(
                status=response.status_code,
                headers=dict(response.headers.items()),
                content_type=response.headers.get("content-type"),
                body_file=f"{stem}.response.bin",
                body_sha256=sha256_bytes(stored_body),
                size=len(stored_body),
                redacted=redacted,
            )

        exchange = HttpExchange(
            key=key,
            attempt=attempt,
            method=method,
            path=path,
            sent_at=sent_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            request=request_record,
            response=response_record,
            request_id=request_id,
            transport_error=transport_error,
        )
        return ExchangeResult(exchange, record_body, stored_body)
