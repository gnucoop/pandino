"""Tests for evaluation.datachat.collectors.artifacts, plot image and CSV export evidence.

A minimal valid PNG is built here with struct/zlib, so no image library is
needed. Covers CURRENT's "b'<base64>'" wrapper, every decode status and the
artifact metadata.
"""

import base64
import hashlib
import struct
import zlib

import pytest

from evaluation.datachat.collectors.artifacts import (
    BASE64_INVALID,
    DECODE_OK,
    NOT_PNG,
    PNG_SIGNATURE,
    WRAPPER_MISSING,
    decode_image_value,
    export_artifact,
    image_artifact,
)
from evaluation.datachat.model import Availability

RID = "0123456789abcdef"


def _chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _png(width=1, height=1):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00" * width for _ in range(height))
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")


def _wrap(data):
    return str(base64.b64encode(data))  # exactly what CURRENT produces


def test_valid_wrapped_png():
    png = _png(3, 2)
    decoded = decode_image_value(_wrap(png))
    assert decoded.status == DECODE_OK
    assert decoded.data == png
    assert (decoded.wrapper_stripped, decoded.png_signature_ok, decoded.detected_format) == (True, True, "png")


@pytest.mark.parametrize("value", [base64.b64encode(_png()).decode(), "b'", "b\"abc\"", "'abc'", "", "B'abc'"])
def test_wrapper_missing_or_malformed(value):
    decoded = decode_image_value(value)
    assert decoded.status == WRAPPER_MISSING
    assert decoded.wrapper_stripped is False
    assert decoded.data is None


@pytest.mark.parametrize("inner", ["not*base64", "abc", "YWJj\\nZGVm"])
def test_invalid_base64(inner):
    decoded = decode_image_value(f"b'{inner}'")
    assert decoded.status == BASE64_INVALID
    assert decoded.wrapper_stripped is True
    assert decoded.data is None


def test_decoded_non_png():
    decoded = decode_image_value(_wrap(b"\xff\xd8\xff\xe0 jpeg-ish"))
    assert decoded.status == NOT_PNG
    assert decoded.png_signature_ok is False
    assert decoded.detected_format == "jpeg"
    assert decoded.data is None


@pytest.mark.parametrize(
    "data",
    [
        PNG_SIGNATURE,
        PNG_SIGNATURE + b"garbage that is not a chunk at all",
        PNG_SIGNATURE + _chunk(b"IDAT", b"x" * 13),
        _png()[:20],
        _png()[:29] + b"\x00\x00\x00\x00" + _png()[33:],  # bad IHDR CRC
        PNG_SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", 0, 1, 8, 0, 0, 0, 0)),
    ],
)
def test_png_signature_without_valid_ihdr(data):
    decoded = decode_image_value(_wrap(data))
    assert decoded.status == NOT_PNG
    assert decoded.png_signature_ok is True
    assert decoded.data is None


def test_decode_rejects_non_str():
    with pytest.raises(TypeError):
        decode_image_value(b"b'abc'")


def test_image_artifact_metadata_ok():
    png = _png()
    record = image_artifact(RID, decode_image_value(_wrap(png)))
    assert record.to_dict() == {
        "kind": "image",
        "source_request_id": RID,
        "status": "PRESENT",
        "path": "artifacts/image.png",
        "sha256": hashlib.sha256(png).hexdigest(),
        "size": len(png),
        "wrapper_stripped": True,
        "decode_status": "OK",
        "png_signature_ok": True,
        "detected_format": "png",
        "http_status": None,
    }


def test_image_artifact_metadata_records_failure_without_payload():
    record = image_artifact(RID, decode_image_value("b'!!!'"))
    assert record.status is Availability.PRESENT
    assert record.decode_status == BASE64_INVALID
    assert (record.path, record.sha256, record.size) == (None, None, None)


def test_export_artifact_ok():
    body = b"region,count\nNorth,10\n"
    record, payload = export_artifact(body, 200, RID)
    assert payload == body
    assert record.status is Availability.PRESENT
    assert record.path == "artifacts/export.csv"
    assert record.sha256 == hashlib.sha256(body).hexdigest()
    assert record.size == len(body)
    assert record.http_status == 200
    assert record.source_request_id == RID


def test_export_artifact_empty_body_is_still_evidence():
    record, payload = export_artifact(b"", 200, RID)
    assert payload == b""
    assert record.size == 0
    assert record.sha256 == hashlib.sha256(b"").hexdigest()


def test_export_artifact_failed_fetch():
    record, payload = export_artifact(b'{"error": "expired"}', 404, RID)
    assert payload is None
    assert record.status is Availability.UNAVAILABLE
    assert record.http_status == 404
    assert record.path is None
