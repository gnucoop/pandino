"""Plot image and CSV export artifact evidence.

CURRENT ``/datachat`` returns a plot as ``response.value == "b'<base64 PNG>'"``
(``str()`` of a bytes literal). ``decode_image_value`` strips exactly that
wrapper, base64-decodes strictly and checks the PNG signature plus a valid
IHDR chunk, using the standard library only.

A decode failure is DataChat/backend-contract evidence for grading, not an
infrastructure failure: the artifact record is still ``PRESENT`` (the
outcome was captured) with ``decode_status`` saying what went wrong and no
image path. The raw HTTP response remains the authoritative evidence.
"""

import base64
import binascii
import struct
import zlib
from dataclasses import dataclass

from evaluation.datachat.evidence import sha256_bytes
from evaluation.datachat.model import ArtifactRecord, Availability

IMAGE_PATH = "artifacts/image.png"
EXPORT_PATH = "artifacts/export.csv"

DECODE_OK = "OK"
WRAPPER_MISSING = "WRAPPER_MISSING"
BASE64_INVALID = "BASE64_INVALID"
NOT_PNG = "NOT_PNG"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SNIFF = (
    (PNG_SIGNATURE, "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)


@dataclass(frozen=True)
class ImageDecode:
    status: str
    wrapper_stripped: bool
    png_signature_ok: bool
    detected_format: str | None
    data: bytes | None  # decoded PNG, only when status == OK


def _detect_format(data: bytes) -> str | None:
    for prefix, name in _SNIFF:
        if data.startswith(prefix):
            return name
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _valid_ihdr(data: bytes) -> bool:
    """First chunk is a 13-byte IHDR with a correct CRC and non-zero dimensions."""
    header = data[8:16]
    if len(header) < 8:
        return False
    length, chunk_type = struct.unpack(">I4s", header)
    if chunk_type != b"IHDR" or length != 13:
        return False
    body = data[16:29]
    crc = data[29:33]
    if len(body) != 13 or len(crc) != 4:
        return False
    if struct.unpack(">I", crc)[0] != zlib.crc32(chunk_type + body):
        return False
    width, height = struct.unpack(">II", body[:8])
    return width > 0 and height > 0


def decode_image_value(value: str) -> ImageDecode:
    if not isinstance(value, str):
        raise TypeError(f"image value must be str, got {type(value).__name__}")
    if not (len(value) >= 3 and value.startswith("b'") and value.endswith("'")):
        return ImageDecode(WRAPPER_MISSING, False, False, None, None)
    encoded = value[2:-1]
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return ImageDecode(BASE64_INVALID, True, False, None, None)
    signature_ok = data.startswith(PNG_SIGNATURE)
    detected = _detect_format(data)
    if not (signature_ok and _valid_ihdr(data)):
        return ImageDecode(NOT_PNG, True, signature_ok, detected, None)
    return ImageDecode(DECODE_OK, True, True, "png", data)


def image_artifact(source_request_id: str | None, decoded: ImageDecode) -> ArtifactRecord:
    """Metadata for ``artifacts/image.json``; path/hash/size only when decoding succeeded."""
    ok = decoded.data is not None
    return ArtifactRecord(
        kind="image",
        source_request_id=source_request_id,
        status=Availability.PRESENT,
        path=IMAGE_PATH if ok else None,
        sha256=sha256_bytes(decoded.data) if ok else None,
        size=len(decoded.data) if ok else None,
        wrapper_stripped=decoded.wrapper_stripped,
        decode_status=decoded.status,
        png_signature_ok=decoded.png_signature_ok,
        detected_format=decoded.detected_format,
    )


def export_artifact(
    body: bytes, http_status: int, source_request_id: str | None
) -> tuple[ArtifactRecord, bytes | None]:
    """Export evidence from an already-fetched ``download_url`` response.

    HTTP 200 → ``PRESENT`` with the body (possibly empty) to persist as
    ``artifacts/export.csv``; any other status → ``UNAVAILABLE`` and no payload.
    """
    if not isinstance(body, (bytes, bytearray)):
        raise TypeError(f"export body must be bytes, got {type(body).__name__}")
    body = bytes(body)
    if http_status != 200:
        return (
            ArtifactRecord(
                kind="export",
                source_request_id=source_request_id,
                status=Availability.UNAVAILABLE,
                http_status=http_status,
            ),
            None,
        )
    return (
        ArtifactRecord(
            kind="export",
            source_request_id=source_request_id,
            status=Availability.PRESENT,
            path=EXPORT_PATH,
            sha256=sha256_bytes(body),
            size=len(body),
            http_status=http_status,
        ),
        body,
    )
