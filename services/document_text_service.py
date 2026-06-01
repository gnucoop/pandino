from werkzeug.datastructures import FileStorage
from typing import TypedDict, Optional
import os
import io
import pymupdf
import pymupdf4llm
from docx import Document as DocxDocument
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.table import Table as DocxTable
from striprtf.striprtf import rtf_to_text


class DocumentInput(TypedDict):
    """
    Represents a document as received from the external request layer.

    It may originate from a file upload or pre-existing text and contains
    heterogeneous, unprocessed data.

    This structure reflects raw, untrusted input and does not guarantee
    consistency, format, or safety of the content.
    """

    content: Optional[str | FileStorage]
    filename: Optional[str]
    source_type: str  # "file" | "text"
    role: Optional[str]


class NormalizedDocument(TypedDict):
    """
    Represents a document after extraction and normalization.

    At this stage, the content is always available as clean text,
    ready to be used by downstream components (e.g. prompt building, LLM).

    This is the standard internal representation used across the
    document comparison pipeline.
    """

    text: str
    filename: Optional[str]
    role: Optional[str]


def _normalize_text(text: str) -> str:
    normalized = text.strip()

    if not normalized:
        raise ValueError("Document text is empty")

    return normalized


def _extract_docx_text(file: FileStorage) -> str:
    """
    Extract plain text from a .docx file.

    Iterates paragraphs and tables in document order using iter_inner_content(),
    which preserves the original reading sequence.

    Known limitation: text inside text boxes and floating shapes is not accessible
    via the python-docx API and will be silently omitted. This is a documented
    upstream limitation with no official workaround.
    """
    document = DocxDocument(io.BytesIO(file.read()))
    parts = []

    for block in document.iter_inner_content():
        if isinstance(block, DocxParagraph):
            text = block.text.strip()
            if text:
                parts.append(text)
        elif isinstance(block, DocxTable):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

    return _normalize_text("\n\n".join(parts))


def _extract_rtf_text(file: FileStorage) -> str:
    """
    Extract plain text from an .rtf file.

    Reads raw bytes and decodes with latin-1 before passing to rtf_to_text().
    latin-1 is used for the initial decode because it is lossless for any byte
    value, preserving the RTF markup intact.

    striprtf handles codepage detection internally via the \ansicpg directive
    and per-font \fcharset entries, covering cp1252 (Windows default), cp1251
    (Cyrillic), cp1250 (Central European) and others.

    errors="ignore" is used to silently drop characters that cannot be mapped
    in the target codepage, which is preferable to raising an exception on
    real-world RTF files produced by Windows applications.
    """
    raw = file.read()
    text = rtf_to_text(raw.decode("latin-1"), errors="ignore")
    return _normalize_text(text)


def _extract_text_from_file(file: FileStorage, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".txt":
        return file.read().decode("utf-8")

    if ext == ".pdf":
        pdf_bytes = file.read()
        pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        return pymupdf4llm.to_markdown(pdf_document)

    if ext == ".docx":
        return _extract_docx_text(file)

    if ext == ".rtf":
        return _extract_rtf_text(file)

    raise NotImplementedError(f"Unsupported file format: {filename}")


def extract_and_normalize_document(input_doc: DocumentInput) -> NormalizedDocument:

    if input_doc["source_type"] == "text":
        content = input_doc.get("content")

        if not isinstance(content, str):

            raise ValueError("Text document is missing content")

        return {
            "text": _normalize_text(content),
            "filename": input_doc.get("filename"),
            "role": input_doc.get("role"),
        }

    elif input_doc["source_type"] == "file":

        file = input_doc.get("content")

        if not isinstance(file, FileStorage):

            raise ValueError("Invalid file object")

        filename = input_doc.get("filename") or file.filename

        if not filename:
            raise ValueError("Filename is missing")

        content = _extract_text_from_file(file, filename)

        if not content.strip():

            raise ValueError("File is empty")

        return {
            "text": _normalize_text(content),
            "filename": filename,
            "role": input_doc.get("role"),
        }

    raise NotImplementedError(f"Unsupported source_type: {input_doc['source_type']}")
