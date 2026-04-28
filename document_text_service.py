from werkzeug.datastructures import FileStorage
from typing import TypedDict, Optional


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


def extract_and_normalize_document(input_doc: DocumentInput) -> NormalizedDocument:

    if input_doc["source_type"] == "text":
        content = input_doc.get("content")

        if not content or not content.strip():
            raise ValueError("Text document is empty or missing content")

        return {
            "text": content.strip(),
            "filename": input_doc.get("filename"),
            "role": input_doc.get("role"),
        }

    elif input_doc["source_type"] == "file":

        file = input_doc.get("content")

        if not isinstance(file, FileStorage):

            raise ValueError("Invalid file object")

        filename = input_doc.get("filename") or file.filename

        if not filename or not filename.endswith(".txt"):

            raise NotImplementedError("Only .txt files are supported for now")

        content = file.read().decode("utf-8")

        if not content.strip():

            raise ValueError("File is empty")

        return {
            "text": content.strip(),
            "filename": filename,
            "role": input_doc.get("role"),
        }

    raise NotImplementedError(f"Unsupported source_type: {input_doc['source_type']}")
