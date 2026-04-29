from typing import TypedDict

from document_text_service import NormalizedDocument


class ComparisonResult(TypedDict):
    """
    Represents the validated output of a document comparison.

    This is the internal contract returned by the comparison service
    after the LLM response has been parsed and validated.
    """

    score: int
    summary: str
    reasoning: str


def _format_documents_for_prompt(documents: list[NormalizedDocument]) -> str:
    formatted_documents: list[str] = []

    for index, document in enumerate(documents, start=1):
        role = document.get("role") or "unspecified"
        filename = document.get("filename") or "unknown"

        formatted_documents.append(
            f"DOCUMENT {index}\n"
            f"Role: {role}\n"
            f"Filename: {filename}\n"
            f"Content:\n{document['text']}"
        )

    return "\n\n---\n\n".join(formatted_documents)


def compare_documents(
    documents: list[NormalizedDocument],
    prompt: str,
    additional_context: str | None = None,
    language: str | None = None,
) -> ComparisonResult:
    if len(documents) < 2:
        raise ValueError("At least two documents are required")

    if not prompt or not prompt.strip():
        raise ValueError("Comparison prompt is required")

    raise NotImplementedError("LLM comparison not implemented yet")
