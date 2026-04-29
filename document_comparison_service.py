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
