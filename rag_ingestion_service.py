import logging
import tempfile
from typing import List, Optional
from dataclasses import dataclass

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ai import choose_emb_model, describe_image, whisper_response
from database_pg import insert_rag_file
from vector_store import (
    MauiVectorStore,
    ensure_pgvector_namespace_ready,
    file_id_from_text,
    merge_segments,
    normalize_table_name,
)


@dataclass(frozen=True)
class RagIngestionResult:
    file_id: str
    file_name: str
    namespace: str
    chunk_count: int
    language: Optional[str]
    tracking_saved: bool


def process_rag_file(
    file,
    url: str,
    namespace: str,
    language: str | None,
    *,
    whisper_model: str | None,
    deepinfra_api_key: str | None,
    vision_provider: str | None,
    vision_model: str | None,
    embedding_provider: str | None,
    embedding_model: str | None,
) -> RagIngestionResult:
    """Process a file for RAG: extract text, split it into chunks, store embeddings, and track the file.

    :param file: Uploaded file object (from request.files).
    :param url: Source identifier to store in chunk metadata.
    :param namespace: Vector store namespace/table name.
    :param language: Optional language code for metadata.
    :param whisper_model: Whisper model name for audio transcription.
    :param deepinfra_api_key: API key for DeepInfra Whisper requests.
    :param vision_provider: Vision provider name for image description.
    :param vision_model: Vision model name for image description.
    :param embedding_provider: Embedding provider name.
    :param embedding_model: Embedding model name.
    :return: Structured ingestion result.
    :raises ValueError: For unsupported file types, missing configuration, or whisper errors.
    """
    chunk_size = 900
    chunk_overlap = 100
    tx_split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    md_split = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    metadata = {"url": url, "mimetype": file.mimetype, "source": file.filename}
    text = ""
    paragraphs: List[Document] = []

    if file.mimetype == "text/plain":
        text = file.stream.read().decode()
        paragraphs = tx_split.split_documents(
            [Document(page_content=text, metadata=metadata)]
        )

    elif file.mimetype == "text/markdown":
        text = file.stream.read().decode()
        paragraphs = md_split.split_documents(
            [Document(page_content=text, metadata=metadata)]
        )

    elif file.mimetype == "application/pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
            file.save(temp.name)
            pages: List[dict] = pymupdf4llm.to_markdown(temp.name, page_chunks=True)  # type: ignore
            page_texts = [p["text"] for p in pages]
            text = "".join(page_texts)
            page_docs = [
                Document(
                    page_content=p["text"],
                    metadata=metadata | {"page": p["metadata"]["page"]},
                )
                for p in pages
            ]
            paragraphs = md_split.split_documents(page_docs)

    elif file.mimetype.startswith("audio"):
        if not whisper_model or not deepinfra_api_key:
            raise ValueError("Missing Whisper configuration")

        resp = whisper_response(file, whisper_model, deepinfra_api_key)
        if resp.status_code != 200:
            raise ValueError("Error whispering audio")

        json = resp.json()
        text = json["text"]
        segments = [
            Document(
                page_content=s["text"],
                metadata=metadata | {"start_time": s["start"]},
            )
            for s in json["segments"]
        ]
        paragraphs = merge_segments(segments, chunk_size)

    elif file.mimetype.startswith("image"):
        text = describe_image(url, vision_provider or "", vision_model or "")
        paragraphs = [
            Document(
                page_content=text,
                metadata=metadata,
            )
        ]

    else:
        raise ValueError(f"Unsupported file type: {file.mimetype}")

    normalized_namespace = normalize_table_name(namespace)
    file_name = file.filename or ""

    if text == "":
        return RagIngestionResult(
            file_id="",
            file_name=file_name,
            namespace=normalized_namespace,
            chunk_count=0,
            language=language,
            tracking_saved=False,
        )

    file_id = file_id_from_text(text, namespace)

    paragraphs = [
        Document(
            page_content=par.page_content,
            metadata=par.metadata
            | {"file_id": file_id}
            | ({"language": language} if language else {}),
        )
        for par in paragraphs
    ]

    embeddings = choose_emb_model(
        embedding_provider or "",
        embedding_model or "",
    )

    ensure_pgvector_namespace_ready(
        embeddings=embeddings,
        table_name=namespace,
    )

    store = MauiVectorStore(embeddings, namespace)
    store.store_paragraphs(paragraphs)

    chunk_count = len(paragraphs)

    tracking_ok = insert_rag_file(
        file_id=file_id,
        file_name=file_name,
        namespace=normalized_namespace,
        chunk_count=chunk_count,
        language=language,
    )

    if not tracking_ok:
        logging.warning(
            "RAG file tracking failed for file_id=%s, namespace=%s",
            file_id,
            normalized_namespace,
        )

    return RagIngestionResult(
        file_id=file_id,
        file_name=file_name,
        namespace=normalized_namespace,
        chunk_count=chunk_count,
        language=language,
        tracking_saved=tracking_ok,
    )
