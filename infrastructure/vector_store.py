from abc import ABC, abstractmethod
from typing import List, Optional
import base64
import hashlib
import re
import uuid

from config import AppConfig
from infrastructure.database_pg import table_exists, pgvector_maui_id_exists
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from usage.embedding_operation_context import OPERATION_PROBE, embedding_operation

PGUSER: Optional[str] = None
PGPWD: Optional[str] = None
PGHOST: Optional[str] = None
PGDB: Optional[str] = None
PGPORT: Optional[str] = None
schema: Optional[str] = None


class VectorStore(ABC):
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    @abstractmethod
    def find_similar_vectors(
        self, text: str, top_k: int, min_similarity: float
    ) -> List[dict]:
        pass

    @abstractmethod
    def store_paragraphs(self, paragraphs: List[Document]) -> None:
        pass


def init(config: AppConfig) -> None:
    """Initialise module-level globals from AppConfig. Must be called once at startup."""
    global PGUSER, PGPWD, PGHOST, PGDB, PGPORT, schema
    PGUSER = config.database.user
    PGPWD = config.database.password
    PGHOST = config.database.host
    PGDB = config.database.db
    PGPORT = config.database.port
    schema = config.database.schema


def create_pgvector_engine() -> PGEngine:
    if PGHOST is None:
        raise RuntimeError("vector_store.init() must be called before use.")
    connection_string = (
        f"postgresql+psycopg://{PGUSER}:{PGPWD}@{PGHOST}:{PGPORT}/{PGDB}"
    )
    return PGEngine.from_connection_string(connection_string)


def ensure_pgvector_namespace_ready(
    embeddings: Embeddings,
    table_name: str,
) -> None:
    if schema is None:
        raise RuntimeError("vector_store.init() must be called before use.")
    normalized_table_name = normalize_table_name(table_name)

    engine = create_pgvector_engine()
    table_already_exists = table_exists(schema, normalized_table_name)

    if not table_already_exists:
        # The probe scope covers the one provider call that actually happens
        # here. The existence check above and the table creation below are
        # not embedding work, and an already-existing namespace makes no
        # provider call at all -- so no probe scope is entered for it.
        with embedding_operation(OPERATION_PROBE):
            vector_size = len(embeddings.embed_query("test"))

        engine.init_vectorstore_table(
            table_name=normalized_table_name,
            vector_size=vector_size,
            schema_name=schema,
        )


def normalize_table_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


class MauiVectorStore(VectorStore):
    def __init__(
        self,
        embeddings: Embeddings,
        table_name: str,
    ):
        super().__init__(embeddings)

        normalized_table_name = normalize_table_name(table_name)

        self.engine = create_pgvector_engine()

        self.table_name = normalized_table_name

        self.store = PGVectorStore.create_sync(  # type: ignore
            engine=self.engine,
            table_name=normalized_table_name,
            embedding_service=embeddings,
        )

    def find_similar_vectors(
        self, text: str, top_k: int, min_similarity: float
    ) -> List[dict]:
        try:
            results = self.store.similarity_search_with_score(
                text,
                k=top_k,
            )

            vectors = []
            for doc, score in results:
                similarity = 1 - score

                if similarity < min_similarity:
                    continue

                vectors.append(
                    {
                        "similarity": similarity,
                        "metadata": doc.metadata,
                    }
                )

            return vectors

        except Exception as e:
            raise RuntimeError(f"Error in find_similar_vectors (V2): {str(e)}")

    def store_paragraphs(self, paragraphs: List[Document]) -> None:
        try:
            new_docs = []
            new_ids = []

            for par in paragraphs:
                maui_id = paragraph_id(par, self.table_name)

                if pgvector_maui_id_exists(self.table_name, maui_id):
                    continue

                metadata = par.metadata | {
                    "text": par.page_content,
                    "maui_id": maui_id,
                }

                new_doc = Document(
                    page_content=par.page_content,
                    metadata=metadata,
                )

                new_docs.append(new_doc)
                new_ids.append(str(uuid.uuid4()))

            if not new_docs:
                return

            self.store.add_documents(new_docs, ids=new_ids)

        except Exception as e:
            raise RuntimeError(f"Error in store_paragraphs (V2): {str(e)}")



def split_text(document: str, paragraph_len: int = 900) -> list[str]:
    if len(document) <= paragraph_len * 2:
        return [document]
    sentences = re.split(r"\.\s+|\n\s*", document)
    paragraphs = []
    par = ""
    for i, s in enumerate(sentences):
        if par == "":
            par = s
        else:
            par += ". " + s
        if len(par) >= paragraph_len or i == len(sentences) - 1:
            paragraphs.append(par)
            if len(s) <= paragraph_len // 3:
                # Next paragraph starts with s, overlapping
                par = s
            else:
                par = ""
    return paragraphs


def merge_segments(
    segments: List[Document], paragraph_len: int = 900
) -> list[Document]:
    segments = segments.copy()
    if len(segments) <= 1:
        return segments
    segments.reverse()
    docs: List[Document] = []
    while len(segments) > 0:
        doc = segments.pop()
        while len(doc.page_content) < paragraph_len and len(segments) > 0:
            doc.page_content += segments.pop().page_content
        docs.append(doc)
    return docs


# Hash a string using SHA256 and encode it in base64
def hash_text(t: str) -> str:
    hash_bytes = hashlib.sha256(t.encode("utf-8")).digest()
    return base64.b64encode(hash_bytes).decode("utf-8")


def paragraph_id(paragraph: Document, namespace: str) -> str:
    return f"{namespace}:{hash_text(paragraph.page_content)}"


def file_id_from_text(text: str, namespace: str) -> str:
    normalized_namespace = normalize_table_name(namespace)
    return f"{normalized_namespace}:{hash_text(text)}"
