from abc import ABC, abstractmethod
from typing import List, Any, cast
import base64
import hashlib
import os
import re
import logging
import uuid
from pinecone import Pinecone
from dotenv import load_dotenv

from database_pg import table_exists, pgvector_maui_id_exists
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore

load_dotenv()

PGUSER = os.environ["PGUSER"]
PGPWD = os.environ["PGPWD"]
PGHOST = os.environ["PGHOST"]
PGDB = os.environ["PGDB"]
PGPORT = os.getenv("PG_PORT", "5432")
schema = os.environ.get("MAUI_SCHEMA", "public")


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


def create_pgvector_engine() -> PGEngine:
    connection_string = (
        f"postgresql+psycopg://{PGUSER}:{PGPWD}@{PGHOST}:{PGPORT}/{PGDB}"
    )
    return PGEngine.from_connection_string(connection_string)


def ensure_pgvector_namespace_ready(
    embeddings: Embeddings,
    table_name: str,
) -> None:
    normalized_table_name = normalize_table_name(table_name)

    engine = create_pgvector_engine()
    table_already_exists = table_exists(schema, normalized_table_name)

    if not table_already_exists:
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


class PineconeStore(VectorStore):
    def __init__(self, embeddings: Embeddings, index_name: str, namespace: str):
        super().__init__(embeddings)
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index = pc.Index(index_name)
        self.namespace = namespace

    def find_similar_vectors(
        self, text: str, top_k: int, min_similarity: float
    ) -> List[dict]:
        try:
            vec = self.embeddings.embed_query(text)
            resp = self.index.query(
                vector=vec,
                top_k=top_k,
                min_similarity=min_similarity,
                namespace=self.namespace,
                include_metadata=True,
            )
            matches = getattr(resp, "matches", [])
            if not matches:
                raise RuntimeError(
                    "Pinecone query result does not contain 'matches' attribute."
                )
            logging.info(
                f"Vector Database query completed, found {len(matches)} matches"
            )

            vectors = []
            for vec in matches:
                vectors.append(
                    {
                        "similarity": vec.score,
                        "metadata": vec.metadata,
                    }
                )
            return vectors
        except Exception as e:
            raise RuntimeError(f"Error in find_similar_vectors: {str(e)}")

    def store_paragraphs(self, paragraphs: List[Document]) -> None:
        batch_size = 100
        for start in range(0, len(paragraphs), batch_size):
            end = min(start + batch_size, len(paragraphs))
            batch = paragraphs[start:end]

            ids = [paragraph_id(par, self.namespace) for par in batch]

            # Check if batch is already in index, to avoid recomputing embeddings
            try:
                fetch_response = self.index.fetch(ids=ids, namespace=self.namespace)
                if fetch_response and len(fetch_response.vectors) == len(ids):
                    print("Batch already present")
                    continue
            except Exception:
                pass

            vectors = self.embeddings.embed_documents(
                [par.page_content for par in batch]
            )
            logging.info(f"Successfully created {len(vectors)} embeddings")

            pc_vectors: list[dict] = [
                {
                    "id": ids[i],
                    "values": vectors[i],
                    "metadata": batch[i].metadata | {"text": batch[i].page_content},
                }
                for i in range(len(batch))
            ]
            try:
                upsert_response = self.index.upsert(
                    vectors=cast(Any, pc_vectors), namespace=self.namespace
                )
                logging.info(
                    f"Successfully upserted {upsert_response['upserted_count']} vectors"
                )
            except Exception as e:
                raise RuntimeError(f"Error upserting vectors: {e}")


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
