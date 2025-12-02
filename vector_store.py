from abc import ABC, abstractmethod
from typing import List, Any, cast
import base64
import hashlib
import os
import re
import logging
from pinecone import Pinecone
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

load_dotenv()

PGUSER = os.environ["PGUSER"]
PGPWD = os.environ["PGPWD"]
PGHOST = os.environ["PGHOST"]
PGDB = os.environ["PGDB"]
PGPORT = os.environ["PG_PORT"]
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


class PGVectorStore(VectorStore):
    def __init__(self, embeddings: Embeddings, collection_name: str):

        connection = (
            f"postgresql+psycopg://{PGUSER}:{PGPWD}@{PGHOST}:{PGPORT}/{PGDB}"
            f"?options=-csearch_path%3D{schema}"
        ) 
        self.collection = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

    def find_similar_vectors(
        self, text: str, top_k: int, min_similarity: float
    ) -> List[dict]:
        try:
            results = self.collection.similarity_search_with_score(query=text, k=top_k)
            logging.info(f"PGVector query completed, found {len(results)} matches")

            vectors = []
            for doc, score in results:
                if (1 - score) < min_similarity:
                    continue
                vectors.append(
                    {
                        "similarity": 1 - score,
                        "metadata": doc.metadata,
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
            ids = [paragraph_id(par, "") for par in batch]

            try:
                # Get existing documents
                existing_docs = self.collection.get_by_ids(ids)

                # Extract existing IDs - the exact method depends on your PGVector implementation
                # Option 1: If the returned docs have id in metadata
                existing_ids = {
                    doc.metadata.get("id")
                    for doc in existing_docs
                    if doc.metadata.get("id")
                }

                # Option 2: If get_by_ids returns a dict mapping ids to docs
                # existing_ids = set(existing_docs.keys()) if isinstance(existing_docs, dict) else set()

                # Option 3: If you need to compare with the original ids list
                # existing_ids = set(ids[:len(existing_docs)]) if existing_docs else set()

                new_docs = []
                for i, par in enumerate(batch):
                    doc_id = ids[i]
                    if doc_id in existing_ids:
                        logging.info(f"Skipping existing document with ID: {doc_id}")
                        continue

                    # Create Document objects instead of dictionaries
                    new_doc = Document(
                        page_content=par.page_content,
                        metadata=par.metadata
                        | {"text": par.page_content, "id": doc_id},
                    )
                    new_docs.append(new_doc)

                if not new_docs:
                    logging.info("Batch already present")
                    continue

                # Add documents with their IDs
                self.collection.add_documents(new_docs, ids=ids[-len(new_docs) :])
                logging.info(
                    f"Successfully added {len(new_docs)} new documents to PGVector"
                )

            except Exception as e:
                logging.error(f"Error in batch starting at {start}: {str(e)}")
                raise RuntimeError(f"Error storing paragraphs to PGVector: {e}")


#    def store_paragraphs(self, paragraphs: List[Document]) -> None:
#        batch_size = 100
#        for start in range(0, len(paragraphs), batch_size):
#            end = min(start + batch_size, len(paragraphs))#
#
#            batch = paragraphs[start:end]
#            ids = [paragraph_id(par, "") for par in batch]
#            try:
#                existing_docs = self.collection.get_by_ids(ids)
#                existing_ids = {doc.id for doc in existing_docs}#
#
#                new_docs = []
#                for i, par in enumerate(batch):
#                    id = ids[i]
#                    if id in existing_ids:
#                        # Skipping already existing paragraph
#                        continue
#                    new_docs.append({
#                        "id": id,
#                        "page_content": par.page_content,
#                        "metadata": par.metadata | {"text": par.page_content},
#                    })
#                if not new_docs:
#                    logging.info("Batch already present")
#                    continue#
#
#                self.collection.add_documents(new_docs)
#                logging.info(f"Successfully added {len(new_docs)} new documents to PGVector")
#            except Exception as e:
#                raise RuntimeError(f"Error storing paragraphs to PGVector: {e}")


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
