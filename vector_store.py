from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from langchain_pinecone import PineconeVectorStore
from langchain_postgres.vectorstores import PGVector
from pinecone import Pinecone
import os
import logging
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
class VectorStore(ABC):
    @abstractmethod
    def similarity_search(self, query: str, top_k: int, min_similarity: float, namespace: str) -> Tuple[List[str], List[float], List[str], List[str], List[str], List[str]]:
        pass

class PineconeStore(VectorStore):
    def __init__(self, index_name: str, embeddings):
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        #self.index = pc.Index(index_name)
        self.index = pc.Index("langchain-test-index")
        self.vector_store = PineconeVectorStore(index=self.index, embedding=embeddings)

    def similarity_search(self, query: str, top_k: int, min_similarity: float, namespace: str) -> Tuple[List[str], List[float], List[str], List[str], List[str], List[str]]:
        # Use the embeddings from initialization
        query_embedding = self.vector_store.embeddings.embed_query(query)
        
        resp = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            min_score=min_similarity
        )

        paragraphs, similarities, pages, sources, urls, mimetypes = [], [], [], [], [], []
        
        if hasattr(resp, 'matches'):
            for vec in resp.matches:
                if vec.score >= min_similarity:
                    paragraphs.append(vec.metadata["text"])
                    similarities.append(vec.score)
                    pages.append(vec.metadata["page"])
                    sources.append(vec.metadata["source"])
                    urls.append(vec.metadata["url"])
                    mimetypes.append(vec.metadata["mimetype"])

        return paragraphs, similarities, sources, pages, urls, mimetypes

class PGVectorStore(VectorStore):
    def __init__(self, collection_name: str, embeddings):
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        self.vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True
        )

    def similarity_search(self, query: str, top_k: int, min_similarity: float, namespace: str, embeddings=None) -> Tuple[List[str], List[float], List[str], List[str], List[str], List[str]]:
        # embeddings parameter is ignored since PGVector handles embedding internally
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, 
            k=top_k,
            score_threshold=min_similarity
        )
        
        paragraphs, similarities, pages, sources, urls, mimetypes = [], [], [], [], [], []
        
        for doc, score in results:
            metadata = doc.metadata
            paragraphs.append(doc.page_content)
            similarities.append(score)
            pages.append(metadata.get("page", ""))
            sources.append(metadata.get("source", ""))
            urls.append(metadata.get("url", ""))
            mimetypes.append(metadata.get("mimetype", ""))

        return paragraphs, similarities, sources, pages, urls, mimetypes

def create_vector_store(db_type: str, store_name: str, embeddings) -> VectorStore:
    if db_type == 'pinecone':
        return PineconeStore(store_name, embeddings)
    elif db_type == 'pgvector':
        return PGVectorStore(store_name, embeddings)
    else:
        raise ValueError(f"Unsupported vector store type: {db_type}")
