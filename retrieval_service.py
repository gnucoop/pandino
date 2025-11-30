"""
retrieval_service.py
--------------------
Centralized service for retrieving vector context from a PGVector collection.

This module defines the *only* component responsible for deciding:
- which embedding model to use,
- how many vectors to retrieve (top_k),
- the minimum similarity threshold,
- and how to query the underlying vector store.

All RAG configuration is fully centralized here and managed via environment
variables. No external component (endpoint, agent, tool, or LLM) can override
these parameters.
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from vector_store import PGVectorStore
from ai import choose_emb_model

load_dotenv()


def retrieve_from_collection(question: str, namespace: str) -> List[Dict[str, Any]]:
    """
    Retrieve vector context for a given question within a specific namespace.

    Parameters
    ----------
    question : str
        The textual query used to compute vector similarity.
    namespace : str
        The PGVector collection name (e.g. "Dino", "Farm", or course-specific namespaces).
        This must be explicitly provided by the calling component (typically a RetrieverTool).

    Returns
    -------
    List[Dict[str, Any]]
        A list of objects, each containing:
        - "similarity": float,
        - "metadata": dict (original stored metadata).
    """

    # === Centralized retrieval configuration ===
    top_k = int(os.getenv("RAG_TOP_K", "3"))
    min_sim = float(os.getenv("RAG_MIN_SIM", "0.5"))

    logging.info(
        f"[retrieval] Query started. namespace={namespace}, top_k={top_k}, min_sim={min_sim}"
    )

    try:
        # === Load embedding model once per call ===
        emb = choose_emb_model(
            os.getenv("COMPLETION_EMBEDDING_MODEL_PROVIDER", "Deepinfra"),
            os.getenv("COMPLETION_EMBEDDING_MODEL", "BAAI/bge-m3")
        )

        # === Initialize vector store ===
        store = PGVectorStore(
            embeddings=emb,
            collection_name=namespace
        )

        # === Perform similarity search ===
        vectors = store.find_similar_vectors(
            text=question,
            top_k=top_k,
            min_similarity=min_sim
        )

        logging.info(
            f"[retrieval] Query completed: {len(vectors)} matches found in '{namespace}'."
        )
        return vectors

    except Exception as e:
        logging.exception("[retrieval] Error during vector retrieval")
        raise RuntimeError(
            f"Error retrieving vectors from namespace '{namespace}': {e}"
        ) from e

