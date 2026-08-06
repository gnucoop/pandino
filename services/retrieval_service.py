"""
retrieval_service.py
--------------------
Centralized service for retrieving vector context from a PGVector collection.

This module defines the *only* component responsible for deciding:
- which embedding model to use,
- how many vectors to retrieve (top_k),
- the minimum similarity threshold,
- and how to query the underlying vector store.

All configuration parameters must be passed explicitly by the caller.
"""

import logging
from typing import List, Dict, Any

from infrastructure.vector_store import MauiVectorStore
from infrastructure.ai import choose_emb_model

logger = logging.getLogger(__name__)


def retrieve_from_collection(
    question: str,
    namespace: str,
    embedding_provider: str,
    embedding_model: str,
    top_k: int,
    min_sim: float,
) -> List[Dict[str, Any]]:
    """
    Retrieve vector context for a given question within a specific namespace.

    Parameters
    ----------
    question : str
        The textual query used to compute vector similarity.
    namespace : str
        The PGVector collection name (e.g. "Dino", "Farm", or course-specific namespaces).
        This must be explicitly provided by the calling component (typically a RetrieverTool).
    embedding_provider : str
        The provider to use for the embedding model.
    embedding_model : str
        The embedding model identifier.
    top_k : int
        Number of top similar vectors to retrieve.
    min_sim : float
        Minimum similarity threshold for returned vectors.

    Returns
    -------
    List[Dict[str, Any]]
        A list of objects, each containing:
        - "similarity": float,
        - "metadata": dict (original stored metadata).
    """

    logger.info(
        "event=retrieval_query_started namespace=%s top_k=%s min_sim=%s",
        namespace,
        top_k,
        min_sim,
    )

    try:
        # === Load embedding model once per call ===
        emb = choose_emb_model(embedding_provider, embedding_model)

        # === Initialize vector store ===
        store = MauiVectorStore(embeddings=emb, table_name=namespace)

        # === Perform similarity search ===
        vectors = store.find_similar_vectors(
            text=question, top_k=top_k, min_similarity=min_sim
        )

        logger.info(
            "event=retrieval_query_completed count=%s namespace=%s",
            len(vectors),
            namespace,
        )
        return vectors

    except Exception as e:
        logger.exception("event=retrieval_query_failed")
        raise RuntimeError(
            f"Error retrieving vectors from namespace '{namespace}': {e}"
        ) from e
