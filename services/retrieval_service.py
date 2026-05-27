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

from vector_store import MauiVectorStore
from ai import choose_emb_model


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

    logging.info(
        f"[retrieval] Query started. namespace={namespace}, top_k={top_k}, min_sim={min_sim}"
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

        logging.info(
            f"[retrieval] Query completed: {len(vectors)} matches found in '{namespace}'."
        )
        return vectors

    except Exception as e:
        logging.exception("[retrieval] Error during vector retrieval")
        raise RuntimeError(
            f"Error retrieving vectors from namespace '{namespace}': {e}"
        ) from e
