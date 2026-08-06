"""
retriever_tool.py
-----------------
Smolagents tool for retrieving vector context from a PGVector collection.
It is a reusable and Flask-independent wrapper based on retrieval_service.
it exposes *only* the question and delegates all retrieval configuration 
(k, min_similarity, model setup) to retrieval_service.py.
"""

import logging
from typing import Any, Dict

from smolagents import Tool
from services.retrieval_service import retrieve_from_collection

logger = logging.getLogger(__name__)


class RetrieverTool(Tool):
    """
    Smolagents tool that queries a PGVector-backed collection.
    The namespace is injected at instantiation time and cannot
    be changed by the LLM.
    """

    name = "retriever"
    description = (
        "Retrieve the most relevant text passages from the knowledge base using "
        "vector similarity search. Use this tool whenever the user asks a question "
        "that requires contextual information from the stored documents. "
        "The only required input is the user question. The retrieval namespace is "
        "preconfigured and cannot be changed by the LLM. The tool returns an object "
        "containing the retrieved vectors and their metadata."
    )
    output_type = "object"

    
    inputs = {
        "question": {
            "type": "string",
            "description": "User question to search for relevant context."
        }
    }

    def __init__(
        self,
        namespace: str,
        embedding_provider: str,
        embedding_model: str,
        top_k: int,
        min_sim: float,
    ) -> None:
        super().__init__()
        self.default_namespace = namespace  # pre-bound namespace
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.min_sim = min_sim

    def forward(self, question: str) -> Dict[str, Any]:
        """
        The tool retrieves relevant vectors using the fixed namespace.
        Parameters k and min_similarity are no longer accepted here,
        because they are centrally managed by retrieval_service.py.
        """

        # Namespace cannot be overridden by the LLM
        effective_namespace = self.default_namespace

        logger.info(
            f"Executing retrieval: namespace={effective_namespace or '(default)'}"
        )

        try:
            vectors = retrieve_from_collection(
                question=question,
                namespace=effective_namespace,  # always enforced
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                top_k=self.top_k,
                min_sim=self.min_sim,
            )

            result = {
                "vectors": vectors,
                "used": {
                    "namespace": effective_namespace or "(default)"
                },
            }

            logger.info(
                f"Retrieved {len(vectors)} results successfully."
            )
            return result

        except Exception as e:
            logger.exception("Error during retrieval execution")
            return {"vectors": [], "error": str(e)}
