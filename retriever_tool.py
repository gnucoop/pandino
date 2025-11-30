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
from dotenv import load_dotenv

from smolagents import Tool
from retrieval_service import retrieve_from_collection


load_dotenv()


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

    def __init__(self, namespace: str) -> None:
        super().__init__()
        self.default_namespace = namespace  # pre-bound namespace

    def forward(self, question: str) -> Dict[str, Any]:
        """
        The tool retrieves relevant vectors using the fixed namespace.
        Parameters k and min_similarity are no longer accepted here,
        because they are centrally managed by retrieval_service.py.
        """

        # Namespace cannot be overridden by the LLM
        effective_namespace = self.default_namespace

        logging.info(
            f"[retriever_tool] Executing retrieval: namespace={effective_namespace or '(default)'}"
        )

        try:
            vectors = retrieve_from_collection(
                question=question,
                namespace=effective_namespace,  # always enforced
            )

            result = {
                "vectors": vectors,
                "used": {
                    "namespace": effective_namespace or "(default)"
                },
            }

            logging.info(
                f"[retriever_tool] Retrieved {len(vectors)} results successfully."
            )
            return result

        except Exception as e:
            logging.exception("[retriever_tool] Error during retrieval execution")
            return {"vectors": [], "error": str(e)}
