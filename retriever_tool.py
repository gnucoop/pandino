"""
retriever_tool.py
-----------------
Smolagents tool for retrieving vector context from a PGVector collection.
It is a reusable and Flask-independent wrapper based on retrieval_service.
"""

import logging
from typing import Any, Dict, List
from dotenv import load_dotenv

from smolagents import Tool
from retrieval_service import retrieve_from_collection


load_dotenv()


class DinoRetrieverTool(Tool):
    """
    Smolagents tool that queries the Dino collection via PGVector.
    Provides the agent with the ability to obtain relevant context for grounded responses.
    """

    name = "retriever"
    description = (
        "Fetch relevant context passages from the Dino knowledge base using vector similarity search. "
        "Use this before answering user questions to ensure your response is grounded in real data."
    )
    output_type = "object"

    inputs = {
        "question": {
            "type": "string",
            "description": "User question or query to search for relevant context."
        },
        "namespace": {
            "type": "string",
            "description": "Target namespace (default: Dino).",
            "nullable": True
        },
        "k": {
            "type": "integer",
            "description": "Number of results to retrieve (default: 3).",
            "nullable": True
        },
        "min_similarity": {
            "type": "number",
            "description": "Minimum similarity threshold (default: 0.5).",
            "nullable": True
        }
    }

    def forward(
        self,
        question: str,
        namespace: str | None = None,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> Dict[str, Any]:
        """
        Retrieves the vectors most similar to the provided question by querying a collection (PGVector).

        :param question: Text of the question or query.
        :param namespace: Name of the collection (e.g., 'course:<slug>').
                          If not specified, uses RAG_DEFAULT_NAMESPACE or 'Dino'.
        :param k: Maximum number of vectors to return.
                  Default: RAG_TOP_K or 3.
        :param min_similarity: Minimum similarity threshold [0.0–1.0].
                               Default: RAG_MIN_SIM or 0.5.
        :return: Dictionary containing the retrieval result
        """
        logging.info(
            f"[retriever_tool] Executing retrieval: namespace={namespace or 'Dino'}, "
            f"k={k or 3}, min_sim={min_similarity or 0.5}"
        )

        try:
            vectors: List[Dict[str, Any]] = retrieve_from_collection(
                question=question,
                namespace=namespace,
                k=k,
                min_sim=min_similarity,
            )

            result = {
                "vectors": vectors,
                "used": {
                    "namespace": namespace or "Dino",
                    "k": k or 3,
                    "min_similarity": min_similarity or 0.5,
                },
            }

            logging.info(f"[retriever_tool] Retrieved {len(vectors)} results successfully.")
            return result

        except Exception as e:
            logging.exception("[retriever_tool] Error during retrieval execution")
            return {"vectors": [], "error": str(e)}


# # --- test manuale da CLI ---
# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python retriever_tool.py '<your question>' [namespace]")
#         sys.exit(1)

#     question = sys.argv[1]
#     namespace = sys.argv[2] if len(sys.argv) > 2 else None

#     tool = DinoRetrieverTool()
#     result = tool.forward(question, namespace)

#     print("\nRetrieval result:")
#     print(f"Found {len(result.get('vectors', []))} vectors")
#     if "error" in result:
#         print(f"Error: {result['error']}")
#     else:
#         for i, r in enumerate(result["vectors"], 1):
#             snippet = r["metadata"].get("text", "")[:100].replace("\n", " ")
#             print(f"{i}. sim={r['similarity']:.3f} | {snippet}...")
