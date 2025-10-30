"""
retrieval_service.py
--------------------
Reusable module for vector context retrieval from a PGVector collection.
This service is agnostic to Flask or Smolagents: it can be imported anywhere.
"""

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from vector_store import PGVectorStore
from ai import choose_emb_model


load_dotenv()


def retrieve_from_collection(
    question: str,
    namespace: str | None = None,
    k: int | None = None,
    min_sim: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves the vectors most similar to the provided question by querying the specified collection (PGVector).

    :param question: Text of the question or query.
    :param namespace: Name of the collection (e.g., 'course:<slug>').
                      If not specified, uses RAG_DEFAULT_NAMESPACE or 'Dino'.
    :param k: Maximum number of vectors to return.
              Default: RAG_TOP_K or 3.
    :param min_sim: Minimum similarity threshold [0.0–1.0].
                    Default: RAG_MIN_SIM or 0.5.
    :return: List of dictionaries {"similarity": float, "metadata": dict}.
    """

    # Setting fallback values
    namespace = namespace or os.getenv("RAG_DEFAULT_NAMESPACE", "Dino")
    k = k or int(os.getenv("RAG_TOP_K", "3"))
    min_sim = min_sim or float(os.getenv("RAG_MIN_SIM", "0.5"))

    logging.info(f"[retrieval] Starting retrieval for namespace={namespace}, k={k}, min_sim={min_sim}")

    try:
        # Instantiate the embedding model
        emb = choose_emb_model(
            os.getenv("COMPLETION_EMBEDDING_MODEL_PROVIDER", "Deepinfra"),
            os.getenv("COMPLETION_EMBEDDING_MODEL", "BAAI/bge-m3")
        )

        # Create a vector store for the namespace
        store = PGVectorStore(embeddings=emb, collection_name=namespace)

        # Retrieve the vectors
        vectors = store.find_similar_vectors(
            text=question,
            top_k=k,
            min_similarity=min_sim
        )

        logging.info(f"[retrieval] Query completed: {len(vectors)} matches found.")
        return vectors

    except Exception as e:
        logging.exception("[retrieval] Error during vector retrieval")
        raise RuntimeError(f"Error retrieving vectors from Dino: {str(e)}") from e


# # Test rapido da linea di comando
# if __name__ == "__main__":
#     import sys

#     # Controllo degli argomenti passati da riga di comando:
#     # il primo argomento (obbligatorio) è la domanda da cercare,
#     # il secondo (opzionale) è il namespace/collezione.
#     if len(sys.argv) < 2:
#         # Se manca la domanda, mostro l'uso corretto e termino con codice di errore
#         print("Usage: python retrieval_service.py '<your question>' [namespace]")
#         sys.exit(1)

#     # Legge la domanda dal primo argomento
#     question = sys.argv[1]
#     # Se è stato passato un namespace lo usa, altrimenti legge la variabile d'ambiente RAG_DEFAULT_NAMESPACE
#     namespace = sys.argv[2] if len(sys.argv) > 2 else os.getenv("RAG_DEFAULT_NAMESPACE", "Dino")

#     # Stampa informativa sul namespace che verrà interrogato
#     print(f"\n Querying namespace: {namespace}")
#     # Esegue la funzione di retrieval definita sopra passando domanda e namespace
#     results = retrieve_from_collection(question, namespace)
#     # Stampa quanti vettori sono stati trovati
#     print(f"\n Found {len(results)} vectors\n")

#     # Itera sui risultati e mostra una breve anteprima del testo e la similarità
#     for i, r in enumerate(results, 1):
#         # Prende dal metadata il campo 'text' (se presente), tronca a 120 caratteri e rimuove newline
#         snippet = r['metadata'].get('text', '')[:120].replace('\n', ' ')
#         # Stampa indice, similarità (formattata a 3 decimali) e l'anteprima del contenuto
#         print(f"{i}. sim={r['similarity']:.3f} | {snippet}...")
