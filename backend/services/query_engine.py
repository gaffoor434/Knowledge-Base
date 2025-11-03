import logging
from typing import List, Dict, Any, Optional, Tuple

from backend.services.embedding_service import generate_query_embedding
from backend.services.qdrant_service import search_similar_chunks
from backend.services.llm_service import answer_from_chunks
from backend.services.hybrid_search import hybrid_search
from backend.services.bm25_service import BM25Service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def query_knowledge_base(
    query: str,
    bm25_service: BM25Service
) -> Tuple[str, List[str]]:
    """
    Queries the knowledge base using a hybrid search approach.
    - Retrieves relevant chunks using BM25 and vector search.
    - Merges and re-ranks the results.
    - Applies a confidence threshold to prevent low-quality answers.
    - Calls the LLM service to generate a grounded answer.
    """
    if not query.strip():
        return "Please provide a valid question.", []

    # 1. Vector Search
    query_embedding = generate_query_embedding(query)
    if query_embedding is None:
        return "Could not generate an embedding for the query.", []
    vector_results = search_similar_chunks(query_embedding, top_k=10)

    # 2. BM25 Search
    bm25_results = bm25_service.query(query, top_k=10)

    # 3. Hybrid Search & Re-ranking
    top_chunks = hybrid_search(
        query_text=query,
        bm25_results=bm25_results,
        vector_results=vector_results,
        top_k=5
    )

    # 4. Confidence Check
    if not top_chunks or (max(r.get("combined", 0.0) for r in top_chunks) < 0.12):
        logger.info("Query returned no results above the confidence threshold.")
        return "I don't know. The provided documents do not contain the information.", []

    # 5. Generate Answer from Chunks
    try:
        result = answer_from_chunks(query, top_chunks)
        answer = result.get("answer", "No answer could be generated.")
        sources = result.get("sources", [])

        logger.info("Successfully generated an answer for the query.")
        return answer, sources

    except Exception as e:
        logger.error(f"Error calling LLM service: {e}", exc_info=True)
        return "An error occurred while generating the answer.", []
