import logging
from typing import List, Dict, Any, Callable

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_scores(results: List[Dict[str, Any]], score_key: str, normalized_key: str):
    """Normalizes scores in a list of results from 0 to 1."""
    if not results:
        return

    scores = [r.get(score_key, 0.0) for r in results if r.get(score_key) is not None]
    if not scores:
        return

    min_score, max_score = min(scores), max(scores)

    for r in results:
        score = r.get(score_key, 0.0)
        if max_score == min_score:
            r[normalized_key] = 0.0 if min_score == 0 else 1.0
        else:
            r[normalized_key] = (score - min_score) / (max_score - min_score)


def hybrid_search(
    query_text: str,
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Performs a hybrid search by merging and re-ranking BM25 and vector search results.
    - Normalizes scores for both search methods.
    - Combines scores with specified weights.
    - Filters out results missing essential metadata.
    - Logs top results for debugging.
    """
    # Normalize BM25 scores
    normalize_scores(bm25_results, score_key="bm25_score", normalized_key="bm25_norm")

    # Vector results from qdrant_service are already normalized to 'vec_norm'
    # If not, you would normalize them here, e.g.:
    # normalize_scores(vector_results, score_key="score", normalized_key="vec_norm")

    # Merge results using a unique identifier (e.g., a tuple of document_name and chunk_index)
    merged = {}

    for r in bm25_results:
        # A unique key for each chunk
        chunk_key = (r.get("document_name"), r.get("chunk_index"))
        if not all(k is not None for k in chunk_key):
            continue
        merged[chunk_key] = {
            "document_name": r.get("document_name"),
            "text": r.get("text"),
            "bm25_norm": r.get("bm25_norm", 0.0),
            "vec_norm": 0.0
        }

    for r in vector_results:
        chunk_key = (r.get("document_name"), r.get("chunk_index"))
        if not all(k is not None for k in chunk_key):
            continue

        if chunk_key in merged:
            merged[chunk_key]["vec_norm"] = r.get("vec_norm", 0.0)
        else:
            merged[chunk_key] = {
                "document_name": r.get("document_name"),
                "text": r.get("text"),
                "bm25_norm": 0.0,
                "vec_norm": r.get("vec_norm", 0.0)
            }

    # Calculate combined score and filter out incomplete entries
    results = []
    for chunk_key, scores in merged.items():
        if scores.get("document_name") and scores.get("text"):
            scores["combined"] = (bm25_weight * scores["bm25_norm"]) + (vector_weight * scores["vec_norm"])
            results.append(scores)

    # Sort by the new combined score
    results.sort(key=lambda x: x["combined"], reverse=True)

    # Log the top results for debugging
    top_results_log = [(r['document_name'], round(r['combined'], 4)) for r in results[:5]]
    logger.debug("Hybrid top results: %s", top_results_log)

    return results[:top_k]
