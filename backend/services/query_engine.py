from typing import List, Tuple, Dict, Any, Optional, Union
import logging
import functools
import time

from backend.services.embedding_service import generate_query_embedding
from backend.services.qdrant_service import search_similar_chunks, count_documents
from backend.services.llm_service import answer_from_chunks
from .hybrid_search import hybrid_search
from .bm25_service import BM25Service
from .query_expansion_service import expand_query
from .utils import dedupe_by_id

logger = logging.getLogger(__name__)

# Cache embeddings to speed up repeated queries
@functools.lru_cache(maxsize=128)
def cached_generate_query_embedding(q_text: str):
    return generate_query_embedding(q_text)


# Lazy-init reranker
_reranker: Optional[Union[object, bool]] = None


def _get_reranker() -> Optional[Union[object, bool]]:
    """Lazily load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("✅ CrossEncoder initialized successfully.")
        except Exception as e:
            logger.warning(f"⚠️ CrossEncoder unavailable, skipping rerank: {e}")
            _reranker = False
    return _reranker


def query_knowledge_base(query: str, bm25_service: Optional[BM25Service] = None) -> Tuple[str, List[str]]:
    """
    Query the knowledge base using hybrid retrieval, reranking, and grounding.
    Returns a tuple: (answer_text, source_documents).
    """
    try:
        start_time = time.time()

        if not query or not query.strip():
            return "Please enter a valid question.", []

        # ✅ Ensure documents exist
        if (bm25_service is None or not bm25_service.docs) and count_documents() == 0:
            return "No documents indexed. Please upload documents first.", []

        # ✅ Qdrant search function
        def qdrant_search_fn(q_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
            emb = cached_generate_query_embedding(q_text)
            if emb is None:
                return []
            vec_hits = search_similar_chunks(emb, top_k=top_k)
            results: List[Dict[str, Any]] = []
            for r in vec_hits or []:
                rid = f"{r.get('document_id', '')}:{r.get('chunk_index', '')}"
                results.append({
                    "id": rid,
                    "text": r.get("text", ""),
                    "score": r.get("score", 0.0),
                    "document_name": r.get("document_name"),
                    "page_number": r.get("page_number")
                })
            return results

        # ✅ Multi-query expansion
        subqueries = expand_query(query) or [query]
        merged: List[Dict[str, Any]] = []

        for sq in subqueries:
            hits = hybrid_search(sq, bm25_service, qdrant_search_fn)
            if hits:
                merged.extend(hits)

        # ✅ Deduplicate and filter
        merged = dedupe_by_id(merged)
        merged = [
            doc for doc in merged
            if doc.get("score", doc.get("combined", 0.0)) >= 0.55
        ]

        if not merged:
            return "No relevant documents found. Please refine your query or upload more documents.", []

        # ✅ Optional reranking
        reranker = _get_reranker()
        if reranker and hasattr(reranker, "predict"):
            pairs = [(query, doc.get("text", "")) for doc in merged]
            try:
                scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
                for doc, sc in zip(merged, scores):
                    doc["rank_score"] = float(sc)
            except Exception as e:
                logger.warning(f"⚠️ Reranking failed: {e}")
        else:
            for doc in merged:
                doc["rank_score"] = doc.get("combined", doc.get("score", 0.0))

        # ✅ Apply weighted scores
        SOURCE_WEIGHTS = {
            "ESM DOCS.docx": 1.3,
            "FAQs - Additional Surveillance Measure (ASM)_14.4.25_NEWTEMP.pdf": 1.0,
            "sample_employee_data.xlsx": 0.8,
        }
        for doc in merged:
            base = doc.get("rank_score", doc.get("combined", 0.0))
            w = SOURCE_WEIGHTS.get(doc.get("document_name", ""), 1.0)
            doc["weighted_score"] = float(base) * float(w)

        merged.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
        top_chunks = merged[:8]

        # ✅ Ensure correct types for answer_from_chunks
        result: Any = answer_from_chunks(query, top_chunks)

        # ✅ Safely handle both dict or string results
        if isinstance(result, dict):
            answer_text = result.get("answer", "No answer generated.")
            sources_data = result.get("sources", [])
        else:
            answer_text = str(result)
            sources_data = []

        # ✅ Extract document names
        source_documents: List[str] = []
        if isinstance(sources_data, list):
            for source in sources_data:
                if isinstance(source, dict):
                    doc_name = source.get("document_name")
                    if doc_name and doc_name not in source_documents:
                        source_documents.append(doc_name)

        # Fallback: if no sources, use chunk docs
        if not source_documents:
            for chunk in top_chunks:
                doc_name = chunk.get("document_name")
                if doc_name and doc_name not in source_documents:
                    source_documents.append(doc_name)

        duration = time.time() - start_time
        logger.info(f"✅ Query processed in {duration:.2f}s | {len(top_chunks)} chunks used.")
        return answer_text, source_documents

    except Exception as e:
        logger.exception("❌ Error querying knowledge base")
        return f"An error occurred while processing your query: {str(e)}", []
