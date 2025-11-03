from .bm25_service import BM25Service

def exact_match_check(docs, query_text):
    q = query_text.strip().lower()
    for d in docs:
        if q in d['text'].lower():
            return {"id": d['id'], "text": d['text']}
    return None

def normalize_scores_bm25(scores):
    vals = [s['bm25_score'] for s in scores]
    if not vals:
        return scores
    mn, mx = min(vals), max(vals)
    for s in scores:
        s['bm25_norm'] = 0.0 if mx == mn else (s['bm25_score'] - mn) / (mx - mn)
    return scores

def normalize_scores_vector(vec_results):
    vals = [r['score'] for r in vec_results]
    if not vals:
        return vec_results
    mn, mx = min(vals), max(vals)
    for r in vec_results:
        r['vec_norm'] = 0.0 if mx == mn else (r['score'] - mn) / (mx - mn)
    return vec_results

def hybrid_search(query_text, bm25_service: BM25Service, qdrant_search_fn, top_k=7,
                  min_combined_score: float = 0.12, require_both: bool = False):
    """
    Enhanced hybrid search: BM25 + Vector -> re-rank -> apply thresholds.
    Returns a list of chunk dicts or an empty list if nothing relevant.
    """
    bm25_results = bm25_service.query(query_text, top_k=top_k) or []
    vector_results = qdrant_search_fn(query_text, top_k=top_k) or []

    # Fail fast if no documents are indexed
    if (not bm25_service.docs) and (not vector_results):
        return []

    # Normalize scores
    bm25_res = normalize_scores_bm25(bm25_results)
    vec_res = normalize_scores_vector(vector_results)

    merged = {}
    for r in bm25_res:
        merged[r['id']] = {
            'id': r['id'],
            'text': r.get('text', ''),
            'bm25_score': r['bm25_score'],
            'bm25_norm': r.get('bm25_norm', 0.0),
            'vec_score': 0.0,
            'vec_norm': 0.0,
            'document_name': r.get('document_name', '')
        }

    for r in vec_res:
        if r['id'] in merged:
            merged[r['id']]['vec_score'] = r.get('score', 0.0)
            merged[r['id']]['vec_norm'] = r.get('vec_norm', 0.0)
        else:
            merged[r['id']] = {
                'id': r['id'],
                'text': r.get('text', ''),
                'bm25_score': 0.0,
                'bm25_norm': 0.0,
                'vec_score': r.get('score', 0.0),
                'vec_norm': r.get('vec_norm', 0.0),
                'document_name': r.get('document_name', '')
            }

    BM25_WEIGHT = 0.6
    VEC_WEIGHT = 0.4
    merged_list = list(merged.values())

    for m in merged_list:
        m['combined'] = BM25_WEIGHT * m.get('bm25_norm', 0.0) + VEC_WEIGHT * m.get('vec_norm', 0.0)

    # Filter by thresholds
    merged_list.sort(key=lambda x: x['combined'], reverse=True)
    merged_list = [m for m in merged_list if m.get('combined', 0.0) >= min_combined_score]

    if require_both:
        merged_list = [m for m in merged_list if m.get('bm25_norm', 0.0) >= 0.08 and m.get('vec_norm', 0.0) >= 0.08]

    return merged_list[:5]
