from rank_bm25 import BM25Okapi
import pickle
import os
import re

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    return text.split()

class BM25Service:
    def __init__(self, docs=None, index_path=None):
        self.index_path = index_path
        self.docs = docs or []
        self._bm25 = None
        self._tokenized_texts = None
        print(f"[BM25] Initialized with index path: {index_path}")
        if index_path and os.path.exists(index_path):
            self._load(index_path)
        elif self.docs:
            self.build(self.docs)

    def build(self, docs):
        self.docs = docs
        corpus = [d['text'] for d in docs]
        self._tokenized_texts = [simple_tokenize(t) for t in corpus]
        self._bm25 = BM25Okapi(self._tokenized_texts)
        if self.index_path:
            self._save(self.index_path)

    def _save(self, index_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "docs": self.docs
            }, f)
        print(f"[BM25] Saved index with {len(self.docs)} docs at {index_path}")

    def _load(self, index_path):
        if not os.path.exists(index_path):
            print(f"[BM25] No index found at {index_path}. Skipping load.")
            return

        with open(index_path, "rb") as f:
            payload = pickle.load(f)

        # Support multiple formats
        if isinstance(payload, dict):
            if "bm25" in payload:
                # New format: serialized BM25 object and docs
                self._bm25 = payload.get("bm25")
                self.docs = payload.get("docs", [])
                self._tokenized_texts = None
                print(f"[BM25] Loaded index with {len(self.docs)} docs.")
            else:
                # Older format in this codebase: tokenized + docs
                self.docs = payload.get('docs', [])
                self._tokenized_texts = payload.get('tokenized')
                if self._tokenized_texts:
                    self._bm25 = BM25Okapi(self._tokenized_texts)
                print(f"[BM25] Loaded legacy index with {len(self.docs)} docs (rebuilt BM25 from tokens).")
        else:
            # Legacy support (payload is a BM25Okapi object)
            self._bm25 = payload
            self.docs = []
            self._tokenized_texts = None
            print("[BM25] Loaded legacy BM25 model without docs.")

    def query(self, query_text, top_k=5):
        if not self._bm25:
            return []
        tokens = simple_tokenize(query_text)
        scores = self._bm25.get_scores(tokens)
        if scores is None or len(scores) == 0:
            return []
        top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_n_idx:
            if 0 <= idx < len(self.docs):
                doc_id = self.docs[idx].get('id')
                doc_text = self.docs[idx].get('text', '')
            else:
                # Fallback when docs are unavailable (legacy BM25 without docs)
                doc_id = str(idx)
                doc_text = ""
            results.append({
                "id": doc_id,
                "text": doc_text,
                "bm25_score": float(scores[idx])
            })
        return results
