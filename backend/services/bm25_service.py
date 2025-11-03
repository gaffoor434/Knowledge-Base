import pickle
import os
import re
import logging
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_tokenize(text: str) -> List[str]:
    """A simple tokenizer that cleans and splits text."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    return text.split()

class BM25Service:
    def __init__(self, index_path: str = "bm25_index.pkl"):
        self.index_path = index_path
        self.docs: List[Dict[str, Any]] = []
        self.doc_names: List[str] = []
        self._bm25: BM25Okapi = None
        self.load_index()

    def build_index(self, docs: List[Dict[str, Any]]):
        """Builds the BM25 index from a list of documents (chunks)."""
        if not docs:
            logger.warning("BM25: No documents provided to build index.")
            return

        self.docs = docs
        self.doc_names = [meta.get("document_name", "unknown") for meta in docs]

        corpus = [d.get('text', '') for d in docs]
        tokenized_corpus = [simple_tokenize(text) for text in corpus]

        self._bm25 = BM25Okapi(tokenized_corpus)
        self.save_index()
        logger.info(f"BM25 index built with {len(self.docs)} documents.")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the index and returns the top k results with metadata."""
        if self._bm25 is None:
            logger.warning("BM25 index not built. Returning empty list.")
            return []

        tokenized_query = simple_tokenize(query_text)
        doc_scores = self._bm25.get_scores(tokenized_query)

        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_n_indices:
            original_doc = self.docs[idx]
            results.append({
                "document_name": original_doc.get("document_name"),
                "chunk_index": original_doc.get("chunk_index"),
                "text": original_doc.get("text"),
                "bm25_score": doc_scores[idx]
            })

        return results

    def save_index(self):
        """Saves the BM25 index and associated documents to a file."""
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump({
                    "bm25": self._bm25,
                    "docs": self.docs
                }, f)
            logger.info(f"BM25 index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}", exc_info=True)

    def load_index(self):
        """Loads the BM25 index from a file."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self._bm25 = data["bm25"]
                    self.docs = data["docs"]
                    self.doc_names = [meta.get("document_name", "unknown") for meta in self.docs]
                logger.info(f"BM25 index loaded from {self.index_path} with {len(self.docs)} documents.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}", exc_info=True)
                self._bm25 = None
                self.docs = []
