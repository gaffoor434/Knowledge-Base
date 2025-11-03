import os
import subprocess
import time
import uuid
import logging
import platform
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.services.embedding_service import generate_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant configuration
QDRANT_PATH = r"/qdrant"
QDRANT_COLLECTION = "knowledge_base"
QDRANT_PORT = 6333
QDRANT_HOST = "localhost"

# Global Qdrant client
client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    global client
    if client is None:
        ensure_qdrant_running()
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10.0)
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    return client


def ensure_qdrant_running():
    try:
        test_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
        test_client.get_collections()
        logger.info("Qdrant is running.")
    except Exception as e:
        logger.info(f"Qdrant not reachable ({e}). Please ensure Qdrant is running manually.")


def ensure_collection_exists():
    global client
    client = get_qdrant_client()
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating collection {QDRANT_COLLECTION}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
            )
        create_payload_indices()
        logger.info(f"Collection '{QDRANT_COLLECTION}' is ready.")
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        raise


def create_payload_indices():
    try:
        client = get_qdrant_client()
        # Common metadata keys we'll index for filters
        indices = [
            ("document_path", models.PayloadSchemaType.KEYWORD),
            ("document_name", models.PayloadSchemaType.KEYWORD),
            ("is_table", models.PayloadSchemaType.BOOL),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("page", models.PayloadSchemaType.INTEGER), # For pdfplumber tables
        ]
        for field_name, schema in indices:
            try:
                client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name=field_name, field_schema=schema)
                logger.info(f"Created payload index for: {field_name}")
            except UnexpectedResponse as e:
                logger.debug(f"Payload index create warning for {field_name}: {e}")
    except Exception as e:
        logger.error(f"Failed to create payload indices: {e}", exc_info=True)


def store_document_embeddings(document_path: str, document_name: str, chunks: List[str],
                              embeddings: Optional[List[List[float]]] = None,
                              metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Store document chunks and embeddings in Qdrant.
    Ensures consistent metadata and logs the operation.
    """
    try:
        client = get_qdrant_client()

        if embeddings is None:
            embeddings = generate_embeddings(chunks)

        if not embeddings or len(embeddings) != len(chunks):
            logger.error("Embeddings length mismatch or empty for %s.", document_name)
            return False

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())

            payload = metadatas[i].copy() if metadatas and i < len(metadatas) else {}

            # Ensure base fields are consistently present
            payload["document_name"] = document_name
            payload["document_path"] = document_path
            payload["chunk_index"] = i
            payload["text"] = chunk

            points.append(models.PointStruct(id=point_id, vector=emb, payload=payload))

        # Upsert points in batches
        batch_size = 128
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=QDRANT_COLLECTION, points=batch, wait=False)

        logger.info("Upserted %d chunks for %s", len(chunks), document_name)
        return True
    except Exception as e:
        logger.error(f"Error storing embeddings for %s: %s", document_name, e, exc_info=True)
        return False


def delete_document(document_path: str) -> bool:
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_path",
                            match=models.MatchValue(value=document_path)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted embeddings for document: {document_path}")
        return True
    except Exception as e:
        logger.exception(f"Error deleting document embeddings: {e}")
        return False


def search_similar_chunks(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        client = get_qdrant_client()
        results = client.search(collection_name=QDRANT_COLLECTION, query_vector=query_embedding, limit=top_k)
        out = []
        for r in results:
            payload = r.payload or {}
            out.append({
                "score": getattr(r, "score", None),
                "document_name": payload.get("document_name"),
                "text": payload.get("text"),
                **payload
            })
        
        # Normalize vector scores
        scores = [r.get('score', 0.0) or 0.0 for r in out]
        if scores:
            min_s, max_s = min(scores), max(scores)
            for r in out:
                score = r.get('score', 0.0) or 0.0
                r['vec_norm'] = 0.0 if max_s == min_s else (score - min_s) / (max_s - min_s)

        return out
    except Exception as e:
        logger.exception(f"Error searching similar chunks: {e}")
        return []


def count_documents() -> int:
    """Counts the total number of points (chunks) in the collection."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name=QDRANT_COLLECTION)
        return int(getattr(info, "points_count", info.get("points_count", 0)))
    except Exception as e:
        logger.exception(f"Error counting documents: {e}")
        return 0
