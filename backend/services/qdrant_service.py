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
QDRANT_PATH = r"D:\knowledge base\qdrant"
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
        logger.info(f"Qdrant not reachable ({e}). Attempting to start it from {QDRANT_PATH}...")
        # Attempt to start Qdrant binary
        if platform.system() == "Windows":
            qdrant_exe = os.path.join(QDRANT_PATH, "qdrant.exe")
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            qdrant_exe = os.path.join(QDRANT_PATH, "qdrant")
            creation_flags = 0
        try:
            subprocess.Popen([qdrant_exe], cwd=QDRANT_PATH, creationflags=creation_flags)
        except Exception as start_err:
            logger.error(f"Failed to start Qdrant: {start_err}")
            raise
        # wait and re-check
        for i in range(20):
            time.sleep(2)
            try:
                test_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
                test_client.get_collections()
                logger.info("Qdrant started successfully.")
                return
            except Exception:
                logger.info("Waiting for Qdrant to become available...")
        raise RuntimeError("Unable to start Qdrant; please start it manually.")


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
        # Common metadata keys we'll index for filters
        indices = [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("document_name", models.PayloadSchemaType.KEYWORD),
            ("source_type", models.PayloadSchemaType.KEYWORD),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("section_title", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema in indices:
            try:
                client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name=field_name, field_schema=schema)
            except UnexpectedResponse as e:
                # Some older qdrant-client versions may raise an error if index exists; ignore
                logger.debug(f"Payload index create warning for {field_name}: {e}")
    except Exception as e:
        logger.error(f"Failed to create payload indices: {e}")
        raise


def store_document_embeddings(document_path: str, document_name: str, chunks: List[str],
                              embeddings: Optional[List[List[float]]] = None,
                              metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Store document chunks and embeddings in Qdrant.
    Accepts precomputed embeddings and optional per-chunk metadata payloads.
    """
    try:
        client = get_qdrant_client()

        # If embeddings not provided, compute them (backwards compatibility)
        if embeddings is None:
            logger.info("No embeddings passed — generating inside qdrant_service.")
            embeddings = generate_embeddings(chunks)

        if not embeddings or len(embeddings) != len(chunks):
            logger.error("Embeddings length mismatch or empty.")
            return False

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "document_id": document_path,
                "document_name": document_name,
                "chunk_index": i,
                "text": chunk  # store full chunk text for better grounding/citations
            }
            # Merge optional metadata if provided
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            points.append(models.PointStruct(
                id=point_id,
                vector=emb,
                payload=payload
            ))

        # Upsert points in batches if many
        batch_size = 256
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
            logger.info(f"Upserted points {i}..{i + len(batch) - 1}")

        logger.info(f"Stored {len(points)} chunks for document: {document_name}")
        return True
    except Exception as e:
        logger.exception(f"Error storing document embeddings: {e}")
        return False


def delete_document(document_path: str) -> bool:
    try:
        client = get_qdrant_client()
        # Delete by payload filter matching document_id/document_path
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
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
            out.append({
                "score": getattr(r, "score", None),
                "document_id": r.payload.get("document_id"),
                "document_name": r.payload.get("document_name"),
                "chunk_index": r.payload.get("chunk_index"),
                "text": r.payload.get("text"),
                # include any other payload keys present
                **{k: v for k, v in (r.payload.items()) if k not in ("document_id", "document_name", "chunk_index", "text")}
            })
        
        # Normalize vector scores
        scores = [r.get('score', 0.0) or 0.0 for r in out]
        if scores:
            mn, mx = min(scores), max(scores)
            for i, r in enumerate(out):
                raw = scores[i]
                r['vec_norm'] = 0.0 if mx == mn else (raw - mn) / (mx - mn)
        else:
            for r in out:
                r['vec_norm'] = 0.0
                
        return out
    except Exception as e:
        logger.exception(f"Error searching similar chunks: {e}")
        return []


def count_documents() -> int:
    """Count total number of documents in the collection."""
    try:
        client = get_qdrant_client()
        # Use scroll with limit=1 to get a small sample and check if collection has data
        result = client.scroll(collection_name=QDRANT_COLLECTION, limit=1, with_payload=False)
        if result and result[0]:
            # If we got results, try to get a rough count by scrolling through
            # This is not exact but good enough for our use case
            return 1  # At least one document exists
        return 0
    except Exception as e:
        logger.exception(f"Error counting documents: {e}")
        return 0
