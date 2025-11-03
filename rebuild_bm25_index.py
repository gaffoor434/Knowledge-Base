import sys
import os
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.qdrant_service import get_qdrant_client, QDRANT_COLLECTION
from backend.services.bm25_service import BM25Service
from qdrant_client.http import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_all_documents_from_qdrant():
    """
    Fetches all document chunks from the Qdrant collection.
    """
    try:
        client = get_qdrant_client()
        logger.info(f"Connecting to Qdrant collection: {QDRANT_COLLECTION}")

        all_points = []
        next_offset = 0
        while True:
            points, next_offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            all_points.extend(points)
            if next_offset is None:
                break
        
        # Convert points to the format expected by BM25Service
        docs = [p.payload for p in all_points if p.payload]
        logger.info(f"Fetched a total of {len(docs)} documents from Qdrant.")
        return docs
        
    except Exception as e:
        logger.error(f"Failed to fetch documents from Qdrant: {e}", exc_info=True)
        return []


def main():
    """
    Main function to rebuild the BM25 index.
    """
    logger.info("Starting BM25 index rebuild process...")
    
    # 1. Fetch all documents from Qdrant
    documents = fetch_all_documents_from_qdrant()
    
    if not documents:
        logger.warning("No documents found in Qdrant. Aborting BM25 index rebuild.")
        return

    # 2. Initialize BM25Service
    # Ensure the index path matches the one used in your application
    bm25_service = BM25Service(index_path="bm25_index.pkl")

    # 3. Build and save the new index
    bm25_service.build_index(documents)

    logger.info("BM25 index has been successfully rebuilt and saved.")


if __name__ == "__main__":
    main()
