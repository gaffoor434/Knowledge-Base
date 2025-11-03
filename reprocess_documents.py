"""
Reprocess all documents with optimized chunk size (400 words)
This clears the existing collection and processes all documents fresh
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.document_processor import process_document
from backend.services.qdrant_service import get_qdrant_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENT_DIR = r"D:\knowledge base\Document for test"
QDRANT_COLLECTION = "knowledge_base"


def reprocess_all_documents():
    """Reprocess all documents with new chunk size"""
    try:
        logger.info("Starting document reprocessing...")
        
        # Get Qdrant client
        client = get_qdrant_client()
        
        # Clear existing collection
        logger.info(f"Clearing existing collection '{QDRANT_COLLECTION}'...")
        try:
            client.delete_collection(collection_name=QDRANT_COLLECTION)
            logger.info("Collection deleted")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
        
        # Recreate collection
        logger.info("Recreating collection...")
        from backend.services.qdrant_service import ensure_collection_exists
        ensure_collection_exists()
        
        # Process all documents
        logger.info(f"Processing documents from {DOCUMENT_DIR}...")
        
        if not os.path.exists(DOCUMENT_DIR):
            logger.error(f"Document directory not found: {DOCUMENT_DIR}")
            return False
        
        files = [f for f in os.listdir(DOCUMENT_DIR) if os.path.isfile(os.path.join(DOCUMENT_DIR, f))]
        supported_ext = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv', '.md', '.json']
        
        doc_files = [f for f in files if any(f.lower().endswith(ext) for ext in supported_ext)]
        
        if not doc_files:
            logger.warning("No documents found to process")
            return False
        
        logger.info(f"Found {len(doc_files)} documents to process")
        
        success_count = 0
        failed_count = 0
        
        for i, filename in enumerate(doc_files, 1):
            file_path = os.path.join(DOCUMENT_DIR, filename)
            logger.info(f"[{i}/{len(doc_files)}] Processing: {filename}")
            
            try:
                result = process_document(file_path)
                if result:
                    success_count += 1
                    logger.info(f"  ✓ Success")
                else:
                    failed_count += 1
                    logger.warning(f"  ✗ Failed")
            except Exception as e:
                failed_count += 1
                logger.error(f"  ✗ Error: {str(e)}")
        
        logger.info("\n" + "="*80)
        logger.info(f"REPROCESSING COMPLETE")
        logger.info(f"  Successfully processed: {success_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Total: {len(doc_files)}")
        logger.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during reprocessing: {str(e)}")
        logger.exception(e)
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DOCUMENT REPROCESSING UTILITY")
    print("Optimized chunk size: 400 words with 50 word overlap")
    print("="*80 + "\n")
    
    confirm = input("This will clear and reprocess all documents. Continue? (yes/no): ")
    
    if confirm.lower() in ['yes', 'y']:
        success = reprocess_all_documents()
        
        if success:
            print("\n✅ Documents reprocessed successfully!")
            print("   Run 'python rebuild_bm25_index.py' to sync BM25 index.\n")
        else:
            print("\n❌ Reprocessing failed. Check logs above.\n")
    else:
        print("\nCancelled by user.\n")
