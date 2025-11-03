import time
import os
import threading
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from backend.services.document_processor import process_document, remove_document

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to monitor
WATCH_PATH = r"D:\knowledge base\Document for test"
# Set to True to use polling observer (more reliable but higher CPU usage)
USE_POLLING_OBSERVER = True
# Polling interval in seconds (only used if USE_POLLING_OBSERVER is True)
POLLING_INTERVAL = 2

class DocumentHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        # Track processed files to avoid duplicate processing
        self.recently_processed = set()
        
    def on_created(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            logger.info(f"File created: {event.src_path}")
            success = process_document(event.src_path)
            if success:
                logger.info(f"Successfully processed new file: {event.src_path}")
            else:
                logger.error(f"Failed to process new file: {event.src_path}")

    def on_modified(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            logger.info(f"File modified: {event.src_path}")
            success = process_document(event.src_path, update=True)
            if success:
                logger.info(f"Successfully updated file: {event.src_path}")
            else:
                logger.error(f"Failed to update file: {event.src_path}")

    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            success = remove_document(event.src_path)
            if success:
                logger.info(f"Successfully removed file from database: {event.src_path}")
            else:
                logger.error(f"Failed to remove file from database: {event.src_path}")
    
    def _should_process(self, file_path):
        """
        Check if file should be processed (avoid duplicates and non-document files)
        """
        # Get filename and extension
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(file_path.lower())
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv']
        
        # Skip files with unsupported extensions
        if ext not in supported_extensions:
            return False
            
        # Skip Microsoft Office temporary files (start with ~$)
        if filename.startswith('~$'):
            logger.debug(f"Skipping temporary Office file: {file_path}")
            return False
            
        # Skip hidden files and system files
        if filename.startswith('.') or filename.startswith('~'):
            logger.debug(f"Skipping hidden/system file: {file_path}")
            return False
            
        # Check if file was recently processed (to avoid duplicate events)
        if file_path in self.recently_processed:
            return False
            
        # Add to recently processed set and schedule removal after 2 seconds
        self.recently_processed.add(file_path)
        threading.Timer(2, lambda: self.recently_processed.remove(file_path)).start()
        
        return True

# Global observer reference to prevent garbage collection
_observer = None

def start_file_watcher():
    """Start the file watcher in a separate thread and return the thread"""
    global _observer
    
    # Create a thread that won't block the main thread
    thread = threading.Thread(target=run_file_watcher, daemon=True)
    thread.start()
    logger.info("File watcher thread started")
    return thread

def run_file_watcher():
    """Run the file watcher with automatic restart on failure"""
    global _observer
    
    while True:
        try:
            # Create observer based on configuration
            if USE_POLLING_OBSERVER:
                _observer = PollingObserver(timeout=POLLING_INTERVAL)
                logger.info(f"Using polling observer with interval {POLLING_INTERVAL}s")
            else:
                _observer = Observer()
                logger.info("Using standard file system observer")
                
            event_handler = DocumentHandler()
            
            # Schedule the observer
            _observer.schedule(event_handler, WATCH_PATH, recursive=True)
            
            # Start the observer
            _observer.start()
            logger.info(f"File watcher started for directory: {WATCH_PATH}")
            
            # Process existing files on startup
            process_existing_files()
            
            # Keep the thread alive
            while True:
                time.sleep(1)
                if not _observer.is_alive():
                    logger.error("Observer stopped unexpectedly, restarting...")
                    break
                    
        except Exception as e:
            logger.error(f"Error in file watcher: {str(e)}")
            
        finally:
            # Clean up before restart
            if _observer:
                try:
                    _observer.stop()
                    _observer.join()
                except Exception as e:
                    logger.error(f"Error stopping observer: {str(e)}")
            
            # Wait before restarting
            logger.info("Restarting file watcher in 5 seconds...")
            time.sleep(5)

def process_existing_files():
    """Process all existing files in the watch directory on startup"""
    logger.info(f"Processing existing files in {WATCH_PATH}")
    
    try:
        for root, _, files in os.walk(WATCH_PATH):
            for filename in files:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    # Skip non-document files
                    _, ext = os.path.splitext(file_path.lower())
                    supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv']
                    
                    # Skip temporary Office files and hidden files
                    if filename.startswith('~$') or filename.startswith('.') or filename.startswith('~'):
                        logger.debug(f"Skipping temporary/hidden file: {file_path}")
                        continue
                    
                    if ext in supported_extensions:
                        logger.info(f"Processing existing file: {file_path}")
                        process_document(file_path)
    except Exception as e:
        logger.error(f"Error processing existing files: {str(e)}")

if __name__ == "__main__":
    # For testing the file watcher independently
    run_file_watcher()