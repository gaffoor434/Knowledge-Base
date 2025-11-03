import logging
import os

LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "processing.log")),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    return logging.getLogger(name)
