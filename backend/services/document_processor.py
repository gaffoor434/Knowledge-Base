import os
import re
import fitz  # PyMuPDF
import pdfplumber
import docx
import pandas as pd
from typing import List, Dict, Any, Tuple
import uuid
import logging

from backend.services.qdrant_service import store_document_embeddings, delete_document
from backend.services.utils import adaptive_sentence_chunks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_document(file_path: str, update: bool = False) -> bool:
    """
    Process a document file, extracting both text and structured table data.
    - Extracts text from PDF, DOCX, TXT.
    - Extracts tables from PDF, DOCX, XLSX.
    - Chunks text and table rows separately.
    - Stores chunks with rich metadata in Qdrant.
    """
    try:
        filename = os.path.basename(file_path)
        if filename.startswith('~$'):
            logger.info(f"Skipping temporary Office file: {file_path}")
            return False

        if update:
            delete_document(file_path)

        document_name = os.path.basename(file_path)
        all_chunks: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        # Extract text and tables based on file type
        text_chunks, table_chunks = extract_text_and_tables(file_path)

        # Process text chunks
        for i, chunk_text in enumerate(text_chunks):
            all_chunks.append(chunk_text)
            metadatas.append({
                "is_table": False,
                "page_number": chunk_text.get("page_number") if isinstance(chunk_text, dict) else None,
                "document_path": file_path,
                "document_name": document_name,
                "chunk_index": i
            })

        # Process table chunks
        for table_meta in table_chunks:
            all_chunks.append(table_meta["text"])
            # Remove the 'text' key from the metadata to avoid duplication
            meta = {k: v for k, v in table_meta.items() if k != 'text'}
            meta.update({
                "is_table": True,
                "document_path": file_path,
                "document_name": document_name
            })
            metadatas.append(meta)

        if not all_chunks:
            logger.warning(f"No chunks generated for {file_path}")
            return False

        # Store in Qdrant
        store_document_embeddings(file_path, document_name, all_chunks, metadatas=metadatas)

        logger.info("Parsed %s: %d chunks (%d tables)", document_name, len(all_chunks), len(table_chunks))
        return True

    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
        return False


def extract_text_and_tables(file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extracts text and tables from a file, returns separate lists."""
    ext = os.path.splitext(file_path)[1].lower()
    text_chunks, table_chunks = [], []
    
    if ext == '.pdf':
        text_chunks, table_chunks = extract_from_pdf(file_path)
    elif ext == '.docx':
        text_chunks, table_chunks = extract_from_docx(file_path)
    elif ext == '.xlsx':
        table_chunks = extract_from_excel(file_path)
    elif ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            text_chunks = adaptive_sentence_chunks(text, min_words=8)
            
    return text_chunks, table_chunks


def extract_from_pdf(file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extracts page text and tables from a PDF."""
    text_chunks, table_chunks = [], []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text from the page
            page_text = page.extract_text() or ""
            if page_text:
                clean_text = preprocess_text(page_text)
                chunks = adaptive_sentence_chunks(clean_text, min_words=8)
                for chunk in chunks:
                    text_chunks.append(chunk)

            # Extract tables from the page
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table: continue
                headers = [str(h).strip() for h in table[0]] if table[0] else []
                for row_idx, row in enumerate(table[1:], start=1):
                    row_data = {headers[j]: str(cell).strip() for j, cell in enumerate(row) if j < len(headers)}
                    row_text = ", ".join([f"{k}: {v}" for k, v in row_data.items()])
                    table_chunks.append({
                        "text": row_text,
                        "is_table": True,
                        "page": i + 1,
                        "table_index": table_idx,
                        "row_index": row_idx,
                        "headers": headers
                    })
    return text_chunks, table_chunks


def extract_from_docx(file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extracts paragraphs and tables from a DOCX."""
    doc = docx.Document(file_path)
    text_chunks, table_chunks = [], []

    # Extract text from paragraphs
    full_text = "\n".join([p.text for p in doc.paragraphs])
    if full_text.strip():
        text_chunks = adaptive_sentence_chunks(preprocess_text(full_text), min_words=8)
        
    # Extract tables
    for table_idx, table in enumerate(doc.tables):
        headers = [cell.text.strip() for cell in table.rows[0].cells]
        for row_idx, row in enumerate(table.rows[1:], start=1):
            row_data = {headers[j]: cell.text.strip() for j, cell in enumerate(row.cells) if j < len(headers)}
            row_text = ", ".join(f"{k}: {v}" for k, v in row_data.items())
            table_chunks.append({
                "text": row_text,
                "is_table": True,
                "table_index": table_idx,
                "row_index": row_idx,
                "headers": headers
            })
    return text_chunks, table_chunks


def extract_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """Extracts rows from each sheet of an Excel file."""
    xls = pd.ExcelFile(file_path)
    table_chunks = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df = df.dropna(how='all')
        if df.empty: continue
        
        headers = [str(h).strip() for h in df.columns]
        for row_idx, row in df.iterrows():
            row_data = {h: str(row[h]).strip() for h in headers if h in row and pd.notna(row[h])}
            row_text = ", ".join([f"{k}: {v}" for k,v in row_data.items()])
            table_chunks.append({
                "text": row_text,
                "is_table": True,
                "sheet_name": sheet_name,
                "row_index": row_idx + 1,
                "headers": headers
            })
    return table_chunks


def preprocess_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = text.strip()
    return text
