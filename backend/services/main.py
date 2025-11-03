from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import sys
import logging
from backend.services.hybrid_search import hybrid_search
from backend.services.bm25_service import BM25Service


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from backend.services.query_engine import query_knowledge_base
from backend.services.file_watcher import start_file_watcher
from backend.services.llm_service import ensure_model_loaded, generate_response
from backend.services.qdrant_service import ensure_collection_exists

# Initialize BM25 index service once
bm25_service = BM25Service(index_path=os.path.join("data", "bm25_index", "bm25_index.pkl"))

# Static directory setup
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = FastAPI(title="Knowledge Base RAG System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[str]

class DocumentInfo(BaseModel):
    filename: str
    path: str
    last_modified: Optional[str] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing Qdrant collection...")
        ensure_collection_exists()

        logger.info("Checking GPT OSS 20B model...")
        ensure_model_loaded()
        logger.info("LLM model loaded successfully.")

        logger.info("Starting file watcher...")
        start_file_watcher()
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Routes
@app.get("/")
async def root():
    return {"message": "Knowledge Base RAG System API is running."}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query knowledge base and use GPT OSS 20B for reasoning answer generation
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # Step 1: Retrieve relevant info from hybrid (BM25 + Qdrant) knowledge base
        context, source_docs = query_knowledge_base(request.query, bm25_service)

        # Step 2: Combine context with query for better reasoning
        final_prompt = f"""
You are a reasoning assistant. Use the given context to answer the question clearly and accurately.

Context:
{context}

Question:
{request.query}

Answer:
"""

        # Step 3: Generate answer via GPT OSS 20B
        answer = generate_response(final_prompt)

        if not answer or len(answer.strip()) < 5:
            answer = "No relevant information found. Please rephrase or try another query."

        return QueryResponse(answer=answer, source_documents=source_docs)

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    doc_dir = r"D:\knowledge base\Document for test"
    try:
        documents = []
        for filename in os.listdir(doc_dir):
            file_path = os.path.join(doc_dir, filename)
            if os.path.isfile(file_path):
                last_modified = os.path.getmtime(file_path)
                documents.append(
                    DocumentInfo(
                        filename=filename,
                        path=file_path,
                        last_modified=str(last_modified)
                    )
                )
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_document(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)
    doc_dir = r"D:\knowledge base\Document for test"
    try:
        if not os.path.exists(file_path):
            alt_path = os.path.join(doc_dir, filename)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/view/{filename}")
async def view_document(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)
    doc_dir = r"D:\knowledge base\Document for test"
    try:
        if not os.path.exists(file_path):
            alt_path = os.path.join(doc_dir, filename)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=file_path, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)