import os
import sys
import uvicorn

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting Knowledge Base RAG System Backend...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)