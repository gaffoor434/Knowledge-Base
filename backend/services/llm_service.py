"""
Robust LLM service supporting:
 - OpenAI (official) via OPENAI_API_KEY / OPENAI_API_BASE
 - OpenRouter via OPENROUTER_API_KEY or OPENAI_API_KEY + OPENAI_API_BASE pointing to OpenRouter base
This module provides:
 - ensure_model_loaded() -> bool
 - generate_response(prompt, model=..., temperature=..., **kwargs) -> str
 - answer_from_chunks(chunks, question, model=...) -> str
 - health_check() -> dict
"""

import os
import logging
import requests
from typing import Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuration via environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")  # e.g. https://openrouter.ai/api/v1 for OpenRouter
DEFAULT_MODEL = os.environ.get("LLM_DEFAULT_MODEL", "openrouter/auto")

# Decide auth and base URL safely
if OPENROUTER_API_KEY:
    API_KEY = OPENROUTER_API_KEY
    API_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
elif OPENAI_API_KEY and OPENAI_API_BASE:
    API_KEY = OPENAI_API_KEY
    API_BASE = (OPENAI_API_BASE or "").rstrip("/")
elif OPENAI_API_KEY:
    API_KEY = OPENAI_API_KEY
    API_BASE = "https://api.openai.com/v1"
else:
    API_KEY = None
    API_BASE = None


# Helper: request headers
def _auth_header():
    if not API_KEY:
        return {}
    return {"Authorization": f"Bearer {API_KEY}"}


# --- Model listing helpers ---
def _list_models_openrouter() -> Optional[dict]:
    """GET /api/v1/models on OpenRouter-compatible base."""
    try:
        if not API_BASE:
            return None
        url = f"{API_BASE.rstrip('/')}/models"
        logger.debug("Checking models at: %s", url)
        r = requests.get(url, headers=_auth_header(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("OpenRouter model list check failed: %s", e)
        return None


def _list_models_openai() -> Optional[dict]:
    """Fallback to OpenAI /models endpoint."""
    try:
        if not API_BASE:
            return None
        url = f"{API_BASE.rstrip('/')}/models"
        r = requests.get(url, headers=_auth_header(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("OpenAI model list check failed: %s", e)
        return None


# --- Public: ensure_model_loaded ---
def ensure_model_loaded(model_name: Optional[str] = None) -> bool:
    """
    Ensure a model is available. Returns True if one of the expected models is reachable.
    Avoids calling incorrect endpoints (fixes 404 problem).
    """
    model_name = model_name or DEFAULT_MODEL
    if not API_KEY or not API_BASE:
        logger.error(
            "No API key or base configured for LLM. "
            "Set OPENAI_API_KEY or OPENROUTER_API_KEY and optionally OPENAI_API_BASE/OPENROUTER_BASE_URL."
        )
        return False

    logger.info("Checking LLM availability on %s (model=%s)...", API_BASE, model_name)

    # First try OpenRouter-style listing (works when API_BASE points at openrouter)
    models_resp = _list_models_openrouter()
    if models_resp:
        model_ids = []
        if isinstance(models_resp.get("data"), list):
            model_ids = [
                m.get("id") or m.get("model_name") or "" for m in models_resp.get("data", [])
            ]
        else:
            model_ids = list(models_resp.keys())
        logger.info("Found %d models from models endpoint.", len(model_ids))
        return True

    # Fallback try generic /models
    models_resp = _list_models_openai()
    if models_resp:
        logger.info("Model listing succeeded (generic).")
        return True

    logger.error("Model check failed: could not contact models endpoint on %s", API_BASE)
    return False


# --- Public: generate_response ---
def generate_response(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512
) -> str:
    model = model or DEFAULT_MODEL
    if not API_KEY or not API_BASE:
        raise RuntimeError("LLM API not configured (missing API key/base).")

    url = f"{API_BASE.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={**_auth_header(), "Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # OpenAI/OpenRouter format
        if "choices" in data and len(data["choices"]) > 0:
            ch0 = data["choices"][0]
            if isinstance(ch0.get("message"), dict):
                return (ch0["message"].get("content") or "").strip()
            return (ch0.get("text") or ch0.get("message") or "").strip()

        if "result" in data and isinstance(data["result"], str):
            return data["result"].strip()

        return str(data)

    except requests.HTTPError as e:
        response_text = ""
        if getattr(e, "response", None) is not None and hasattr(e.response, "text"):
            response_text = e.response.text
        logger.exception("LLM request failed (HTTP): %s %s", e, response_text)
        raise
    except Exception as e:
        logger.exception("LLM request failed: %s", e)
        raise


# --- NEW: Combine chunks and query ---
def answer_from_chunks(chunks: List[str], question: str, model: Optional[str] = None) -> str:
    """
    Combine text chunks and user question into one prompt, then call generate_response().
    This is used by query_engine to synthesize an answer from retrieved knowledge base chunks.
    """
    if not chunks:
        return "No relevant information found in the knowledge base."

    # Ensure all chunks are strings
    chunks = [str(c) for c in chunks if c]

    # Combine chunks with separators for context
    context_text = "\n\n".join(chunks[:10])  # limit to 10 chunks to avoid overlong prompts
    prompt = (
        "You are an intelligent assistant. Use the following context to answer the question accurately.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and factually, only using information from the context provided."
    )

    try:
        response = generate_response(prompt, model=model)
        return response.strip()
    except Exception as e:
        logger.exception("Error generating answer from chunks: %s", e)
        return "Sorry, an error occurred while generating the answer."

# --- Health check endpoint ---
def health_check() -> dict:
    ok = ensure_model_loaded()
    return {
        "llm_ok": ok,
        "api_base": API_BASE,
        "has_api_key": bool(API_KEY),
    }


# --- Quick test ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(health_check())
    if API_KEY and API_BASE:
        try:
            print("Test generate...", generate_response("Say hello in one sentence.", max_tokens=30))
        except Exception as e:
            print("Test generate failed:", e)
