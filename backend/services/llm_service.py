import logging
import re
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This is a placeholder for your actual LLM client, e.g., OpenAI, Anthropic, etc.
# from your_llm_client import call_llm

STOPWORDS = {'a', 'an', 'the', 'is', 'in', 'it', 'of', 'for', 'on', 'with', 'what', 'who', 'all', 'list'}

def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """
    (Placeholder) Replace this with your actual LLM API call.
    """
    logger.info("Making a call to the LLM with temperature=%f", temperature)

    evidence_section = prompt.split("EVIDENCE:")
    if len(evidence_section) < 2:
        logger.warning("No EVIDENCE section found in the prompt.")
        return "I don't know. The provided documents do not contain this information."

    evidence = evidence_section[1]
    question_section = prompt.split("Question:")
    if len(question_section) < 2:
        logger.warning("No Question section found in the prompt.")
        return "I don't know. The provided documents do not contain this information."

    question = question_section[1].split("\n\n")[0].strip()
    # Clean question and extract keywords
    cleaned_question = re.sub(r"[^\w\s']", '', question.lower()) # keep apostrophes for possessives
    cleaned_question = re.sub(r"'s\b", '', cleaned_question) # remove possessives
    keywords = {word for word in cleaned_question.split() if word not in STOPWORDS and len(word) >= 2}

    logger.info(f"Question: '{question}', Keywords: {keywords}")

    scored_chunks = []
    for chunk in evidence.split("---"):
        if not chunk.strip():
            continue

        chunk_text = ""
        for line in chunk.strip().split('\n'):
            if line.startswith("Text:"):
                chunk_text = line[len("Text:"):].strip()
                break

        if not chunk_text:
            continue

        score = 0
        cleaned_chunk = chunk_text.lower()
        for keyword in keywords:
            if keyword in cleaned_chunk:
                score += 1

        logger.info(f"Chunk: '{chunk_text}', Score: {score}")
        scored_chunks.append((score, chunk_text))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    if not scored_chunks or not keywords or scored_chunks[0][0] == 0:
        logger.info("No relevant chunks found. Returning 'I don't know'.")
        return "I don't know. The provided documents do not contain this information."

    top_chunk_text = scored_chunks[0][1]
    logger.info(f"Top chunk: '{top_chunk_text}'")
    return top_chunk_text


def answer_from_chunks(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates an answer strictly from the provided chunks.
    If the answer isn't found, it returns a specific fallback message.
    """
    if not chunks:
        return {
            "answer": "I don't know. The provided documents do not contain this information.",
            "sources": []
        }

    # Combine chunks into an evidence string
    evidence = ""
    for c in chunks:
        evidence += f"---\n"
        evidence += f"Source: {c.get('document_name', 'unknown')}\n"
        if c.get('chunk_index') is not None:
            evidence += f"ChunkIndex: {c.get('chunk_index')}\n"
        if c.get('text'):
            evidence += f"Text: {c.get('text')}\n"

    # Construct the prompt with strict instructions
    prompt = (
        "Use ONLY the information in the following documents to answer the question.\n"
        "If the answer is not found in the EVIDENCE, reply exactly with:\n"
        "\"I don't know. The provided documents do not contain this information.\"\n\n"
        f"Question: {question}\n\n"
        f"EVIDENCE:\n{evidence}"
    )

    # Call the LLM with the constructed prompt
    try:
        response = call_llm(prompt, temperature=0.0)
        logger.info("Successfully generated an answer for the query.")
    except Exception as e:
        logger.error("LLM API call failed: %s", e, exc_info=True)
        response = "Error: Could not connect to the language model."

    # Extract unique sources
    sources = []
    if "I don't know" not in response:
        # Fallback to check if response is a substring of any chunk text
        sources = sorted(list({c.get("document_name", "unknown") for c in chunks if c.get("document_name") and response in c.get('text', '')}))


    return {"answer": response, "sources": sources}
