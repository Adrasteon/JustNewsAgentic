
import os
import logging
from datetime import datetime
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

FEEDBACK_LOG = os.environ.get("MEMORY_FEEDBACK_LOG", "./feedback_memory.log")
EMBEDDING_MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory.tools")

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def get_embedding_model():
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed.")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# The following are stubs that should call the FastAPI endpoints or be used for local testing/mocking.

def save_article(content: str, metadata: dict) -> dict:
    """Stub for saving an article. In production, call the FastAPI endpoint."""
    logger.info(f"Stub: save_article called with content length {len(content)}")
    log_feedback("save_article", {"content_len": len(content), "metadata": metadata})
    return {"id": 0}

def get_article(article_id: int) -> dict:
    """Stub for retrieving an article. In production, call the FastAPI endpoint."""
    logger.info(f"Stub: get_article called for id {article_id}")
    log_feedback("get_article", {"article_id": article_id})
    return {"id": article_id, "content": "", "metadata": {}}

def vector_search_articles(query: str, top_k: int = 5) -> list:
    """Stub for vector search. In production, call the FastAPI endpoint."""
    logger.info(f"Stub: vector_search_articles called for query '{query}'")
    log_feedback("vector_search_articles", {"query": query, "top_k": top_k})
    return []

def log_training_example(task: str, input: dict, output: dict, critique: str) -> dict:
    """Stub for logging a training example. In production, call the FastAPI endpoint."""
    logger.info(f"Stub: log_training_example called for task {task}")
    log_feedback("log_training_example", {"task": task, "input": input, "output": output, "critique": critique})
    return {"status": "logged", "task": task}