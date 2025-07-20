

# main.py for Memory Agent

import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# Removed unused import
try:
    import numpy as np
    import faiss
except ImportError:
    np = None
    faiss = None

# Optional import for embedding model
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ...existing code...

    # Removed unused import
    EMBEDDING_MODEL_NAME = os.environ.get("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FEEDBACK_LOG = os.environ.get("MEMORY_FEEDBACK_LOG", "./feedback_memory.log")

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST"),
            database=os.environ.get("POSTGRES_DB"),
            user=os.environ.get("POSTGRES_USER"),
            password=os.environ.get("POSTGRES_PASSWORD")
        )
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to PostgreSQL database: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

def get_embedding_model():
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed.")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

class Article(BaseModel):
    content: str
    metadata: dict

@app.post("/save_article")
def save_article(article: Article):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO articles (content, metadata) VALUES (%s, %s) RETURNING id", 
                    (article.content, article.metadata))
        result = cur.fetchone()
        if result:
            # If using default cursor, result is a tuple; if RealDictCursor, it's a dict
            if isinstance(result, dict):
                article_id = result.get('id')
            else:
                article_id = result[0]
        else:
            article_id = None
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Saved article with id: {article_id}")
        return {"id": article_id}
    except Exception as e:
        logger.error(f"An error occurred while saving the article: {e}")
        raise HTTPException(status_code=500, detail="Error saving article")

@app.get("/get_article/{article_id}")
def get_article(article_id: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        article = cur.fetchone()
        cur.close()
        conn.close()
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        logger.info(f"Retrieved article with id: {article_id}")
        return article
    except Exception as e:
        logger.error(f"An error occurred while retrieving the article: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving article")


# --- MCP Tool: Vector Search Articles ---

@app.post("/vector_search_articles")
def vector_search_articles(query: str, top_k: int = 5):
    """
    Semantic search over articles using embeddings and pgvector.
    """
    logger.info(f"Vector searching articles for query: {query}, top_k: {top_k}")
    try:
        model = get_embedding_model()
        query_emb = model.encode([query])[0].tolist()
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # pgvector: '<->' is the Euclidean distance operator
        cur.execute(
            """
            SELECT id, content, metadata, embedding, (embedding <-> %s) AS distance
            FROM articles
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (query_emb, query_emb, top_k)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()
        # Log retrieval for feedback loop
        log_feedback("vector_search", {"query": query, "top_k": top_k, "results": [r["id"] for r in results]})
        return results
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail="Error in vector search")


# --- MCP Tool: Log Training Example ---
@app.post("/log_training_example")
def log_training_example(task: str, input: dict, output: dict, critique: str):
    """
    Log a new training example for feedback and model improvement.
    Stores in DB and logs to feedback file for continual learning.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO training_examples (task, input, output, critique, created_at) VALUES (%s, %s, %s, %s, %s)",
            (task, input, output, critique, datetime.utcnow())
        )
        conn.commit()
        cur.close()
        conn.close()
        log_feedback("log_training_example", {"task": task, "input": input, "output": output, "critique": critique})
        logger.info(f"Logged training example for task: {task}")
        return {"status": "logged", "task": task}
    except Exception as e:
        logger.error(f"Error logging training example: {e}")
        raise HTTPException(status_code=500, detail="Error logging training example")

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")
