
from typing import List
# main.py for Memory Agent
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

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
    # Placeholder: In production, use pgvector or similar for semantic search
    logger.info(f"Vector searching articles for query: {query}, top_k: {top_k}")
    # Return dummy results
    return [{"id": i, "score": 1.0 - i * 0.1} for i in range(1, top_k + 1)]

# --- MCP Tool: Log Training Example ---
@app.post("/log_training_example")
def log_training_example(task: str, input: dict, output: dict, critique: str):
    # Placeholder: In production, log to a training examples table
    logger.info(f"Logging training example for task: {task}")
    return {"status": "logged", "task": task}
