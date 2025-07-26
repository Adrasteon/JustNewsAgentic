"""
Main file for the Memory Agent.
"""
import logging
import os
import requests
from contextlib import asynccontextmanager
from datetime import datetime

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

from tools import (get_embedding_model, log_feedback,
                   log_training_example, save_article, vector_search_articles)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
MEMORY_AGENT_PORT = int(os.environ.get("MEMORY_AGENT_PORT", 8007))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://mcp_bus:8000")

# Pydantic models
class Article(BaseModel):
    content: str
    metadata: dict

class TrainingExample(BaseModel):
    task: str
    input: dict
    output: dict
    critique: str

class VectorSearch(BaseModel):
    query: str
    top_k: int = 5

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "agent_name": agent_name,
            "agent_address": agent_address,
            "tools": tools,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data)
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to PostgreSQL database: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    logger.info("Memory agent is starting up.")
    # Register agent with MCP Bus
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="memory",
            agent_address=f"http://memory:{MEMORY_AGENT_PORT}",
            tools=[
                "save_article",
                "get_article",
                "vector_search_articles",
                "log_training_example",
            ],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    yield
    logger.info("Memory agent is shutting down.")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/save_article")
def save_article_endpoint(article: Article):
    """Saves an article to the database."""
    return save_article(article.content, article.metadata)

@app.get("/get_article/{article_id}")
def get_article_endpoint(article_id: int):
    """Retrieves an article from the database."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
            article = cur.fetchone()
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            return article
    finally:
        conn.close()

@app.post("/vector_search_articles")
def vector_search_articles_endpoint(search: VectorSearch):
    """Performs a vector search for articles."""
    return vector_search_articles(search.query, search.top_k)

@app.post("/log_training_example")
def log_training_example_endpoint(example: TrainingExample):
    """Logs a training example to the database."""
    return log_training_example(
        example.task, example.input, example.output, example.critique
    )
