"""
Main file for the Memory Agent.
"""

import json
import logging
import os
from contextlib import asynccontextmanager

import psycopg2
import requests
from fastapi import FastAPI, HTTPException
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from agents.memory.tools import (
    log_training_example,
    save_article,
    vector_search_articles,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Readiness flag
ready = False

# Environment variables
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
MEMORY_AGENT_PORT = int(os.environ.get("MEMORY_AGENT_PORT", 8007))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")


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


class ToolCall(BaseModel):
    args: list
    kwargs: dict


class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(
                f"{self.base_url}/register", json=registration_data, timeout=(2, 5)
            )
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise


# Use connection pooling for database connections
def get_db_connection():
    """Establishes a connection to the PostgreSQL database with pooling."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            options="-c search_path=public",
        )
        logger.info("Database connection established successfully.")
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
            agent_address=f"http://localhost:{MEMORY_AGENT_PORT}",
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
    # Mark ready after startup and optional registration
    global ready
    ready = True
    yield
    logger.info("Memory agent is shutting down.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/ready")
def ready_endpoint():
    """Readiness endpoint for startup gating."""
    return {"ready": ready}


@app.post("/save_article")
def save_article_endpoint(request: dict):
    """Saves an article to the database. Handles both direct calls and MCP Bus format."""
    try:
        # Handle MCP Bus format: {"args": [...], "kwargs": {...}}
        if "args" in request and "kwargs" in request:
            if len(request["args"]) > 0:
                article_data = request["args"][0]
            else:
                article_data = request["kwargs"]
        else:
            # Direct call format
            article_data = request

        # Create Article object from the data
        article = Article(**article_data)
        return save_article(article.content, article.metadata)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving article: {str(e)}")


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
def vector_search_articles_endpoint(request: dict):
    """Performs a vector search for articles. Handles both direct calls and MCP Bus format."""
    try:
        # Handle MCP Bus format: {"args": [...], "kwargs": {...}}
        if "args" in request and "kwargs" in request:
            if len(request["args"]) > 0:
                search_data = request["args"][0]
            else:
                search_data = request["kwargs"]
        else:
            # Direct call format
            search_data = request

        # Create VectorSearch object from the data
        search = VectorSearch(**search_data)
        return vector_search_articles(search.query, search.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error searching articles: {str(e)}"
        )


@app.post("/log_training_example")
def log_training_example_endpoint(example: TrainingExample):
    """Logs a training example to the database."""
    return log_training_example(
        example.task, example.input, example.output, example.critique
    )


# Improved error handling and logging
@app.post("/store_article")
def store_article_endpoint(call: ToolCall):
    """Stores an article in the database."""
    try:
        article_data = call.kwargs
        if not article_data.get("content") or not article_data.get("metadata"):
            raise ValueError("Missing required fields: 'content' or 'metadata'")

        # Serialize metadata to JSON string
        metadata_json = json.dumps(article_data["metadata"])

        # Save article with serialized metadata
        result = save_article(article_data["content"], metadata_json)
        logger.info(f"Article stored successfully: {result}")
        return result
    except ValueError as ve:
        logger.warning(f"Validation error in store_article: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"An error occurred in store_article: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
