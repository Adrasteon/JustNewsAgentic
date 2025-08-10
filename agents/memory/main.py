"""
Main file for the Memory Agent.
"""
import json
import logging
import os
import requests
from contextlib import asynccontextmanager
from datetime import datetime

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from fastapi import FastAPI, HTTPException, Response, Header
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

from .tools import (
    get_embedding_model,
    log_feedback,
    log_training_example,
    save_article,
    vector_search_articles,
)
from common.observability import MetricsCollector, request_timing_middleware
from common.tracing import init_tracing, add_tracing_middleware
from common.security import get_service_headers, require_service_token, HEADER_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Readiness flag and metrics
ready = False
metrics = {
    "warmups_total": 0,
    "db_health_checks_total": 0,
    "db_last_available": 0,
}

# Environment variables
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
MEMORY_AGENT_PORT = int(os.environ.get("MEMORY_AGENT_PORT", 8007))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

# Optional connection pool (initialized on first use)
DB_POOL: SimpleConnectionPool | None = None

def release_connection(conn) -> None:
    """Release a DB connection back to the pool or close it."""
    try:
        if DB_POOL is not None:
            DB_POOL.putconn(conn)
        else:
            conn.close()
    except Exception:
        pass
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
                f"{self.base_url}/register",
                json=registration_data,
                headers=get_service_headers(),
                timeout=(2, 5),
            )
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

# Use connection pooling for database connections
def _init_pool(minconn: int = 1, maxconn: int = 5) -> None:
    """Initialize a global PostgreSQL connection pool if possible."""
    global DB_POOL
    if DB_POOL is not None:
        return
    try:
        DB_POOL = SimpleConnectionPool(
            minconn,
            maxconn,
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=3,
            options='-c search_path=public',
        )
        logger.info("PostgreSQL connection pool initialized")
    except Exception as e:
        logger.warning(f"Could not initialize DB pool: {e}")
        DB_POOL = None


def get_db_connection():
    """Establishes a connection to the PostgreSQL database with pooling."""
    # Prefer pool if available
    global DB_POOL
    if DB_POOL is None:
        _init_pool()
    if DB_POOL is not None:
        try:
            return DB_POOL.getconn()
        except Exception as e:
            logger.warning(f"DB pool unavailable, falling back to direct connect: {e}")
            DB_POOL = None
    # Fallback direct connection
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=3,
            options='-c search_path=public',
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

# Observability: request metrics
collector = MetricsCollector(agent="memory")
request_timing_middleware(app, collector)
if init_tracing("memory"):
    add_tracing_middleware(app, "memory")

@app.get("/health")
def health(x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
    """Health check endpoint."""
    # Optional enforcement: only if token configured
    try:
        require_service_token(x_service_token)
    except HTTPException as e:
        # For health, allow anonymous if token not set; if set and invalid, still expose 200 with status
        logger.warning(f"Service token validation failed for health check: {e.detail}")
        pass
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint(x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
    """Readiness endpoint for startup gating."""
    try:
        require_service_token(x_service_token)
    except HTTPException:
        pass
    # Vector-store readiness: check DB connectivity and vector extension
    db_ok = False
    vector_ok = False
    conn = None
    try:
        conn = get_db_connection()
        db_ok = True
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector');")
            vector_ok = bool(cur.fetchone()[0])
    except Exception:
        db_ok = False
        vector_ok = False
    finally:
        if conn:
            try:
                release_connection(conn)
            except Exception:
                pass
    return {"ready": ready and db_ok and vector_ok, "db_ok": db_ok, "vector_ok": vector_ok}

@app.post("/warmup")
def warmup(x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
    """Minimal warmup to touch lazy paths without heavy DB ops."""
    try:
        require_service_token(x_service_token)
    except HTTPException:
        pass
    try:
        # Light imports only
        from .tools import get_embedding_model  # noqa: F401
    except Exception:
        pass
    metrics["warmups_total"] += 1
    return {"warmed": True}

@app.get("/metrics")
def metrics_endpoint(x_service_token: str | None = Header(default=None, alias=HEADER_NAME)) -> Response:
    """Prometheus-style metrics for the Memory agent.

    Exposes counters and lightweight gauges, including DB pool status if available.
    """
    pool_in_use = 0
    pool_available = 0
    try:
        if DB_POOL is not None:
            # psycopg2 SimpleConnectionPool doesn't expose public counters; use internals safely
            pool_in_use = len(getattr(DB_POOL, "_used", []))
            pool_available = len(getattr(DB_POOL, "_pool", []))
    except Exception:
        # Best-effort: keep zeros if inspection fails
        pass

    body = (
        f"memory_warmups_total {metrics['warmups_total']}\n"
        f"memory_db_health_checks_total {metrics['db_health_checks_total']}\n"
        f"memory_db_last_available {metrics['db_last_available']}\n"
        f"memory_db_pool_in_use {pool_in_use}\n"
        f"memory_db_pool_available {pool_available}\n"
    )
    # Append request metrics from collector
    body += collector.render()
    return Response(content=body, media_type="text/plain; version=0.0.4")

@app.get("/db/health")
def db_health(x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
    """Database health probe. Always 200 with availability status.

    Returns:
        {"available": bool, "version": Optional[str], "error": Optional[str]}
    """
    metrics["db_health_checks_total"] += 1
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
        metrics["db_last_available"] = 1
        return {"available": True, "version": version}
    except Exception as e:
        metrics["db_last_available"] = 0
        # Do not fail the endpoint to keep orchestration simple
        return {"available": False, "error": str(e)}
    finally:
        if conn:
            try:
                release_connection(conn)
            except Exception:
                pass

@app.post("/db/init")
def db_init(schema_path: str | None = None, x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
    """Initialize the PostgreSQL schema by executing the provided SQL file.

    Args:
        schema_path: Optional path to a .sql file. If not provided, defaults to
                     scripts/init_postgres_schema.sql relative to repo dir.
    """
    # Resolve default path
    if not schema_path:
        repo_dir = os.environ.get("REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
        schema_path = os.path.join(repo_dir, "scripts", "init_postgres_schema.sql")

    if not os.path.exists(schema_path):
        raise HTTPException(status_code=400, detail=f"Schema file not found: {schema_path}")

    sql_text = ""
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            sql_text = f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read schema file: {e}")

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql_text)
        conn.commit()
        return {"status": "ok", "applied": True, "path": schema_path}
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Schema apply failed: {e}")
    finally:
        if conn:
            release_connection(conn)

@app.post("/save_article")
def save_article_endpoint(request: dict, x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
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
def get_article_endpoint(article_id: int, x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
    """Retrieves an article from the database."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
            article = cur.fetchone()
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            return article
    finally:
        if conn:
            release_connection(conn)

@app.post("/vector_search_articles")
def vector_search_articles_endpoint(request: dict, x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
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
        raise HTTPException(status_code=400, detail=f"Error searching articles: {str(e)}")

@app.post("/log_training_example")
def log_training_example_endpoint(example: TrainingExample, x_service_token: str | None = Header(default=None, alias=HEADER_NAME)):
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
