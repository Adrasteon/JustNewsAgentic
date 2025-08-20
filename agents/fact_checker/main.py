"""
Main file for the Fact-Checker Agent.
"""
# main.py for Fact-Checker Agent
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from datetime import datetime
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ready = False

# Environment variables
FACT_CHECKER_AGENT_PORT = int(os.environ.get("FACT_CHECKER_AGENT_PORT", 8003))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data, timeout=(2, 5))
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fact Checker agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="fact_checker",
            agent_address=f"http://localhost:{FACT_CHECKER_AGENT_PORT}",
            tools=["verify_facts", "validate_sources", "validate_is_news_gpu", "verify_claims_gpu"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    global ready
    ready = True
    yield
    logger.info("Fact Checker agent is shutting down.")

# Initialize FastAPI with the lifespan context manager
app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.post("/validate_is_news")
def validate_is_news(call: ToolCall):
    try:
        from tools import validate_is_news
        logger.info(f"Calling validate_is_news with args: {call.args} and kwargs: {call.kwargs}")
        return validate_is_news(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in validate_is_news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_claims")
def verify_claims(call: ToolCall):
    try:
        from tools import verify_claims
        logger.info(f"Calling verify_claims with args: {call.args} and kwargs: {call.kwargs}")
        return verify_claims(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in verify_claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_claims")
def validate_claims_endpoint(request: dict):
    """Validates claims in the provided content."""
    try:
        # Handle MCP Bus format: {"args": [...], "kwargs": {...}}
        if "args" in request and len(request["args"]) > 0:
            content = request["args"][0]
        elif "kwargs" in request and "content" in request["kwargs"]:
            content = request["kwargs"]["content"]
            logger.debug(f"validate_claims received content of length {len(content)}")
        else:
            raise ValueError("Missing 'content' in request")

        # Perform validation logic (mocked for now)
        validation_score = 0.85  # Example score
        return {"validation_score": validation_score}
    except ValueError as ve:
        logger.warning(f"Validation error in validate_claims: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"An error occurred in validate_claims: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# GPU-Enhanced Endpoints (New V4 Performance)
@app.post("/validate_is_news_gpu")
def validate_is_news_gpu(call: ToolCall):
    try:
        from gpu_tools import validate_is_news_detailed
        logger.info(f"Calling GPU validate_is_news with args: {call.args} and kwargs: {call.kwargs}")
        return validate_is_news_detailed(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in GPU validate_is_news: {e}")
        # Fallback to original implementation
        from tools import validate_is_news
        return validate_is_news(*call.args, **call.kwargs)

@app.post("/verify_claims_gpu")
def verify_claims_gpu(call: ToolCall):
    try:
        from gpu_tools import verify_claims_detailed
        logger.info(f"Calling GPU verify_claims with args: {call.args} and kwargs: {call.kwargs}")
        return verify_claims_detailed(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in GPU verify_claims: {e}")
        # Fallback to original implementation
        from tools import verify_claims
        return verify_claims(*call.args, **call.kwargs)

@app.get("/performance/stats")
def get_performance_stats():
    """Get GPU acceleration performance statistics"""
    try:
        from gpu_tools import get_fact_checker_performance
        return get_fact_checker_performance()
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return {"error": str(e), "gpu_available": False}

@app.post("/log_feedback")
def log_feedback(call: ToolCall):
    try:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "feedback": call.kwargs.get("feedback")
        }
        logger.info(f"Logging feedback: {feedback_data}")
        return feedback_data
    except Exception as e:
        logger.error(f"An error occurred while logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))