"""
Main file for the Analyst Agent.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from hybrid_tools_v4 import (
    identify_entities,
    log_feedback,
    score_bias,
    score_sentiment,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
ANALYST_AGENT_PORT = int(os.environ.get("ANALYST_AGENT_PORT", 8004))
MODEL_PATH = os.environ.get("MISTRAL_7B_PATH", "./models/mistral-7b-instruct-v0.2")
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://mcp_bus:8000")

# Pydantic models
class ToolCall(BaseModel):
    args: list
    kwargs: dict

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    logger.info("Analyst agent is starting up.")
    # Note: Model will be loaded lazily when first needed
    logger.info("Model loading deferred to first use.")

    # Register agent with MCP Bus
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="analyst",
            agent_address=f"http://analyst:{ANALYST_AGENT_PORT}",
            tools=["score_bias", "score_sentiment", "identify_entities"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")

    yield

    logger.info("Analyst agent is shutting down.")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/score_bias")
def score_bias_endpoint(call: ToolCall):
    """Scores the bias of a given text."""
    try:
        return score_bias(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in score_bias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score_sentiment")
def score_sentiment_endpoint(call: ToolCall):
    """Scores the sentiment of a given text."""
    try:
        return score_sentiment(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in score_sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify_entities")
def identify_entities_endpoint(call: ToolCall):
    """Identifies entities in a given text."""
    try:
        return identify_entities(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in identify_entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_feedback")
def log_feedback_endpoint(call: ToolCall):
    """Logs feedback."""
    try:
        feedback = call.kwargs.get("feedback", {})
        log_feedback("log_feedback", feedback)
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"An error occurred while logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))