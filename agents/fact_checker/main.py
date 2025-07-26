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

# Environment variables
FACT_CHECKER_AGENT_PORT = int(os.environ.get("FACT_CHECKER_AGENT_PORT", 8003))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://mcp_bus:8000")

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

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fact Checker agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="fact_checker",
            agent_address=f"http://fact_checker:{FACT_CHECKER_AGENT_PORT}",
            tools=["verify_facts", "validate_sources"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    yield
    logger.info("Fact Checker agent is shutting down.")

# Initialize FastAPI with the lifespan context manager
app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

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