"""
Main file for the Critic Agent.
"""
# main.py for Critic Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
import requests
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
CRITIC_AGENT_PORT = int(os.environ.get("CRITIC_AGENT_PORT", 8002))
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Critic agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="critic",
            agent_address=f"http://critic:{CRITIC_AGENT_PORT}",
            tools=["critique_synthesis", "critique_neutrality"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    yield
    logger.info("Critic agent is shutting down.")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.post("/critique_synthesis")
def critique_synthesis(call: ToolCall):
    try:
        from tools import critique_synthesis
        logger.info(f"Calling critique_synthesis with args: {call.args} and kwargs: {call.kwargs}")
        return critique_synthesis(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in critique_synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/critique_neutrality")
def critique_neutrality(call: ToolCall):
    try:
        from tools import critique_neutrality
        logger.info(f"Calling critique_neutrality with args: {call.args} and kwargs: {call.kwargs}")
        return critique_neutrality(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in critique_neutrality: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add feedback logging for Critic Agent
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