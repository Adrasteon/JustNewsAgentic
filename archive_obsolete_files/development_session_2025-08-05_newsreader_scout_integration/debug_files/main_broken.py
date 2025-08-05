"""
Main file for the Chief Editor Agent.
""        # Register with the MCP Bus
        mcp_bus_client.register_agent(
            agent_name="chief_editor",
            agent_address=f"http://localhost:{CHIEF_EDITOR_AGENT_PORT}",
            tools=["coordinate_editorial_workflow", "manage_content_lifecycle"]
        )ain.py for Chief Editor Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
CHIEF_EDITOR_AGENT_PORT = int(os.environ.get("CHIEF_EDITOR_AGENT_PORT", 8001))
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
            response = requests.post(f"{self.base_url}/register", json=registration_data)
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Chief Editor agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="chief_editor",
            agent_address=f"http://localhost:{CHIEF_EDITOR_AGENT_PORT}",
            tools=["request_story_brief", "publish_story"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    yield
    # Shutdown logic
    logger.info("Chief Editor agent is shutting down.")

# Initialize FastAPI with the lifespan context manager
app = FastAPI(lifespan=lifespan)

# Add health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Define ToolCall model
class ToolCall(BaseModel):
    args: list
    kwargs: dict

# Import tools functions
from tools import request_story_brief as request_story_brief_tool, publish_story as publish_story_tool

# Adjust endpoints for tools
@app.post("/request_story_brief")
def request_story_brief(tool_call: ToolCall):
    try:
        topic = tool_call.kwargs.get("topic", tool_call.args[0] if tool_call.args else "")
        scope = tool_call.kwargs.get("scope", tool_call.args[1] if len(tool_call.args) > 1 else "general")
        brief = request_story_brief_tool(topic, scope)
        return {"status": "success", "brief": brief}
    except Exception as e:
        logger.error(f"Error in request_story_brief: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/publish_story")
def publish_story(tool_call: ToolCall):
    try:
        story_id = tool_call.kwargs.get("story_id", tool_call.args[0] if tool_call.args else "")
        result = publish_story_tool(story_id)
        return result
    except Exception as e:
        logger.error(f"Error in publish_story: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add feedback logging for Chief Editor Agent
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