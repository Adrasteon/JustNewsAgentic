"""
Main file for the Chief Editor Agent.
"""
# main.py for Chief Editor Agent
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ready = False

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
            response = requests.post(f"{self.base_url}/register", json=registration_data, timeout=(2, 5))
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    global ready
    ready = True
    yield
    
    logger.info("Chief Editor agent is shutting down.")

# Initialize FastAPI with the lifespan context manager
app = FastAPI(title="Chief Editor Agent", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

# Pydantic models
class ToolCall(BaseModel):
    args: list
    kwargs: dict

class StoryBrief(BaseModel):
    topic: str
    deadline: str
    priority: str

@app.post("/request_story_brief")
def request_story_brief(call: ToolCall):
    """Request a story brief from another agent"""
    try:
        # Implementation for requesting story briefs
        logger.info("Requesting story brief")
        return {"status": "success", "message": "Story brief requested"}
    except Exception as e:
        logger.error(f"Error requesting story brief: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/publish_story")
def publish_story(call: ToolCall):
    """Publish a finalized story"""
    try:
        # Implementation for publishing stories
        logger.info("Publishing story")
        return {"status": "success", "message": "Story published"}
    except Exception as e:
        logger.error(f"Error publishing story: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/coordinate_editorial_workflow")
def coordinate_editorial_workflow(call: ToolCall):
    """Coordinate the editorial workflow between agents"""
    try:
        # Implementation for coordinating workflow
        logger.info("Coordinating editorial workflow")
        return {"status": "success", "message": "Workflow coordinated"}
    except Exception as e:
        logger.error(f"Error coordinating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manage_content_lifecycle")
def manage_content_lifecycle(call: ToolCall):
    """Manage the lifecycle of content through the system"""
    try:
        # Implementation for managing content lifecycle
        logger.info("Managing content lifecycle")
        return {"status": "success", "message": "Content lifecycle managed"}
    except Exception as e:
        logger.error(f"Error managing content lifecycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=CHIEF_EDITOR_AGENT_PORT)
