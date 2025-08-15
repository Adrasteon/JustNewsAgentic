"""
Main file for the Dashboard Agent.
"""

import logging
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config, save_config

# Configure logging
logging.basicConfig(
    filename="dashboard_agent.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("dashboard_agent")

# Load configuration
config = load_config()
DASHBOARD_AGENT_PORT = config.get("dashboard_port", 8010)
MCP_BUS_URL = config.get("mcp_bus_url", "http://localhost:8000")


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
                f"{self.base_url}/register", json=registration_data
            )
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Failed to register {agent_name} with MCP Bus: {e}", exc_info=True
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="dashboard",
            agent_address=f"http://localhost:{DASHBOARD_AGENT_PORT}",
            tools=["get_status", "send_command", "receive_logs"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(
            f"MCP Bus unavailable: {e}. Running in standalone mode.", exc_info=True
        )
    yield
    logger.info("Dashboard agent is shutting down.")
    save_config(config)

    if __name__ == "__main__":
        import os

        port = int(
            os.environ.get("DASHBOARD_AGENT_PORT", config.get("dashboard_port", 8011))
        )
        uvicorn.run(
            "agents.dashboard.main:app", host="0.0.0.0", port=port, reload=False
        )


app = FastAPI(lifespan=lifespan)


# Health endpoint for dashboard agent status
@app.get("/health")
def health():
    """Health check endpoint for dashboard agent."""
    return {"status": "ok"}


class ToolCall(BaseModel):
    args: list
    kwargs: dict


@app.get("/get_status")
def get_status():
    """Fetch the status of all agents."""
    try:
        response = requests.get(f"{MCP_BUS_URL}/agents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(
            f"An error occurred while fetching agent status: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send_command")
def send_command(call: ToolCall):
    """Send a command to another agent."""
    try:
        response = requests.post(f"{MCP_BUS_URL}/call", json=call.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred while sending a command: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
