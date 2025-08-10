"""
Main file for the Dashboard Agent.
"""

import logging
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import requests
from contextlib import asynccontextmanager
from config import load_config, save_config
from common.observability import MetricsCollector, request_timing_middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
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
            response = requests.post(f"{self.base_url}/register", json=registration_data)
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
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
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    yield
    logger.info("Dashboard agent is shutting down.")
    save_config(config)

app = FastAPI(lifespan=lifespan)

# Observability
collector = MetricsCollector("dashboard")
request_timing_middleware(app, collector)

ready = True
legacy_metrics = {"warmups_total": 0}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

@app.post("/warmup")
def warmup():
    legacy_metrics["warmups_total"] += 1
    collector.inc("warmups_total")
    return {"warmed": True}

@app.get("/metrics")
def metrics_endpoint() -> Response:
    body = collector.render() + f"dashboard_warmups_total {legacy_metrics['warmups_total']}\n"
    return Response(content=body, media_type="text/plain; version=0.0.4")

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
        logger.error(f"An error occurred while fetching agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_command")
def send_command(call: ToolCall):
    """Send a command to another agent."""
    try:
        response = requests.post(f"{MCP_BUS_URL}/call", json=call.dict())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred while sending a command: {e}")
        raise HTTPException(status_code=500, detail=str(e))
