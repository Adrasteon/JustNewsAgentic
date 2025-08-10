"""
Main file for the MCP Bus.
"""
# main.py for MCP Message Bus
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import requests
import time
import atexit
import logging
from contextlib import asynccontextmanager

app = FastAPI()
ready = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agents = {}
cb_state = {}
CB_FAIL_THRESHOLD = 3
CB_COOLDOWN_SEC = 10

# Simple in-process metrics (Prometheus text format)
metrics = {
    "mcp_agents_registered": 0,
    "mcp_requests_total": 0,
    "mcp_errors_total": 0,
}

class Agent(BaseModel):
    name: str
    address: str

class ToolCall(BaseModel):
    agent: str
    tool: str
    args: list
    kwargs: dict

@app.post("/register")
def register_agent(agent: Agent):
    logger.info(f"Registering agent: {agent.name} at {agent.address}")
    agents[agent.name] = agent.address
    # Reset circuit breaker on registration
    cb_state[agent.name] = {"fails": 0, "open_until": 0}
    metrics["mcp_agents_registered"] += 1
    return {"status": "ok"}

@app.post("/call")
def call_tool(call: ToolCall):
    if call.agent not in agents:
        raise HTTPException(status_code=404, detail=f"Agent not found: {call.agent}")
    
    agent_name = call.agent
    agent_address = agents[agent_name]

    # Circuit breaker check
    state = cb_state.get(agent_name, {"fails": 0, "open_until": 0})
    now = time.time()
    if state.get("open_until", 0) > now:
        raise HTTPException(status_code=503, detail=f"Circuit open for agent {agent_name}")

    payload = {"args": call.args, "kwargs": call.kwargs}
    url = f"{agent_address}/{call.tool}"
    timeout = (3, 10)  # (connect, read)

    # Simple retry with backoff
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            # Success: reset failures
            cb_state[agent_name] = {"fails": 0, "open_until": 0}
            metrics["mcp_requests_total"] += 1
            return response.json()
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            time.sleep(0.2 * (2 ** attempt))

    # Failure after retries: increment failure count
    fails = state.get("fails", 0) + 1
    if fails >= CB_FAIL_THRESHOLD:
        cb_state[agent_name] = {"fails": 0, "open_until": now + CB_COOLDOWN_SEC}
        logger.warning(f"Circuit opened for {agent_name} for {CB_COOLDOWN_SEC}s after failures")
    else:
        cb_state[agent_name] = {"fails": fails, "open_until": 0}

    metrics["mcp_errors_total"] += 1
    raise HTTPException(status_code=502, detail=f"Tool call failed: {last_error}")

@app.get("/agents")
def get_agents():
    return agents
    

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}


@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Return Prometheus-style metrics in text/plain"""
    # Render minimal text exposition
    lines = [
        f"mcp_agents_registered {metrics['mcp_agents_registered']}",
        f"mcp_requests_total {metrics['mcp_requests_total']}",
        f"mcp_errors_total {metrics['mcp_errors_total']}",
    ]
    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4")

@asynccontextmanager
async def lifespan(app):
    logger.info("MCP_Bus is starting up.")
    try:
        response = requests.get("http://localhost:8000/register", timeout=(2, 5))
        response.raise_for_status()
        logger.info("MCP Bus connected successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to MCP Bus: {e}")
    global ready
    ready = True
    yield
    logger.info("MCP_Bus is shutting down.")

atexit.register(lambda: logger.info("MCP_Bus has exited."))
