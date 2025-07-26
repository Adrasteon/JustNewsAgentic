"""
Main file for the MCP Bus.
"""
# main.py for MCP Message Bus
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import atexit
import logging
from contextlib import asynccontextmanager

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agents = {}

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
    return {"status": "ok"}

@app.post("/call")
def call_tool(call: ToolCall):
    if call.agent not in agents:
        raise HTTPException(status_code=404, detail=f"Agent not found: {call.agent}")
    
    agent_address = agents[call.agent]
    try:
        response = requests.post(f"{agent_address}/{call.tool}", json={"args": call.args, "kwargs": call.kwargs})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
def get_agents():
    return agents
    

@app.get("/health")
def health():
    return {"status": "ok"}

@asynccontextmanager
async def lifespan(app):
    logger.info("MCP_Bus is starting up.")
    try:
        response = requests.get("http://localhost:8000/register")
        response.raise_for_status()
        logger.info("MCP Bus connected successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to MCP Bus: {e}")
    yield
    logger.info("MCP_Bus is shutting down.")

atexit.register(lambda: logger.info("MCP_Bus has exited."))
