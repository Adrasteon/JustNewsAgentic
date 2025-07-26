"""
Main file for the MCP Bus.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import requests
import uvicorn
import threading
import os
import time
import logging

app = FastAPI()

# Store registered agents
agents = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory store for agent registrations
registered_agents = {}

AGENT_URLS = {
    "chief_editor": "http://chief_editor:8001",
    "scout": "http://scout:8002",
    "fact_checker": "http://fact_checker:8003",
    "analyst": "http://analyst:8004",
    "synthesizer": "http://synthesizer:8005",
    "critic": "http://critic:8006",
    "memory": "http://memory:8007",
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/agents")
def get_agents():
    """Get all registered agents and their details."""
    return agents

class ToolCallRequest(BaseModel):
    agent: str
    tool: str
    args: list = []
    kwargs: dict = {}

@app.post("/call")
def call_tool(request: ToolCallRequest):
    """Call a tool on a specific agent."""
    if request.agent not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent} not found")
    
    agent_info = agents[request.agent]
    agent_address = agent_info["address"]
    
    # Check if tool exists for this agent
    if request.tool not in [tool["name"] for tool in agent_info["tools"]]:
        raise HTTPException(status_code=404, detail=f"Tool {request.tool} not found for agent {request.agent}")
    
    # Call the tool endpoint on the agent
    try:
        tool_payload = {
            "args": request.args,
            "kwargs": request.kwargs
        }
        resp = requests.post(f"{agent_address}/{request.tool}", json=tool_payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Call to {request.agent}.{request.tool} failed: {e}")

class AgentRegistration(BaseModel):
    agent_name: str
    agent_address: str
    tools: list

@app.post("/register")
def register_agent(registration: AgentRegistration):
    agents[registration.agent_name] = {
        "address": registration.agent_address,
        "tools": registration.tools,
    }
    logger.info(f"Agent {registration.agent_name} registered with address {registration.agent_address} and tools {registration.tools}")
    return {"message": "Agent registered successfully"}

# Example endpoint: relay a message to another agent (stub for now)
class RelayRequest(BaseModel):
    target_agent: str
    endpoint: str
    payload: Dict[str, Any]

@app.post("/relay")
def relay_message(request: RelayRequest):
    # In a real implementation, this would forward the payload to the target agent
    # and return the response. For now, just echo the request.
        target_url = AGENT_URLS.get(request.target_agent)
        if not target_url:
            raise HTTPException(status_code=404, detail=f"Unknown target agent: {request.target_agent}")
        url = f"{target_url}{request.endpoint}"
        try:
            resp = requests.post(url, json=request.payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Relay to {url} failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
