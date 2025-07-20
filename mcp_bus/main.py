
# main.py for MCP Message Bus
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

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
    print(f"Registering agent: {agent.name} at {agent.address}")
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
