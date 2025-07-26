"""
Main file for the Synthesizer Agent.
"""
# main.py for Synthesizer Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from datetime import datetime
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SYNTHESIZER_AGENT_PORT = int(os.environ.get("SYNTHESIZER_AGENT_PORT", 8005))
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
    logger.info("Synthesizer agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="synthesizer",
            agent_address=f"http://synthesizer:{SYNTHESIZER_AGENT_PORT}",
            tools=["cluster_articles", "neutralize_text", "aggregate_cluster"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")

    # Model downloading (if applicable)
    SYNTHESIZER_MODEL_PATH = './models/synthesizer-model'
    SYNTHESIZER_MODEL_URL = 'https://example.com/path/to/synthesizer-model'
    if not os.path.exists(SYNTHESIZER_MODEL_PATH):
        logger.info(f"Model not found at {SYNTHESIZER_MODEL_PATH}. Downloading...")
        try:
            response = requests.get(SYNTHESIZER_MODEL_URL, stream=True)
            response.raise_for_status()
            with open(SYNTHESIZER_MODEL_PATH, 'wb') as model_file:
                for chunk in response.iter_content(chunk_size=8192):
                    model_file.write(chunk)
            logger.info(f"Model downloaded to {SYNTHESIZER_MODEL_PATH}.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")

    yield
    logger.info("Synthesizer agent is shutting down.")

app = FastAPI(lifespan=lifespan)

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.post("/aggregate_cluster")
def aggregate_cluster_endpoint(call: ToolCall):
    try:
        from tools import aggregate_cluster
        logger.info(f"Calling aggregate_cluster with args: {call.args} and kwargs: {call.kwargs}")
        return aggregate_cluster(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in aggregate_cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster_articles")
def cluster_articles_endpoint(call: ToolCall):
    try:
        from tools import cluster_articles
        logger.info(f"Calling cluster_articles with args: {call.args} and kwargs: {call.kwargs}")
        return cluster_articles(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in cluster_articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neutralize_text")
def neutralize_text_endpoint(call: ToolCall):
    try:
        from tools import neutralize_text
        logger.info(f"Calling neutralize_text with args: {call.args} and kwargs: {call.kwargs}")
        return neutralize_text(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in neutralize_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))