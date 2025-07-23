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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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
            "tool": call.kwargs.get("tool"),
            "args": call.args,
            "outcome": call.kwargs.get("outcome"),
            "timestamp": datetime.now().isoformat()
        }
        with open(os.environ.get("CRITIC_FEEDBACK_LOG", "./feedback_critic.log"), "a") as log_file:
            log_file.write(f"{feedback_data}\n")
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"An error occurred in log_feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def check_and_download_model(model_path, download_url):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        response = requests.get(download_url, stream=True)
        with open(model_path, 'wb') as model_file:
            for chunk in response.iter_content(chunk_size=8192):
                model_file.write(chunk)
        print(f"Model downloaded to {model_path}.")

# Example usage
CRITIC_MODEL_PATH = './models/critic-model'
CRITIC_MODEL_URL = 'https://example.com/path/to/critic-model'
check_and_download_model(CRITIC_MODEL_PATH, CRITIC_MODEL_URL)

try:
    # Attempt to register tools with MCP Bus
    mcp_url = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
    response = requests.post(f"{mcp_url}/register", json={"agent": "critic", "tools": ["critique_synthesis", "critique_neutrality"]})
    response.raise_for_status()
    logger.info("Registered tools with MCP Bus.")
except Exception as e:
    logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")