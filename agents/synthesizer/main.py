"""
Main file for the Synthesizer Agent.
"""
# main.py for Synthesizer Agent

# main.py for Synthesizer Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/aggregate_cluster")
def aggregate_cluster_endpoint(call: ToolCall):
    try:
        from tools import aggregate_cluster
        logger.info(f"Calling aggregate_cluster with args: {call.args} and kwargs: {call.kwargs}")
        return aggregate_cluster(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in aggregate_cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))

try:
    # Attempt to register tools with MCP Bus
    mcp_url = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
    response = requests.post(f"{mcp_url}/register", json={"agent": "synthesizer", "tools": ["cluster_articles", "neutralize_text", "aggregate_cluster"]})
    response.raise_for_status()
    logger.info("Registered tools with MCP Bus.")
except Exception as e:
    logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")

def check_and_download_model(model_path, download_url):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        response = requests.get(download_url, stream=True)
        with open(model_path, 'wb') as model_file:
            for chunk in response.iter_content(chunk_size=8192):
                model_file.write(chunk)
        print(f"Model downloaded to {model_path}.")

# Example usage
SYNTHESIZER_MODEL_PATH = './models/synthesizer-model'
SYNTHESIZER_MODEL_URL = 'https://example.com/path/to/synthesizer-model'
check_and_download_model(SYNTHESIZER_MODEL_PATH, SYNTHESIZER_MODEL_URL)