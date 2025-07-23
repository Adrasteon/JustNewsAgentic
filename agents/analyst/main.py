"""
Main file for the Analyst Agent.
"""
# main.py for Analyst Agent
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

@app.post("/score_bias")
def score_bias(call: ToolCall):
    try:
        from tools import score_bias
        logger.info(f"Calling score_bias with args: {call.args} and kwargs: {call.kwargs}")
        return score_bias(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in score_bias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score_sentiment")
def score_sentiment(call: ToolCall):
    try:
        from tools import score_sentiment
        logger.info(f"Calling score_sentiment with args: {call.args} and kwargs: {call.kwargs}")
        return score_sentiment(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in score_sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify_entities")
def identify_entities(call: ToolCall):
    try:
        from tools import identify_entities
        logger.info(f"Calling identify_entities with args: {call.args} and kwargs: {call.kwargs}")
        return identify_entities(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in identify_entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add feedback logging for Analyst Agent
@app.post("/log_feedback")
def log_feedback(call: ToolCall):
    try:
        feedback_data = {
            "tool": call.kwargs.get("tool"),
            "args": call.args,
            "outcome": call.kwargs.get("outcome"),
            "timestamp": datetime.now().isoformat()
        }
        with open(os.environ.get("ANALYST_FEEDBACK_LOG", "./feedback_analyst.log"), "a") as log_file:
            log_file.write(f"{feedback_data}\n")
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"An error occurred in log_feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

try:
    # Attempt to register tools with MCP Bus
    mcp_url = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
    response = requests.post(f"{mcp_url}/register", json={"agent": "analyst", "tools": ["score_bias", "score_sentiment", "identify_entities"]})
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
MISTRAL_7B_PATH = './models/mistral-7b-instruct-v0.2'
MISTRAL_7B_URL = 'https://example.com/path/to/mistral-7b-model'
check_and_download_model(MISTRAL_7B_PATH, MISTRAL_7B_URL)