"""
Main file for the Chief Editor Agent.
"""
# main.py for Chief Editor Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from datetime import datetime

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

@app.post("/request_story_brief")
def request_story_brief(call: ToolCall):
    try:
        from tools import request_story_brief
        logger.info(f"Calling request_story_brief with args: {call.args} and kwargs: {call.kwargs}")
        return request_story_brief(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in request_story_brief: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/publish_story")
def publish_story(call: ToolCall):
    try:
        from tools import publish_story
        logger.info(f"Calling publish_story with args: {call.args} and kwargs: {call.kwargs}")
        return publish_story(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in publish_story: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add feedback logging for Chief Editor Agent
@app.post("/log_feedback")
def log_feedback(call: ToolCall):
    try:
        feedback_data = {
            "tool": call.kwargs.get("tool"),
            "args": call.args,
            "outcome": call.kwargs.get("outcome"),
            "timestamp": datetime.now().isoformat()
        }
        with open(os.environ.get("CHIEF_EDITOR_FEEDBACK_LOG", "./feedback_chief_editor.log"), "a") as log_file:
            log_file.write(f"{feedback_data}\n")
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"An error occurred in log_feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

try:
    # Attempt to register tools with MCP Bus
    mcp_url = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
    response = requests.post(f"{mcp_url}/register", json={"agent": "chief_editor", "tools": ["request_story_brief", "publish_story"]})
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
CHIEF_EDITOR_MODEL_PATH = './models/chief-editor-model'
CHIEF_EDITOR_MODEL_URL = 'https://example.com/path/to/chief-editor-model'
check_and_download_model(CHIEF_EDITOR_MODEL_PATH, CHIEF_EDITOR_MODEL_URL)