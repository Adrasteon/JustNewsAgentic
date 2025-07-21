"""
Main file for the Chief Editor Agent.
"""
# main.py for Chief Editor Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os

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