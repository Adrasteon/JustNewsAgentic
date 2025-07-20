
# main.py for Chief Editor Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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