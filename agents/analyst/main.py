"""
Main file for the Analyst Agent.
"""
# main.py for Analyst Agent
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