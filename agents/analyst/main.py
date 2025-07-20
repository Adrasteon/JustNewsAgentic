
# main.py for Analyst Agent
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