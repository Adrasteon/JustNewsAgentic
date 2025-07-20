
# main.py for Critic Agent
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