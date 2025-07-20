# main.py for Synthesizer Agent

# main.py for Synthesizer Agent
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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