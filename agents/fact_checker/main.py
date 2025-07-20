
# main.py for Fact-Checker Agent
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

@app.post("/validate_is_news")
def validate_is_news(call: ToolCall):
    try:
        from tools import validate_is_news
        logger.info(f"Calling validate_is_news with args: {call.args} and kwargs: {call.kwargs}")
        return validate_is_news(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in validate_is_news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_claims")
def verify_claims(call: ToolCall):
    try:
        from tools import verify_claims
        logger.info(f"Calling verify_claims with args: {call.args} and kwargs: {call.kwargs}")
        return verify_claims(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in verify_claims: {e}")
        raise HTTPException(status_code=500, detail=str(e))