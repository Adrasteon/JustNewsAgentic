"""
Main file for the Scout Agent.
"""
# main.py for Scout Agent
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

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.post("/discover_sources")
def discover_sources(call: ToolCall):
    try:
        from tools import discover_sources
        logger.info(f"Calling discover_sources with args: {call.args} and kwargs: {call.kwargs}")
        return discover_sources(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in discover_sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl_url")
def crawl_url(call: ToolCall):
    try:
        from tools import crawl_url
        logger.info(f"Calling crawl_url with args: {call.args} and kwargs: {call.kwargs}")
        return crawl_url(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in crawl_url: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deep_crawl_site")
def deep_crawl_site(call: ToolCall):
    try:
        from tools import deep_crawl_site
        logger.info(f"Calling deep_crawl_site with args: {call.args} and kwargs: {call.kwargs}")
        return deep_crawl_site(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in deep_crawl_site: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
