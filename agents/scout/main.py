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
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SCOUT_AGENT_PORT = int(os.environ.get("SCOUT_AGENT_PORT", 8002))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data)
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Scout agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="scout",
            agent_address=f"http://localhost:{SCOUT_AGENT_PORT}",
            tools=[
                "discover_sources", "crawl_url", "deep_crawl_site", "enhanced_deep_crawl_site",
                "intelligent_source_discovery", "intelligent_content_crawl", 
                "intelligent_batch_analysis",
                "production_crawl_ultra_fast", "get_production_crawler_info"
            ],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    yield
    logger.info("Scout agent is shutting down.")

app = FastAPI(lifespan=lifespan)

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

@app.post("/enhanced_deep_crawl_site")
async def enhanced_deep_crawl_site_endpoint(call: ToolCall):
    try:
        from tools import enhanced_deep_crawl_site
        logger.info(f"Calling enhanced_deep_crawl_site with args: {call.args} and kwargs: {call.kwargs}")
        return await enhanced_deep_crawl_site(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in enhanced_deep_crawl_site: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intelligent_source_discovery")
def intelligent_source_discovery_endpoint(call: ToolCall):
    try:
        from tools import intelligent_source_discovery
        logger.info(f"Calling intelligent_source_discovery with args: {call.args} and kwargs: {call.kwargs}")
        return intelligent_source_discovery(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in intelligent_source_discovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intelligent_content_crawl")
def intelligent_content_crawl_endpoint(call: ToolCall):
    try:
        from tools import intelligent_content_crawl
        logger.info(f"Calling intelligent_content_crawl with args: {call.args} and kwargs: {call.kwargs}")
        return intelligent_content_crawl(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in intelligent_content_crawl: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intelligent_batch_analysis")
def intelligent_batch_analysis_endpoint(call: ToolCall):
    try:
        from tools import intelligent_batch_analysis
        logger.info(f"Calling intelligent_batch_analysis with args: {call.args} and kwargs: {call.kwargs}")
        return intelligent_batch_analysis(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in intelligent_batch_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/log_feedback")
def log_feedback(call: ToolCall):
    try:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "feedback": call.kwargs.get("feedback")
        }
        logger.info(f"Logging feedback: {feedback_data}")
        return feedback_data
    except Exception as e:
        logger.error(f"An error occurred while logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# PRODUCTION CRAWLER ENDPOINTS
# =============================================================================

@app.post("/production_crawl_ultra_fast")
async def production_crawl_ultra_fast_endpoint(call: ToolCall):
    try:
        from tools import production_crawl_ultra_fast
        logger.info(f"Calling production_crawl_ultra_fast with args: {call.args} and kwargs: {call.kwargs}")
        return await production_crawl_ultra_fast(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in production_crawl_ultra_fast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_production_crawler_info")
def get_production_crawler_info_endpoint(call: ToolCall):
    try:
        from tools import get_production_crawler_info
        logger.info(f"Calling get_production_crawler_info with args: {call.args} and kwargs: {call.kwargs}")
        return get_production_crawler_info(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in get_production_crawler_info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
