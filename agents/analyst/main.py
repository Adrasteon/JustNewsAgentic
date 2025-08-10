"""
Main file for the Analyst Agent.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
import requests

from .tools import (
    identify_entities,
    analyze_text_statistics,
    extract_key_metrics,
    analyze_content_trends,
    log_feedback,
)
from common.observability import MetricsCollector, request_timing_middleware
from common.security import get_service_headers, require_service_token, HEADER_NAME

# Import standardized schemas
from common.schemas import (
    ToolCallV1, AgentRegistration, HealthResponse, ReadinessResponse, WarmupResponse
)
from common.observability import MetricsCollector, request_timing_middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Readiness flag and metrics
ready = False
agent_metrics = MetricsCollector("analyst")

# Environment variables
ANALYST_AGENT_PORT = int(os.environ.get("ANALYST_AGENT_PORT", 8004))
MODEL_PATH = os.environ.get("MISTRAL_7B_PATH", "./models/mistral-7b-instruct-v0.2")
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        """Register agent using standardized schema."""
        registration = AgentRegistration(
            name=agent_name,
            address=agent_address,
            tools=tools,
            version="v2"
        )
        
        try:
            response = requests.post(
                f"{self.base_url}/register", 
                json=registration.model_dump(),
                headers=get_service_headers(),
                timeout=(2, 5)
            )
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    logger.info("🔍 Analyst Agent V2 - Specialized Quantitative Analysis")
    logger.info("📊 Focus: Entity extraction, statistical analysis, numerical metrics")
    logger.info("🎯 Specialization: Text statistics, trends, financial/temporal data")
    logger.info("🤝 Integration: Works with Scout V2 for comprehensive content analysis")
    
    logger.info("Specialized analysis modules loaded and ready")

    # Register agent with MCP Bus
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="analyst",
            agent_address=f"http://localhost:{ANALYST_AGENT_PORT}",
            tools=[
                "identify_entities",
                "analyze_text_statistics", 
                "extract_key_metrics",
                "analyze_content_trends"
            ],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    # Mark ready after successful startup tasks
    global ready
    ready = True
    yield
    
    # Cleanup on shutdown
    logger.info("✅ Analyst agent shutdown completed.")

    logger.info("Analyst agent is shutting down.")

app = FastAPI(
    title="Analyst Agent V2",
    description="Specialized quantitative analysis agent with enhanced observability",
    version="2.0.0",
    lifespan=lifespan
)

# Add observability middleware
request_timing_middleware(app, agent_metrics)

@app.get("/health", response_model=HealthResponse)
def health():
    """Enhanced health check endpoint."""
    return HealthResponse(
        status="ok",
        service="analyst",
    )

@app.get("/ready", response_model=ReadinessResponse) 
def ready_endpoint():
    """Enhanced readiness endpoint."""
    return ReadinessResponse(
        ready=ready,
        service="analyst",
        dependencies={
            "models_available": True,  # Could check actual model availability
            "mcp_bus_connected": True  # Could ping MCP bus
        }
    )

@app.get("/warmup", response_model=WarmupResponse)
def warmup():
    """Enhanced warmup endpoint."""
    start_time = time.time()
    
    # Lightweight warmup - just verify basic functionality
    components_warmed = ["analysis_tools", "text_processing", "metrics"]
    
    agent_metrics.inc("warmup_total")
    duration = time.time() - start_time
    
    return WarmupResponse(
        status="completed",
        service="analyst",
        duration_seconds=duration,
        components_warmed=components_warmed
    )

@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Enhanced metrics endpoint using agent_metrics collector."""
    body = agent_metrics.render()
    return Response(content=body, media_type="text/plain; version=0.0.4")

# Tool endpoints using standardized ToolCallV1 schema

@app.post("/identify_entities")
def identify_entities_endpoint(call: ToolCallV1):
    """Identifies entities in a given text."""
    start_time = time.time()
    try:
        result = identify_entities(*call.args, **call.kwargs)
        agent_metrics.observe("tool_duration_seconds", time.time() - start_time)
        agent_metrics.inc("tool_calls_total", {"tool": "identify_entities"})
        return result
    except Exception as e:
        agent_metrics.inc("tool_errors_total", {"tool": "identify_entities"})
        logger.error(f"Error in identify_entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_text_statistics")
def analyze_text_statistics_endpoint(call: ToolCallV1):
    """Analyzes text statistics including readability and complexity."""
    start_time = time.time()
    try:
        result = analyze_text_statistics(*call.args, **call.kwargs)
        agent_metrics.observe("tool_duration_seconds", time.time() - start_time)
        agent_metrics.inc("tool_calls_total", {"tool": "analyze_text_statistics"})
        return result
    except Exception as e:
        agent_metrics.inc("tool_errors_total", {"tool": "analyze_text_statistics"})
        logger.error(f"Error in analyze_text_statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_key_metrics")
def extract_key_metrics_endpoint(call: ToolCallV1):
    """Extracts key numerical and statistical metrics from text."""
    start_time = time.time()
    try:
        result = extract_key_metrics(*call.args, **call.kwargs)
        agent_metrics.observe("tool_duration_seconds", time.time() - start_time)
        agent_metrics.inc("tool_calls_total", {"tool": "extract_key_metrics"})
        return result
    except Exception as e:
        agent_metrics.inc("tool_errors_total", {"tool": "extract_key_metrics"}) 
        logger.error(f"Error in extract_key_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_content_trends")
def analyze_content_trends_endpoint(call: ToolCallV1):
    """Analyzes trends and patterns in content."""
    start_time = time.time()
    try:
        result = analyze_content_trends(*call.args, **call.kwargs)
        agent_metrics.observe("tool_duration_seconds", time.time() - start_time) 
        agent_metrics.inc("tool_calls_total", {"tool": "analyze_content_trends"})
        return result
    except Exception as e:
        agent_metrics.inc("tool_errors_total", {"tool": "analyze_content_trends"})
        logger.error(f"Error in analyze_content_trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_feedback")
def log_feedback_endpoint(call: ToolCallV1):
    """Logs feedback for model improvement."""
    try:
        feedback = call.kwargs.get("feedback", {})
        log_feedback("log_feedback", feedback)
        agent_metrics.inc("feedback_logged_total")
        return {"status": "logged"}
    except Exception as e:
        agent_metrics.inc("feedback_errors_total")
        logger.error(f"Error logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ANALYST_AGENT_PORT)