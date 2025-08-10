"""
Main file for the MCP Bus with idempotency, enhanced observability, and standardized schemas.
"""
# main.py for MCP Message Bus
from fastapi import FastAPI, HTTPException, Response, Header
from fastapi.middleware.cors import CORSMiddleware
import requests
import time
import atexit
import logging
import hashlib
import json
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Set
from threading import Lock

# Import standardized schemas
from common.schemas import (
    ToolCallV1, AgentRegistration, MCPResponse, ErrorResponse,
    HealthResponse, ReadinessResponse, WarmupResponse
)
from common.observability import MetricsCollector, request_timing_middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
ready = False
agents = {}
cb_state = {}
CB_FAIL_THRESHOLD = 3
CB_COOLDOWN_SEC = 10

# Idempotency cache with TTL
idempotency_cache: Dict[str, Dict[str, Any]] = {}
idempotency_lock = Lock()
IDEMPOTENCY_TTL_SECONDS = 300  # 5 minutes

# Enhanced metrics with observability
metrics_collector = MetricsCollector("mcp_bus")

# Create FastAPI app with enhanced middleware
app = FastAPI(
    title="MCP Bus",
    description="Model Context Protocol Message Bus with idempotency and observability",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
request_timing_middleware(app, metrics_collector)

def _generate_cache_key(agent: str, tool: str, args: list, kwargs: dict, idempotency_key: Optional[str] = None) -> str:
    """Generate cache key for idempotency checks."""
    if idempotency_key:
        # Use provided idempotency key
        return f"idem:{agent}:{tool}:{idempotency_key}"
    else:
        # Generate key from request content hash
        content = json.dumps({"agent": agent, "tool": tool, "args": args, "kwargs": kwargs}, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"hash:{agent}:{tool}:{content_hash}"


def _cleanup_expired_cache():
    """Remove expired entries from idempotency cache."""
    now = time.time()
    expired_keys = []
    
    with idempotency_lock:
        for key, entry in idempotency_cache.items():
            if entry["expires_at"] < now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del idempotency_cache[key]
    
    if expired_keys:
        logger.debug(f"Cleaned up {len(expired_keys)} expired idempotency cache entries")


def _check_idempotency_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Check if request is duplicate and return cached response if found."""
    _cleanup_expired_cache()
    
    with idempotency_lock:
        entry = idempotency_cache.get(cache_key)
        if entry and entry["expires_at"] > time.time():
            metrics_collector.inc("idempotency_hits_total")
            logger.info(f"Idempotency cache hit for key: {cache_key[:32]}...")
            return entry["response"]
    
    return None


def _store_in_idempotency_cache(cache_key: str, response: Dict[str, Any]):
    """Store response in idempotency cache."""
    expires_at = time.time() + IDEMPOTENCY_TTL_SECONDS
    
    with idempotency_lock:
        idempotency_cache[cache_key] = {
            "response": response,
            "expires_at": expires_at
        }
        metrics_collector.inc("idempotency_stores_total")


@app.post("/register", response_model=MCPResponse)
def register_agent(agent: AgentRegistration):
    """Register an agent with enhanced logging and metrics."""
    logger.info(f"Registering agent: {agent.name} at {agent.address} (version: {agent.version})")
    
    agents[agent.name] = {
        "address": agent.address,
        "tools": agent.tools or [],
        "version": agent.version,
        "registered_at": time.time()
    }
    
    # Reset circuit breaker on registration
    cb_state[agent.name] = {"fails": 0, "open_until": 0}
    
    metrics_collector.inc("agents_registered_total")
    metrics_collector.set_gauge("agents_active_count", len(agents))
    
    return MCPResponse(
        status="success",
        data={"message": f"Agent {agent.name} registered successfully"}
    )

@app.post("/call", response_model=MCPResponse)
def call_tool(call: ToolCallV1, x_idempotency_key: Optional[str] = Header(None, alias="X-Idempotency-Key")):
    """Enhanced tool call with idempotency support and better observability."""
    start_time = time.time()
    
    # Use header idempotency key if provided, otherwise use the one in body
    effective_idempotency_key = x_idempotency_key or call.idempotency_key
    
    if call.agent not in agents:
        metrics_collector.inc("errors_total", {"error_type": "agent_not_found"})
        raise HTTPException(
            status_code=404, 
            detail=f"Agent not found: {call.agent}"
        )
    
    agent_name = call.agent
    agent_info = agents[agent_name]
    agent_address = agent_info["address"]

    # Generate cache key for idempotency
    cache_key = _generate_cache_key(
        call.agent, call.tool, call.args, call.kwargs, effective_idempotency_key
    )
    
    # Check idempotency cache
    if effective_idempotency_key:
        cached_response = _check_idempotency_cache(cache_key)
        if cached_response:
            metrics_collector.observe("call_duration_seconds", time.time() - start_time)
            return MCPResponse(**cached_response)

    # Circuit breaker check
    state = cb_state.get(agent_name, {"fails": 0, "open_until": 0})
    now = time.time()
    if state.get("open_until", 0) > now:
        metrics_collector.inc("circuit_breaker_rejections_total", {"agent": agent_name})
        raise HTTPException(
            status_code=503, 
            detail=f"Circuit open for agent {agent_name}"
        )

    payload = {"args": call.args, "kwargs": call.kwargs}
    url = f"{agent_address}/{call.tool}"
    timeout = (3, 10)  # (connect, read)

    # Simple retry with backoff
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            
            # Success: reset failures and record metrics
            cb_state[agent_name] = {"fails": 0, "open_until": 0}
            duration = time.time() - start_time
            
            metrics_collector.inc("requests_total", {"agent": agent_name, "tool": call.tool})
            metrics_collector.observe("call_duration_seconds", duration)
            metrics_collector.set_gauge("circuit_breaker_state", 0)  # 0 = closed
            
            # Prepare response
            mcp_response = MCPResponse(
                status="success",
                data=response.json(),
                timestamp=time.time()
            )
            
            # Store in idempotency cache if key was provided
            if effective_idempotency_key:
                _store_in_idempotency_cache(cache_key, mcp_response.model_dump())
            
            return mcp_response
            
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            metrics_collector.inc("retry_attempts_total", {"agent": agent_name, "attempt": str(attempt + 1)})
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(0.2 * (2 ** attempt))

    # Failure after retries: increment failure count and update circuit breaker
    fails = state.get("fails", 0) + 1
    duration = time.time() - start_time
    
    if fails >= CB_FAIL_THRESHOLD:
        cb_state[agent_name] = {"fails": 0, "open_until": now + CB_COOLDOWN_SEC}
        metrics_collector.set_gauge("circuit_breaker_state", 1)  # 1 = open
        logger.warning(f"Circuit opened for {agent_name} for {CB_COOLDOWN_SEC}s after {fails} failures")
    else:
        cb_state[agent_name] = {"fails": fails, "open_until": 0}

    metrics_collector.inc("errors_total", {"agent": agent_name, "error_type": "request_failed"})
    metrics_collector.observe("call_duration_seconds", duration)
    
    raise HTTPException(
        status_code=502, 
        detail=f"Tool call failed after {3} attempts: {last_error}"
    )

@app.get("/agents")
def get_agents():
    """Get all registered agents with enhanced information."""
    return {
        "agents": agents,
        "count": len(agents),
        "timestamp": time.time()
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Enhanced health check with service information."""
    return HealthResponse(
        status="ok",
        service="mcp_bus",
        timestamp=time.time()
    )


@app.get("/ready", response_model=ReadinessResponse)
def ready_endpoint():
    """Enhanced readiness check with dependency status."""
    return ReadinessResponse(
        ready=ready,
        service="mcp_bus",
        timestamp=time.time(),
        dependencies={
            "agents_count": len(agents),
            "cache_size": len(idempotency_cache)
        }
    )


@app.get("/warmup", response_model=WarmupResponse)
def warmup():
    """Warmup endpoint for MCP Bus."""
    start_time = time.time()
    
    # Lightweight warmup - just verify basic functionality
    components_warmed = ["routing", "circuit_breakers", "idempotency_cache"]
    
    metrics_collector.inc("warmup_total")
    duration = time.time() - start_time
    
    return WarmupResponse(
        status="completed",
        service="mcp_bus", 
        timestamp=time.time(),
        duration_seconds=duration,
        components_warmed=components_warmed
    )


@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Enhanced Prometheus-style metrics with histograms and circuit breaker state."""
    # Add circuit breaker state metrics
    for agent_name, state in cb_state.items():
        is_open = 1 if state.get("open_until", 0) > time.time() else 0
        metrics_collector.set_gauge(f"circuit_breaker_open", is_open)
        metrics_collector.set_gauge(f"circuit_breaker_fails", state.get("fails", 0))
    
    # Add cache metrics
    metrics_collector.set_gauge("idempotency_cache_size", len(idempotency_cache))
    
    # Render all metrics
    body = metrics_collector.render()
    return Response(content=body, media_type="text/plain; version=0.0.4")

@asynccontextmanager
async def lifespan(app):
    """Enhanced startup and shutdown lifecycle management."""
    logger.info("MCP_Bus is starting up.")
    global ready
    
    # Initialize metrics
    metrics_collector.inc("startup_total")
    
    # Mark as ready
    ready = True
    logger.info("MCP_Bus is ready to accept connections.")
    
    yield
    
    # Cleanup on shutdown
    logger.info("MCP_Bus is shutting down.")
    with idempotency_lock:
        idempotency_cache.clear()
    logger.info("Idempotency cache cleared.")


# Set the lifespan for the FastAPI app
app.router.lifespan_context = lifespan

atexit.register(lambda: logger.info("MCP_Bus has exited."))
