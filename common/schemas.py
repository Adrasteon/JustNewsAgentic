"""Standardized schemas for all MCP agents and bus communication.

This module provides versioned Pydantic models for consistent communication
between the MCP Bus and all agents. All services should use these schemas
to ensure API compatibility and contract validation.

Version: v1 (initial standardization)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import time


# Core MCP Communication Models

class ToolCallV1(BaseModel):
    """Standardized tool call format for MCP communication.
    
    This replaces individual ToolCall models in each service with a 
    centralized, versioned schema.
    """
    agent: str = Field(..., description="Target agent name")
    tool: str = Field(..., description="Tool/endpoint name to call")  
    args: List[Any] = Field(default_factory=list, description="Positional arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments")
    
    # Idempotency support
    idempotency_key: Optional[str] = Field(None, description="Optional idempotency key for deduplication")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent": "analyst",
                "tool": "analyze_sentiment", 
                "args": ["Article content here"],
                "kwargs": {"model": "v4"},
                "idempotency_key": "analyze_sentiment_abc123"
            }
        }
    )


# Health Check Models

class HealthResponse(BaseModel):
    """Standard health check response."""
    status: str = Field("ok", description="Health status: 'ok' or 'error'")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    service: Optional[str] = Field(None, description="Service name")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "timestamp": 1691234567.89,
                "service": "mcp_bus"
            }
        }
    )


class ReadinessResponse(BaseModel):
    """Standard readiness check response."""
    ready: bool = Field(..., description="True if service is ready to handle requests")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    service: Optional[str] = Field(None, description="Service name")
    dependencies: Optional[Dict[str, bool]] = Field(None, description="Dependency readiness status")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ready": True,
                "timestamp": 1691234567.89,
                "service": "analyst",
                "dependencies": {"models_loaded": True, "gpu_available": True}
            }
        }
    )


class WarmupResponse(BaseModel):
    """Standard warmup response."""
    status: str = Field(..., description="Warmup status: 'completed', 'in_progress', 'failed'")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    service: Optional[str] = Field(None, description="Service name")
    duration_seconds: Optional[float] = Field(None, description="Warmup duration")
    components_warmed: Optional[List[str]] = Field(None, description="List of warmed components")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "completed",
                "timestamp": 1691234567.89,
                "service": "synthesizer",
                "duration_seconds": 12.5,
                "components_warmed": ["models", "gpu_context", "cache"]
            }
        }
    )


# Agent Registration Models

class AgentRegistration(BaseModel):
    """Agent registration with MCP Bus."""
    name: str = Field(..., description="Agent name (unique identifier)")
    address: str = Field(..., description="Agent base URL (e.g., 'http://localhost:8004')")
    tools: Optional[List[str]] = Field(None, description="List of available tools/endpoints")
    version: Optional[str] = Field("v1", description="Agent API version")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "analyst",
                "address": "http://localhost:8004", 
                "tools": ["analyze_sentiment", "extract_entities"],
                "version": "v1"
            }
        }
    )


# Response Models

class MCPResponse(BaseModel):
    """Standard MCP operation response."""
    status: str = Field(..., description="Operation status: 'success' or 'error'")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracing")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "data": {"sentiment": "positive", "confidence": 0.85},
                "timestamp": 1691234567.89,
                "request_id": "req_abc123"
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracing")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Agent not found: invalid_agent",
                "error_code": "AGENT_NOT_FOUND",
                "timestamp": 1691234567.89,
                "request_id": "req_xyz789",
                "details": {"agent": "invalid_agent", "available_agents": ["analyst", "scout"]}
            }
        }
    )


# Legacy compatibility - keep existing ToolCall for gradual migration
ToolCall = ToolCallV1  # Alias for backward compatibility

# Versioned models for future expansion
__all__ = [
    "ToolCallV1", "ToolCall",  # Core communication
    "HealthResponse", "ReadinessResponse", "WarmupResponse",  # Health checks
    "AgentRegistration",  # Registration
    "MCPResponse", "ErrorResponse"  # Responses
]