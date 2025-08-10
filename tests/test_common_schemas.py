"""Tests for common schemas module."""

import pytest
import time
from pydantic import ValidationError

from common.schemas import (
    ToolCallV1, AgentRegistration, MCPResponse, ErrorResponse,
    HealthResponse, ReadinessResponse, WarmupResponse
)


class TestToolCallV1:
    """Tests for the standardized ToolCallV1 schema."""
    
    def test_basic_tool_call(self):
        """Test basic tool call creation."""
        call = ToolCallV1(
            agent="analyst",
            tool="analyze_sentiment",
            args=["test content"],
            kwargs={"model": "v4"}
        )
        
        assert call.agent == "analyst"
        assert call.tool == "analyze_sentiment"
        assert call.args == ["test content"]
        assert call.kwargs == {"model": "v4"}
        assert call.idempotency_key is None
    
    def test_tool_call_with_idempotency_key(self):
        """Test tool call with idempotency key."""
        call = ToolCallV1(
            agent="synthesizer",
            tool="create_summary",
            args=["content"],
            idempotency_key="summary_abc123"
        )
        
        assert call.idempotency_key == "summary_abc123"
    
    def test_tool_call_defaults(self):
        """Test tool call with default values."""
        call = ToolCallV1(agent="scout", tool="discover")
        
        assert call.args == []
        assert call.kwargs == {}
        assert call.idempotency_key is None
    
    def test_tool_call_validation(self):
        """Test tool call validation."""
        with pytest.raises(ValidationError):
            ToolCallV1()  # Missing required fields
        
        # Test that the schema accepts empty strings (this is valid in V2)
        call = ToolCallV1(agent="", tool="test")
        assert call.agent == ""
        assert call.tool == "test"


class TestHealthCheckSchemas:
    """Tests for health check related schemas."""
    
    def test_health_response(self):
        """Test HealthResponse schema."""
        response = HealthResponse(service="test_service")
        
        assert response.status == "ok"
        assert response.service == "test_service"
        assert isinstance(response.timestamp, float)
    
    def test_readiness_response(self):
        """Test ReadinessResponse schema."""
        response = ReadinessResponse(
            ready=True,
            service="analyst",
            dependencies={"models_loaded": True}
        )
        
        assert response.ready is True
        assert response.service == "analyst" 
        assert response.dependencies == {"models_loaded": True}
    
    def test_warmup_response(self):
        """Test WarmupResponse schema."""
        response = WarmupResponse(
            status="completed",
            service="synthesizer",
            duration_seconds=5.2,
            components_warmed=["models", "cache"]
        )
        
        assert response.status == "completed"
        assert response.duration_seconds == 5.2
        assert response.components_warmed == ["models", "cache"]


class TestAgentRegistration:
    """Tests for AgentRegistration schema."""
    
    def test_basic_registration(self):
        """Test basic agent registration."""
        reg = AgentRegistration(
            name="analyst",
            address="http://localhost:8004"
        )
        
        assert reg.name == "analyst"
        assert reg.address == "http://localhost:8004"
        assert reg.version == "v1"  # Default value
        assert reg.tools is None
    
    def test_registration_with_tools(self):
        """Test registration with tools list."""
        reg = AgentRegistration(
            name="scout",
            address="http://localhost:8002",
            tools=["discover_sources", "crawl_content"],
            version="v2"
        )
        
        assert reg.tools == ["discover_sources", "crawl_content"]
        assert reg.version == "v2"


class TestResponseSchemas:
    """Tests for response schemas."""
    
    def test_mcp_response_success(self):
        """Test successful MCP response."""
        response = MCPResponse(
            status="success",
            data={"result": "analysis complete"},
            request_id="req_123"
        )
        
        assert response.status == "success"
        assert response.data == {"result": "analysis complete"}
        assert response.request_id == "req_123"
        assert response.error is None
    
    def test_mcp_response_error(self):
        """Test error MCP response."""
        response = MCPResponse(
            status="error",
            error="Agent not found",
            request_id="req_456"
        )
        
        assert response.status == "error"
        assert response.error == "Agent not found"
        assert response.data is None
    
    def test_error_response(self):
        """Test ErrorResponse schema."""
        error = ErrorResponse(
            error="Invalid request format",
            error_code="INVALID_REQUEST",
            details={"field": "agent", "message": "required"}
        )
        
        assert error.error == "Invalid request format"
        assert error.error_code == "INVALID_REQUEST"
        assert error.details == {"field": "agent", "message": "required"}


class TestSchemaCompatibility:
    """Tests for backward compatibility."""
    
    def test_toolcall_alias(self):
        """Test that ToolCall alias works."""
        from common.schemas import ToolCall
        
        call = ToolCall(agent="test", tool="test_tool")
        assert isinstance(call, ToolCallV1)
        assert call.agent == "test"
        assert call.tool == "test_tool"


class TestSchemaValidation:
    """Tests for schema validation edge cases."""
    
    def test_timestamp_defaults(self):
        """Test that timestamps are properly set."""
        health = HealthResponse()
        readiness = ReadinessResponse(ready=True)
        
        # Timestamps should be recent (within last second)
        now = time.time()
        assert abs(health.timestamp - now) < 1.0
        assert abs(readiness.timestamp - now) < 1.0
    
    def test_optional_fields(self):
        """Test handling of optional fields."""
        # Test with minimal required fields
        call = ToolCallV1(agent="test", tool="test")
        assert call.args == []
        assert call.kwargs == {}
        
        health = HealthResponse()
        assert health.service is None
        
        error = ErrorResponse(error="test error")
        assert error.error_code is None
        assert error.details is None