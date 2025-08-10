"""Tests for MCP Bus idempotency and enhanced functionality."""

import pytest
import requests
import time
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import the FastAPI app and related functions from MCP Bus
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mcp_bus.main import (
    app, _generate_cache_key, _check_idempotency_cache, 
    _store_in_idempotency_cache, idempotency_cache, idempotency_lock
)
from common.schemas import ToolCallV1, AgentRegistration


class TestIdempotencyFunctionality:
    """Tests for idempotency key functionality."""
    
    def setup_method(self):
        """Clear idempotency cache before each test."""
        with idempotency_lock:
            idempotency_cache.clear()
    
    def test_generate_cache_key_with_idempotency_key(self):
        """Test cache key generation with idempotency key."""
        key = _generate_cache_key(
            agent="analyst",
            tool="analyze",
            args=["content"],
            kwargs={"model": "v4"},
            idempotency_key="test_key_123"
        )
        
        assert key == "idem:analyst:analyze:test_key_123"
    
    def test_generate_cache_key_without_idempotency_key(self):
        """Test cache key generation without idempotency key (content hash)."""
        key1 = _generate_cache_key(
            agent="analyst",
            tool="analyze",
            args=["content"],
            kwargs={"model": "v4"}
        )
        
        key2 = _generate_cache_key(
            agent="analyst", 
            tool="analyze",
            args=["content"],
            kwargs={"model": "v4"}
        )
        
        # Same content should generate same key
        assert key1 == key2
        assert key1.startswith("hash:analyst:analyze:")
        assert len(key1.split(":")[-1]) == 16  # 16-char hash
    
    def test_generate_cache_key_different_content(self):
        """Test that different content generates different keys."""
        key1 = _generate_cache_key("analyst", "analyze", ["content1"], {})
        key2 = _generate_cache_key("analyst", "analyze", ["content2"], {})
        
        assert key1 != key2
    
    def test_idempotency_cache_store_and_retrieve(self):
        """Test storing and retrieving from idempotency cache."""
        cache_key = "test:key:123"
        response_data = {"status": "success", "data": {"result": "test"}}
        
        # Store response
        _store_in_idempotency_cache(cache_key, response_data)
        
        # Retrieve response
        cached = _check_idempotency_cache(cache_key)
        assert cached == response_data
    
    def test_idempotency_cache_expiry(self):
        """Test that cache entries expire correctly."""
        cache_key = "test:expire:123"
        response_data = {"status": "success", "data": {"result": "test"}}
        
        # Store with short TTL by manipulating the cache directly
        with idempotency_lock:
            idempotency_cache[cache_key] = {
                "response": response_data,
                "expires_at": time.time() - 1  # Already expired
            }
        
        # Should return None for expired entry
        cached = _check_idempotency_cache(cache_key)
        assert cached is None
    
    def test_idempotency_cache_miss(self):
        """Test cache miss for non-existent key."""
        cached = _check_idempotency_cache("non:existent:key")
        assert cached is None


class TestMCPBusEndpoints:
    """Tests for MCP Bus endpoints with enhanced functionality."""
    
    def setup_method(self):
        """Setup test client and clear state."""
        self.client = TestClient(app)
        # Clear global state
        from mcp_bus.main import agents, cb_state
        agents.clear()
        cb_state.clear()
        with idempotency_lock:
            idempotency_cache.clear()
    
    def test_health_endpoint(self):
        """Test enhanced health endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "mcp_bus"
        assert "timestamp" in data
    
    def test_ready_endpoint(self):
        """Test enhanced readiness endpoint.""" 
        response = self.client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert "ready" in data
        assert data["service"] == "mcp_bus"
        assert "dependencies" in data
        assert "agents_count" in data["dependencies"]
    
    def test_warmup_endpoint(self):
        """Test warmup endpoint."""
        response = self.client.get("/warmup")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "completed"
        assert data["service"] == "mcp_bus"
        assert "duration_seconds" in data
        assert "components_warmed" in data
        assert isinstance(data["components_warmed"], list)
    
    def test_agent_registration(self):
        """Test enhanced agent registration."""
        agent_data = {
            "name": "test_agent",
            "address": "http://localhost:8999",
            "tools": ["test_tool"],
            "version": "v1"
        }
        
        response = self.client.post("/register", json=agent_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "test_agent registered successfully" in data["data"]["message"]
    
    def test_get_agents_enhanced(self):
        """Test enhanced agents endpoint."""
        # Register an agent first
        agent_data = {
            "name": "test_agent",
            "address": "http://localhost:8999"
        }
        self.client.post("/register", json=agent_data)
        
        response = self.client.get("/agents")
        assert response.status_code == 200
        
        data = response.json()
        assert "agents" in data
        assert "count" in data  
        assert "timestamp" in data
        assert data["count"] == 1
        assert "test_agent" in data["agents"]
    
    @patch('requests.post')
    def test_tool_call_with_idempotency_header(self, mock_post):
        """Test tool call with idempotency key in header."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response
        
        # Register agent
        agent_data = {"name": "test_agent", "address": "http://localhost:8999"}
        self.client.post("/register", json=agent_data)
        
        # Make tool call with idempotency key in header
        call_data = {
            "agent": "test_agent",
            "tool": "test_tool",
            "args": ["test"],
            "kwargs": {}
        }
        
        headers = {"X-Idempotency-Key": "test_idem_key_123"}
        
        # First call
        response1 = self.client.post("/call", json=call_data, headers=headers)
        assert response1.status_code == 200
        
        # Second call with same idempotency key should return cached response
        response2 = self.client.post("/call", json=call_data, headers=headers)
        assert response2.status_code == 200
        assert response2.json() == response1.json()
        
        # Verify the actual agent was only called once due to caching
        assert mock_post.call_count == 1
    
    @patch('requests.post')
    def test_tool_call_with_idempotency_in_body(self, mock_post):
        """Test tool call with idempotency key in request body."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response
        
        # Register agent
        agent_data = {"name": "test_agent", "address": "http://localhost:8999"}
        self.client.post("/register", json=agent_data)
        
        # Make tool call with idempotency key in body
        call_data = {
            "agent": "test_agent",
            "tool": "test_tool", 
            "args": ["test"],
            "kwargs": {},
            "idempotency_key": "body_idem_key_456"
        }
        
        # First call
        response1 = self.client.post("/call", json=call_data)
        assert response1.status_code == 200
        
        # Second call should return cached response
        response2 = self.client.post("/call", json=call_data) 
        assert response2.status_code == 200
        assert response2.json() == response1.json()
        
        # Agent should only be called once
        assert mock_post.call_count == 1
    
    def test_tool_call_agent_not_found(self):
        """Test tool call with non-existent agent."""
        call_data = {
            "agent": "non_existent",
            "tool": "test_tool",
            "args": [],
            "kwargs": {}
        }
        
        response = self.client.post("/call", json=call_data)
        assert response.status_code == 404
        assert "Agent not found" in response.json()["detail"]
    
    def test_metrics_endpoint_enhanced(self):
        """Test enhanced metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
        
        # Verify metrics format
        content = response.content.decode()
        assert "mcp_bus_" in content  # Should contain metrics with agent prefix


class TestCircuitBreaker:
    """Tests for circuit breaker functionality in enhanced MCP Bus."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
        # Clear global state
        from mcp_bus.main import agents, cb_state
        agents.clear()
        cb_state.clear()
    
    @patch('requests.post')
    def test_circuit_breaker_opens_after_failures(self, mock_post):
        """Test that circuit breaker opens after repeated failures."""
        # Setup mock to always fail
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        # Register agent
        agent_data = {"name": "failing_agent", "address": "http://localhost:8999"}
        self.client.post("/register", json=agent_data)
        
        call_data = {
            "agent": "failing_agent", 
            "tool": "test_tool",
            "args": [],
            "kwargs": {}
        }
        
        # Make several failing calls to trigger circuit breaker
        for _ in range(3):  # Should trigger circuit breaker after 3 failures
            response = self.client.post("/call", json=call_data)
            assert response.status_code == 502  # Bad Gateway
        
        # Next call should be rejected by circuit breaker
        response = self.client.post("/call", json=call_data)
        assert response.status_code == 503  # Service Unavailable
        assert "Circuit open" in response.json()["detail"]