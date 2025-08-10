"""Common security utilities for inter-service auth and request validation.

Implements a simple shared-secret authentication mechanism for inter-service
requests. Services should pass the token via the `X-Service-Token` header.

Environment variables:
- MCP_SERVICE_TOKEN: Shared secret token for authenticating service calls.

Usage:
- On FastAPI endpoints: call `require_service_token(header_value)` early.
- On clients: use `get_service_headers()` in HTTP requests.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from fastapi import HTTPException


SERVICE_TOKEN_ENV = "MCP_SERVICE_TOKEN"
HEADER_NAME = "X-Service-Token"


def get_expected_token() -> Optional[str]:
    """Return the expected service token from environment or None if unset."""
    token = os.environ.get(SERVICE_TOKEN_ENV)
    return token


def require_service_token(header_value: Optional[str]) -> None:
    """Validate the provided header token against the expected shared secret.

    Raises:
        HTTPException: 401 if token missing; 403 if token invalid.
    """
    expected = get_expected_token()
    if not expected:
        # Auth disabled (token not configured). Allow the request.
        return
    if header_value is None:
        raise HTTPException(status_code=401, detail="Missing service token")
    if header_value != expected:
        raise HTTPException(status_code=403, detail="Invalid service token")


def get_service_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return headers including the service token when configured."""
    headers: Dict[str, str] = {}
    expected = get_expected_token()
    if expected:
        headers[HEADER_NAME] = expected
    if extra:
        headers.update(extra)
    return headers


__all__ = [
    "SERVICE_TOKEN_ENV",
    "HEADER_NAME",
    "get_expected_token",
    "require_service_token",
    "get_service_headers",
]
