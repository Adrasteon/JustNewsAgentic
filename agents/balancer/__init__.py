"""Balancer agent package.

Minimal FastAPI scaffold for MCP-style local dispatch tests.
"""

# Re-export the primary submodules so tests can do `from agents.balancer import main`
from . import main  # noqa: F401
from . import tools  # noqa: F401

__all__ = ["main", "tools"]
