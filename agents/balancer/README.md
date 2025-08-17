# Balancer agent (scaffold)

This is a minimal FastAPI scaffold for the Balancer agent used in MCP tests.

Endpoints:
- GET /health — simple health check
- POST /call?name=<tool> — call a tool implemented in `agents.balancer.tools`

Run locally:

```bash
# From repository root
uvicorn agents.balancer.main:app --reload --port 8009
```

Notes
-----

- `agents.balancer.main:app` is an ASGI application (FastAPI implements the ASGI interface). Run it with an ASGI server such as `uvicorn` or `hypercorn` to serve HTTP and WebSocket requests.
- The module is import-safe: if FastAPI is not installed the module still exposes Python-callable handlers (`health()` and `call_tool()`) so tests or other agents can call the functions directly without running an HTTP server.

Example (direct call from Python):

```python
from agents.balancer import main
print(main.health())
call = main.ToolCall(args=[1,2,3], kwargs={})
print(main.call_tool('echo', call))
```
