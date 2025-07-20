
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import requests
import uvicorn


app = FastAPI()

AGENT_URLS = {
    "chief_editor": "http://chief_editor:8001",
    "scout": "http://scout:8002",
    "fact_checker": "http://fact_checker:8003",
    "analyst": "http://analyst:8004",
    "synthesizer": "http://synthesizer:8005",
    "critic": "http://critic:8006",
    "memory": "http://memory:8007",
}

@app.get("/health")
def health():
    return {"status": "ok"}

# Example endpoint: relay a message to another agent (stub for now)
class RelayRequest(BaseModel):
    target_agent: str
    endpoint: str
    payload: Dict[str, Any]

@app.post("/relay")
def relay_message(request: RelayRequest):
    # In a real implementation, this would forward the payload to the target agent
    # and return the response. For now, just echo the request.
        target_url = AGENT_URLS.get(request.target_agent)
        if not target_url:
            raise HTTPException(status_code=404, detail=f"Unknown target agent: {request.target_agent}")
        url = f"{target_url}{request.endpoint}"
        try:
            resp = requests.post(url, json=request.payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Relay to {url} failed: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
