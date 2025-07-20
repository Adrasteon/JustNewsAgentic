from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# Example endpoint: relay a message to another agent (stub for now)
class RelayRequest(BaseModel):
    target_agent: str
    payload: Dict[str, Any]

@app.post("/relay")
def relay_message(request: RelayRequest):
    # In a real implementation, this would forward the payload to the target agent
    # and return the response. For now, just echo the request.
    return {"relayed_to": request.target_agent, "payload": request.payload}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
