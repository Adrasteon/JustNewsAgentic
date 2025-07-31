"""
Reasoning Agent (Nucleoid)
Purpose: Symbolic reasoning, fact validation, contradiction detection, and explainability for news analysis
GPU Status: ❌ CPU Only (symbolic logic)
Performance: Fast for logic/rule queries; not GPU-accelerated
V4 Compliance: Designed for multi-agent orchestration, FastAPI, and MCP bus integration
Dependencies: nucleoid, fastapi, pydantic, uvicorn
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, List
import nucleoid
import os

app = FastAPI(title="JustNews V4 Reasoning Agent (Nucleoid)")

# --- Nucleoid Engine Initialization ---
engine = nucleoid.Engine()

# --- Pydantic Models ---
class Fact(BaseModel):
    data: Dict[str, Any]

class Rule(BaseModel):
    rule: str

class Query(BaseModel):
    query: str

class Evaluate(BaseModel):
    expression: str

# --- API Endpoints ---
@app.post("/add_fact")
def add_fact(fact: Fact):
    try:
        engine.add_fact(fact.data)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/add_facts")
def add_facts(facts: List[Fact]):
    try:
        for fact in facts:
            engine.add_fact(fact.data)
        return {"status": "ok", "count": len(facts)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/add_rule")
def add_rule(rule: Rule):
    try:
        engine.add_rule(rule.rule)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
def query(q: Query):
    try:
        result = engine.query(q.query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate")
def evaluate(expr: Evaluate):
    try:
        result = engine.evaluate(expr.expression)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

# --- MCP Bus Integration Example ---
@app.post("/register")
def register():
    # Example: Return agent metadata for MCP bus
    return {
        "agent": "reasoning",
        "tools": ["add_fact", "add_facts", "add_rule", "query", "evaluate"],
        "status": "ok"
    }

@app.post("/call")
def call_tool(request: Request):
    # MCP bus pattern: expects {"tool": ..., "args": [...], "kwargs": {...}}
    data = request.json()
    tool = data.get("tool")
    args = data.get("args", [])
    kwargs = data.get("kwargs", {})
    if tool == "add_fact":
        return add_fact(Fact(**args[0]))
    elif tool == "add_facts":
        return add_facts([Fact(**f) for f in args[0]])
    elif tool == "add_rule":
        return add_rule(Rule(**args[0]))
    elif tool == "query":
        return query(Query(**args[0]))
    elif tool == "evaluate":
        return evaluate(Evaluate(**args[0]))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")

# --- Startup Logic (Optional) ---
@app.on_event("startup")
def startup_event():
    # Optionally preload rules/facts from file or env
    preload = os.environ.get("REASONING_AGENT_PRELOAD")
    if preload and os.path.exists(preload):
        with open(preload, "r") as f:
            for line in f:
                if line.strip().startswith("RULE:"):
                    engine.add_rule(line.strip()[5:])
                elif line.strip().startswith("FACT:"):
                    # Expecting JSON dict after FACT:
                    import json
                    engine.add_fact(json.loads(line.strip()[5:]))

# --- Run with: uvicorn agents.reasoning.nucleoid_service:app --reload ---
