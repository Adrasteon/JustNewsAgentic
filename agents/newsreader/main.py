"""
MCP Bus Integration for LLaVA NewsReader Agent
Integrates with JustNews V4 MCP Bus system
"""

from fastapi import FastAPI
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava_newsreader_agent import LlavaNewsReaderAgent, NewsExtractionRequest

# Global agent instance
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global agent
    print("🚀 Initializing NewsReader Agent for MCP Bus")
    agent = LlavaNewsReaderAgent()
    print("✅ NewsReader Agent initialized")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down NewsReader Agent")
    # Cleanup if needed
    print("✅ NewsReader Agent shutdown complete")

app = FastAPI(
    title="NewsReader Agent", 
    description="LLaVA-based news content extraction",
    lifespan=lifespan
)

class ToolCall(BaseModel):
    args: List[Any]
    kwargs: Dict[str, Any]

@app.post("/extract_news_content")
async def extract_news_content_endpoint(call: ToolCall):
    """Extract news content from URL - MCP Bus compatible"""
    from llava_newsreader_agent import extract_news_content
    return extract_news_content(*call.args, **call.kwargs)

@app.post("/capture_screenshot")
async def capture_screenshot(call: ToolCall):
    """Capture webpage screenshot - MCP Bus compatible"""
    if not agent:
        return {"error": "Agent not initialized"}
    
    url = call.args[0] if call.args else call.kwargs.get("url")
    screenshot_path = call.args[1] if len(call.args) > 1 else call.kwargs.get("screenshot_path", "page_llava.png")
    
    try:
        result_path = await agent.capture_screenshot(url, screenshot_path)
        return {"screenshot_path": result_path, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/analyze_screenshot")
async def analyze_screenshot(call: ToolCall):
    """Analyze screenshot with LLaVA - MCP Bus compatible"""
    if not agent:
        return {"error": "Agent not initialized"}
    
    image_path = call.args[0] if call.args else call.kwargs.get("image_path")
    
    try:
        result = agent.extract_content_with_llava(image_path)
        return result
    except Exception as e:
        return {"error": str(e), "success": False}

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "agent": "newsreader",
        "model": "llava-v1.6-mistral-7b-optimized",
        "environment": "rapids-25.06",
        "lifespan": "modern"
    }

# Tool functions for direct import
async def extract_news_content(url: str, screenshot_path: str = None) -> Dict[str, Any]:
    """Extract news content from URL"""
    global agent
    if not agent:
        agent = LlavaNewsReaderAgent()
    
    result = await agent.process_news_url(url, screenshot_path)
    return result.dict()

async def capture_webpage_screenshot(url: str, output_path: str = "page_llava.png") -> str:
    """Capture webpage screenshot"""
    global agent
    if not agent:
        agent = LlavaNewsReaderAgent()
    
    return await agent.capture_screenshot(url, output_path)

def analyze_image_content(image_path: str) -> Dict[str, str]:
    """Analyze image content with LLaVA"""
    global agent
    if not agent:
        agent = LlavaNewsReaderAgent()
    
    return agent.extract_content_with_llava(image_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
