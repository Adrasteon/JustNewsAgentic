# Model loading for Chief Editor Agent (Llama-3-70B-Instruct)
import os
import logging
import requests
from datetime import datetime

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

MODEL_NAME = "meta-llama/Llama-3-70B-Instruct"
MODEL_PATH = os.environ.get("LLAMA_3_70B_PATH", "./models/llama-3-70b-instruct")

def get_llama_model():
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers library is not installed.")
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Downloading {MODEL_NAME} to {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
    else:
        print(f"Loading {MODEL_NAME} from local cache {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://mcp_bus:8000")
FEEDBACK_LOG = os.environ.get("CHIEF_EDITOR_FEEDBACK_LOG", "./feedback_chief_editor.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chief_editor.tools")

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def request_story_brief(topic: str, scope: str):
    """Generate a story brief based on the given topic and scope."""
    logger.info(f"Requesting story brief for topic: {topic}, scope: {scope}")
    # Example logic: Combine topic and scope into a brief
    brief = f"Story brief for topic '{topic}' within scope '{scope}'."
    log_feedback("request_story_brief", {"topic": topic, "scope": scope, "brief": brief})
    return brief

def publish_story(story_id: str):
    """
    Publishes a story by notifying the Librarian Agent and updating the Memory Agent via the MCP bus.
    Logs feedback for continual learning.
    """
    logger.info(f"[ChiefEditor] Publishing story with ID: {story_id}")
    payload = {
        "agent": "librarian",
        "tool": "update_story_timeline",
        "args": [story_id],
        "kwargs": {}
    }
    try:
        resp = requests.post(f"{MCP_BUS_URL}/call", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        log_feedback("publish_story", {"story_id": story_id, "result": result})
        return {
            "status": "published",
            "story_id": story_id,
            "mcp_result": result,
            "message": "Librarian Agent notified and story status updated via MCP bus."
        }
    except Exception as e:
        logger.error(f"Error calling MCP bus for publish_story: {e}")
        log_feedback("publish_story_error", {"story_id": story_id, "error": str(e)})
        return {
            "status": "error",
            "story_id": story_id,
            "error": str(e)
        }