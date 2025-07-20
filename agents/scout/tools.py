# Model loading for Scout Agent (Llama-3-8B-Instruct)
import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

MODEL_NAME = "meta-llama/Llama-3-8B-Instruct"
MODEL_PATH = os.environ.get("LLAMA_3_8B_PATH", "./models/llama-3-8b-instruct")

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
# tools.py for Scout Agent

import logging
import requests
from datetime import datetime

from crawl4ai.adaptive_crawler import AdaptiveCrawler, AdaptiveConfig

SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
FEEDBACK_LOG = os.environ.get("SCOUT_FEEDBACK_LOG", "./feedback_scout.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scout.tools")

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def discover_sources(query: str) -> list[str]:
    """
    Discovers news sources for a given query using SerpAPI.
    """
    logger.info(f"[ScoutAgent] Discovering sources for query: {query}")
    if not SERPAPI_KEY:
        logger.error("SERPAPI_KEY not set in environment.")
        log_feedback("discover_sources_error", {"query": query, "error": "SERPAPI_KEY not set"})
        return []
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "num": 10
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        links = [r["link"] for r in data.get("organic_results", []) if "link" in r]
        log_feedback("discover_sources", {"query": query, "results": links})
        return links
    except Exception as e:
        logger.error(f"An error occurred during web search: {e}")
        log_feedback("discover_sources_error", {"query": query, "error": str(e)})
        return []

import asyncio

async def async_crawl_url(url: str, extraction_prompt: str | None) -> str:
    """
    Asynchronously crawl a given URL and extract content using crawl4ai AdaptiveCrawler.
    Supports custom extraction prompts.
    """
    logger.info(f"[ScoutAgent] Crawling URL: {url} with prompt: {extraction_prompt}")
    try:
        config = AdaptiveConfig()
        crawler = AdaptiveCrawler(config=config)
        state = await crawler.digest(start_url=url, query=extraction_prompt or "")
        content = "\n".join([r.content for r in getattr(state, 'results', []) if hasattr(r, 'content')])
        log_feedback("crawl_url", {"url": url, "prompt": extraction_prompt, "content_len": len(content)})
        return content
    except Exception as e:
        logger.error(f"An error occurred during crawling: {e}")
        log_feedback("crawl_url_error", {"url": url, "prompt": extraction_prompt, "error": str(e)})
        return ""

def crawl_url(url: str, extraction_prompt: str | None) -> str:
    """
    Synchronous wrapper for async_crawl_url.
    """
    return asyncio.run(async_crawl_url(url, extraction_prompt))

async def async_deep_crawl_site(domain: str, keywords: list[str]) -> list[str]:
    """
    Asynchronously perform a deep crawl on a specific website for given keywords using crawl4ai AdaptiveCrawler.
    """
    logger.info(f"[ScoutAgent] Deep crawling domain: {domain} for keywords: {keywords}")
    try:
        config = AdaptiveConfig()
        crawler = AdaptiveCrawler(config=config)
        start_url = f"https://{domain}"
        state = await crawler.digest(start_url=start_url, query=" ".join(keywords))
        links = [r.url for r in getattr(state, 'results', []) if hasattr(r, 'url')]
        log_feedback("deep_crawl_site", {"domain": domain, "keywords": keywords, "results": links})
        return links
    except Exception as e:
        logger.error(f"An error occurred during deep crawl: {e}")
        log_feedback("deep_crawl_site_error", {"domain": domain, "keywords": keywords, "error": str(e)})
        return []

def deep_crawl_site(domain: str, keywords: list[str]) -> list[str]:
    """
    Synchronous wrapper for async_deep_crawl_site.
    """
    return asyncio.run(async_deep_crawl_site(domain, keywords))
