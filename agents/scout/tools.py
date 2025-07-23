# Model loading for Scout Agent (Llama-3-8B-Instruct)
import os
import logging
import requests
from datetime import datetime

FEEDBACK_LOG = os.environ.get("SCOUT_FEEDBACK_LOG", "./feedback_scout.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scout.tools")

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def discover_sources(*args, **kwargs):
    """
    Discover sources for a given query using the crawl4ai Docker container.
    """
    logger.info(f"[ScoutAgent] Discovering sources with args: {args}, kwargs: {kwargs}")
    try:
        # Call the running crawl4ai container
        response = requests.post("http://localhost:32768/discover_sources", json={"args": args, "kwargs": kwargs})
        response.raise_for_status()
        links = response.json()
        log_feedback("discover_sources", {"args": args, "results": links})
        return links
    except Exception as e:
        logger.error(f"An error occurred during web search: {e}")
        log_feedback("discover_sources_error", {"args": args, "error": str(e)})
        return []

def crawl_url(*args, **kwargs):
    """
    Crawl a given URL and extract content. Interacts with crawl4ai Docker container.
    """
    logger.info(f"[ScoutAgent] Crawling URL with args: {args}, kwargs: {kwargs}")
    try:
        # Call the running crawl4ai container
        response = requests.post("http://localhost:32768/crawl_url", json={"args": args, "kwargs": kwargs})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred during crawling: {e}")
        log_feedback("crawl_url_error", {"args": args, "error": str(e)})
        return []

def deep_crawl_site(*args, **kwargs):
    """
    Perform a deep crawl on a specific website for given keywords. Interacts with crawl4ai Docker container.
    """
    logger.info(f"[ScoutAgent] Deep crawling site with args: {args}, kwargs: {kwargs}")
    try:
        # Call the running crawl4ai container
        response = requests.post("http://localhost:32768/deep_crawl_site", json={"args": args, "kwargs": kwargs})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred during deep crawl: {e}")
        log_feedback("deep_crawl_site_error", {"args": args, "error": str(e)})
        return []
