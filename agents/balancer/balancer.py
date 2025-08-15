"""
Balancer Agent V1 - News Article Neutralization and Integration
=============================================================

Architecture: Uses JustNews production models for sentiment, bias, fact-checking, summarization, and semantic embeddings. Integrates a new quote extraction model. MCP bus compatible.

Models:
- Sentiment: RoBERTa (Analyst Agent)
- Bias: martin-ha/toxic-comment-model (Scout Agent)
- Fact Checking: DistilBERT, RoBERTa, BERT-large, SentenceTransformers, spaCy NER (Fact Checker Agent)
- Summarization/Neutralization: BART, T5 (Synthesizer Agent)
- Embeddings: SentenceTransformers (Synthesizer/Memory)
- Quote Extraction: Jean-Baptiste/roberta-large-ner-quotations (new)

Status: V1 Prototype - MCP bus ready
"""

import structlog
from typing import List, Dict, Any
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests

# MCP Bus Integration
import requests as mcp_requests
MCP_BUS_URL = "http://localhost:8000"  # Update as needed

logger = structlog.get_logger("balancer_agent")

def get_device() -> int:
    return 0 if torch.cuda.is_available() else -1

# Model initialization (JustNews conventions)
def get_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=get_device())
    except Exception as e:
        logger.error("model_load_error", model="sentiment", error=str(e), status="error")
        raise RuntimeError(f"Failed to load sentiment model: {e}")

def get_bias_pipeline():
    try:
        return pipeline("text-classification", model="martin-ha/toxic-comment-model", device=get_device())
    except Exception as e:
        logger.error("model_load_error", model="bias", error=str(e), status="error")
        raise RuntimeError(f"Failed to load bias model: {e}")

def get_fact_checker_pipelines():
    try:
        return {
            "distilbert": pipeline("text-classification", model="distilbert-base-uncased", device=get_device()),
            "roberta": pipeline("text-classification", model="roberta-base", device=get_device()),
            "bert_large": pipeline("text-classification", model="bert-large-uncased", device=get_device()),
            "sentence_transformer": SentenceTransformer("all-MiniLM-L6-v2", device=get_device()),
            # spaCy NER would be loaded separately if needed
        }
    except Exception as e:
        logger.error("model_load_error", model="fact_checker", error=str(e), status="error")
        raise RuntimeError(f"Failed to load fact checker models: {e}")

def get_summarization_pipeline():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", device=get_device())
    except Exception as e:
        logger.error("model_load_error", model="summarization", error=str(e), status="error")
        raise RuntimeError(f"Failed to load summarization model: {e}")

def get_neutralization_pipeline():
    try:
        return pipeline("text2text-generation", model="google/flan-t5-base", device=get_device())
    except Exception as e:
        logger.error("model_load_error", model="neutralization", error=str(e), status="error")
        raise RuntimeError(f"Failed to load neutralization model: {e}")

def get_quote_extraction_pipeline():
    try:
        return pipeline("token-classification", model="Jean-Baptiste/roberta-large-ner-quotations", device=get_device())
    except Exception as e:
        logger.error("model_load_error", model="quote_extraction", error=str(e), status="error")
        raise RuntimeError(f"Failed to load quote extraction model: {e}")

def fetch_article(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        article = ' '.join(paragraphs)
        logger.info("article_fetched", url=url, status="success")
        return article
    except Exception as e:
        logger.error("article_fetch_error", url=url, error=str(e), status="error")
        return ""

def call_agent_tool(agent: str, tool: str, *args, **kwargs) -> Any:
    payload = {
        "agent": agent,
        "tool": tool,
        "args": list(args),
        "kwargs": kwargs
    }
    response = mcp_requests.post(f"{MCP_BUS_URL}/call", json=payload)
    response.raise_for_status()
    return response.json()


# --- Multi-Agent Delegation ---
def analyze_article(text: str) -> Dict[str, Any]:
    """Delegate analysis to Analyst, Fact Checker, and Synthesizer agents via MCP bus."""
    try:
        sentiment = call_agent_tool("analyst", "score_sentiment", text)
    except Exception as e:
        logger.error("analyst_sentiment_error", error=str(e), status="error")
        sentiment = get_sentiment_pipeline()(text)
    try:
        bias = call_agent_tool("analyst", "score_bias", text)
    except Exception as e:
        logger.error("analyst_bias_error", error=str(e), status="error")
        bias = get_bias_pipeline()(text)
    try:
        fact_result = call_agent_tool("fact_checker", "verify_claims", text)
    except Exception as e:
        logger.error("fact_checker_error", error=str(e), status="error")
        fact_result = {"error": str(e)}
    try:
        summary = call_agent_tool("synthesizer", "summarize_content_bart", text)
    except Exception as e:
        logger.error("synthesizer_error", error=str(e), status="error")
        summary = summarize_article(text)
    logger.info("multi_agent_analysis_completed", status="success")
    return {"sentiment": sentiment, "bias": bias, "fact": fact_result, "summary": summary}

def extract_quotes(text: str) -> List[str]:
    quote_pipe = get_quote_extraction_pipeline()
    results = quote_pipe(text)
    quotes = [r['word'] for r in results if r['entity_group'] == 'QUOTE']
    logger.info("quotes_extracted", num_quotes=len(quotes), status="success")
    return quotes

def summarize_article(text: str) -> str:
    summarizer = get_summarization_pipeline()
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def neutralize_text(text: str) -> str:
    neutralizer = get_neutralization_pipeline()
    result = neutralizer(text)
    return result[0]['generated_text']


def log_gpu_usage(operation: str):
    """Log current GPU memory usage for operation."""
    try:
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"GPU memory usage for {operation}: {memory_used:.2f} GB")
    except Exception as e:
        logger.warning(f"GPU usage logging failed: {e}")

def batch_fetch_articles(urls: List[str], batch_size: int = 4) -> List[str]:
    """Batch fetch articles with configurable batch size and GPU logging."""
    articles = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        batch_articles = [fetch_article(url) for url in batch]
        articles.extend(batch_articles)
        log_gpu_usage(f"batch_fetch_articles batch {i//batch_size+1}")
    return articles


def log_training_example(task: str, input: dict, output: dict, critique: str = ""):
    """Log balancing actions and outcomes for continuous learning."""
    try:
        # Example: send to training system via MCP bus
        payload = {
            "task": task,
            "input": input,
            "output": output,
            "critique": critique
        }
        response = mcp_requests.post(f"{MCP_BUS_URL}/training/log_example", json=payload)
        response.raise_for_status()
        logger.info(f"Logged training example for task: {task}")
    except Exception as e:
        logger.warning(f"Training log failed: {e}")

def generate_balanced_article(main_article: str, quotes: List[str], alt_statements: List[str]) -> str:
    summary = summarize_article(main_article)
    neutral = neutralize_text(summary)
    article = neutral + "\n\nCounter-balancing statements:\n"
    for q in quotes:
        article += f"- {q}\n"
    for s in alt_statements:
        article += f"- {s}\n"
    logger.info("balanced_article_generated", num_quotes=len(quotes), num_alt_statements=len(alt_statements), status="success")
    # Log for continuous learning
    log_training_example(
        task="balance_article",
        input={"main_article": main_article, "quotes": quotes, "alt_statements": alt_statements},
        output={"balanced_article": article},
        critique=""
    )
    return article

def main():
    main_url = "https://www.bbc.co.uk/news/articles/c36jkydpxx6o"
    alt_urls = [
        "https://www.cambridge-news.co.uk/news/local-news/saved-bus-routes-see-no-32258242",
        "https://www.elystandard.co.uk/news/25387266.cambridgeshire-mayor-gets-funding-village-bus-routes/"
    ]
    main_article = fetch_article(main_url)
    analysis = analyze_article(main_article)
    alt_articles = batch_fetch_articles(alt_urls)
    quotes = []
    for alt in alt_articles:
        quotes.extend(extract_quotes(alt))
    balanced_article = generate_balanced_article(main_article, quotes, alt_articles)
    print(balanced_article)


# MCP Bus Registration and Health Check (JustNews V4 pattern)
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn

# FastAPI app initialization (must be before handlers and endpoints)

# FastAPI app initialization with slowapi rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom error handler for request validation
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("request_validation_error", error=str(exc), status="error")
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "detail": exc.errors(),
            "body": exc.body,
        },
    )

# Custom error handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error("http_exception", error=str(exc.detail), status="error", code=exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "HTTP_ERROR",
            "detail": exc.detail,
        },
    )

class AgentRegistration(BaseModel):
    agent_name: str
    description: str
    version: str
    endpoints: Dict[str, str]

@app.post("/register")
def register_agent(reg: AgentRegistration) -> Dict[str, Any]:
    """Register balancer agent with MCP bus."""
    logger.info(f"Agent registered: {reg.agent_name} v{reg.version}")
    return {"status": "success", "agent": reg.agent_name}

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint for MCP bus compliance."""
    return {"status": "ok", "agent": "balancer"}

# --- MCP Bus Health Check and Status Endpoint ---
import time

def check_mcp_bus_health() -> Dict[str, Any]:
    """Check MCP bus health by querying /health endpoint."""
    try:
        start = time.time()
        response = mcp_requests.get(f"{MCP_BUS_URL}/health", timeout=2)
        response.raise_for_status()
        latency = time.time() - start
        return {"status": "ok", "latency_sec": round(latency, 3)}
    except Exception as e:
        logger.error("mcp_bus_health_error", error=str(e), status="error")
        return {"status": "error", "detail": str(e)}

def get_model_status() -> Dict[str, Any]:
    """Check if all models are loaded and available."""
    status = {}
    for name, loader in {
        "sentiment": get_sentiment_pipeline,
        "bias": get_bias_pipeline,
        "fact_checker": get_fact_checker_pipelines,
        "summarization": get_summarization_pipeline,
        "neutralization": get_neutralization_pipeline,
        "quote_extraction": get_quote_extraction_pipeline
    }.items():
        try:
            loader()
            status[name] = "ok"
        except Exception as e:
            status[name] = f"error: {e}"
    return status

@app.get("/status")
def status_endpoint() -> Dict[str, Any]:
    """Report agent health, MCP bus connectivity, and model status."""
    mcp_status = check_mcp_bus_health()
    model_status = get_model_status()
    return {
        "agent": "balancer",
        "mcp_bus": mcp_status,
        "models": model_status
    }

# --- Resource Monitoring Endpoint ---
import psutil

@app.get("/resource_status")
def resource_status() -> Dict[str, Any]:
    """Report current CPU, memory, and GPU usage."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    memory_gb = mem.used / 1024**3
    gpu_available = torch.cuda.is_available()
    gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3 if gpu_available else 0.0
    return {
        "cpu_percent": cpu_percent,
        "memory_gb": round(memory_gb, 2),
        "gpu_available": gpu_available,
        "gpu_memory_gb": round(gpu_memory_gb, 2)
    }

if __name__ == "__main__":
    # Start FastAPI app for MCP bus integration
    uvicorn.run(app, host="0.0.0.0", port=8009)
