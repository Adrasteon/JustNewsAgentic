# Optimized Fact Checker Configuration
# Phase 1 Memory Optimization: DialoGPT-large → DialoGPT-medium + Context optimization

import os
import logging
from datetime import datetime
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

# PHASE 1 OPTIMIZATIONS APPLIED
MODEL_NAME = "microsoft/DialoGPT-medium"  # Downgraded from DialoGPT-large (Scout pre-filtering enables this)
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/dialogpt-medium")
OPTIMIZED_MAX_LENGTH = 1512  # Reduced from 2048 (news articles are typically shorter)
OPTIMIZED_BATCH_SIZE = 16    # Memory-efficient batching

FEEDBACK_LOG = os.environ.get("FACT_CHECKER_FEEDBACK_LOG", "./feedback_fact_checker.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fact_checker.tools")

def get_dialog_model():
    """Load optimized DialoGPT-medium model with memory-efficient configuration."""
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

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def validate_is_news(content: str) -> bool:
    """Validate if the given content qualifies as news."""
    logger.info(f"Validating content for news: {content[:50]}...")
    keywords = ["breaking", "report", "headline", "news"]
    is_news = any(keyword in content.lower() for keyword in keywords)
    log_feedback("validate_is_news", {"content": content, "is_news": is_news})
    return is_news

def verify_claims(claims: list[str], sources: list[str]) -> dict:
    """Uses optimized model to verify claims with reduced memory footprint."""
    logger.info(f"Verifying claims: {claims} with sources: {sources}")
    try:
        model, tokenizer = get_dialog_model()
        if pipeline is None:
            raise ImportError("transformers pipeline is not available.")
        
        # Use optimized pipeline configuration
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_length=OPTIMIZED_MAX_LENGTH,
            batch_size=OPTIMIZED_BATCH_SIZE
        )
        
        joined_sources = "\n".join(sources)
        results = {}
        for claim in claims:
            prompt = f"Verify the following claim against the provided sources.\nClaim: {claim}\nSources: {joined_sources}\nIs this claim supported? Answer 'verified' or 'not verified'."
            output = pipe(prompt, max_new_tokens=64)[0]["generated_text"].strip().lower()
            results[claim] = "verified" if "verified" in output else "not verified"
        
        log_feedback("verify_claims", {"claims": claims, "sources": sources, "results": results})
        return results
    except Exception as e:
        logger.error(f"Error in verify_claims: {e}")
        log_feedback("verify_claims_error", {"claims": claims, "sources": sources, "error": str(e)})
        return {claim: "error" for claim in claims}
