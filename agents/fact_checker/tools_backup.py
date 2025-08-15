"""
Fact Checker V2 - Production-Ready Multi-Model AI Architecture
Specialized fact verification with 5 AI models matching Scout V2 standard

AI Models:
1. DistilBERT-base: Fact verification (factual/questionable classification)
2. RoBERTa-base: Source credibility assessment (reliability scoring)
3. BERT-large: Contradiction detection (logical consistency)
4. SentenceTransformers: Evidence retrieval (semantic search)
5. spaCy NER: Claim extraction (verifiable claims identification)

Performance: Production-ready with GPU acceleration and professional error handling
V4 Compliance: TensorRT-ready multi-model architecture with MCP bus integration
Dependencies: transformers, sentence-transformers, spacy, torch, numpy
"""

import logging
import os
from datetime import datetime

# Import V2 Engine
try:
    from agents.fact_checker.fact_checker_v2_engine import (
        get_fact_checker_engine,
        initialize_fact_checker_v2,
    )

    FACT_CHECKER_V2_AVAILABLE = True
except ImportError as e:
    FACT_CHECKER_V2_AVAILABLE = False
    logging.error(f"❌ Fact Checker V2 Engine not available: {e}")

# Fallback imports for compatibility
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

# Environment configuration
FEEDBACK_LOG = os.environ.get(
    "FACT_CHECKER_FEEDBACK_LOG", "./feedback_fact_checker.log"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fact_checker.tools")

# Initialize V2 Engine on module load
if FACT_CHECKER_V2_AVAILABLE:
    initialize_fact_checker_v2()
    logger.info("🚀 Fact Checker V2 Engine initialized with 5 AI models")
else:
    logger.warning("⚠️ Running in fallback mode - V2 engine unavailable")


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
            batch_size=OPTIMIZED_BATCH_SIZE,
        )

        joined_sources = "\n".join(sources)
        results = {}
        for claim in claims:
            prompt = f"Verify the following claim against the provided sources.\nClaim: {claim}\nSources: {joined_sources}\nIs this claim supported? Answer 'verified' or 'not verified'."
            output = (
                pipe(prompt, max_new_tokens=64)[0]["generated_text"].strip().lower()
            )
            results[claim] = "verified" if "verified" in output else "not verified"

        log_feedback(
            "verify_claims", {"claims": claims, "sources": sources, "results": results}
        )
        return results
    except Exception as e:
        logger.error(f"Error in verify_claims: {e}")
        log_feedback(
            "verify_claims_error",
            {"claims": claims, "sources": sources, "error": str(e)},
        )
        return {claim: "error" for claim in claims}
