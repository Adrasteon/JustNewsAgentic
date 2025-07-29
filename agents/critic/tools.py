# Optimized Critic Configuration
# Phase 1 Memory Optimization: Context and batch size optimization

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
MODEL_NAME = "microsoft/DialoGPT-medium"
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/dialogpt-medium")
OPTIMIZED_MAX_LENGTH = 1512  # Reduced from 2048 (analysis tasks need moderate context)
OPTIMIZED_BATCH_SIZE = 8     # Memory-efficient batching for critique tasks

FEEDBACK_LOG = os.environ.get("CRITIC_FEEDBACK_LOG", "./feedback_critic.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("critic.tools")

def get_dialog_model():
    """Load optimized DialoGPT-medium model with memory-efficient configuration."""
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers library is not installed.")
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        logger.info(f"Downloading {MODEL_NAME} to {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
    else:
        logger.info(f"Loading {MODEL_NAME} from local cache {MODEL_PATH}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

def log_feedback(event: str, details: dict):
    """Log feedback for continual learning and retraining."""
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def critique_synthesis(summary: str, source_texts: list[str]) -> str:
    """Use optimized model to critique summaries with reduced memory footprint."""
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
        
        joined_sources = "\n".join(source_texts)
        prompt = f"Critique the following summary for hallucinations, omissions, and factual consistency.\nSummary: {summary}\nSources: {joined_sources}"
        result = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
        
        # Optional: Fact-checking pipeline integration
        fact_check_result = None
        if os.environ.get("CRITIC_USE_FACTCHECK", "0") == "1":
            try:
                fact_check_result = "[Fact-checking not implemented]"
            except Exception as fc_err:
                fact_check_result = f"Fact-checking error: {fc_err}"
        
        log_feedback("critique_synthesis", {"summary": summary, "sources": source_texts, "output": result, "fact_check": fact_check_result})
        return result + (f"\nFact-check: {fact_check_result}" if fact_check_result else "")
    except Exception as e:
        logger.error(f"Error in critique_synthesis: {e}")
        log_feedback("critique_synthesis_error", {"summary": summary, "error": str(e)})
        return "[Error in critique_synthesis]"

def critique_neutrality(original_text: str, neutralized_text: str) -> str:
    """Use optimized model to critique neutrality with memory efficiency."""
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
        
        prompt = f"Compare the following original and neutralized texts. Flag any factual drift or loss of nuance.\nOriginal: {original_text}\nNeutralized: {neutralized_text}"
        result = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
        log_feedback("critique_neutrality", {"original": original_text, "neutralized": neutralized_text, "output": result})
        return result
    except Exception as e:
        logger.error(f"Error in critique_neutrality: {e}")
        log_feedback("critique_neutrality_error", {"original": original_text, "error": str(e)})
        return "[Error in critique_neutrality]"
