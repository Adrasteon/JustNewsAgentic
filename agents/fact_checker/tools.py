
# Model loading for Fact-Checker Agent (Mistral-7B-Instruct-v0.2)
import os
import logging
from datetime import datetime
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.pipelines import pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PATH = os.environ.get("MISTRAL_7B_PATH", "./models/mistral-7b-instruct-v0.2")

def get_mistral_model():
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
# tools.py for Fact-Checker Agent

FEEDBACK_LOG = os.environ.get("FACT_CHECKER_FEEDBACK_LOG", "./feedback_fact_checker.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fact_checker.tools")

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def validate_is_news(content: str) -> bool:
    """
    Uses the LLM to validate if the given content is a news article.
    """
    logger.info(f"Validating if content is a news article: '{content[:50]}...' ")
    try:
        model, tokenizer = get_mistral_model()
        if pipeline is None:
            raise ImportError("transformers pipeline is not available.")
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        prompt = f"Is the following text a news article? Answer yes or no.\nText: {content}"
        result = pipe(prompt, truncation=True, max_length=512)[0]
        is_news = 'yes' in result['label'].lower() or result.get('score', 0) > 0.5
        log_feedback("validate_is_news", {"content": content[:100], "result": is_news, "raw": result})
        return is_news
    except Exception as e:
        logger.error(f"Error in validate_is_news: {e}")
        log_feedback("validate_is_news_error", {"content": content[:100], "error": str(e)})
        return False

def verify_claims(claims: list[str], sources: list[str]) -> dict:
    """
    Uses the LLM to verify a list of claims against a list of sources.
    """
    logger.info(f"Verifying claims: {claims} with sources: {sources}")
    try:
        model, tokenizer = get_mistral_model()
        if pipeline is None:
            raise ImportError("transformers pipeline is not available.")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
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