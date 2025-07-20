# Model loading for Analyst Agent (Mistral-7B-Instruct-v0.2)
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

# tools.py for Analyst Agent


FEEDBACK_LOG = os.environ.get("ANALYST_FEEDBACK_LOG", "./feedback_analyst.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyst.tools")

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def score_bias(text: str) -> float:
    """
    Uses the LLM to score the bias of a given text.
    """
    logger.info(f"Scoring bias for text: '{text[:50]}...' ")
    try:
        model, tokenizer = get_mistral_model()
        if pipeline is None:
            raise ImportError("transformers pipeline is not available.")
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        prompt = f"Classify the following text for bias on a scale from 0 (neutral) to 1 (high bias).\nText: {text}"
        result = pipe(prompt, truncation=True, max_length=512)[0]
        score = float(result.get('score', 0.5))
        log_feedback("score_bias", {"text": text[:100], "score": score, "raw": result})
        return score
    except Exception as e:
        logger.error(f"Error in score_bias: {e}")
        log_feedback("score_bias_error", {"text": text[:100], "error": str(e)})
        return 0.5

def score_sentiment(text: str) -> float:
    """
    Uses the LLM to score the sentiment of a given text.
    """
    logger.info(f"Scoring sentiment for text: '{text[:50]}...' ")
    try:
        model, tokenizer = get_mistral_model()
        if pipeline is None:
            raise ImportError("transformers pipeline is not available.")
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        result = pipe(text, truncation=True, max_length=512)[0]
        # If the model returns a label, map it to a float score
        label = result.get('label', '').lower()
        if label in ["positive", "pos"]:
            score = 0.8
        elif label in ["negative", "neg"]:
            score = 0.2
        else:
            score = 0.5
        log_feedback("score_sentiment", {"text": text[:100], "score": score, "raw": result})
        return score
    except Exception as e:
        logger.error(f"Error in score_sentiment: {e}")
        log_feedback("score_sentiment_error", {"text": text[:100], "error": str(e)})
        return 0.5

def identify_entities(text: str) -> list[str]:
    """
    Uses the LLM to identify entities in a given text.
    """
    logger.info(f"Identifying entities in text: '{text[:50]}...' ")
    try:
        model, tokenizer = get_mistral_model()
        if pipeline is None:
            raise ImportError("transformers pipeline is not available.")
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        result = pipe(text, truncation=True, max_length=512)
        entities = list(set([r['word'] for r in result if r.get('entity_group', '').startswith('PER') or r.get('entity_group', '').startswith('ORG') or r.get('entity_group', '').startswith('LOC')]))
        log_feedback("identify_entities", {"text": text[:100], "entities": entities, "raw": result})
        return entities
    except Exception as e:
        logger.error(f"Error in identify_entities: {e}")
        log_feedback("identify_entities_error", {"text": text[:100], "error": str(e)})
        return []