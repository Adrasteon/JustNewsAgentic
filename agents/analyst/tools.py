# tools.py for Analyst Agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def score_bias(text: str) -> float:
    """
    Scores the bias of a given text.
    Uses a simple rule-based check as a placeholder for model-based logic.
    """
    logger.info(f"Scoring bias for text: '{text[:50]}...' ")
    # Simple rule: if 'allegedly' or 'reportedly' present, lower bias
    if any(word in text.lower() for word in ["allegedly", "reportedly"]):
        return 0.2
    # If 'clearly', 'obviously', 'undeniably' present, higher bias
    if any(word in text.lower() for word in ["clearly", "obviously", "undeniably"]):
        return 0.8
    return 0.5

def score_sentiment(text: str) -> float:
    """
    Scores the sentiment of a given text.
    Uses a simple rule-based check as a placeholder for model-based logic.
    """
    logger.info(f"Scoring sentiment for text: '{text[:50]}...' ")
    # Simple rule: count positive/negative words
    positive = ["good", "great", "positive", "success", "win"]
    negative = ["bad", "poor", "negative", "fail", "loss"]
    pos_count = sum(word in text.lower() for word in positive)
    neg_count = sum(word in text.lower() for word in negative)
    if pos_count > neg_count:
        return 0.8
    if neg_count > pos_count:
        return 0.2
    return 0.5

def identify_entities(text: str) -> list[str]:
    """
    Identifies entities in a given text.
    Uses a simple rule-based check as a placeholder for model-based logic.
    """
    logger.info(f"Identifying entities in text: '{text[:50]}...' ")
    # Simple rule: extract capitalized words as entities
    entities = [word for word in text.split() if word.istitle()]
    return list(set(entities))