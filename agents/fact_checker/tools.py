# tools.py for Fact-Checker Agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_is_news(content: str) -> bool:
    """
    Validates if the given content is a news article.
    Uses a simple rule-based check as a placeholder for model-based logic.
    """
    logger.info(f"Validating if content is a news article: '{content[:50]}...' ")
    # Simple rule: must contain at least 100 words and mention a date or location
    if len(content.split()) < 100:
        logger.info("Content too short to be a news article.")
        return False
    keywords = ["202", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "New York", "London", "Paris"]
    if not any(k in content for k in keywords):
        logger.info("Content does not mention a date or location.")
        return False
    return True

def verify_claims(claims: list[str], sources: list[str]) -> dict:
    """
    Verifies a list of claims against a list of sources.
    Uses a simple keyword match as a placeholder for model-based logic.
    """
    logger.info(f"Verifying claims: {claims} with sources: {sources}")
    results = {}
    for claim in claims:
        found = any(claim.lower() in src.lower() for src in sources)
        results[claim] = "verified" if found else "not verified"
    return results