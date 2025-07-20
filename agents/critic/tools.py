# tools.py for Critic Agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def critique_synthesis(summary: str, source_ids: list[str]) -> str:
    """
    Critiques a synthesis of a story.
    Uses a simple rule-based check as a placeholder for model-based logic.
    """
    logger.info(f"Critiquing synthesis for summary: '{summary[:50]}...' with sources: {source_ids}")
    # Simple rule: if summary is too short, flag as incomplete
    if len(summary.split()) < 50:
        return "Synthesis is too short and may lack detail."
    return "Synthesis appears sufficiently detailed."

def critique_neutrality(original_text: str, neutralized_text: str) -> str:
    """
    Critiques the neutrality of a text.
    Uses a simple rule-based check as a placeholder for model-based logic.
    """
    logger.info(f"Critiquing neutrality for text: '{original_text[:50]}...' ")
    # Simple rule: if strong adjectives/adverbs remain, flag as not neutral
    strong_words = ["amazing", "terrible", "clearly", "obviously", "undeniably"]
    if any(word in neutralized_text.lower() for word in strong_words):
        return "Neutralized text still contains strong language."
    return "Text appears neutral."