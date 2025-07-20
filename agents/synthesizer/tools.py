# tools.py for Synthesizer Agent

# MCP Tool Definitions

def cluster_articles(article_ids: list[str]) -> list[list[str]]:
    # Placeholder: group articles into a single cluster
    # In production, use embeddings and clustering
    if not article_ids:
        return []
    return [article_ids]

def neutralize_text(text: str) -> str:
    # Placeholder: remove strong adjectives/adverbs
    import re
    neutral = re.sub(r'\b(amazing|terrible|clearly|obviously|undeniably)\b', '', text, flags=re.IGNORECASE)
    return neutral.strip()

def aggregate_cluster(article_ids: list[str]) -> str:
    # Placeholder: concatenate article IDs as a summary
    if not article_ids:
        return ""
    return f"Summary of articles: {', '.join(article_ids)}"
