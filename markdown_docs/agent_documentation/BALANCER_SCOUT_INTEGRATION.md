# Balancer Agent V1 - Enhanced Integration with Scout Agent

## Overview
The Balancer agent is responsible for ensuring news analysis is unbiased, balanced, and robust by integrating multiple sources and applying its own multi-model analysis pipeline. With the latest improvements, the Balancer agent now leverages the Scout agent's advanced source discovery and intelligence scoring, while retaining its own analysis functions for sentiment, bias, fact-checking, summarization, and quote extraction.

## Key Changes & Improvements

### 1. Scout Intelligence Integration
- The Balancer agent now calls the Scout agent to discover related articles and sources for any main news item.
- For each source, the full Scout intelligence output (including scores, recommendations, and metadata) is retrieved and stored.
- Scout's analysis is used to prioritize sources and provide additional context, but does not replace the Balancer's own analysis pipeline.

### 2. Dual Analysis Pipeline
- For every alternative source, the Balancer agent runs its own analysis (sentiment, bias, fact-checking, summarization, quote extraction) in addition to storing Scout's intelligence.
- Both sets of results are logged and returned in the API response, providing transparency and richer context for downstream reporting and UI.

### 3. Transparent Logging & Reporting
- All analyses (Scout and Balancer) are logged for each source, enabling auditability and future enhancements in reporting or visualization.
- The API response for `/web_search_balance` now includes a list of sources, each with Scout and Balancer analysis, plus extracted quotes.

### 4. Balanced Article Generation
- The final balanced article is generated using the Balancer's own pipeline, with Scout's recommendations and scores available for reference and prioritization.
- This ensures that the Balancer agent remains the central authority for bias mitigation and content neutralization.

## Relationship to Scout Agent
- **Scout Agent Role:** Provides advanced source discovery, web crawling, and AI-powered intelligence scoring for news articles.
- **Balancer Agent Role:** Integrates sources from Scout, applies its own multi-model analysis, and generates balanced, neutralized articles.
- **Integration Pattern:** Scout's output is used to enhance, not replace, the Balancer's analysis. Both sets of results are available for transparency and future UI/reporting improvements.

## Example API Response Structure
```json
{
  "status": "success",
  "main_url": "...",
  "sources": [
    {
      "url": "...",
      "scout": { ... },
      "balancer": { ... },
      "quotes": [ ... ]
    },
    ...
  ],
  "balanced_article": "..."
}
```

## Logging Example
- Each source's Scout and Balancer analysis is logged:
  - `logger.info("source_analysis", url=..., scout=..., balancer=...)`

## Benefits
- **Transparency:** Both Scout and Balancer analyses are available for review.
- **Auditability:** All intelligence and analysis steps are logged.
- **Extensibility:** Future UI and reporting can leverage both sets of metrics.
- **Robustness:** Balancer retains full control over bias mitigation and content neutralization.

## References
- See `agents/balancer/balancer.py` for implementation details.
- See `agents/scout/main.py` for Scout agent API and intelligence output.

---
*Last updated: August 15, 2025*
*Author: GitHub Copilot*
