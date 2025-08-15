"""
Scout Agent Production Crawlers Module

This module provides production-scale news crawling capabilities for the Scout Agent,
integrating high-speed web scraping with the existing Crawl4AI deep crawling system.

Key Components:
- orchestrator.py: Main orchestration and multi-site coordination
- sites/bbc_crawler.py: Ultra-fast BBC crawling (8.14+ articles/second)
- sites/bbc_ai_crawler.py: AI-enhanced BBC crawling (0.86+ articles/second)

Future Expansion:
- sites/cnn_crawler.py: CNN news crawling
- sites/reuters_crawler.py: Reuters news crawling
- sites/guardian_crawler.py: Guardian news crawling
- sites/nytimes_crawler.py: New York Times crawling

Performance Targets:
- Ultra-fast mode: 8+ articles/second per site
- AI-enhanced mode: 0.8+ articles/second per site
- Multi-site concurrent crawling supported
- Production-scale capacity: 100K+ articles/day
"""

from .orchestrator import ProductionCrawlerOrchestrator

__version__ = "1.0.0"
__all__ = ["ProductionCrawlerOrchestrator"]
