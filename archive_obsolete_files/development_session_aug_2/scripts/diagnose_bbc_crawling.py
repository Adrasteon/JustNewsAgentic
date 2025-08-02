#!/usr/bin/env python3
"""
Diagnostic test for BBC crawling to understand content extraction issues
"""

import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
import json

async def diagnose_bbc_crawling():
    """Diagnose why BBC pages return empty content."""
    
    test_urls = [
        "https://www.bbc.co.uk/news",
        "https://www.bbc.co.uk/news/world",
        "https://www.bbc.co.uk/news/business",
        "https://www.bbc.co.uk/news/technology"
    ]
    
    print("🔍 Diagnosing BBC Crawling Issues...")
    print("=" * 60)
    
    async with AsyncWebCrawler() as crawler:
        for url in test_urls:
            print(f"\n📰 Testing: {url}")
            
            try:
                # Basic crawl
                config = CrawlerRunConfig(
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    cache_mode=CacheMode.BYPASS,
                    verbose=True
                )
                
                result = await crawler.arun(url, config=config)
                
                print(f"✅ Success: {result.success}")
                print(f"📊 Status Code: {getattr(result, 'status_code', 'N/A')}")
                print(f"📄 Title: {getattr(result, 'title', 'No title')[:100]}")
                print(f"📝 Content Length: {len(getattr(result, 'content', ''))}")
                print(f"🏷️ Metadata Keys: {list(getattr(result, 'metadata', {}).keys())}")
                
                # Show first 500 chars of content
                content = getattr(result, 'content', '')
                if content:
                    print(f"📰 Content Preview: {content[:500]}...")
                else:
                    print("❌ No content extracted")
                
                # Show metadata sample
                metadata = getattr(result, 'metadata', {})
                if metadata:
                    print(f"🏷️ Metadata Sample:")
                    for key, value in list(metadata.items())[:5]:
                        print(f"   {key}: {str(value)[:100]}")
                
                # Show raw HTML sample if available
                raw_html = getattr(result, 'raw_html', '')
                if raw_html:
                    print(f"🔍 Raw HTML Length: {len(raw_html)}")
                    print(f"🔍 Raw HTML Preview: {raw_html[:300]}...")
                else:
                    print("❌ No raw HTML available")
                
            except Exception as e:
                print(f"❌ Error crawling {url}: {e}")
            
            print("-" * 40)
    
    # Test specific article URL
    print(f"\n🎯 Testing specific BBC article...")
    specific_urls = [
        "https://www.bbc.co.uk/news/articles/cx2l98zddpno",  # Example article format
        "https://www.bbc.com/news/world",  # Try .com instead of .co.uk
    ]
    
    async with AsyncWebCrawler() as crawler:
        for url in specific_urls:
            print(f"\n📰 Testing specific: {url}")
            try:
                config = CrawlerRunConfig(
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    cache_mode=CacheMode.BYPASS,
                    verbose=True
                )
                
                result = await crawler.arun(url, config=config)
                
                print(f"✅ Success: {result.success}")
                print(f"📝 Content Length: {len(getattr(result, 'content', ''))}")
                
                content = getattr(result, 'content', '')
                if content:
                    print(f"📰 Content Preview: {content[:500]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(diagnose_bbc_crawling())
