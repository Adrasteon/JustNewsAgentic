#!/usr/bin/env python3
"""
Find Current Article URLs and Test Crawl4AI Properly
"""

import asyncio
from crawl4ai import AsyncWebCrawler
import re

async def test_with_current_articles():
    """Find current articles and test Crawl4AI content extraction"""
    
    print("🔍 TESTING CRAWL4AI WITH CURRENT ARTICLES")
    print("=" * 60)
    
    # Test with different news sites and their current content
    news_sites = [
        {
            "name": "BBC World",
            "section_url": "https://www.bbc.com/news/world",
            "article_pattern": r'https://www\.bbc\.com/news/world-[\w-]+-\d+'
        },
        {
            "name": "Reuters World", 
            "section_url": "https://www.reuters.com/world/",
            "article_pattern": r'https://www\.reuters\.com/world/[\w-]+/[\w-]+-\d{4}-\d{2}-\d{2}/'
        }
    ]
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        for site in news_sites:
            print(f"\n📰 TESTING: {site['name']}")
            print("-" * 40)
            
            # First get the section page
            try:
                section_result = await crawler.arun(
                    url=site['section_url'],
                    bypass_cache=True
                )
                
                if section_result.success:
                    print(f"✅ Section page loaded: {site['section_url']}")
                    
                    # Extract article URLs from the HTML
                    article_urls = []
                    if section_result.html:
                        # Look for article URLs in the HTML
                        article_matches = re.findall(r'href="([^"]*)"', section_result.html)
                        
                        for match in article_matches:
                            # Make absolute URLs
                            if match.startswith('/'):
                                if 'bbc.com' in site['section_url']:
                                    match = 'https://www.bbc.com' + match
                                elif 'reuters.com' in site['section_url']:
                                    match = 'https://www.reuters.com' + match
                            
                            # Check if it matches article pattern
                            if ('bbc.com/news/' in match and len(match.split('/')) >= 5 and 
                                match.split('/')[-1].isdigit() and len(match.split('/')[-1]) >= 8):
                                article_urls.append(match)
                            elif ('reuters.com' in match and '/world/' in match and 
                                  len(match.split('/')) >= 6):
                                article_urls.append(match)
                    
                    # Remove duplicates and test first few
                    article_urls = list(set(article_urls))[:3]
                    print(f"Found {len(article_urls)} potential articles")
                    
                    for i, article_url in enumerate(article_urls):
                        await test_single_article(crawler, article_url, f"{site['name']} Article {i+1}")
                        
                else:
                    print(f"❌ Failed to load section: {section_result.error_message}")
                    
            except Exception as e:
                print(f"❌ Error with {site['name']}: {e}")

async def test_single_article(crawler, url, test_name):
    """Test a single article URL with Crawl4AI"""
    
    print(f"\n📄 {test_name}: {url[:60]}...")
    
    try:
        result = await crawler.arun(
            url=url,
            bypass_cache=True,
            word_count_threshold=50,
            # Try different extraction settings
            remove_overlay_elements=True,
            simulate_user=True
        )
        
        if result.success:
            title = result.metadata.get('title', 'No title')
            print(f"   ✅ Title: {title[:50]}...")
            
            # Analyze content quality
            if result.cleaned_html:
                cleaned_len = len(result.cleaned_html)
                word_count = len(result.cleaned_html.split())
                
                print(f"   📊 Cleaned HTML: {cleaned_len:,} chars, {word_count} words")
                
                # Check for article indicators
                has_paragraphs = result.cleaned_html.count('<p>') > 3
                has_minimal_html = result.cleaned_html.count('<') < (cleaned_len / 100)  # Less than 1% HTML tags
                has_content = word_count > 100
                
                # Look for actual article content patterns
                content_preview = result.cleaned_html[:500]
                article_indicators = ['said', 'according to', 'reported', 'announced', 'stated']
                has_news_language = any(indicator in content_preview.lower() for indicator in article_indicators)
                
                print(f"   🔍 Content Analysis:")
                print(f"      Paragraphs: {'✅' if has_paragraphs else '❌'}")
                print(f"      Minimal HTML: {'✅' if has_minimal_html else '❌'}")
                print(f"      Sufficient content: {'✅' if has_content else '❌'}")
                print(f"      News language: {'✅' if has_news_language else '❌'}")
                
                if has_content and has_minimal_html and has_news_language:
                    print(f"   ✅ HIGH QUALITY ARTICLE CONTENT!")
                    
                    # Show clean content sample
                    clean_text = re.sub(r'<[^>]+>', '', result.cleaned_html)
                    sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 20]
                    
                    print(f"   📖 Content Sample:")
                    for i, sentence in enumerate(sentences[:3]):
                        print(f"      {i+1}. {sentence[:80]}...")
                    
                    return True
                else:
                    print(f"   ⚠️ Content needs improvement")
            
            # Also check markdown as comparison
            if result.markdown:
                markdown_len = len(result.markdown)
                print(f"   📊 Markdown: {markdown_len:,} chars")
                
                if markdown_len > 0 and '<' not in result.markdown:
                    print(f"   ✅ Markdown is clean")
        else:
            print(f"   ❌ Failed: {result.error_message}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

async def test_extraction_strategies():
    """Test different Crawl4AI extraction strategies"""
    
    print(f"\n🧪 TESTING EXTRACTION STRATEGIES")
    print("=" * 40)
    
    # Use a reliable news URL for testing
    test_url = "https://www.bbc.com/news/world"  # Section page with articles
    
    strategies = [
        {"name": "Default", "params": {}},
        {"name": "CSS Selector", "params": {"css_selector": "article, .story-body, .post-content"}},
        {"name": "Clean Extraction", "params": {
            "exclude_tags": ['nav', 'header', 'footer', 'aside', 'menu', 'advertisement'],
            "remove_overlay_elements": True,
            "simulate_user": True
        }}
    ]
    
    async with AsyncWebCrawler() as crawler:
        for strategy in strategies:
            print(f"\n🔧 Strategy: {strategy['name']}")
            
            try:
                result = await crawler.arun(
                    url=test_url,
                    bypass_cache=True,
                    **strategy['params']
                )
                
                if result.success:
                    cleaned_len = len(result.cleaned_html) if result.cleaned_html else 0
                    markdown_len = len(result.markdown) if result.markdown else 0
                    
                    print(f"   📊 Cleaned HTML: {cleaned_len:,} chars")
                    print(f"   📊 Markdown: {markdown_len:,} chars")
                    
                    # Quick quality check
                    if result.cleaned_html:
                        html_tags = result.cleaned_html.count('<')
                        tag_ratio = html_tags / max(cleaned_len, 1) * 100
                        print(f"   📊 HTML tag ratio: {tag_ratio:.1f}%")
                        
                        if tag_ratio < 5:  # Less than 5% HTML tags
                            print(f"   ✅ Good content extraction")
                        else:
                            print(f"   ⚠️ High HTML content")
                else:
                    print(f"   ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 GOAL: Test Crawl4AI with current articles to find best approach")
    
    asyncio.run(test_with_current_articles())
    asyncio.run(test_extraction_strategies())
    
    print(f"\n💡 CONCLUSIONS:")
    print("1. Identify which content extraction method works best")
    print("2. Determine if cleaned_html vs markdown is better for articles")
    print("3. Update Scout Agent with proper extraction configuration")
    
    print("\n" + "="*60)
